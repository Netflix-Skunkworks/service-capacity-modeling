"""Demand-driven planning: models state requirements, one fitter picks shapes.

Sketch of an alternative architecture. Today the planner calls every model
once per (instance, drive), and each model sizes that one shape itself. Here a
model never sees an instance:

1. ``DemandModel.demand(desires, drive)`` returns a per-zone ``Demand`` (what
   the workload needs, as intervals) and ``NodeRules`` (the few constraints
   that depend on the node itself).
2. ``fit`` turns demand into a node count for any shape: the count is set by
   the most constrained resource (the "binding" dimension).
3. ``plan`` fits every shape, picks the cheapest as the primary, and lists
   alternatives: other shapes that can stand in for a primary node at the
   same count.

CPU is measured in normalized compute units (NCU). One NCU is one core of
``default_reference_shape`` (2.3 GHz, IPC 1.0). A vCPU on another instance is
worth ``cpu_ghz * cpu_ipc_scale / 2.3`` NCU, the same conversion as
``normalize_cores``. An m6id.2xlarge (8 vCPU at 3.5 GHz, IPC 1.15) supplies
14 NCU; an m7a.2xlarge (8 vCPU at 3.7 GHz, IPC 1.95) supplies about 25.

Demand intervals carry two meanings. ``mid`` sizes the primary. ``low`` is
the floor every node must meet, so any shape at or above ``low / count`` on
every dimension can replace a primary node, whether as an ICE fallback, in a
temporarily mixed fleet, or as a container request. A wider gap between
``low`` and ``mid`` allows weaker fallbacks and costs more primary capacity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from dataclasses import field
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Protocol
from typing import Sequence
from typing import Tuple
from typing import Union

from service_capacity_modeling.interface import Bottleneck
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import default_reference_shape
from service_capacity_modeling.interface import Drive
from service_capacity_modeling.interface import DriveType
from service_capacity_modeling.interface import Excuse
from service_capacity_modeling.interface import Instance
from service_capacity_modeling.interface import Interval
from service_capacity_modeling.enum_utils import StrEnum


class Resource(StrEnum):
    """A resource a node supplies and a workload demands."""

    ncu = "ncu"
    mem_gib = "mem_gib"
    disk_gib = "disk_gib"
    network_mbps = "network_mbps"


# Excuses carry the existing Bottleneck enum, which has no network value.
_BOTTLENECK: Dict[Resource, Bottleneck] = {
    Resource.ncu: Bottleneck.cpu,
    Resource.mem_gib: Bottleneck.memory,
    Resource.disk_gib: Bottleneck.disk_capacity,
}

# The resources in a fixed order, so the binding dimension is deterministic.
DIMENSIONS: Tuple[Resource, ...] = (
    Resource.ncu,
    Resource.mem_gib,
    Resource.disk_gib,
    Resource.network_mbps,
)


@dataclass(frozen=True)
class Demand:
    """What one zone of the workload needs, independent of any instance."""

    ncu: Interval
    mem_gib: Interval
    disk_gib: Interval
    network_mbps: Interval

    def at(self, point: str) -> Dict[Resource, float]:
        """Demand per dimension at ``point`` ("low", "mid" or "high")."""
        return {
            Resource.ncu: getattr(self.ncu, point),
            Resource.mem_gib: getattr(self.mem_gib, point),
            Resource.disk_gib: getattr(self.disk_gib, point),
            Resource.network_mbps: getattr(self.network_mbps, point),
        }


def _no_reserve(instance: Instance) -> float:
    del instance
    return 0.0


def _any_count(count: int) -> int:
    return count


def _accept_all(instance: Instance, drive: Drive) -> Optional[str]:
    del instance, drive
    return None


@dataclass(frozen=True)
class NodeRules:
    """Constraints that depend on the node, not on total demand.

    Keep this small. Every rule here is model knowledge the fitter must apply
    to shapes the model never saw. If a model needs many rules, re-running the
    model per shape is the better design for it.
    """

    # Memory the node uses before the workload sees any, e.g. heap, OS and
    # sidecars. A function of the node because heap is often sized from RAM.
    reserve_mem_gib: Callable[[Instance], float] = _no_reserve
    # Cap on data per node, e.g. to bound streaming time when a node is replaced.
    max_data_per_node_gib: float = math.inf
    # Round the raw count, e.g. to an even number or a power of two.
    cluster_size: Callable[[int], int] = _any_count
    min_count: int = 1
    # Hard rejections, returning a reason without numbers, e.g. a RAM floor.
    rejects: Callable[[Instance, Drive], Optional[str]] = _accept_all


class DemandModel(Protocol):
    """What a model implements in this design instead of ``capacity_plan``."""

    def demand(
        self, desires: CapacityDesires, drive: Drive
    ) -> Optional[Tuple[Demand, NodeRules]]:
        """Per-zone demand on this kind of drive, or None if it cannot run there.

        Demand depends on the drive, not the instance: working sets that must
        stay in RAM shrink as drive read latency falls.
        """


def ncu_per_node(instance: Instance) -> float:
    """CPU an instance supplies, in reference cores."""
    reference = default_reference_shape
    speed = instance.cpu_ghz * instance.cpu_ipc_scale
    return instance.cpu * speed / (reference.cpu_ghz * reference.cpu_ipc_scale)


def supply(instance: Instance, drive: Drive, rules: NodeRules) -> Dict[Resource, float]:
    """What one node supplies to the workload on each dimension."""
    if drive.drive_type == DriveType.attached_ssd:
        # Attached volumes grow on demand, so only the data cap and the
        # volume's own limit bound capacity.
        disk_gib = min(rules.max_data_per_node_gib, float(drive.max_scale_size_gib))
    else:
        local = instance.drive.size_gib if instance.drive is not None else 0
        disk_gib = min(rules.max_data_per_node_gib, float(local))
    return {
        Resource.ncu: ncu_per_node(instance),
        Resource.mem_gib: max(0.0, instance.ram_gib - rules.reserve_mem_gib(instance)),
        Resource.disk_gib: disk_gib,
        Resource.network_mbps: instance.net_mbps,
    }


@dataclass(frozen=True)
class Fit:
    """A shape that meets demand, with what each node needs and supplies."""

    instance: Instance
    drive: Drive
    count: int
    binding: Resource
    need_per_node: Dict[Resource, float]
    supply_per_node: Dict[Resource, float]
    annual_cost: float


def _annual_cost(
    instance: Instance, drive: Drive, count: int, disk_per_node_gib: float
) -> float:
    cost = instance.annual_cost
    if drive.drive_type == DriveType.attached_ssd:
        # Capacity only. IOPS and throughput pricing would come from NodeRules.
        cost += disk_per_node_gib * drive.annual_cost_per_gib
    return cost * count


def fit(
    demand: Demand,
    rules: NodeRules,
    instance: Instance,
    drive: Drive,
    count: Optional[int] = None,
) -> Union[Fit, Excuse]:
    """Size ``instance`` for ``demand``, or explain why it cannot serve it.

    With ``count`` unset, returns the smallest count that meets ``mid`` demand.
    With ``count`` set, checks whether that many nodes meet ``low`` demand,
    which is how a shape qualifies as an alternative at the primary's count.
    """
    name, drive_name = instance.name, drive.name
    reason = rules.rejects(instance, drive)
    if reason is not None:
        return Excuse(instance=name, drive=drive_name, reason=reason)

    have = supply(instance, drive, rules)
    for dim in DIMENSIONS:
        if have[dim] <= 0:
            return Excuse(
                instance=name,
                drive=drive_name,
                reason=f"Supplies no {dim.value}",
                bottleneck=_BOTTLENECK.get(dim),
            )

    if count is None:
        need = demand.at("mid")
        by_dim = {dim: math.ceil(need[dim] / have[dim]) for dim in DIMENSIONS}
        raw = max(by_dim.values())
        count = max(rules.cluster_size(raw), rules.min_count)
        binding = max(DIMENSIONS, key=lambda d: by_dim[d])
    else:
        need = demand.at("low")
        short = [d for d in DIMENSIONS if have[d] * count < need[d]]
        if short:
            return Excuse(
                instance=name,
                drive=drive_name,
                reason=f"Below the per-node {short[0].value} floor",
                context={
                    "count": count,
                    "need_per_node": need[short[0]] / count,
                    "supply_per_node": have[short[0]],
                },
                bottleneck=_BOTTLENECK.get(short[0]),
            )
        binding = max(DIMENSIONS, key=lambda d: need[d] / have[d])

    need_per_node = {dim: need[dim] / count for dim in DIMENSIONS}
    return Fit(
        instance=instance,
        drive=drive,
        count=count,
        binding=binding,
        need_per_node=need_per_node,
        supply_per_node=have,
        annual_cost=_annual_cost(
            instance, drive, count, need_per_node[Resource.disk_gib]
        ),
    )


@dataclass(frozen=True)
class DemandPlan:
    """The primary shape, the shapes that can stand in for it, and why others can't."""

    demand: Demand
    primary: Fit
    alternatives: List[Fit] = field(default_factory=list)
    excuses: List[Excuse] = field(default_factory=list)


def _compatible(instance: Instance, drive: Drive) -> bool:
    if drive.drive_type == DriveType.attached_ssd:
        return (
            not drive.compatible_families
            or instance.family in drive.compatible_families
        )
    return instance.drive is not None and instance.drive.name == drive.name


def plan(
    model: DemandModel,
    desires: CapacityDesires,
    instances: Iterable[Instance],
    drives: Sequence[Drive],
    max_alternatives: int = 3,
) -> Optional[DemandPlan]:
    """Fit every shape, pick the cheapest as primary, then list alternatives.

    Demand is computed once per drive, not once per instance. Alternatives
    share the primary's drive and count, are at most one per family, and are
    ordered by cost.
    """
    instances = list(instances)
    excuses: List[Excuse] = []
    best = _cheapest(model, desires, instances, drives, excuses)
    if best is None:
        return None
    primary, demand, rules = best
    alternatives = _alternatives(primary, demand, rules, instances, excuses)
    return DemandPlan(
        demand=demand,
        primary=primary,
        alternatives=alternatives[:max_alternatives],
        excuses=excuses,
    )


def _cheapest(
    model: DemandModel,
    desires: CapacityDesires,
    instances: List[Instance],
    drives: Sequence[Drive],
    excuses: List[Excuse],
) -> Optional[Tuple[Fit, Demand, NodeRules]]:
    """Cheapest fit across every drive and instance, with the demand behind it."""
    best: Optional[Tuple[Fit, Demand, NodeRules]] = None
    for drive in drives:
        stated = model.demand(desires, drive)
        if stated is None:
            continue
        demand, rules = stated
        for instance in instances:
            if not _compatible(instance, drive):
                continue
            result = fit(demand, rules, instance, drive)
            if isinstance(result, Excuse):
                excuses.append(result)
            elif best is None or result.annual_cost < best[0].annual_cost:
                best = (result, demand, rules)
    return best


def _alternatives(
    primary: Fit,
    demand: Demand,
    rules: NodeRules,
    instances: List[Instance],
    excuses: List[Excuse],
) -> List[Fit]:
    """Other families that meet the floor at the primary's drive and count."""
    alternatives: Dict[str, Fit] = {}
    for instance in instances:
        if instance.family == primary.instance.family:
            continue
        if not _compatible(instance, primary.drive):
            continue
        result = fit(demand, rules, instance, primary.drive, count=primary.count)
        if isinstance(result, Excuse):
            excuses.append(result)
            continue
        kept = alternatives.get(instance.family)
        if kept is None or result.annual_cost < kept.annual_cost:
            alternatives[instance.family] = result

    return sorted(alternatives.values(), key=lambda f: f.annual_cost)
