"""Demand-driven planning: models state requirements, one fitter sizes shapes.

A model that implements ``CapacityModel.demand`` never sizes an instance
itself. It returns:

- ``Demand``: what the workload needs for one cluster scope (a zone, or the
  region for regional clusters), as intervals, independent of any instance.
- ``NodeRules``: the few constraints that depend on the node, such as memory
  reserved from node RAM, a cap on data per node, hard rejections, and the
  ``layout`` rule that turns need and supply into a node count and placement.

``fit`` sizes any shape against that demand. Because the same demand and
rules size every shape, the planner can ask a second question cheaply: which
other shapes can host the *same layout* as a chosen plan? Those are the
plan's alternatives, used as fallbacks when the primary type is unavailable.

CPU is measured in normalized compute units (NCU). One NCU is one core of
``default_reference_shape`` (2.3 GHz, IPC 1.0). A vCPU on another instance is
worth ``cpu_ghz * cpu_ipc_scale / 2.3`` NCU, the same conversion as
``normalize_cores``. An m6id.2xlarge (8 vCPU at 3.5 GHz, IPC 1.15) supplies
14 NCU; an m7a.2xlarge (8 vCPU at 3.7 GHz, IPC 1.95) supplies about 25.

``fit`` sizes and checks at the demand's ``mid``, so an alternative never
serves less than the plan targets. Models that do not state demand still get
alternatives: the planner plans them again on each shape at the same count.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

from service_capacity_modeling.enum_utils import StrEnum
from service_capacity_modeling.interface import AlternativeInstance
from service_capacity_modeling.interface import Bottleneck
from service_capacity_modeling.interface import default_reference_shape
from service_capacity_modeling.interface import Drive
from service_capacity_modeling.interface import DriveType
from service_capacity_modeling.interface import Excuse
from service_capacity_modeling.interface import Instance
from service_capacity_modeling.interface import Interval


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

# The resources in a fixed order, so the binding resource is deterministic.
DIMENSIONS: Tuple[Resource, ...] = (
    Resource.ncu,
    Resource.mem_gib,
    Resource.disk_gib,
    Resource.network_mbps,
)

Amounts = Dict[Resource, float]


@dataclass(frozen=True)
class Demand:
    """What one cluster scope needs, independent of any instance."""

    ncu: Interval
    mem_gib: Interval
    disk_gib: Interval
    network_mbps: Interval

    def mid(self) -> Amounts:
        """Demand per resource at each interval's ``mid``."""
        return {
            Resource.ncu: self.ncu.mid,
            Resource.mem_gib: self.mem_gib.mid,
            Resource.disk_gib: self.disk_gib.mid,
            Resource.network_mbps: self.network_mbps.mid,
        }


@dataclass(frozen=True)
class Layout:
    """How a cluster is laid out: the node count plus model-specific placement.

    ``params`` holds whatever a layout rule needs to check that another shape
    can host the same placement, and is copied into ``cluster_params``.
    """

    count: int
    params: Dict[str, Any] = field(default_factory=dict)
    # What each node actually carries when that is not demand / count, for
    # example a replicated partition placement. Omitted resources use
    # demand / count.
    need_per_node: Dict[Resource, float] = field(default_factory=dict)


# (instance, need, supply per node, pinned layout or None) -> layout or a
# number-free reason the shape cannot host it.
LayoutRule = Callable[
    [Instance, Amounts, Amounts, Optional[Layout]], Union[Layout, str]
]


def _no_reserve(instance: Instance) -> float:
    del instance
    return 0.0


def _accept_all(instance: Instance, drive: Drive) -> Optional[str]:
    del instance, drive
    return None


@dataclass(frozen=True)
class NodeRules:
    """Constraints that depend on the node, not on total demand.

    Keep this small. Every rule here is model knowledge the fitter applies to
    shapes the model never saw.
    """

    # Node count and placement from need and supply, or a pinned layout check.
    layout: LayoutRule
    # Memory the node uses before the workload sees any (OS, heap, sidecars).
    reserve_mem_gib: Callable[[Instance], float] = _no_reserve
    # Effective cap on data per node, buffer already applied.
    max_data_per_node_gib: float = math.inf
    # Hard rejections, returning a number-free reason.
    rejects: Callable[[Instance, Drive], Optional[str]] = _accept_all


def ncu_per_node(instance: Instance) -> float:
    """CPU an instance supplies, in reference cores."""
    reference = default_reference_shape
    speed = instance.cpu_ghz * instance.cpu_ipc_scale
    return instance.cpu * speed / (reference.cpu_ghz * reference.cpu_ipc_scale)


def cores_for_ncu(ncu: float, instance: Instance) -> int:
    """Whole vCPUs of ``instance`` that supply ``ncu``.

    Demand usually reaches NCU from whole cores of another shape, so the exact
    answer is often an integer that float division lands just above. The
    relative tolerance keeps that from rounding up a whole extra core.
    """
    cores = (
        ncu * default_reference_shape.cpu_ghz * default_reference_shape.cpu_ipc_scale
    )
    cores /= instance.cpu_ghz * instance.cpu_ipc_scale
    return max(1, math.ceil(cores * (1 - 1e-9)))


def supply(instance: Instance, drive: Drive, rules: NodeRules) -> Amounts:
    """What one node supplies to the workload on each resource."""
    if drive.drive_type == DriveType.attached_ssd:
        disk_gib = min(rules.max_data_per_node_gib, float(drive.max_scale_size_gib))
    else:
        local = instance.drive.size_gib if instance.drive is not None else 0
        disk_gib = min(rules.max_data_per_node_gib, local)
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
    layout: Layout
    binding: Resource
    need_per_node: Amounts
    supply_per_node: Amounts

    @property
    def count(self) -> int:
        return self.layout.count

    @property
    def annual_cost(self) -> float:
        cost = self.instance.annual_cost
        if self.drive.drive_type == DriveType.attached_ssd:
            # Capacity only; IOPS and throughput pricing are not modeled here.
            disk = self.need_per_node[Resource.disk_gib]
            cost += disk * self.drive.annual_cost_per_gib
        return cost * self.count


def fit(
    demand: Demand,
    rules: NodeRules,
    instance: Instance,
    drive: Drive,
    pinned: Optional[Layout] = None,
) -> Union[Fit, Excuse]:
    """Size ``instance`` for ``demand``, or explain why it cannot serve it.

    Without ``pinned``, returns the layout rule's smallest layout. With
    ``pinned``, checks that ``instance`` can host that exact layout, which is
    how a shape qualifies as an alternative.
    """
    reason = rules.rejects(instance, drive)
    if reason is not None:
        return Excuse(instance=instance.name, drive=drive.name, reason=reason)

    need = demand.mid()
    have = supply(instance, drive, rules)
    for dim in DIMENSIONS:
        if have[dim] <= 0 < need[dim]:
            return Excuse(
                instance=instance.name,
                drive=drive.name,
                reason=f"Supplies no {dim.value}",
                bottleneck=_BOTTLENECK.get(dim),
            )

    layout = rules.layout(instance, need, have, pinned)
    if isinstance(layout, str):
        short: Optional[Resource] = next(
            (d for d in DIMENSIONS if d.value in layout), None
        )
        return Excuse(
            instance=instance.name,
            drive=drive.name,
            reason=layout,
            context={
                "count": pinned.count if pinned else None,
                "supply_per_node": {d.value: have[d] for d in DIMENSIONS},
            },
            bottleneck=_BOTTLENECK.get(short) if short else None,
        )

    need_per_node = {
        d: layout.need_per_node.get(d, need[d] / layout.count) for d in DIMENSIONS
    }
    # A layout rule decides placement, but every node must still supply what it
    # carries on every resource, including ones the rule does not model.
    short = next(
        (d for d in DIMENSIONS if need_per_node[d] > have[d] * (1 + 1e-9)), None
    )
    if short is not None:
        return Excuse(
            instance=instance.name,
            drive=drive.name,
            reason=f"Cannot host the layout: {short.value}",
            context={
                "count": layout.count,
                "need_per_node": need_per_node[short],
                "supply_per_node": have[short],
            },
            bottleneck=_BOTTLENECK.get(short),
        )
    binding = max(
        DIMENSIONS,
        key=lambda d: need_per_node[d] / have[d] if have[d] > 0 else 0.0,
    )
    return Fit(
        instance=instance,
        drive=drive,
        layout=layout,
        binding=binding,
        need_per_node=need_per_node,
        supply_per_node=have,
    )


def fit_all(
    demand: Demand,
    rules: NodeRules,
    instances: Iterable[Instance],
    pinned: Optional[Layout] = None,
    drive: Optional[Drive] = None,
    *,
    key: Callable[[Fit], Any] = lambda f: f.annual_cost,
) -> Tuple[List[Fit], List[Excuse]]:
    """Fit every instance; fits are ordered by ``key``, cheapest first by
    default.

    This is the entry point for callers that choose shapes themselves, for
    example an autoscaler that holds the node count fixed and picks a type for
    a target NCU with its own ranking. ``drive`` is the attached drive;
    without it each instance uses its own local drive and instances without
    one are skipped.
    """
    fits: List[Fit] = []
    excuses: List[Excuse] = []
    for instance in instances:
        node_drive = drive or instance.drive
        if node_drive is None:
            continue
        result = fit(demand, rules, instance, node_drive, pinned=pinned)
        if isinstance(result, Excuse):
            excuses.append(result)
        else:
            fits.append(result)
    fits.sort(key=key)
    return fits, excuses


def to_alternative(result: Fit) -> AlternativeInstance:
    """The planner-facing record of an alternative."""
    return AlternativeInstance(
        instance=result.instance.name,
        count=result.count,
        annual_cost=result.annual_cost,
        binding=result.binding.value,
        need_per_node={d.value: v for d, v in result.need_per_node.items()},
        supply_per_node={d.value: v for d, v in result.supply_per_node.items()},
    )
