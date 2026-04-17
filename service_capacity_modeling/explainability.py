"""Explainability types for the capacity planner.

**Experimental** — this API may change.

This module contains the family graph (FamilyTrait, FamilyEdge, FamilyGraph),
ExplainedPlans, and ExplainedUncertainPlans — types used to explain *why*
the planner rejected certain instance/drive combinations and what
alternatives exist.

Core contract types (Bottleneck, Excuse) live in interface.py because they
are part of the CapacityModel.capacity_plan() return type.

Consumer usage::

    from service_capacity_modeling.capacity_planner import planner
    from service_capacity_modeling.models.plan_comparison import compare_plans

    # Rejection explanations + family graph
    explained = planner.plan_certain_explained(
        model_name="org.netflix.cassandra",
        region="us-east-1",
        desires=desires,
        extra_model_arguments=extra,
    )

    # Current-vs-recommended comparison (separate concern)
    baseline = planner.extract_baseline_plan(
        model_name="org.netflix.cassandra",
        region="us-east-1",
        desires=desires,
        extra_model_arguments=extra,
    )
    comparison = compare_plans(baseline, explained.plans[0])

    # Serialize both for downstream consumers
    explained.model_dump_json()
    comparison.model_dump_json()

    # Uncertain (stochastic) explained mode
    explained_uncertain = planner.plan_explained(
        model_name="org.netflix.cassandra",
        region="us-east-1",
        desires=desires,
        extra_model_arguments=extra,
    )
    explained_uncertain.plan          # UncertainCapacityPlan
    explained_uncertain.excuses       # deduped across all simulations
    explained_uncertain.family_graph  # hardware trade-off graph
"""

from __future__ import annotations

import re
from typing import Any
from typing import Dict
from typing import FrozenSet
from typing import List
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple

from service_capacity_modeling.interface import Bottleneck
from service_capacity_modeling.interface import CapacityPlan
from service_capacity_modeling.interface import CountedExcuse
from service_capacity_modeling.interface import DriveType
from service_capacity_modeling.interface import ExcludeUnsetModel
from service_capacity_modeling.interface import Excuse
from service_capacity_modeling.interface import ExcuseTag
from service_capacity_modeling.interface import Hardware
from service_capacity_modeling.interface import Instance
from service_capacity_modeling.interface import RegretPlanSummary
from service_capacity_modeling.interface import UncertainCapacityPlan
from service_capacity_modeling.interface import WorldRef


class FamilyTrait(ExcludeUnsetModel):
    """Intrinsic hardware properties of an instance family.

    All numeric values are derived from hardware shapes data via
    from_instance(). Within an AWS family, ratios (ram/vcpu, disk/vcpu,
    cost/vcpu) are constant across instance sizes due to linear pricing.
    """

    family: str
    memory_gib_per_vcpu: float
    has_local_disk: bool
    local_disk_gib_per_vcpu: Optional[float] = None
    drive_type: Optional[DriveType] = None
    cost_per_vcpu_annual: Optional[float] = None
    """Annual cost per vCPU derived from loaded pricing (may be internal)."""

    @classmethod
    def from_instance(cls, instance: Instance) -> FamilyTrait:
        """Derive family traits from any instance in the family.

        Since ratios are constant within a family, any size works.
        Cost uses whatever pricing is loaded (public or internal).
        """
        drive = instance.drive
        has_local = drive is not None
        return cls(
            family=instance.family,
            memory_gib_per_vcpu=round(instance.ram_gib / instance.cpu, 2),
            has_local_disk=has_local,
            local_disk_gib_per_vcpu=(
                round(drive.size_gib / instance.cpu, 1)
                if drive is not None and drive.size_gib > 0
                else None
            ),
            drive_type=drive.drive_type if drive is not None else None,
            cost_per_vcpu_annual=(
                round(instance.annual_cost / instance.cpu, 2)
                if instance.annual_cost and instance.cpu
                else None
            ),
        )


class FamilyEdge(ExcludeUnsetModel):
    """A directed trade-off edge between two instance families.

    Edges encode hardware topology: what improves and what degrades
    when switching from one family to another. All attributes are
    derived at runtime from FamilyTrait hardware data by FamilyGraph.build().
    """

    from_family: str
    to_family: str
    trade_off: Optional[str] = None
    improves: List[Bottleneck] = []
    degrades: List[Bottleneck] = []


def _family_generation(family: str) -> Optional[int]:
    """Extract generation number from AWS family name (e.g. 'r7a' → 7)."""
    m = re.search(r"\d+", family)
    return int(m.group()) if m else None


def _derive_edge_attributes(  # noqa: C901  # pylint: disable=too-many-branches
    from_trait: FamilyTrait,
    to_trait: FamilyTrait,
) -> Tuple[List[Bottleneck], List[Bottleneck]]:
    """Derive improves/degrades for a family pair from their hardware traits."""
    improves: List[Bottleneck] = []
    degrades: List[Bottleneck] = []

    # cost — from loaded pricing (may be internal)
    if from_trait.cost_per_vcpu_annual and to_trait.cost_per_vcpu_annual:
        if to_trait.cost_per_vcpu_annual < from_trait.cost_per_vcpu_annual:
            improves.append(Bottleneck.cost)
        elif to_trait.cost_per_vcpu_annual > from_trait.cost_per_vcpu_annual:
            degrades.append(Bottleneck.cost)

    # memory
    if to_trait.memory_gib_per_vcpu > from_trait.memory_gib_per_vcpu:
        improves.append(Bottleneck.memory)
    elif to_trait.memory_gib_per_vcpu < from_trait.memory_gib_per_vcpu:
        degrades.append(Bottleneck.memory)

    # disk_capacity — local vs EBS
    if from_trait.has_local_disk and to_trait.has_local_disk:
        from_disk = from_trait.local_disk_gib_per_vcpu or 0
        to_disk = to_trait.local_disk_gib_per_vcpu or 0
        if to_disk > from_disk:
            improves.append(Bottleneck.disk_capacity)
        elif to_disk < from_disk:
            degrades.append(Bottleneck.disk_capacity)
    elif from_trait.has_local_disk and not to_trait.has_local_disk:
        improves.append(Bottleneck.disk_capacity)  # EBS: flexible sizing
    elif not from_trait.has_local_disk and to_trait.has_local_disk:
        degrades.append(Bottleneck.disk_capacity)  # local: fixed size

    # disk_iops — local NVMe vs EBS is a qualitative change (latency curve,
    # not just peak IOPS). EBS-to-EBS and local-to-local are omitted: max
    # IOPS is rarely the bottleneck and latency curves are modeled identically.
    if from_trait.has_local_disk and not to_trait.has_local_disk:
        degrades.append(Bottleneck.disk_iops)
    elif not from_trait.has_local_disk and to_trait.has_local_disk:
        improves.append(Bottleneck.disk_iops)

    # generation — derived from family name (r7a=7, r6a=6, i4i=4, ...)
    from_gen = _family_generation(from_trait.family)
    to_gen = _family_generation(to_trait.family)
    if from_gen is not None and to_gen is not None:
        if to_gen > from_gen:
            improves.append(Bottleneck.generation)
        elif to_gen < from_gen:
            degrades.append(Bottleneck.generation)

    return improves, degrades


class FamilyGraph(ExcludeUnsetModel):
    """Soft DAG of instance family trade-off relationships.

    Nodes are FamilyTraits (derived from hardware shapes).
    Edges are FamilyEdges (derived from trait comparisons).
    """

    traits: Dict[str, FamilyTrait] = {}
    edges: List[FamilyEdge] = []

    def suggest_alternatives(self, excuse: Excuse) -> List[FamilyEdge]:
        """Return edges to families that improve the excuse's bottleneck."""
        if not excuse.bottleneck:
            return []
        excuse_family = excuse.instance.rsplit(".", 1)[0]
        return [
            e
            for e in self.edges
            if e.from_family == excuse_family and excuse.bottleneck in e.improves
        ]

    @classmethod
    def build(
        cls,
        excuses: Sequence[Excuse],
        hardware: Hardware,
        preferred_families: Optional[FrozenSet[str]],
    ) -> FamilyGraph:
        """Build an M×N FamilyGraph from derived hardware traits.

        All edge attributes (cost, memory, disk_capacity, disk_iops, generation)
        are derived at runtime from FamilyTrait values — no hardcoded trade-off
        tables. The current cluster's family is always included even if it falls
        outside preferred_families, so consumers always see why their current
        shape was rejected.

        The graph is always populated from preferred_families. If preferred_families
        is None (model has no declared preference) the base is empty, so only the
        current cluster's family appears as a node (if current_clusters is set).
        This prevents stateless models from inheriting storage-optimized families.
        """
        base = preferred_families if preferred_families is not None else frozenset()
        current_shape_families: Set[str] = {
            e.instance.rsplit(".", 1)[0]
            for e in excuses
            if ExcuseTag.current_shape in e.tags
        }
        included = base | current_shape_families

        # Index one instance per family — O(M) single pass
        family_first: Dict[str, Any] = {}
        for inst in hardware.instances.values():
            fam = inst.family
            if fam in included and fam not in family_first:
                family_first[fam] = inst

        traits = {
            fam: FamilyTrait.from_instance(family_first[fam])
            for fam in included
            if fam in family_first
        }

        # M×N directed edges — all pairs, attributes fully derived
        edges: List[FamilyEdge] = []
        for from_fam, from_trait in traits.items():
            for to_fam, to_trait in traits.items():
                if from_fam == to_fam:
                    continue
                improves, degrades = _derive_edge_attributes(from_trait, to_trait)
                edges.append(
                    FamilyEdge(
                        from_family=from_fam,
                        to_family=to_fam,
                        improves=improves,
                        degrades=degrades,
                    )
                )

        return cls(traits=traits, edges=edges)


# Default preferred family set for stateful datastores (Cassandra, Kafka, EVCache).
# Covers the full decision space: one representative per
# {memory-class × storage-class × generation-tier}.
# Not appropriate for stateless services (DGW, Java apps) — those models
# should override preferred_families() with compute/general families only.
# "n"-suffix (enhanced-network) families are intentionally excluded.
STATEFUL_DATASTORE_FAMILIES: FrozenSet[str] = frozenset(
    {
        "c6a",
        "c7a",  # compute-optimized EBS (~1.9 GiB/vCPU)
        "m6a",
        "m7a",  # general-purpose EBS (~3.8 GiB/vCPU)
        "m6id",  # general-purpose local NVMe (~3.8 GiB/vCPU)
        "r6a",
        "r7a",  # memory-optimized EBS (~7.6 GiB/vCPU)
        "r5d",
        "r6id",  # memory-optimized local NVMe (~7.6–8.0 GiB/vCPU)
        "i4i",
        "i3en",  # storage-optimized local NVMe
    }
)

# Preferred family set for stateless services (DGW, Java apps, NodeQuark).
# EBS-only: no local NVMe (storage density is irrelevant), no storage-optimized
# families (i4i/i3en). Covers the compute × memory × generation axes.
# Models override preferred_families() to return this set.
STATELESS_SERVICE_FAMILIES: FrozenSet[str] = frozenset(
    {
        "c6a",
        "c7a",  # compute-optimized EBS (~1.9 GiB/vCPU) — CPU-bound services
        "m6a",
        "m7a",  # general-purpose EBS (~3.8 GiB/vCPU) — balanced workloads
        "r6a",
        "r7a",  # memory-optimized EBS (~7.6 GiB/vCPU) — heap-heavy services
    }
)


def deduplicate_excuses(excuses: Sequence[Excuse]) -> Sequence[Excuse]:
    """Deduplicate excuses by (instance, drive, reason) across simulations."""
    seen: Set[Tuple[str, str, str, Optional[Bottleneck], Tuple[str, ...]]] = set()
    result_by_key: Dict[
        Tuple[str, str, str, Optional[Bottleneck], Tuple[str, ...]], Excuse
    ] = {}
    ordered_keys: List[Tuple[str, str, str, Optional[Bottleneck], Tuple[str, ...]]] = []
    for exc in excuses:
        key = (
            exc.instance,
            exc.drive,
            exc.reason,
            exc.bottleneck,
            tuple(sorted(tag.value for tag in exc.tags)),
        )
        if key not in seen:
            seen.add(key)
            result_by_key[key] = exc.model_copy(deep=True)
            ordered_keys.append(key)
            continue
        current = result_by_key[key]
        if current.context and exc.context and current.context != exc.context:
            current.context = {}
    return [result_by_key[key] for key in ordered_keys]


def count_excuses(excuses: Sequence[Excuse]) -> Sequence[CountedExcuse]:
    """Count excuse frequency across simulations without world provenance."""
    return count_world_excuses(
        [
            (WorldRef(world_id=f"w-local-{index}", world_label="local"), exc)
            for index, exc in enumerate(excuses)
        ]
    )


def count_world_excuses(
    world_excuses: Sequence[Tuple[WorldRef, Excuse]],
) -> Sequence[CountedExcuse]:
    """Count excuses across worlds and retain bounded example world refs."""
    keys_in_order: List[
        Tuple[str, str, str, Optional[Bottleneck], Tuple[str, ...]]
    ] = []
    counted: Dict[
        Tuple[str, str, str, Optional[Bottleneck], Tuple[str, ...]], CountedExcuse
    ] = {}
    worlds_seen: Dict[
        Tuple[str, str, str, Optional[Bottleneck], Tuple[str, ...]], Set[str]
    ] = {}
    max_example_worlds = 3

    for world, exc in world_excuses:
        key = (
            exc.instance,
            exc.drive,
            exc.reason,
            exc.bottleneck,
            tuple(sorted(tag.value for tag in exc.tags)),
        )
        if key not in counted:
            counted[key] = CountedExcuse(
                **exc.model_dump(),
                occurrence_count=0,
                world_count=0,
                example_worlds=[],
            )
            worlds_seen[key] = set()
            keys_in_order.append(key)
        current = counted[key]
        current.occurrence_count += 1
        if world.world_id not in worlds_seen[key]:
            worlds_seen[key].add(world.world_id)
            current.world_count += 1
            if len(current.example_worlds) < max_example_worlds:
                current.example_worlds = [*current.example_worlds, world]
        if current.context and exc.context and current.context != exc.context:
            current.context = {}

    return [counted[key] for key in keys_in_order]


class ExplainedPlans(ExcludeUnsetModel):
    """Plans + excuses + family context.

    Structured data for programmatic consumers. Serialize with
    .model_dump() / .model_dump_json().
    """

    plans: Sequence[CapacityPlan]
    excuses: Sequence[Excuse] = []
    family_graph: FamilyGraph = FamilyGraph()


class ExplainedUncertainPlans(ExcludeUnsetModel):
    """Uncertain plans + excuses + family context.

    Mirrors ExplainedPlans but wraps UncertainCapacityPlan instead
    of deterministic plans. Returned by plan_explained().
    """

    plan: UncertainCapacityPlan
    excuses: Sequence[Excuse] = []
    excuse_summary: Sequence[CountedExcuse] = []
    family_graph: FamilyGraph = FamilyGraph()
    least_regret_summaries: Sequence[RegretPlanSummary] = []
