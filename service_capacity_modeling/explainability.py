"""Explainability types for the capacity planner.

**Experimental** — this API may change.

This module contains the family graph (FamilyTrait, FamilyEdge, FamilyGraph)
and ExplainedPlans — types used to explain *why* the planner rejected
certain instance/drive combinations and what alternatives exist.

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
"""

from __future__ import annotations

from typing import Dict
from typing import FrozenSet
from typing import List
from typing import Optional
from typing import Sequence

from service_capacity_modeling.interface import Bottleneck
from service_capacity_modeling.interface import CapacityPlan
from service_capacity_modeling.interface import DriveType
from service_capacity_modeling.interface import ExcludeUnsetModel
from service_capacity_modeling.interface import Excuse
from service_capacity_modeling.interface import Instance


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
    when switching from one family to another. Human-authored for
    non-derivable facts (disk type, generation, IOPS).

    Cost is intentionally excluded — it is derived at runtime from
    loaded pricing (which may be internal) and annotated onto edges
    by _build_family_graph() in capacity_planner.py.
    """

    from_family: str
    to_family: str
    trade_off: Optional[str] = None
    improves: List[Bottleneck] = []
    degrades: List[Bottleneck] = []


class FamilyGraph(ExcludeUnsetModel):
    """Soft DAG of instance family trade-off relationships.

    Nodes are FamilyTraits (derived from hardware shapes).
    Edges are FamilyEdges (authored domain knowledge).
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


# Library-level default family set.  Models override via preferred_families().
# One representative per {memory-class × storage-class × generation-tier};
# "n"-suffix (enhanced-network) families are intentionally excluded.
KNOWN_DATASTORE_FAMILIES: FrozenSet[str] = frozenset(
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


class ExplainedPlans(ExcludeUnsetModel):
    """Plans + excuses + family context.

    Structured data for programmatic consumers. Serialize with
    .model_dump() / .model_dump_json().
    """

    plans: Sequence[CapacityPlan]
    excuses: Sequence[Excuse] = []
    family_graph: FamilyGraph = FamilyGraph()
