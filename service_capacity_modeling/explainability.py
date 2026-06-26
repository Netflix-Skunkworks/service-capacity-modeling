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
    explained_uncertain.excuses_by_model  # deduped across all simulations
    explained_uncertain.family_graph  # hardware trade-off graph
"""

from __future__ import annotations

import json
import re
from typing import Any
from typing import Dict
from typing import FrozenSet
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple

from pydantic import Field

from service_capacity_modeling.interface import Bottleneck
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import CapacityPlan
from service_capacity_modeling.interface import CapacityRegretParameters
from service_capacity_modeling.interface import CapacityRequirement
from service_capacity_modeling.interface import ExcuseSummary
from service_capacity_modeling.interface import DriveType
from service_capacity_modeling.interface import ExcludeUnsetModel
from service_capacity_modeling.interface import Excuse
from service_capacity_modeling.interface import ExcuseTag
from service_capacity_modeling.interface import Hardware
from service_capacity_modeling.interface import Instance
from service_capacity_modeling.interface import Interval
from service_capacity_modeling.interface import RegretPlanSummary
from service_capacity_modeling.interface import UncertainCapacityPlan
from service_capacity_modeling.interface import SampleRef
from service_capacity_modeling.models import CapacityModel
from service_capacity_modeling.models.common import merge_plan


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


# Return type of Excuse.dedupe_key(); see that method for why these fields.
_ExcuseKey = Tuple[str, str, str, Optional[Bottleneck], Tuple[str, ...]]


class SampledExcuse(NamedTuple):
    source_model: str
    model_sample: SampleRef
    excuse: Excuse


def deduplicate_excuses(excuses: Sequence[Excuse]) -> Sequence[Excuse]:
    """Deduplicate excuses by (instance, drive, reason, bottleneck, tags).

    The dedup key includes bottleneck and tags so excuses with the same
    instance/drive/reason but different bottleneck or tag sets are preserved
    as separate entries. When duplicate excuses have conflicting context
    dicts, the context is cleared.
    """
    seen: Set[_ExcuseKey] = set()
    result_by_key: Dict[_ExcuseKey, Excuse] = {}
    ordered_keys: List[_ExcuseKey] = []
    for exc in excuses:
        key = exc.dedupe_key()
        if key not in seen:
            seen.add(key)
            result_by_key[key] = exc.model_copy(deep=True)
            ordered_keys.append(key)
            continue
        current = result_by_key[key]
        # Sample-specific numbers can disagree. Keep the excuse, drop arbitrary detail.
        if current.context != exc.context:
            current.context = {}
    return [result_by_key[key] for key in ordered_keys]


def count_sample_excuses(
    sample_excuses: Sequence[SampledExcuse],
) -> Sequence[ExcuseSummary]:
    """Count excuses across samples and retain bounded example sample refs."""
    keys_in_order: List[Tuple[str, _ExcuseKey]] = []
    counted: Dict[Tuple[str, _ExcuseKey], ExcuseSummary] = {}
    samples_seen: Dict[Tuple[str, _ExcuseKey], Set[str]] = {}
    max_example_samples = 3

    for sampled_excuse in sample_excuses:
        exc = sampled_excuse.excuse
        key = (sampled_excuse.source_model, exc.dedupe_key())
        if key not in counted:
            counted[key] = ExcuseSummary(
                **exc.model_dump(),
                source_model=sampled_excuse.source_model,
                occurrence_count=0,
                sample_count=0,
                example_samples=[],
            )
            samples_seen[key] = set()
            keys_in_order.append(key)
        current = counted[key]
        current.occurrence_count += 1
        sample = sampled_excuse.model_sample
        if sample.sample_id not in samples_seen[key]:
            # Count sampled inputs separately from raw repeated excuse events.
            samples_seen[key].add(sample.sample_id)
            current.sample_count += 1
            if len(current.example_samples) < max_example_samples:
                current.example_samples = [*current.example_samples, sample]
        # Sample-specific numbers can disagree. Keep the excuse, drop arbitrary detail.
        if current.context != exc.context:
            current.context = {}

    return [counted[key] for key in keys_in_order]


###############################################################################
#                    Uncertain Planner Regret Explainability                  #
###############################################################################


class SampledPlan(ExcludeUnsetModel):
    sample: SampleRef
    desires: CapacityDesires
    plan: CapacityPlan


class RegretCandidate(ExcludeUnsetModel):
    """Internal regret record for one candidate sampled from one input."""

    sample: SampleRef
    plan: CapacityPlan
    desires: CapacityDesires
    total_regret: float
    regret_components: Dict[str, float] = Field(default_factory=dict)


class MergedRegretCandidate(ExcludeUnsetModel):
    """Merged plan plus regret breadcrumbs across composed models."""

    samples: Sequence[SampleRef] = Field(default_factory=list)
    plan: CapacityPlan
    total_regret: float
    regret_components_by_model: Dict[str, Dict[str, float]] = Field(
        default_factory=dict
    )


def regret_detailed(
    capacity_plans: Sequence[SampledPlan],
    regret_params: CapacityRegretParameters,
    model: CapacityModel,
) -> Sequence[RegretCandidate]:
    """Return per-candidate regret totals plus per-component totals."""
    plans_by_regret: List[RegretCandidate] = []

    for proposed_sample in capacity_plans:
        total_regret = 0.0
        component_totals: Dict[str, float] = {}
        for optimal_sample in capacity_plans:
            components = model.regret(
                regret_params=regret_params,
                optimal_plan=optimal_sample.plan,
                proposed_plan=proposed_sample.plan,
            )
            total_regret += sum(components.values())
            for component, value in components.items():
                component_totals[component] = (
                    component_totals.get(component, 0.0) + value
                )

        plans_by_regret.append(
            RegretCandidate(
                sample=proposed_sample.sample,
                plan=proposed_sample.plan,
                desires=proposed_sample.desires,
                total_regret=total_regret,
                regret_components=dict(sorted(component_totals.items())),
            )
        )

    plans_by_regret.sort(key=lambda candidate: candidate.total_regret)
    return plans_by_regret


def _mean_component_maps(
    component_maps: Sequence[Dict[str, float]],
) -> Dict[str, float]:
    if not component_maps:
        return {}
    totals: Dict[str, float] = {}
    for component_map in component_maps:
        for component, value in component_map.items():
            totals[component] = totals.get(component, 0.0) + value
    count = float(len(component_maps))
    return {component: value / count for component, value in sorted(totals.items())}


class _RegretSummaryAccumulator(ExcludeUnsetModel):
    plan: CapacityPlan
    sample_count: int = 0
    sum_total_regret: float = 0.0
    example_samples: List[SampleRef] = Field(default_factory=list)
    regret_components_by_model_samples: Dict[str, List[Dict[str, float]]] = Field(
        default_factory=dict
    )

    def add(self, candidate: MergedRegretCandidate) -> None:
        self.sample_count += 1
        self.sum_total_regret += candidate.total_regret

        for candidate_sample in candidate.samples:
            if len(self.example_samples) >= 3:
                break
            if all(
                sample.sample_id != candidate_sample.sample_id
                for sample in self.example_samples
            ):
                self.example_samples.append(candidate_sample)

        for model_name, components in candidate.regret_components_by_model.items():
            samples = self.regret_components_by_model_samples.setdefault(model_name, [])
            samples.append(dict(components))

    def to_summary(self) -> RegretPlanSummary:
        if self.sample_count == 0:
            raise RuntimeError("Cannot summarize regret without any sampled inputs")

        mean_by_model = {
            model_name: _mean_component_maps(samples)
            for model_name, samples in self.regret_components_by_model_samples.items()
        }

        return RegretPlanSummary(
            plan=self.plan,
            sample_count=self.sample_count,
            mean_total_regret=self.sum_total_regret / self.sample_count,
            mean_regret_components_by_model=mean_by_model,
            example_samples=self.example_samples,
        )


def merge_plan_components(plans: Sequence[CapacityPlan]) -> CapacityPlan:
    if not plans:
        raise ValueError("Cannot merge an empty plan sequence")

    merged_plan = plans[0]
    for plan in plans[1:]:
        next_plan = merge_plan(merged_plan, plan)
        if next_plan is None:
            raise RuntimeError("Failed to merge composed capacity plans")
        merged_plan = next_plan
    return merged_plan


def merge_regret_candidates_positional(
    regret_details_by_model: Dict[str, Sequence[RegretCandidate]],
    zonal_requirements: Dict[str, Dict[str, List[Interval]]],
    regional_requirements: Dict[str, Dict[str, List[Interval]]],
) -> List[MergedRegretCandidate]:
    """Positional merge of per-model regret lists - same pairing as _merge_models.

    Each sub-model's regret list is sorted by total regret, so position i of
    the Cassandra list is paired with position i of EVCache, etc. This matches
    the zip(*plans_by_model) merge that produces least_regret in plan() and
    plan_explained(), so the resulting trace IDs align with least_regret.
    """
    model_names = list(regret_details_by_model)
    if not model_names:
        return []

    counts = {
        name: len(candidates) for name, candidates in regret_details_by_model.items()
    }
    if any(count == 0 for count in counts.values()):
        counts_description = ", ".join(
            f"{name}={count}" for name, count in sorted(counts.items())
        )
        raise RuntimeError(
            "Cannot merge composed regret provenance with invalid candidate "
            f"counts: {counts_description}"
        )

    def accumulate_requirements(plan: CapacityPlan) -> None:
        for plan_requirements, accum in (
            (plan.requirements.zonal, zonal_requirements),
            (plan.requirements.regional, regional_requirements),
        ):
            for requirement in plan_requirements:
                by_field = accum.setdefault(requirement.requirement_type, {})
                for field in sorted(CapacityRequirement.model_fields):
                    value = getattr(requirement, field)
                    if isinstance(value, Interval):
                        by_field.setdefault(field, []).append(value)

    merged: List[MergedRegretCandidate] = []
    # Intentional zip: explain the same composed candidates the planner returned.
    for components in zip(*(regret_details_by_model[name] for name in model_names)):
        candidate = MergedRegretCandidate(
            samples=[detail.sample for detail in components],
            plan=merge_plan_components([detail.plan for detail in components]),
            total_regret=sum(detail.total_regret for detail in components),
            regret_components_by_model={
                name: dict(detail.regret_components)
                for name, detail in zip(model_names, components)
            },
        )
        merged.append(candidate)
        accumulate_requirements(candidate.plan)

    return merged


def _candidate_shape_id(plan: CapacityPlan) -> str:
    """Stable key for grouping the same candidate shape across simulations."""
    # Regret summaries follow candidate topology; prices vary without changing shape.
    cost_fields = frozenset(
        {
            "annual_cost",
            "annual_cost_override",
            "annual_cost_per_gib",
            "annual_cost_per_read_io",
            "annual_cost_per_write_io",
            "annual_costs",
            "total_annual_cost",
        }
    )

    def remove_costs(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                k: remove_costs(v) for k, v in value.items() if k not in cost_fields
            }
        if isinstance(value, list):
            return [remove_costs(v) for v in value]
        return value

    return json.dumps(
        remove_costs(plan.candidate_clusters.model_dump(mode="json")),
        sort_keys=True,
        separators=(",", ":"),
    )


def summarize_regret_candidates(
    candidates: Sequence[MergedRegretCandidate],
) -> Dict[str, RegretPlanSummary]:
    grouped: Dict[str, _RegretSummaryAccumulator] = {}

    for candidate in candidates:
        # Same shape can show up from many samples; summarize that as one option.
        shape_id = _candidate_shape_id(candidate.plan)
        if shape_id not in grouped:
            grouped[shape_id] = _RegretSummaryAccumulator(plan=candidate.plan)
        grouped[shape_id].add(candidate)

    return {shape_id: group.to_summary() for shape_id, group in grouped.items()}


def summaries_for_least_regret(
    least_regret: Sequence[CapacityPlan],
    regret_summary_map: Dict[str, RegretPlanSummary],
) -> List[RegretPlanSummary]:
    summaries: List[RegretPlanSummary] = []
    missing: List[Tuple[int, str]] = []

    for index, plan in enumerate(least_regret):
        shape_id = _candidate_shape_id(plan)
        summary = regret_summary_map.get(shape_id)
        if summary is None:
            missing.append((index, shape_id))
            continue
        summaries.append(summary)

    if missing:
        missing_descriptions = ", ".join(
            f"index={index} candidate_shape_id={shape_id}"
            for index, shape_id in missing
        )
        raise RuntimeError(
            "Missing regret summaries for least_regret plans. "
            "The composed plan merge and provenance merge diverged: "
            f"{missing_descriptions}"
        )

    return summaries


def considered_alternative_summaries(
    least_regret: Sequence[CapacityPlan],
    regret_summary_map: Dict[str, RegretPlanSummary],
    max_results: int,
) -> List[RegretPlanSummary]:
    selected_shape_ids = {_candidate_shape_id(plan) for plan in least_regret}
    alternatives = [
        summary
        for shape_id, summary in regret_summary_map.items()
        if shape_id not in selected_shape_ids
    ]
    alternatives.sort(key=lambda summary: summary.mean_total_regret)
    return alternatives[:max_results]


class ExplainedPlans(ExcludeUnsetModel):
    """Plans + sub-model excuses + family context.

    Structured data for programmatic consumers. Serialize with
    .model_dump() / .model_dump_json().
    """

    plans: Sequence[CapacityPlan]
    excuses_by_model: Dict[str, Sequence[Excuse]] = Field(default_factory=dict)
    family_graph: FamilyGraph = FamilyGraph()


class ExplainedUncertainPlans(ExcludeUnsetModel):
    """Uncertain plans + sub-model excuses + family context.

    Mirrors ExplainedPlans but wraps UncertainCapacityPlan instead
    of deterministic plans. Returned by plan_explained().
    """

    plan: UncertainCapacityPlan
    excuses_by_model: Dict[str, Sequence[Excuse]] = Field(default_factory=dict)
    excuse_summary: Sequence[ExcuseSummary] = Field(default_factory=list)
    family_graph: FamilyGraph = FamilyGraph()
    least_regret_summaries: Sequence[RegretPlanSummary] = Field(default_factory=list)
    considered_alternatives: Sequence[RegretPlanSummary] = Field(default_factory=list)
