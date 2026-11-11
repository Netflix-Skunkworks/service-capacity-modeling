"""Explainability types for the capacity planner.

**Experimental** — this API may change.

This module contains ExplainedPlans and ExplainedUncertainPlans, types used
to explain *why* the planner rejected certain instance/drive combinations.
FamilyTrait, FamilyEdge and FamilyGraph are deprecated and always empty.

Core contract types (Bottleneck, Excuse) live in interface.py because they
are part of the CapacityModel.capacity_plan() return type.

Consumer usage::

    from service_capacity_modeling.capacity_planner import planner
    from service_capacity_modeling.models.plan_comparison import compare_plans

    # Rejection explanations
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
"""

from __future__ import annotations

import json
from typing import Any
from typing import Dict
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
from service_capacity_modeling.interface import Interval
from service_capacity_modeling.interface import RegretPlanSummary
from service_capacity_modeling.interface import UncertainCapacityPlan
from service_capacity_modeling.interface import SampleRef
from service_capacity_modeling.models import CapacityModel
from service_capacity_modeling.models.common import merge_plan


class FamilyTrait(ExcludeUnsetModel):
    """Deprecated: the planner no longer builds a family graph.

    Kept so callers that construct or deserialize FamilyGraph keep working.
    """

    family: str
    memory_gib_per_vcpu: float
    has_local_disk: bool
    local_disk_gib_per_vcpu: Optional[float] = None
    drive_type: Optional[DriveType] = None
    cost_per_vcpu_annual: Optional[float] = None


class FamilyEdge(ExcludeUnsetModel):
    """Deprecated: the planner no longer builds a family graph."""

    from_family: str
    to_family: str
    trade_off: Optional[str] = None
    improves: List[Bottleneck] = []
    degrades: List[Bottleneck] = []


class FamilyGraph(ExcludeUnsetModel):
    """Deprecated: always empty, and will be removed.

    The graph compared families by per-vCPU ratios and the digit in the family
    name, so it disagreed with the per-node checks that actually reject a
    type, such as a RAM floor per node. Excuses explain each rejection.
    """

    traits: Dict[str, FamilyTrait] = {}
    edges: List[FamilyEdge] = []


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
    regret_plan: Optional[CapacityPlan] = None


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
                optimal_plan=optimal_sample.regret_plan or optimal_sample.plan,
                proposed_plan=proposed_sample.regret_plan or proposed_sample.plan,
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
    # Deprecated: always empty.
    family_graph: FamilyGraph = FamilyGraph()


class ExplainedUncertainPlans(ExcludeUnsetModel):
    """Uncertain plans + sub-model excuses + family context.

    Mirrors ExplainedPlans but wraps UncertainCapacityPlan instead
    of deterministic plans. Returned by plan_explained().
    """

    plan: UncertainCapacityPlan
    excuses_by_model: Dict[str, Sequence[Excuse]] = Field(default_factory=dict)
    excuse_summary: Sequence[ExcuseSummary] = Field(default_factory=list)
    # Deprecated: always empty.
    family_graph: FamilyGraph = FamilyGraph()
    least_regret_summaries: Sequence[RegretPlanSummary] = Field(default_factory=list)
    considered_alternatives: Sequence[RegretPlanSummary] = Field(default_factory=list)
