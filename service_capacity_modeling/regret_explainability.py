from __future__ import annotations

import json
from typing import Any
from typing import Dict
from typing import Iterator
from typing import List
from typing import Sequence
from typing import Tuple

from pydantic import Field

from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import CapacityPlan
from service_capacity_modeling.interface import CapacityRegretParameters
from service_capacity_modeling.interface import CapacityRequirement
from service_capacity_modeling.interface import ExcludeUnsetModel
from service_capacity_modeling.interface import Interval
from service_capacity_modeling.interface import RegretPlanSummary
from service_capacity_modeling.interface import SampleRef
from service_capacity_modeling.models import CapacityModel
from service_capacity_modeling.models.common import merge_plan


_SIGNATURE_EXCLUDED_FIELDS = frozenset(
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


def _without_signature_excluded_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            k: _without_signature_excluded_fields(v)
            for k, v in value.items()
            if k not in _SIGNATURE_EXCLUDED_FIELDS
        }
    if isinstance(value, list):
        return [_without_signature_excluded_fields(v) for v in value]
    return value


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
    regret_components: Dict[str, float] = {}


class MergedRegretCandidate(ExcludeUnsetModel):
    """Merged plan plus regret breadcrumbs across composed models."""

    samples: Sequence[SampleRef] = Field(default_factory=list)
    plan: CapacityPlan
    total_regret: float
    regret_components_by_model: Dict[str, Dict[str, float]] = {}


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


def _aggregate_component_maps(
    component_maps: Sequence[Dict[str, float]],
) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    for component_map in component_maps:
        for component, value in component_map.items():
            totals[component] = totals.get(component, 0.0) + value
    return dict(sorted(totals.items()))


def _mean_component_maps(
    component_maps: Sequence[Dict[str, float]],
) -> Dict[str, float]:
    if not component_maps:
        return {}
    totals = _aggregate_component_maps(component_maps)
    count = float(len(component_maps))
    return {component: value / count for component, value in totals.items()}


class _RegretSummaryAccumulator(ExcludeUnsetModel):
    plan: CapacityPlan
    selected_total_regret: float
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
            selected_total_regret=self.selected_total_regret,
            mean_total_regret=self.sum_total_regret / self.sample_count,
            mean_regret_components_by_model=mean_by_model,
            example_samples=self.example_samples,
        )


def plan_signature(plan: CapacityPlan) -> str:
    """Topology signature for grouping plans while ignoring cost and rank."""
    return json.dumps(
        _without_signature_excluded_fields(
            plan.candidate_clusters.model_dump(mode="json")
        ),
        sort_keys=True,
        separators=(",", ":"),
    )


def _add_requirement(
    requirement: CapacityRequirement, accum: Dict[str, Dict[str, List[Interval]]]
) -> None:
    if requirement.requirement_type not in accum:
        accum[requirement.requirement_type] = {}

    requirements = accum[requirement.requirement_type]

    for field in sorted(CapacityRequirement.model_fields):
        d = getattr(requirement, field)
        if isinstance(d, Interval):
            if field not in requirements:
                requirements[field] = [d]
            else:
                requirements[field].append(d)


def add_plan_requirements(
    plan: CapacityPlan,
    zonal_requirements: Dict[str, Dict[str, List[Interval]]],
    regional_requirements: Dict[str, Dict[str, List[Interval]]],
) -> None:
    for req in plan.requirements.zonal:
        _add_requirement(req, zonal_requirements)
    for req in plan.requirements.regional:
        _add_requirement(req, regional_requirements)


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


def _candidate_counts_description(
    regret_details_by_model: Dict[str, Sequence[RegretCandidate]],
) -> str:
    return ", ".join(
        f"{name}={len(candidates)}"
        for name, candidates in sorted(regret_details_by_model.items())
    )


def _add_requirements_from_positional_candidates(
    regret_details_by_model: Dict[str, Sequence[RegretCandidate]],
    zonal_requirements: Dict[str, Dict[str, List[Interval]]],
    regional_requirements: Dict[str, Dict[str, List[Interval]]],
) -> None:
    for candidate in _iter_regret_candidates_positional(regret_details_by_model):
        add_plan_requirements(candidate.plan, zonal_requirements, regional_requirements)


def _merged_regret_candidate(
    model_names: Sequence[str],
    components: Sequence[RegretCandidate],
) -> MergedRegretCandidate:
    return MergedRegretCandidate(
        samples=[detail.sample for detail in components],
        plan=merge_plan_components([detail.plan for detail in components]),
        total_regret=sum(d.total_regret for d in components),
        regret_components_by_model={
            name: dict(detail.regret_components)
            for name, detail in zip(model_names, components)
        },
    )


def _validate_regret_candidate_counts(
    regret_details_by_model: Dict[str, Sequence[RegretCandidate]],
) -> None:
    counts = {
        name: len(candidates) for name, candidates in regret_details_by_model.items()
    }
    expected_count = next(iter(counts.values()))
    if any(count == 0 or count != expected_count for count in counts.values()):
        raise RuntimeError(
            "Cannot merge composed regret provenance with invalid candidate "
            f"counts: {_candidate_counts_description(regret_details_by_model)}"
        )


def _iter_regret_candidates_positional(
    regret_details_by_model: Dict[str, Sequence[RegretCandidate]],
) -> Iterator[MergedRegretCandidate]:
    model_names = list(regret_details_by_model)
    if not model_names:
        return

    _validate_regret_candidate_counts(regret_details_by_model)
    lists = [regret_details_by_model[name] for name in model_names]
    for components in zip(*lists):
        yield _merged_regret_candidate(model_names, components)


def _merge_regret_candidates_cross_product_bounded(
    regret_details_by_model: Dict[str, Sequence[RegretCandidate]],
    max_per_model: int,
    max_results: int,
) -> List[MergedRegretCandidate]:
    model_names = list(regret_details_by_model)
    if len(model_names) != 2:
        raise ValueError("Cross-product regret merge requires exactly two models")
    if any(len(regret_details_by_model[name]) == 0 for name in model_names):
        raise RuntimeError(
            "Cannot merge composed regret provenance with empty candidate counts: "
            f"{_candidate_counts_description(regret_details_by_model)}"
        )

    left_model, right_model = model_names
    left_candidates = regret_details_by_model[left_model][:max_per_model]
    right_candidates = regret_details_by_model[right_model][:max_per_model]
    merged = [
        _merged_regret_candidate(model_names, (left, right))
        for left in left_candidates
        for right in right_candidates
    ]
    merged.sort(key=lambda candidate: candidate.total_regret)
    return merged[:max_results]


def merge_regret_candidates_bounded(
    regret_details_by_model: Dict[str, Sequence[RegretCandidate]],
    zonal_requirements: Dict[str, Dict[str, List[Interval]]],
    regional_requirements: Dict[str, Dict[str, List[Interval]]],
    max_per_model: int,
    max_results: int,
) -> List[MergedRegretCandidate]:
    if len(regret_details_by_model) == 2:
        _add_requirements_from_positional_candidates(
            regret_details_by_model=regret_details_by_model,
            zonal_requirements=zonal_requirements,
            regional_requirements=regional_requirements,
        )
        return _merge_regret_candidates_cross_product_bounded(
            regret_details_by_model=regret_details_by_model,
            max_per_model=max_per_model,
            max_results=max_results,
        )

    return merge_regret_candidates_positional(
        regret_details_by_model=regret_details_by_model,
        zonal_requirements=zonal_requirements,
        regional_requirements=regional_requirements,
    )


def merge_regret_candidates_positional(
    regret_details_by_model: Dict[str, Sequence[RegretCandidate]],
    zonal_requirements: Dict[str, Dict[str, List[Interval]]],
    regional_requirements: Dict[str, Dict[str, List[Interval]]],
) -> List[MergedRegretCandidate]:
    """Positional merge of per-model regret lists — same pairing as _merge_models.

    Each sub-model's regret list is sorted by total regret, so position i of
    the Cassandra list is paired with position i of EVCache, etc. This matches
    the zip(*plans_by_model) merge that produces least_regret in plan() and
    plan_explained(), so the resulting signatures align with least_regret.
    """
    model_names = list(regret_details_by_model)
    if not model_names:
        return []

    merged = list(_iter_regret_candidates_positional(regret_details_by_model))
    for candidate in merged:
        add_plan_requirements(candidate.plan, zonal_requirements, regional_requirements)
    return merged


def summarize_regret_candidates(
    candidates: Sequence[MergedRegretCandidate],
) -> Dict[str, RegretPlanSummary]:
    grouped: Dict[str, _RegretSummaryAccumulator] = {}

    for candidate in candidates:
        signature = plan_signature(candidate.plan)
        if signature not in grouped:
            grouped[signature] = _RegretSummaryAccumulator(
                plan=candidate.plan,
                selected_total_regret=candidate.total_regret,
            )
        grouped[signature].add(candidate)

    return {signature: group.to_summary() for signature, group in grouped.items()}


def summaries_for_least_regret(
    least_regret: Sequence[CapacityPlan],
    regret_summary_map: Dict[str, RegretPlanSummary],
) -> List[RegretPlanSummary]:
    summaries: List[RegretPlanSummary] = []
    missing: List[Tuple[int, str]] = []

    for index, plan in enumerate(least_regret):
        signature = plan_signature(plan)
        summary = regret_summary_map.get(signature)
        if summary is None:
            missing.append((index, signature))
            continue
        summaries.append(summary)

    if missing:
        missing_descriptions = ", ".join(
            f"index={index} signature={signature}" for index, signature in missing
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
    selected_signatures = {plan_signature(plan) for plan in least_regret}
    alternatives = [
        summary
        for signature, summary in regret_summary_map.items()
        if signature not in selected_signatures
    ]
    alternatives.sort(key=lambda summary: summary.selected_total_regret)
    return alternatives[:max_results]
