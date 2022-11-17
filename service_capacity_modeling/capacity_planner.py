# -*- coding: utf-8 -*-
import functools
import logging
from hashlib import blake2b
from typing import Any
from typing import Callable
from typing import Dict
from typing import Generator
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple

import numpy as np

from service_capacity_modeling.hardware import HardwareShapes
from service_capacity_modeling.hardware import shapes
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import CapacityPlan
from service_capacity_modeling.interface import CapacityRegretParameters
from service_capacity_modeling.interface import CapacityRequirement
from service_capacity_modeling.interface import certain_float
from service_capacity_modeling.interface import DataShape
from service_capacity_modeling.interface import Interval
from service_capacity_modeling.interface import interval
from service_capacity_modeling.interface import interval_percentile
from service_capacity_modeling.interface import Lifecycle
from service_capacity_modeling.interface import PlanExplanation
from service_capacity_modeling.interface import QueryPattern
from service_capacity_modeling.interface import RegionContext
from service_capacity_modeling.interface import Requirements
from service_capacity_modeling.interface import UncertainCapacityPlan
from service_capacity_modeling.models import CapacityModel
from service_capacity_modeling.models.common import merge_plan
from service_capacity_modeling.models.org import netflix
from service_capacity_modeling.models.utils import reduce_by_family
from service_capacity_modeling.stats import dist_for_interval

logger = logging.getLogger(__name__)


def simulate_interval(
    interval: Interval, name: str
) -> Callable[[int], Sequence[Interval]]:
    if interval.can_simulate:
        # We need to convert the name of the field to a positive 32 bit seed
        # and we don't really need collision prevention so just take a 24 bit
        # digest and enforce it is not signed
        seed = int.from_bytes(
            blake2b(name.encode(), digest_size=3).digest(),
            byteorder="big",
            signed=False,
        )

        def sim_uncertan(count: int) -> Sequence[Interval]:
            return [
                certain_float(s)
                for s in dist_for_interval(interval, seed=seed).rvs(count)
            ]

        return sim_uncertan

    else:

        def sim_certain(count: int) -> Sequence[Interval]:
            return [interval] * count

        return sim_certain


# Take uncertain inputs and simulate a desired number of certain inputs
def model_desires(
    desires: CapacityDesires, num_sims: int = 1000
) -> Generator[CapacityDesires, None, None]:
    query_pattern = desires.query_pattern.copy(deep=True)
    data_shape = desires.data_shape.copy(deep=True)

    query_pattern_simulation = {}
    for field in sorted(query_pattern.__fields__):
        d = getattr(query_pattern, field)
        if isinstance(d, Interval):
            query_pattern_simulation[field] = simulate_interval(d, field)(num_sims)
        else:
            query_pattern_simulation[field] = [d] * num_sims

    data_shape_simulation = {}
    for field in sorted(data_shape.__fields__):
        d = getattr(data_shape, field)
        if isinstance(d, Interval):
            data_shape_simulation[field] = simulate_interval(d, field)(num_sims)
        else:
            data_shape_simulation[field] = [d] * num_sims

    for sim in range(num_sims):
        query_pattern = QueryPattern(
            **{
                f: query_pattern_simulation[f][sim]
                for f in sorted(query_pattern.__fields__)
            }
        )
        data_shape = DataShape(
            **{f: data_shape_simulation[f][sim] for f in sorted(data_shape.__fields__)}
        )

        d = desires.copy(exclude={"query_pattern", "data_shape"})
        d.query_pattern = query_pattern
        d.data_shape = data_shape
        yield d


def model_desires_percentiles(
    desires: CapacityDesires,
    percentiles: Sequence[int] = (5, 25, 50, 75, 95),
) -> Tuple[Sequence[CapacityDesires], CapacityDesires]:
    query_pattern = desires.query_pattern.copy(deep=True)
    data_shape = desires.data_shape.copy(deep=True)

    query_pattern_simulation = {}
    query_pattern_means = {}
    for field in sorted(query_pattern.__fields__):
        d = getattr(query_pattern, field)
        if isinstance(d, Interval):
            query_pattern_means[field] = certain_float(d.mid)
            if d.confidence <= 0.99:
                samples = dist_for_interval(d).rvs(1028)
                query_pattern_simulation[field] = interval_percentile(
                    samples, percentiles
                )
                continue
        query_pattern_simulation[field] = [d] * len(percentiles)
        query_pattern_means[field] = d

    data_shape_simulation = {}
    data_shape_means = {}
    for field in sorted(data_shape.__fields__):
        d = getattr(data_shape, field)
        if isinstance(d, Interval):
            data_shape_means[field] = certain_float(d.mid)
            if d.confidence <= 0.99:
                samples = dist_for_interval(d).rvs(1028)
                data_shape_simulation[field] = interval_percentile(samples, percentiles)
                continue
        data_shape_simulation[field] = [d] * len(percentiles)
        data_shape_means[field] = d

    results = []
    for i in range(len(percentiles)):
        query_pattern = QueryPattern(
            **{
                f: query_pattern_simulation[f][i]
                for f in sorted(query_pattern.__fields__)
            }
        )
        data_shape = DataShape(
            **{f: data_shape_simulation[f][i] for f in sorted(data_shape.__fields__)}
        )
        d = desires.copy(deep=True)
        d.query_pattern = query_pattern
        d.data_shape = data_shape
        results.append(d)

    mean_qp = QueryPattern(
        **{f: query_pattern_means[f] for f in sorted(query_pattern.__fields__)}
    )
    mean_ds = DataShape(
        **{f: data_shape_means[f] for f in sorted(data_shape.__fields__)}
    )
    d = desires.copy(deep=True)
    d.query_pattern = mean_qp
    d.data_shape = mean_ds

    return results, d


def _allow_hardware(
    name: str,
    lifecycle: Lifecycle,
    allowed_names: Sequence[str],
    allowed_lifecycles: Sequence[Lifecycle],
) -> bool:
    # If the user has explicitly asked for particular families instead
    # of all lifecycles filter based on that
    if allowed_names:
        if name not in allowed_names:
            return False
    # Otherwise consider lifecycle (default)
    else:
        if lifecycle not in allowed_lifecycles:
            return False
    return True


def _regret(
    capacity_plans: Sequence[Tuple[CapacityDesires, CapacityPlan]],
    regret_params: CapacityRegretParameters,
    model: CapacityModel,
) -> Sequence[Tuple[CapacityPlan, CapacityDesires, float]]:
    plans_by_regret = []

    # Unfortunately has to be O(N^2) since regret isn't symmetric.
    # We could create the entire NxN regret matrix and use
    # einsum('ij->i') to quickly do a row wise sum, but that would
    # require a _lot_ more memory than this ...
    regret = np.zeros(len(capacity_plans), dtype=np.float64)
    for i, proposed_plan in enumerate(capacity_plans):
        for j, optimal_plan in enumerate(capacity_plans):
            if j == i:
                regret[j] = 0

            regret[j] = sum(
                model.regret(
                    regret_params=regret_params,
                    optimal_plan=optimal_plan[1],
                    proposed_plan=proposed_plan[1],
                ).values()
            )
        plans_by_regret.append(
            (proposed_plan[1], proposed_plan[0], np.einsum("i->", regret))
        )

    plans_by_regret.sort(key=lambda p: p[2])
    return plans_by_regret


def _add_requirement(requirement, accum):
    if requirement.requirement_type not in accum:
        accum[requirement.requirement_type] = {}

    requirements = accum[requirement.requirement_type]

    for field in sorted(requirement.__fields__):
        d = getattr(requirement, field)
        if isinstance(d, Interval):
            if field not in requirements:
                requirements[field] = [d]
            else:
                requirements[field].append(d)


def _merge_models(plans_by_model, zonal_requirements, regional_requirements):
    capacity_plans = []
    for composed in zip(*filter(lambda x: x, plans_by_model)):
        merged_plans = [functools.reduce(merge_plan, composed)]
        if len(merged_plans) == 0:
            continue

        capacity_plans.append(merged_plans[0])
        plan_requirements = merged_plans[0].requirements
        for req in plan_requirements.zonal:
            _add_requirement(req, zonal_requirements)
        for req in plan_requirements.regional:
            _add_requirement(req, regional_requirements)
    return capacity_plans


def _in_allowed(inp: str, allowed: Sequence[str]) -> bool:
    if not allowed:
        return True
    else:
        return inp in allowed


class CapacityPlanner:
    def __init__(
        self,
        default_num_simulations=128,
        default_num_results=2,
        default_lifecycles=(Lifecycle.stable, Lifecycle.beta),
    ):
        self._shapes: HardwareShapes = shapes
        self._models: Dict[str, CapacityModel] = {}

        self._default_num_simulations = default_num_simulations
        self._default_num_results = default_num_results
        self._default_regret_params = CapacityRegretParameters()
        self._default_lifecycles = default_lifecycles

    def register_group(self, group: Callable[[], Dict[str, CapacityModel]]):
        for name, model in group().items():
            self.register_model(name, model)

    def register_model(self, name: str, capacity_model: CapacityModel):
        self._models[name] = capacity_model

    @property
    def models(self) -> Dict[str, CapacityModel]:
        return self._models

    @property
    def hardware_shapes(self) -> HardwareShapes:
        return self._shapes

    def plan_certain(
        self,
        model_name: str,
        region: str,
        desires: CapacityDesires,
        lifecycles: Optional[Sequence[Lifecycle]] = None,
        instance_families: Optional[List[str]] = None,
        drives: Optional[List[str]] = None,
        num_results: Optional[int] = None,
        num_regions: int = 3,
        extra_model_arguments: Optional[Dict[str, Any]] = None,
    ) -> Sequence[CapacityPlan]:
        if model_name not in self._models:
            raise ValueError(
                f"model_name={model_name} does not exist. "
                f"Try {sorted(list(self._models.keys()))}"
            )

        extra_model_arguments = extra_model_arguments or {}
        lifecycles = lifecycles or self._default_lifecycles

        results = []

        for sub_model, sub_desires in self._sub_models(
            model_name=model_name,
            desires=desires,
            extra_model_arguments=extra_model_arguments,
        ):
            results.append(
                self._plan_certain(
                    model_name=sub_model,
                    region=region,
                    desires=sub_desires,
                    num_results=num_results,
                    num_regions=num_regions,
                    extra_model_arguments=extra_model_arguments,
                    lifecycles=lifecycles,
                    instance_families=instance_families,
                    drives=drives,
                )
            )

        return [functools.reduce(merge_plan, composed) for composed in zip(*results)]

    def _plan_certain(
        self,
        model_name: str,
        region: str,
        desires: CapacityDesires,
        num_results: Optional[int] = None,
        num_regions: int = 3,
        lifecycles: Optional[Sequence[Lifecycle]] = None,
        instance_families: Optional[Sequence[str]] = None,
        drives: Optional[Sequence[str]] = None,
        extra_model_arguments: Optional[Dict[str, Any]] = None,
    ) -> Sequence[CapacityPlan]:
        extra_model_arguments = extra_model_arguments or {}
        lifecycles = lifecycles or self._default_lifecycles
        instance_families = instance_families or []
        drives = drives or []

        hardware = self._shapes.region(region)
        num_results = num_results or self._default_num_results

        context = RegionContext(
            zones_in_region=hardware.zones_in_region,
            services={n: s.copy(deep=True) for n, s in hardware.services.items()},
            num_regions=num_regions,
        )

        # Applications often set fixed reservations of heap or OS memory, we
        # should not even bother with shapes that don't meet the minimums
        per_instance_mem = (
            desires.data_shape.reserved_instance_app_mem_gib
            + desires.data_shape.reserved_instance_system_mem_gib
        )

        plans = []
        for instance in hardware.instances.values():
            if not _allow_hardware(
                instance.family, instance.lifecycle, instance_families, lifecycles
            ):
                continue

            if per_instance_mem > instance.ram_gib:
                continue

            for drive in hardware.drives.values():
                if not _allow_hardware(drive.name, drive.lifecycle, drives, lifecycles):
                    continue

                plan = self._models[model_name].capacity_plan(
                    instance=instance,
                    drive=drive,
                    context=context,
                    desires=desires,
                    extra_model_arguments=extra_model_arguments,
                )
                if plan is not None:
                    plans.append(plan)

        # lowest cost first
        plans.sort(key=lambda plan: plan.candidate_clusters.total_annual_cost)

        return reduce_by_family(plans)[:num_results]

    # pylint: disable-msg=too-many-locals
    def plan(
        self,
        model_name: str,
        region: str,
        desires: CapacityDesires,
        percentiles: Tuple[int, ...] = (5, 50, 95),
        simulations: Optional[int] = None,
        num_results: Optional[int] = None,
        num_regions: int = 3,
        lifecycles: Optional[Sequence[Lifecycle]] = None,
        instance_families: Optional[Sequence[str]] = None,
        drives: Optional[Sequence[str]] = None,
        regret_params: Optional[CapacityRegretParameters] = None,
        extra_model_arguments: Optional[Dict[str, Any]] = None,
        explain: bool = False,
    ) -> UncertainCapacityPlan:
        extra_model_arguments = extra_model_arguments or {}

        if not all(0 <= p <= 100 for p in percentiles):
            raise ValueError("percentiles must be an integer in the range [0, 100]")
        if model_name not in self._models:
            raise ValueError(
                f"model_name={model_name} does not exist. "
                f"Try {sorted(list(self._models.keys()))}"
            )

        regret_params = regret_params or self._default_regret_params
        simulations = simulations or self._default_num_simulations
        num_results = num_results or self._default_num_results
        lifecycles = lifecycles or self._default_lifecycles

        # requirement types -> values
        zonal_requirements: Dict[str, Dict] = {}
        regional_requirements: Dict[str, Dict] = {}

        regret_clusters_by_model: Dict[
            str, Sequence[Tuple[CapacityPlan, CapacityDesires, float]]
        ] = {}
        for sub_model, sub_desires in self._sub_models(
            model_name=model_name,
            desires=desires,
            extra_model_arguments=extra_model_arguments,
        ):
            model_plans: List[Tuple[CapacityDesires, Sequence[CapacityPlan]]] = []
            for sim_desires in model_desires(sub_desires, simulations):
                model_plans.append(
                    (
                        sim_desires,
                        self._plan_certain(
                            model_name=sub_model,
                            region=region,
                            desires=sim_desires,
                            num_results=1,
                            num_regions=num_regions,
                            extra_model_arguments=extra_model_arguments,
                            lifecycles=lifecycles,
                            instance_families=instance_families,
                            drives=drives,
                        ),
                    )
                )
            regret_clusters_by_model[sub_model] = _regret(
                capacity_plans=[
                    (sim_desires, plan[0]) for sim_desires, plan in model_plans if plan
                ],
                regret_params=regret_params,
                model=self._models[sub_model],
            )

        # Now accumulate across the composed models and return the top N
        # by distinct hardware type
        least_regret = reduce_by_family(
            _merge_models(
                # First param is the actual plan which we care about
                [
                    [plan[0] for plan in component]
                    for component in regret_clusters_by_model.values()
                ],
                zonal_requirements,
                regional_requirements,
            )
        )[:num_results]

        low_p, high_p = sorted(percentiles)[0], sorted(percentiles)[-1]

        final_zonal = []
        final_regional = []
        for req_type, samples in zonal_requirements.items():
            req = CapacityRequirement(
                core_reference_ghz=desires.core_reference_ghz,
                requirement_type=req_type,
                **{
                    k: interval(samples=[i.mid for i in v], low_p=low_p, high_p=high_p)
                    for k, v in samples.items()
                },
            )
            final_zonal.append(req)
        for req_type, samples in regional_requirements.items():
            req = CapacityRequirement(
                requirement_type=req_type,
                core_reference_ghz=desires.core_reference_ghz,
                **{
                    k: interval(samples=[i.mid for i in v], low_p=low_p, high_p=high_p)
                    for k, v in samples.items()
                },
            )
            final_regional.append(req)

        final_requirement = Requirements(zonal=final_zonal, regional=final_regional)

        percentile_inputs, mean_desires = model_desires_percentiles(
            desires=desires, percentiles=sorted(percentiles)
        )
        percentile_plans = {}
        for index, percentile in enumerate(percentiles):
            percentile_plans[percentile] = self.plan_certain(
                model_name=model_name,
                region=region,
                desires=percentile_inputs[index],
                extra_model_arguments=extra_model_arguments,
                num_regions=num_regions,
            )

        result = UncertainCapacityPlan(
            requirements=final_requirement,
            least_regret=least_regret,
            mean=self.plan_certain(
                model_name=model_name,
                region=region,
                desires=mean_desires,
                extra_model_arguments=extra_model_arguments,
                num_regions=num_regions,
            ),
            percentiles=percentile_plans,
            explanation=PlanExplanation(
                regret_params=regret_params,
                desires_by_model={
                    model: desires.merge_with(
                        self._models[model].default_desires(
                            desires, extra_model_arguments
                        )
                    )
                    for model in regret_clusters_by_model
                },
            ),
        )
        if explain:
            result.explanation.regret_clusters_by_model = regret_clusters_by_model
            result.explanation.context["regret"] = least_regret

        return result

    def _sub_models(
        self,
        model_name: str,
        desires: CapacityDesires,
        extra_model_arguments: Dict[str, Any],
    ):
        queue: List[Tuple[CapacityDesires, str]] = [(desires, model_name)]
        models_used = []

        while queue:
            parent_desires, sub_model = queue.pop()
            # prevent infinite loop of models for now
            if sub_model in models_used:
                continue
            models_used.append(sub_model)

            sub_desires = parent_desires.merge_with(
                self._models[sub_model].default_desires(
                    parent_desires, extra_model_arguments
                )
            )

            # We might have to compose this model with others depending on
            # the user requirement
            queue.extend(
                [
                    (modify_child_desires(desires), child_model)
                    for child_model, modify_child_desires in self._models[
                        sub_model
                    ].compose_with(desires, extra_model_arguments)
                ]
            )

            yield sub_model, sub_desires


planner = CapacityPlanner()
planner.register_group(netflix.models)
