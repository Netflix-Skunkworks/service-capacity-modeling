# -*- coding: utf-8 -*-
import functools
import logging
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


def simulate_interval(interval: Interval) -> Callable[[int], Sequence[Interval]]:
    if interval.can_simulate:

        def sim_uncertan(count: int) -> Sequence[Interval]:
            sims = dist_for_interval(interval).rvs(count)
            return [certain_float(s) for s in sims]

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
            query_pattern_simulation[field] = simulate_interval(d)(num_sims)
        else:
            query_pattern_simulation[field] = [d] * num_sims

    data_shape_simulation = {}
    for field in sorted(data_shape.__fields__):
        d = getattr(data_shape, field)
        if isinstance(d, Interval):
            data_shape_simulation[field] = simulate_interval(d)(num_sims)
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


def _least_regret(
    capacity_plans: Sequence[CapacityPlan],
    regret_params: CapacityRegretParameters,
    model: CapacityModel,
    num_results: int,
) -> Sequence[CapacityPlan]:
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
                    optimal_plan=optimal_plan,
                    proposed_plan=proposed_plan,
                ).values()
            )
        plans_by_regret.append((proposed_plan, np.einsum("i->", regret)))

    plans_by_regret.sort(key=lambda p: p[1])
    return reduce_by_family(p[0] for p in plans_by_regret)[:num_results]


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


def _merge_models(raw_capacity_plans, zonal_requirements, regional_requirements):
    capacity_plans = []
    for components in raw_capacity_plans:
        merged_plans = [
            functools.reduce(merge_plan, composed) for composed in zip(*components)
        ]
        if len(merged_plans) == 0:
            continue

        capacity_plans.append(merged_plans[0])
        plan_requirements = merged_plans[0].requirements
        for req in plan_requirements.zonal:
            _add_requirement(req, zonal_requirements)
        for req in plan_requirements.regional:
            _add_requirement(req, regional_requirements)
    return capacity_plans


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
        num_results: Optional[int] = None,
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
            model_name, desires, extra_model_arguments
        ):
            results.append(
                self._plan_certain(
                    model_name=sub_model,
                    region=region,
                    desires=sub_desires,
                    num_results=num_results,
                    lifecycles=lifecycles,
                    extra_model_arguments=extra_model_arguments,
                )
            )

        return [functools.reduce(merge_plan, composed) for composed in zip(*results)]

    def _plan_certain(
        self,
        model_name: str,
        region: str,
        desires: CapacityDesires,
        num_results: Optional[int] = None,
        lifecycles: Optional[Sequence[Lifecycle]] = None,
        extra_model_arguments: Optional[Dict[str, Any]] = None,
    ) -> Sequence[CapacityPlan]:
        extra_model_arguments = extra_model_arguments or {}
        lifecycles = lifecycles or self._default_lifecycles

        hardware = self._shapes.region(region)
        num_results = num_results or self._default_num_results

        context = RegionContext(
            zones_in_region=hardware.zones_in_region,
            services={n: s.copy(deep=True) for n, s in hardware.services.items()},
        )

        # Applications often set fixed reservations of heap or OS memory, we
        # should not even bother with shapes that don't meet the minimums
        per_instance_mem = (
            desires.data_shape.reserved_instance_app_mem_gib
            + desires.data_shape.reserved_instance_system_mem_gib
        )

        plans = []
        for instance in hardware.instances.values():
            if instance.lifecycle not in lifecycles:
                continue
            if per_instance_mem > instance.ram_gib:
                continue

            for drive in hardware.drives.values():
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
        lifecycles: Optional[Sequence[Lifecycle]] = None,
        regret_params: Optional[CapacityRegretParameters] = None,
        extra_model_arguments: Optional[Dict[str, Any]] = None,
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

        raw_capacity_plans: List[List[Sequence[CapacityPlan]]] = []
        for _ in range(simulations):
            raw_capacity_plans.append([])

        for sub_model, sub_desires in self._sub_models(
            model_name, desires, extra_model_arguments
        ):
            for j, sim_desires in enumerate(model_desires(sub_desires, simulations)):
                raw_capacity_plans[j].append(
                    self._plan_certain(
                        model_name=sub_model,
                        region=region,
                        desires=sim_desires,
                        num_results=1,
                        extra_model_arguments=extra_model_arguments,
                    )
                )

        # Now accumulate across the composed models
        capacity_plans = _merge_models(
            raw_capacity_plans, zonal_requirements, regional_requirements
        )

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
            )

        result = UncertainCapacityPlan(
            requirements=final_requirement,
            least_regret=_least_regret(
                capacity_plans,
                regret_params,
                self._models[model_name],
                num_results,
            ),
            mean=self.plan_certain(
                model_name=model_name,
                region=region,
                desires=mean_desires,
                extra_model_arguments=extra_model_arguments,
            ),
            percentiles=percentile_plans,
        )
        return result

    def _sub_models(self, model_name, desires, extra_model_arguments):
        queue: List[Tuple[Any, str, Optional[Callable[[CapacityDesires], None]]]] = [
            (desires, model_name, None)
        ]
        models_used = []

        while queue:
            parent_desires, sub_model, modify_sub_desires = queue.pop()
            # prevent infinite loop of models for now
            if sub_model in models_used:
                continue
            models_used.append(sub_model)

            # start with a copy of the parent desires
            sub_desires = parent_desires.copy(deep=True)
            if modify_sub_desires:
                # apply composite model specific transform
                modify_sub_desires(sub_desires)
            # then apply model defaults because it could change the defaults applied
            sub_defaults = self._models[sub_model].default_desires(
                sub_desires, extra_model_arguments
            )
            # apply the defaults to the desires to allow the model code to be simpler
            sub_desires = sub_desires.merge_with(sub_defaults)

            # We might have to compose this model with others depending on
            # the user requirement
            queue.extend(
                [
                    (sub_desires, child_model, modify_child_desires)
                    for child_model, modify_child_desires in self._models[
                        sub_model
                    ].compose_with(sub_desires, extra_model_arguments)
                ]
            )

            yield sub_model, sub_desires


planner = CapacityPlanner()
planner.register_group(netflix.models)
