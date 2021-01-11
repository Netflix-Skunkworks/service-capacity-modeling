# -*- coding: utf-8 -*-
import logging
from typing import Callable
from typing import Dict
from typing import Generator
from typing import Optional
from typing import Sequence
from typing import Tuple

from service_capacity_modeling.hardware import HardwareShapes
from service_capacity_modeling.hardware import shapes
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import CapacityPlan
from service_capacity_modeling.interface import CapacityRegretParameters
from service_capacity_modeling.interface import CapacityRequirement
from service_capacity_modeling.interface import certain_float
from service_capacity_modeling.interface import DataShape
from service_capacity_modeling.interface import GlobalHardware
from service_capacity_modeling.interface import Interval
from service_capacity_modeling.interface import interval
from service_capacity_modeling.interface import interval_percentile
from service_capacity_modeling.interface import QueryPattern
from service_capacity_modeling.interface import UncertainCapacityPlan
from service_capacity_modeling.models import CapacityModel
from service_capacity_modeling.models.org import netflix
from service_capacity_modeling.models.utils import reduce_by_family
from service_capacity_modeling.stats import gamma_for_interval

logger = logging.getLogger(__name__)


def simulate_interval(interval: Interval) -> Callable[[int], Sequence[Interval]]:
    if interval.can_simulate:

        def sim_uncertan(count: int) -> Sequence[Interval]:
            sims = gamma_for_interval(interval).rvs(count)
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
    query_pattern = desires.query_pattern.copy()
    data_shape = desires.data_shape.copy()

    query_pattern_simulation = {}
    for field in query_pattern.__fields__:
        d = getattr(query_pattern, field)
        if isinstance(d, Interval):
            query_pattern_simulation[field] = simulate_interval(d)(num_sims)
        else:
            query_pattern_simulation[field] = [d] * num_sims

    data_shape_simulation = {}
    for field in data_shape.__fields__:
        d = getattr(data_shape, field)
        if isinstance(d, Interval):
            data_shape_simulation[field] = simulate_interval(d)(num_sims)
        else:
            data_shape_simulation[field] = [d] * num_sims

    for sim in range(num_sims):
        query_pattern = QueryPattern(
            **{f: query_pattern_simulation[f][sim] for f in query_pattern.__fields__}
        )
        data_shape = DataShape(
            **{f: data_shape_simulation[f][sim] for f in data_shape.__fields__}
        )

        d = desires.copy()
        d.query_pattern = query_pattern
        d.data_shape = data_shape
        yield d


def model_desires_percentiles(
    desires: CapacityDesires,
    percentiles: Sequence[int] = (5, 25, 50, 75, 95),
) -> Tuple[Sequence[CapacityDesires], CapacityDesires]:
    query_pattern = desires.query_pattern.copy()
    data_shape = desires.data_shape.copy()

    query_pattern_simulation = {}
    query_pattern_means = {}
    for field in query_pattern.__fields__:
        d = getattr(query_pattern, field)
        if isinstance(d, Interval):
            query_pattern_means[field] = certain_float(d.mid)
            if d.confidence < 0.99:
                samples = gamma_for_interval(d).rvs(10000)
                query_pattern_simulation[field] = interval_percentile(
                    samples, percentiles
                )
                continue
        query_pattern_simulation[field] = [d] * len(percentiles)
        query_pattern_means[field] = d

    data_shape_simulation = {}
    data_shape_means = {}
    for field in data_shape.__fields__:
        d = getattr(data_shape, field)
        if isinstance(d, Interval):
            data_shape_means[field] = certain_float(d.mid)
            if d.confidence < 0.99:
                samples = gamma_for_interval(d).rvs(10000)
                data_shape_simulation[field] = interval_percentile(samples, percentiles)
                continue
        data_shape_simulation[field] = [d] * len(percentiles)
        data_shape_means[field] = d

    results = []
    for i in range(len(percentiles)):
        query_pattern = QueryPattern(
            **{f: query_pattern_simulation[f][i] for f in query_pattern.__fields__}
        )
        data_shape = DataShape(
            **{f: data_shape_simulation[f][i] for f in data_shape.__fields__}
        )
        d = desires.copy()
        d.query_pattern = query_pattern
        d.data_shape = data_shape
        results.append(d)

    mean_qp = QueryPattern(
        **{f: query_pattern_means[f] for f in query_pattern.__fields__}
    )
    mean_ds = DataShape(**{f: data_shape_means[f] for f in data_shape.__fields__})
    d = desires.copy()
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
    for i, proposed_plan in enumerate(capacity_plans):
        regret = 0
        for j, optimal_plan in enumerate(capacity_plans):
            if j == i:
                continue

            regret += model.regret(
                regret_params=regret_params,
                optimal_plan=optimal_plan,
                proposed_plan=proposed_plan,
            )

        plans_by_regret.append((proposed_plan, regret))

    plans_by_regret.sort(key=lambda d: d[1])
    plans = [p[0] for p in plans_by_regret]
    return reduce_by_family(plans)[:num_results]


class CapacityPlanner:
    def __init__(self):
        self._shapes: HardwareShapes = shapes
        self._models: Dict[str, CapacityModel] = dict()

        self._default_num_simulations = 200
        self._default_num_results = 3
        self._regret_params = CapacityRegretParameters()

    def register_group(self, group: Callable[[], Dict[str, CapacityModel]]):
        for name, model in group().items():
            self.register_model(name, model)

    def register_model(self, name: str, capacity_model: CapacityModel):
        self._models[name] = capacity_model

    @property
    def hardware_shapes(self) -> GlobalHardware:
        return self._shapes.hardware

    def plan_certain(
        self,
        model_name: str,
        region: str,
        desires: CapacityDesires,
        num_results: Optional[int] = None,
        **model_kwargs
    ) -> Sequence[CapacityPlan]:
        hardware = self._shapes.region(region)
        num_results = num_results or self._default_num_results

        plans = []
        j = 0
        for instance in hardware.instances.values():
            for drive in hardware.drives.values():
                j += 1
                plan = self._models[model_name].capacity_plan(
                    instance=instance, drive=drive, desires=desires, **model_kwargs
                )
                if plan is not None:
                    plans.append(plan)

        # lowest cost first
        plans.sort(key=lambda plan: plan.candidate_clusters.total_annual_cost.mid)

        return reduce_by_family(plans)[:num_results]

    # pylint: disable-msg=too-many-locals
    def plan(
        self,
        model_name: str,
        region: str,
        desires: CapacityDesires,
        percentiles: Tuple[int, int] = (5, 25, 50, 75, 95),
        simulations: Optional[int] = None,
        num_results: Optional[int] = None,
        **model_kwargs
    ) -> UncertainCapacityPlan:

        if not all([0 <= p <= 100 for p in percentiles]):
            raise ValueError("percentiles must be an integer in the range [0, 100]")

        simulations = simulations or self._default_num_simulations
        num_results = num_results or self._default_num_results

        requirements = {}
        # desires -> Optional[CapacityPlan]
        capacity_plans = []

        for sim_desires in model_desires(desires, simulations):
            plans = self.plan_certain(
                model_name=model_name,
                region=region,
                desires=sim_desires,
                **model_kwargs,
            )
            if len(plans) == 0:
                continue

            capacity_plans.append(plans[0])
            requirement = plans[0].requirement

            for field in requirement.__fields__:
                d = getattr(requirement, field)
                if isinstance(d, Interval):
                    if field not in requirements:
                        requirements[field] = [d]
                    else:
                        requirements[field].append(d)

        low_p, high_p = sorted(percentiles)[0], sorted(percentiles)[-1]

        caps = {
            k: interval(samples=[i.mid for i in v], low_p=low_p, high_p=high_p)
            for k, v in requirements.items()
        }

        final_requirement = CapacityRequirement(
            core_reference_ghz=desires.core_reference_ghz, **caps
        )

        percentile_inputs, mean_desires = model_desires_percentiles(
            desires=desires, percentiles=sorted(percentiles)
        )
        percentile_plans = {}
        for index, percentile in enumerate(percentiles):
            percentile_plans[percentile] = self.plan_certain(
                model_name=model_name,
                region=region,
                desires=percentile_inputs[index],
                **model_kwargs,
            )

        result = UncertainCapacityPlan(
            requirement=final_requirement,
            least_regret=_least_regret(
                capacity_plans,
                self._regret_params,
                self._models[model_name],
                num_results,
            ),
            mean=self.plan_certain(
                model_name=model_name,
                region=region,
                desires=mean_desires,
                **model_kwargs,
            ),
            percentiles=percentile_plans,
        )
        return result

    @property
    def models(self) -> Sequence[str]:
        return self._models.keys()


planner = CapacityPlanner()
planner.register_group(netflix.models)
