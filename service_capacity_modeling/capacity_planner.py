# -*- coding: utf-8 -*-
import functools
import logging
import math
from hashlib import blake2b
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import Generator
from typing import List
from typing import Optional
from typing import Sequence
from typing import Set
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
from service_capacity_modeling.interface import Drive
from service_capacity_modeling.interface import Hardware
from service_capacity_modeling.interface import Instance
from service_capacity_modeling.interface import Interval
from service_capacity_modeling.interface import interval
from service_capacity_modeling.interface import Lifecycle
from service_capacity_modeling.interface import PlanExplanation
from service_capacity_modeling.interface import Platform
from service_capacity_modeling.interface import QueryPattern
from service_capacity_modeling.interface import RegionContext
from service_capacity_modeling.interface import Requirements
from service_capacity_modeling.interface import UncertainCapacityPlan
from service_capacity_modeling.models import CapacityModel
from service_capacity_modeling.models.common import merge_plan
from service_capacity_modeling.models.org import netflix
from service_capacity_modeling.models.utils import reduce_by_family
from service_capacity_modeling.stats import dist_for_interval
from service_capacity_modeling.stats import interval_percentile

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

        def sim_uncertain(count: int) -> Sequence[Interval]:
            return [
                certain_float(s)
                for s in dist_for_interval(interval, seed=seed).rvs(count)
            ]

        return sim_uncertain

    else:

        def sim_certain(count: int) -> Sequence[Interval]:
            return [interval] * count

        return sim_certain


# Take uncertain inputs and simulate a desired number of certain inputs
# e.g. read_per_second[100, 1000] -> [rps[107, 107], rps[756, 756], ...]
# num_sims concrete values are generated.
def model_desires(
    desires: CapacityDesires, num_sims: int = 1000
) -> Generator[CapacityDesires, None, None]:
    query_pattern = desires.query_pattern.model_copy(deep=True)
    data_shape = desires.data_shape.model_copy(deep=True)

    query_pattern_simulation = {}
    for field in sorted(query_pattern.model_fields):
        d = getattr(query_pattern, field)
        if isinstance(d, Interval):
            query_pattern_simulation[field] = simulate_interval(d, field)(num_sims)
        else:
            query_pattern_simulation[field] = [d] * num_sims

    data_shape_simulation = {}
    for field in sorted(data_shape.model_fields):
        d = getattr(data_shape, field)
        if isinstance(d, Interval):
            data_shape_simulation[field] = simulate_interval(d, field)(num_sims)
        else:
            data_shape_simulation[field] = [d] * num_sims

    for sim in range(num_sims):
        query_pattern = QueryPattern(
            **{
                f: query_pattern_simulation[f][sim]
                for f in sorted(query_pattern.model_fields)
            }
        )
        data_shape = DataShape(
            **{
                f: data_shape_simulation[f][sim]
                for f in sorted(data_shape.model_fields)
            }
        )

        d = desires.model_copy(update={"query_pattern": None, "data_shape": None})
        d.query_pattern = query_pattern
        d.data_shape = data_shape
        yield d


def model_desires_percentiles(
    desires: CapacityDesires,
    percentiles: Sequence[int] = (5, 25, 50, 75, 95),
) -> Tuple[Sequence[CapacityDesires], CapacityDesires]:
    query_pattern = desires.query_pattern.model_copy(deep=True)
    data_shape = desires.data_shape.model_copy(deep=True)

    query_pattern_simulation = {}
    query_pattern_means = {}
    for field in sorted(query_pattern.model_fields):
        d = getattr(query_pattern, field)
        if isinstance(d, Interval):
            query_pattern_simulation[field] = interval_percentile(d, percentiles)
            if d.can_simulate:
                query_pattern_means[field] = certain_float(d.mid)
            else:
                query_pattern_means[field] = d
        else:
            query_pattern_simulation[field] = [d] * len(percentiles)
            query_pattern_means[field] = d

    data_shape_simulation = {}
    data_shape_means = {}
    for field in sorted(data_shape.model_fields):
        d = getattr(data_shape, field)
        if isinstance(d, Interval):
            data_shape_simulation[field] = interval_percentile(d, percentiles)
            if d.can_simulate:
                data_shape_means[field] = certain_float(d.mid)
            else:
                data_shape_means[field] = d
        else:
            data_shape_simulation[field] = [d] * len(percentiles)
            data_shape_means[field] = d

    results = []
    for i in range(len(percentiles)):
        try:
            query_pattern = QueryPattern(
                **{
                    f: query_pattern_simulation[f][i]
                    for f in sorted(query_pattern.model_fields)
                }
            )
        except Exception as exp:
            raise exp
        data_shape = DataShape(
            **{f: data_shape_simulation[f][i] for f in sorted(data_shape.model_fields)}
        )
        d = desires.model_copy(deep=True)
        d.query_pattern = query_pattern
        d.data_shape = data_shape
        results.append(d)

    mean_qp = QueryPattern(
        **{f: query_pattern_means[f] for f in sorted(query_pattern.model_fields)}
    )
    mean_ds = DataShape(
        **{f: data_shape_means[f] for f in sorted(data_shape.model_fields)}
    )
    d = desires.model_copy(deep=True)
    d.query_pattern = mean_qp
    d.data_shape = mean_ds

    return results, d


def _set_instance_objects(
    desires: CapacityDesires,
    hardware: Hardware,
):
    if desires.current_clusters:
        for zonal_cluster_capacity in desires.current_clusters.zonal:
            if zonal_cluster_capacity.cluster_instance_name in hardware.instances:
                zonal_cluster_capacity.cluster_instance = hardware.instances[
                    zonal_cluster_capacity.cluster_instance_name
                ]
            else:
                raise ValueError(
                    f"Model not trained to right size clusters that are of instance"
                    f" types {zonal_cluster_capacity.cluster_instance_name}"
                )
        for regional_cluster_capacity in desires.current_clusters.regional:
            if regional_cluster_capacity.cluster_instance_name in hardware.instances:
                regional_cluster_capacity.cluster_instance = hardware.instances[
                    regional_cluster_capacity.cluster_instance_name
                ]
            else:
                raise ValueError(
                    f"Model not trained to right size clusters that are of instance"
                    f" types {regional_cluster_capacity.cluster_instance_name}"
                )


def _allow_instance(
    instance: Instance,
    allowed_names: Sequence[str],
    allowed_lifecycles: Sequence[Lifecycle],
    allowed_platforms: Set[Platform],
) -> bool:
    # If the user has explicitly asked for particular families instead
    # of all lifecycles filter based on that
    if allowed_names:
        if instance.name not in allowed_names:
            if instance.family not in allowed_names:
                return False
    # Otherwise consider lifecycle (default) and platform
    else:
        if instance.lifecycle not in allowed_lifecycles:
            return False
        if allowed_platforms.isdisjoint(instance.platforms):
            return False

    return True


def _allow_drive(
    drive: Drive,
    allowed_names: Sequence[str],
    allowed_lifecycles: Sequence[Lifecycle],
    allowed_drives: Set[str],
) -> bool:
    # If the user has explicitly asked for particular families instead
    # of all lifecycles filter based on that
    if allowed_names:
        if drive.name not in allowed_names:
            return False
    # Otherwise consider lifecycle (default)
    else:
        if drive.lifecycle not in allowed_lifecycles:
            return False

    if drive.name not in allowed_drives:
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

    for field in sorted(requirement.model_fields):
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

    def instance(self, name: str, region: Optional[str] = None) -> Instance:
        return self.hardware_shapes.instance(name, region=region)

    def _plan_percentiles(  # pylint: disable=too-many-positional-arguments
        self,
        model_name: str,
        percentiles: Tuple[int, ...],
        region: str,
        desires: CapacityDesires,
        lifecycles: Optional[Sequence[Lifecycle]] = None,
        instance_families: Optional[Sequence[str]] = None,
        drives: Optional[Sequence[str]] = None,
        num_results: Optional[int] = None,
        num_regions: int = 3,
        extra_model_arguments: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Sequence[CapacityPlan], Dict[int, Sequence[CapacityPlan]]]:
        if model_name not in self._models:
            raise ValueError(
                f"model_name={model_name} does not exist. "
                f"Try {sorted(list(self._models.keys()))}"
            )

        extra_model_arguments = extra_model_arguments or {}
        lifecycles = lifecycles or self._default_lifecycles

        model_mean_desires: Dict[str, CapacityDesires] = {}
        sorted_percentiles = sorted(percentiles)
        model_percentile_desires: List[Dict[str, CapacityDesires]] = []
        for _ in sorted_percentiles:
            model_percentile_desires.append({})

        for sub_model, sub_desires in self._sub_models(
            model_name=model_name,
            desires=desires,
            extra_model_arguments=extra_model_arguments,
        ):
            percentile_inputs, mean_desires = model_desires_percentiles(
                desires=sub_desires, percentiles=sorted_percentiles
            )
            model_mean_desires[sub_model] = mean_desires
            index = 0
            for percentile_input in percentile_inputs:
                model_percentile_desires[index][sub_model] = percentile_input
                index = index + 1

        mean_plan = self._mean_plan(
            drives,
            extra_model_arguments,
            instance_families,
            lifecycles,
            num_regions,
            num_results,
            region,
            model_mean_desires,
        )
        percentile_plans = self._group_plans_by_percentile(
            drives,
            extra_model_arguments,
            instance_families,
            lifecycles,
            num_regions,
            num_results,
            region,
            model_percentile_desires,
            sorted_percentiles,
        )

        return mean_plan, percentile_plans

    def _group_plans_by_percentile(  # pylint: disable=too-many-positional-arguments
        self,
        drives,
        extra_model_arguments,
        instance_families,
        lifecycles,
        num_regions,
        num_results,
        region,
        model_percentile_desires,
        sorted_percentiles,
    ):
        percentile_plans = {}
        for index, percentile in enumerate(sorted_percentiles):
            percentile_plan = []
            for percentile_sub_model, percentile_sub_desire in model_percentile_desires[
                index
            ].items():
                percentile_sub_plan = self._plan_certain(
                    model_name=percentile_sub_model,
                    region=region,
                    desires=percentile_sub_desire,
                    num_results=num_results,
                    num_regions=num_regions,
                    extra_model_arguments=extra_model_arguments,
                    lifecycles=lifecycles,
                    instance_families=instance_families,
                    drives=drives,
                )
                if percentile_sub_plan:
                    percentile_plan.append(percentile_sub_plan)

            percentile_plans[percentile] = cast(
                Sequence[CapacityPlan],
                [
                    functools.reduce(merge_plan, composed)
                    for composed in zip(*percentile_plan)
                ],
            )
        return percentile_plans

    def _mean_plan(  # pylint: disable=too-many-positional-arguments
        self,
        drives,
        extra_model_arguments,
        instance_families,
        lifecycles,
        num_regions,
        num_results,
        region,
        model_mean_desires,
    ):
        mean_plans = []
        for mean_sub_model, mean_sub_desire in model_mean_desires.items():
            mean_sub_plan = self._plan_certain(
                model_name=mean_sub_model,
                region=region,
                desires=mean_sub_desire,
                num_results=num_results,
                num_regions=num_regions,
                extra_model_arguments=extra_model_arguments,
                lifecycles=lifecycles,
                instance_families=instance_families,
                drives=drives,
            )
            if mean_sub_plan:
                mean_plans.append(mean_sub_plan)
        mean_plan = cast(
            Sequence[CapacityPlan],
            [functools.reduce(merge_plan, composed) for composed in zip(*mean_plans)],
        )
        return mean_plan

    def plan_certain(  # pylint: disable=too-many-positional-arguments
        self,
        model_name: str,
        region: str,
        desires: CapacityDesires,
        lifecycles: Optional[Sequence[Lifecycle]] = None,
        instance_families: Optional[Sequence[str]] = None,
        drives: Optional[Sequence[str]] = None,
        num_results: Optional[int] = None,
        num_regions: int = 3,
        extra_model_arguments: Optional[Dict[str, Any]] = None,
        max_results_per_family: int = 1,
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
            sub_plan = self._plan_certain(
                model_name=sub_model,
                region=region,
                desires=sub_desires,
                num_results=num_results,
                num_regions=num_regions,
                extra_model_arguments=extra_model_arguments,
                lifecycles=lifecycles,
                instance_families=instance_families,
                drives=drives,
                max_results_per_family=max_results_per_family,
            )
            if sub_plan:
                results.append(sub_plan)

        return [functools.reduce(merge_plan, composed) for composed in zip(*results)]

    def _plan_certain(  # pylint: disable=too-many-positional-arguments
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
        max_results_per_family: int = 1,
    ) -> Sequence[CapacityPlan]:
        extra_model_arguments = extra_model_arguments or {}
        model = self._models[model_name]

        plans = []
        for instance, drive, context in self.generate_scenarios(
            model, region, desires, num_regions, lifecycles, instance_families, drives
        ):
            plan = model.capacity_plan(
                instance=instance,
                drive=drive,
                context=context,
                desires=desires,
                extra_model_arguments=extra_model_arguments,
            )
            if plan is not None:
                plans.append(plan)

        # lowest cost first
        plans.sort(key=lambda p: (p.rank, p.candidate_clusters.total_annual_cost))

        num_results = num_results or self._default_num_results
        return reduce_by_family(plans, max_results_per_family=max_results_per_family)[
            :num_results
        ]

    # Calculates the minimum cpu, memory, and network requirements based on desires.
    def _per_instance_requirements(self, desires) -> Tuple[int, float]:

        # Applications often set fixed reservations of heap or OS memory
        per_instance_mem = (
            desires.data_shape.reserved_instance_app_mem_gib
            + desires.data_shape.reserved_instance_system_mem_gib
        )

        # Applications often require a minimum amount of true parallelism
        per_instance_cores = int(
            math.ceil(
                desires.query_pattern.estimated_read_parallelism.mid
                + desires.query_pattern.estimated_write_parallelism.mid
            )
        )

        current_capacity = (
            None
            if desires.current_clusters is None
            else (
                desires.current_clusters.zonal[0]
                if len(desires.current_clusters.zonal)
                else desires.current_clusters.regional[0]
            )
        )
        # Return early if we dont have current_capacity set.
        if current_capacity is None or current_capacity.cluster_instance is None:
            return (per_instance_cores, per_instance_mem)

        # Calculate memory requirements based on current capacity
        current_memory_utilization_gib = current_capacity.memory_utilization_gib.high
        per_instance_mem = max(per_instance_mem, current_memory_utilization_gib)

        return (per_instance_cores, per_instance_mem)

    def generate_scenarios(  # pylint: disable=too-many-positional-arguments
        self,
        model,
        region,
        desires,
        num_regions,
        lifecycles,
        instance_families,
        drives,
    ):
        lifecycles = lifecycles or self._default_lifecycles
        instance_families = instance_families or []
        drives = drives or []

        hardware = self._shapes.region(region)

        context = RegionContext(
            zones_in_region=hardware.zones_in_region,
            services={n: s.model_copy(deep=True) for n, s in hardware.services.items()},
            num_regions=num_regions,
        )

        allowed_platforms: Set[Platform] = set(model.allowed_platforms())
        allowed_drives: Set[str] = set(drives or [])
        for drive_name in model.allowed_cloud_drives():
            if drive_name is None:
                allowed_drives = set()
                break
            allowed_drives.add(drive_name)
        if len(allowed_drives) == 0:
            allowed_drives.update(hardware.drives.keys())

        # Set current instance object if exists
        _set_instance_objects(desires, hardware)

        # We should not even bother with shapes that don't meet the minimums
        (
            per_instance_cores,
            per_instance_mem,
        ) = self._per_instance_requirements(desires)

        if model.run_hardware_simulation():
            for instance in hardware.instances.values():
                if not _allow_instance(
                    instance, instance_families, lifecycles, allowed_platforms
                ):
                    continue

                # If the instance doesn't have enough vertical resources, pass on it
                if (
                    per_instance_mem > instance.ram_gib
                    or per_instance_cores > instance.cpu
                ):
                    continue

                for drive in hardware.drives.values():
                    if not _allow_drive(drive, drives, lifecycles, allowed_drives):
                        continue

                    yield instance, drive, context
        else:
            instance = Instance.get_managed_instance()
            drive = Drive.get_managed_drive()
            yield instance, drive, context

    # pylint: disable=too-many-locals
    def plan(  # pylint: disable=too-many-positional-arguments
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
        max_results_per_family: int = 1,
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
        desires_by_model: Dict[str, CapacityDesires] = {}
        for sub_model, sub_desires in self._sub_models(
            model_name=model_name,
            desires=desires,
            extra_model_arguments=extra_model_arguments,
        ):
            desires_by_model[sub_model] = sub_desires
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
                            max_results_per_family=max_results_per_family,
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
            ),
            max_results_per_family=max_results_per_family,
        )[:num_results]

        low_p, high_p = sorted(percentiles)[0], sorted(percentiles)[-1]

        final_zonal = []
        final_regional = []
        for req_type, samples in zonal_requirements.items():
            req = CapacityRequirement(
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
                **{
                    k: interval(samples=[i.mid for i in v], low_p=low_p, high_p=high_p)
                    for k, v in samples.items()
                },
            )
            final_regional.append(req)

        final_requirement = Requirements(zonal=final_zonal, regional=final_regional)

        mean_plan, percentile_plans = self._plan_percentiles(
            model_name=model_name,
            percentiles=percentiles,
            region=region,
            desires=desires,
            extra_model_arguments=extra_model_arguments,
            num_regions=num_regions,
            instance_families=instance_families,
        )

        result = UncertainCapacityPlan(
            requirements=final_requirement,
            least_regret=least_regret,
            mean=mean_plan,
            percentiles=percentile_plans,
            explanation=PlanExplanation(
                regret_params=regret_params,
                desires_by_model={
                    model: desires.merge_with(
                        self._models[model].default_desires(
                            desires_by_model[model], extra_model_arguments
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
