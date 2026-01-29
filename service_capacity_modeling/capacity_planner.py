# -*- coding: utf-8 -*-
# pylint: disable=too-many-lines
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
from service_capacity_modeling.interface import ClusterCapacity
from service_capacity_modeling.interface import Clusters
from service_capacity_modeling.interface import CurrentClusterCapacity
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
from service_capacity_modeling.interface import RegionClusterCapacity
from service_capacity_modeling.interface import RegionContext
from service_capacity_modeling.interface import Requirements
from service_capacity_modeling.interface import ServiceCapacity
from service_capacity_modeling.interface import UncertainCapacityPlan
from service_capacity_modeling.interface import ZoneClusterCapacity
from service_capacity_modeling.models import CapacityModel
from service_capacity_modeling.models import CostAwareModel
from service_capacity_modeling.models.common import get_disk_size_gib
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
    for field in sorted(QueryPattern.model_fields):
        d = getattr(query_pattern, field)
        if isinstance(d, Interval):
            query_pattern_simulation[field] = simulate_interval(d, field)(num_sims)
        else:
            query_pattern_simulation[field] = [d] * num_sims

    data_shape_simulation = {}
    for field in sorted(DataShape.model_fields):
        d = getattr(data_shape, field)
        if isinstance(d, Interval):
            data_shape_simulation[field] = simulate_interval(d, field)(num_sims)
        else:
            data_shape_simulation[field] = [d] * num_sims

    for sim in range(num_sims):
        query_pattern = QueryPattern(
            **{
                f: query_pattern_simulation[f][sim]
                for f in sorted(QueryPattern.model_fields)
            }
        )
        data_shape = DataShape(
            **{f: data_shape_simulation[f][sim] for f in sorted(DataShape.model_fields)}
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
    for field in sorted(QueryPattern.model_fields):
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
    for field in sorted(DataShape.model_fields):
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
                    for f in sorted(QueryPattern.model_fields)
                }
            )
        except Exception as exp:
            raise exp
        data_shape = DataShape(
            **{f: data_shape_simulation[f][i] for f in sorted(DataShape.model_fields)}
        )
        d = desires.model_copy(deep=True)
        d.query_pattern = query_pattern
        d.data_shape = data_shape
        results.append(d)

    mean_qp = QueryPattern(
        **{f: query_pattern_means[f] for f in sorted(QueryPattern.model_fields)}
    )
    mean_ds = DataShape(
        **{f: data_shape_means[f] for f in sorted(DataShape.model_fields)}
    )
    d = desires.model_copy(deep=True)
    d.query_pattern = mean_qp
    d.data_shape = mean_ds

    return results, d


def _set_instance_objects(
    desires: CapacityDesires,
    hardware: Hardware,
) -> None:
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


def _extract_cluster_plan(
    clusters: Sequence[CurrentClusterCapacity],
    hardware: Hardware,
    is_zonal: bool,
) -> Tuple[List[ClusterCapacity], List[CapacityRequirement]]:
    """Extract CapacityPlan components from current deployment.

    Takes what's currently deployed and builds the ClusterCapacity and
    CapacityRequirement objects needed for a CapacityPlan.

    Drives are priced using hardware.price_drive() to get catalog pricing.
    Cluster annual_cost is computed automatically from instance + drives.

    Args:
        clusters: Current cluster capacities (must have cluster_type set)
        hardware: Hardware catalog for the region (used to price drives)
        is_zonal: True for ZoneClusterCapacity, False for RegionClusterCapacity

    Returns:
        Tuple of (capacities, requirements) for building CapacityPlan

    Raises:
        ValueError: If any cluster is missing cluster_type
    """
    capacities: List[ClusterCapacity] = []
    requirements: List[CapacityRequirement] = []

    for current in clusters:
        if current.cluster_type is None:
            raise ValueError(
                f"cluster_type is required for baseline extraction. "
                f"Cluster '{current.cluster_instance_name}' is missing cluster_type."
            )
        cluster_type = current.cluster_type
        instance = current.cluster_instance
        if instance is None:
            raise ValueError(
                f"cluster_instance not resolved for '{current.cluster_instance_name}'"
            )
        count = int(current.cluster_instance_count.mid)

        # Price the drive from hardware catalog (gets annual_cost_per_gib etc.)
        attached_drives = []
        if current.cluster_drive is not None:
            attached_drives.append(hardware.price_drive(current.cluster_drive))

        disk_gib = get_disk_size_gib(current.cluster_drive, instance)

        capacity_cls = ZoneClusterCapacity if is_zonal else RegionClusterCapacity
        capacities.append(
            capacity_cls(
                cluster_type=cluster_type,
                count=count,
                instance=instance,
                attached_drives=attached_drives,
            )
        )

        requirements.append(
            CapacityRequirement(
                requirement_type=cluster_type,
                reference_shape=instance,
                cpu_cores=certain_float(instance.cpu * count),
                mem_gib=certain_float(instance.ram_gib * count),
                network_mbps=certain_float(instance.net_mbps * count),
                disk_gib=certain_float(disk_gib * count),
            )
        )

    return capacities, requirements


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


def _merge_models(
    plans_by_model: List[List[CapacityPlan]],
    zonal_requirements: Dict[str, Dict[str, List[Interval]]],
    regional_requirements: Dict[str, Dict[str, List[Interval]]],
) -> List[CapacityPlan]:
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
        default_num_simulations: int = 128,
        default_num_results: int = 2,
        default_lifecycles: Tuple[Lifecycle, ...] = (Lifecycle.stable, Lifecycle.beta),
    ) -> None:
        self._shapes: HardwareShapes = shapes
        self._models: Dict[str, CapacityModel] = {}
        self._cluster_types: Dict[str, str] = {}  # cluster_type -> model_name

        self._default_num_simulations = default_num_simulations
        self._default_num_results = default_num_results
        self._default_regret_params = CapacityRegretParameters()
        self._default_lifecycles = default_lifecycles

    def register_group(self, group: Callable[[], Dict[str, CapacityModel]]) -> None:
        for name, model in group().items():
            self.register_model(name, model)

    def register_model(self, name: str, capacity_model: CapacityModel) -> None:
        if isinstance(capacity_model, CostAwareModel):
            # Validate required attributes
            sn = getattr(capacity_model, "service_name", None)
            ct = getattr(capacity_model, "cluster_type", None)
            if not sn or not ct:
                raise ValueError(
                    f"CostAwareModel '{name}' must define service_name and "
                    f"cluster_type (got service_name={sn!r}, cluster_type={ct!r})"
                )

            # Duplicate cluster_type would cause double-counting
            if ct in self._cluster_types:
                raise ValueError(
                    f"Duplicate cluster_type '{ct}': '{name}' "
                    f"conflicts with '{self._cluster_types[ct]}'. Must be unique "
                    f"to avoid double-counting costs."
                )
            self._cluster_types[ct] = name

        self._models[name] = capacity_model

    @property
    def models(self) -> Dict[str, CapacityModel]:
        return self._models

    @property
    def hardware_shapes(self) -> HardwareShapes:
        return self._shapes

    def instance(self, name: str, region: Optional[str] = None) -> Instance:
        return self.hardware_shapes.instance(name, region=region)

    def _prepare_context(
        self,
        region: str,
        num_regions: int,
    ) -> Tuple[Hardware, RegionContext]:
        """Prepare hardware and region context for capacity planning.

        Loads hardware catalog for region and creates RegionContext.
        """
        hardware = self._shapes.region(region)
        context = RegionContext(
            zones_in_region=hardware.zones_in_region,
            services={n: s.model_copy(deep=True) for n, s in hardware.services.items()},
            num_regions=num_regions,
        )
        return hardware, context

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
        drives: Optional[Sequence[str]],
        extra_model_arguments: Dict[str, Any],
        instance_families: Optional[Sequence[str]],
        lifecycles: Sequence[Lifecycle],
        num_regions: int,
        num_results: Optional[int],
        region: str,
        model_percentile_desires: Any,
        sorted_percentiles: List[int],
    ) -> Dict[int, Sequence[CapacityPlan]]:
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
        drives: Optional[Sequence[str]],
        extra_model_arguments: Dict[str, Any],
        instance_families: Optional[Sequence[str]],
        lifecycles: Sequence[Lifecycle],
        num_regions: int,
        num_results: Optional[int],
        region: str,
        model_mean_desires: Dict[str, CapacityDesires],
    ) -> Sequence[CapacityPlan]:
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

    def _get_model_costs(
        self,
        *,
        model_name: str,
        context: RegionContext,
        desires: CapacityDesires,
        zonal_clusters: Sequence[ClusterCapacity],
        regional_clusters: Sequence[ClusterCapacity],
        extra_model_arguments: Dict[str, Any],
    ) -> Tuple[Dict[str, float], List[ServiceCapacity]]:
        """Get total costs for a model and any models it composes with."""
        costs: Dict[str, float] = {}
        services: List[ServiceCapacity] = []

        for sub_model_name, sub_desires in self._sub_models(
            model_name, desires, extra_model_arguments
        ):
            sub_model = self._models[sub_model_name]
            if not isinstance(sub_model, CostAwareModel):
                raise TypeError(
                    f"Sub-model '{sub_model_name}' does not implement CostAwareModel. "
                    f"All models in the composition tree must implement cost methods."
                )

            model_costs = sub_model.cluster_costs(
                service_type=sub_model.service_name,
                zonal_clusters=zonal_clusters,
                regional_clusters=regional_clusters,
            )
            costs.update(model_costs)

            model_services = sub_model.service_costs(
                service_type=sub_model.service_name,
                context=context,
                desires=sub_desires,
                extra_model_arguments=extra_model_arguments,
            )
            for svc in model_services:
                costs[svc.service_type] = svc.annual_cost
            services.extend(model_services)

        return costs, services

    def extract_baseline_plan(  # pylint: disable=too-many-positional-arguments
        self,
        model_name: str,
        region: str,
        desires: CapacityDesires,
        num_regions: int = 3,
        extra_model_arguments: Optional[Dict[str, Any]] = None,
    ) -> CapacityPlan:
        """Extract baseline plan from current clusters using model cost methods.

        This converts the current deployment (from desires.current_clusters) into
        a CapacityPlan that can be compared against recommendations. Uses model-
        specific cost methods for accurate pricing.

        Note: Only works for models with CostAwareModel mixin (EVCache, Kafka,
        Cassandra, Key-Value). Other models will raise AttributeError.

        Supports composite models (like Key-Value) that have both zonal and
        regional clusters - each model's cluster_costs filters by cluster_type.

        Args:
            model_name: Registered model name (e.g., "org.netflix.cassandra")
            region: AWS region for pricing
            desires: CapacityDesires with current_clusters populated
            num_regions: For cross-region cost calculation (default: 3)
            extra_model_arguments: Model-specific arguments (e.g., copies_per_region)

        Returns:
            CapacityPlan with costs from model.cluster_costs() and model.service_costs()

        Raises:
            ValueError: If model_name not found or current_clusters invalid
            AttributeError: If model doesn't have CostAwareModel mixin
        """
        extra_model_arguments = extra_model_arguments or {}
        if model_name not in self._models:
            raise ValueError(
                f"model_name={model_name} does not exist. "
                f"Try {sorted(list(self._models.keys()))}"
            )

        model = self._models[model_name]
        if not isinstance(model, CostAwareModel):
            raise TypeError(f"Model '{model_name}' must implement CostAwareModel mixin")

        if desires.current_clusters is None:
            raise ValueError(
                "Cannot extract baseline: desires.current_clusters is None. "
                "This function requires an existing deployment to compare against."
            )
        if not desires.current_clusters.zonal and not desires.current_clusters.regional:
            raise ValueError(
                "Cannot extract baseline: desires.current_clusters has no zonal "
                "or regional clusters defined."
            )

        hardware, context = self._prepare_context(region, num_regions)
        _set_instance_objects(
            desires, hardware
        )  # Resolve instance refs in current_clusters

        zonal_capacities: List[ClusterCapacity] = []
        zonal_requirements: List[CapacityRequirement] = []
        regional_capacities: List[ClusterCapacity] = []
        regional_requirements: List[CapacityRequirement] = []

        if desires.current_clusters.zonal:
            zonal_capacities, zonal_requirements = _extract_cluster_plan(
                desires.current_clusters.zonal, hardware, is_zonal=True
            )
        if desires.current_clusters.regional:
            regional_capacities, regional_requirements = _extract_cluster_plan(
                desires.current_clusters.regional, hardware, is_zonal=False
            )

        costs, services = self._get_model_costs(
            model_name=model_name,
            context=context,
            desires=desires,
            zonal_clusters=zonal_capacities,
            regional_clusters=regional_capacities,
            extra_model_arguments=extra_model_arguments,
        )

        return CapacityPlan(
            requirements=Requirements(
                zonal=zonal_requirements,
                regional=regional_requirements,
            ),
            candidate_clusters=Clusters(
                annual_costs=costs,
                zonal=zonal_capacities,
                regional=regional_capacities,
                services=services,
            ),
        )

    # Calculates the minimum cpu, memory, and network requirements based on desires.
    def _per_instance_requirements(self, desires: CapacityDesires) -> Tuple[int, float]:
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
        model: CapacityModel,
        region: str,
        desires: CapacityDesires,
        num_regions: int,
        lifecycles: Optional[Sequence[Lifecycle]],
        instance_families: Optional[Sequence[str]],
        drives: Optional[Sequence[str]],
    ) -> Generator[Tuple[Instance, Drive, RegionContext], None, None]:
        lifecycles = lifecycles or self._default_lifecycles
        instance_families = instance_families or []
        drives = drives or []

        hardware, context = self._prepare_context(region, num_regions)
        _set_instance_objects(
            desires, hardware
        )  # Resolve instance refs if current_clusters exists

        allowed_platforms: Set[Platform] = set(model.allowed_platforms())
        allowed_drives: Set[str] = set(drives or [])
        for drive_name in model.allowed_cloud_drives():
            if drive_name is None:
                allowed_drives = set()
                break
            allowed_drives.add(drive_name)
        if len(allowed_drives) == 0:
            allowed_drives.update(hardware.drives.keys())

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
        zonal_requirements: Dict[str, Dict[str, List[Interval]]] = {}
        regional_requirements: Dict[str, Dict[str, List[Interval]]] = {}

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
    ) -> Generator[Tuple[str, CapacityDesires], None, None]:
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
