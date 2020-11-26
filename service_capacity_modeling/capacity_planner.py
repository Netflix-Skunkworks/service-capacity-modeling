# -*- coding: utf-8 -*-
import logging
from typing import Callable
from typing import Dict
from typing import Generator
from typing import Optional
from typing import Sequence
from typing import Tuple

from service_capacity_modeling.capacity_models.cassandra import (
    estimate_cassandra_cluster_zone,
)
from service_capacity_modeling.capacity_models.cassandra import (
    estimate_cassandra_requirement,
)
from service_capacity_modeling.capacity_models.stateless_java_app import (
    estimate_java_app_region,
)
from service_capacity_modeling.capacity_models.stateless_java_app import (
    estimate_java_app_requirement,
)
from service_capacity_modeling.capacity_models.utils import reduce_by_family
from service_capacity_modeling.hardware import HardwareShapes
from service_capacity_modeling.hardware import shapes
from service_capacity_modeling.models import CapacityDesires
from service_capacity_modeling.models import CapacityPlan
from service_capacity_modeling.models import CapacityRequirement
from service_capacity_modeling.models import certain_float
from service_capacity_modeling.models import Clusters
from service_capacity_modeling.models import DataShape
from service_capacity_modeling.models import GlobalHardware
from service_capacity_modeling.models import Interval
from service_capacity_modeling.models import QueryPattern
from service_capacity_modeling.stats import gamma_for_interval

logger = logging.getLogger(__name__)


def simulate_interval(interval: Interval) -> Callable[[int], Sequence[Interval]]:
    if interval.confidence > 0.99:

        def simulate(count: int) -> Sequence[Interval]:
            return [interval] * count

        return simulate
    else:

        def simulate(count: int) -> Sequence[Interval]:
            sims = gamma_for_interval(interval).rvs(count)
            return [certain_float(s) for s in sims]

        return simulate


# Take uncertain inputs and simulate a desired number of certain inputs
def model_inputs(
    desires: CapacityDesires, num_sims: int = 100
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


class CapacityPlanner:
    def __init__(self):
        self._shapes: HardwareShapes = shapes
        self._models: Dict[
            str, Tuple[Callable[..., CapacityRequirement], Callable[..., CapacityPlan]]
        ] = {}

        self.simulations = 1000
        self.results = 4

    def register_model(
        self,
        name: str,
        capacity_model: Callable[..., CapacityRequirement],
        provision_model: Callable[..., Optional[Clusters]],
    ):
        self._models[name] = (capacity_model, provision_model)

    @property
    def hardware_shapes(self) -> GlobalHardware:
        return self._shapes.hardware

    def model_capacity(
        self, model_name: str, desires: CapacityDesires, *args, **kwargs
    ) -> CapacityRequirement:
        return self._models[model_name][0](desires=desires, *args, **kwargs)

    def provision(
        self,
        model_name: str,
        region: str,
        requirement: CapacityRequirement,
        *args,
        **kwargs
    ) -> Sequence[Clusters]:
        hardware = self._shapes.region(region)

        topologies = []
        for instance in hardware.instances.values():
            for drive in hardware.drives.values():
                topo = self._models[model_name][1](
                    instance=instance,
                    drive=drive,
                    requirement=requirement,
                    *args,
                    **kwargs
                )
                if topo is not None:
                    topologies.append(topo)
        topologies.sort(key=lambda topo: topo.total_annual_cost.mid)
        return reduce_by_family(topologies)[: self.results]

    def plan_certain(
        self, model_name: str, region: str, desires: CapacityDesires, *args, **kwargs
    ) -> CapacityPlan:
        requirement = self.model_capacity(
            model_name=model_name, desires=desires, *args, **kwargs
        )
        clusters = self.provision(
            model_name=model_name,
            region=region,
            requirement=requirement,
            *args,
            **kwargs
        )
        return CapacityPlan(requirement=requirement, candidate_clusters=clusters)

    #        cores, mem, disk, network, cost = [], [], [], [], []
    #        for sim_desires in model_inputs(desires, self.simulations):
    #            plan = self._models[model_name](
    #                # Pass Hardware
    #                hardware=hardware,
    #                # Pass CapacityDesires
    #                desires=sim_desires,
    #                # Pass arbitrary additional things
    #                *args,
    #                **kwargs
    #            )
    #            capacity_plans.append(plan)
    #            cores.append(plan.capacity_requirement.cpu_cores)
    #            mem.append(plan.capacity_requirement.mem_gib)
    #            disk.append(plan.capacity_requirement.disk_gib)
    #            network.append(plan.capacity_requirement.mem_gib)
    #            if (plan.clusters):
    #                cost.append(plan.clusters[0].annual_cost)
    #
    #        sim_cores = Interval(
    #            low=np.percentiles(cores, [5]),
    #            mid=np.mean(cores),
    #            high=np.percentiles(cores, [95])
    #            confidence=0.9
    #        )
    #        sim_mem = Interval(
    #            low=np.percentiles(mem, [5]),
    #            mid=np.mean(mem),
    #            high=np.percentiles(mem, [95])
    #            confidence=0.9
    #        )
    #        sim_disk = Interval(
    #            low=np.percentiles(mem, [5]),
    #            mid=np.mean(mem),
    #            high=np.percentiles(mem, [95])
    #            confidence=0.9
    #        )
    #        sim_net = Interval(
    #            low=np.percentiles(network, [5]),
    #            mid=np.mean(network),
    #            high=np.percentiles(network, [95]),
    #            confidence=0.9
    #        )
    #        sim_cost = Interval(
    #            low=np.percentiles(cost, [5]),
    #            mid=np.mean(cost),
    #            high=np.percentiles(network, [95]),
    #            confidence=0.9
    #        )
    #        return None

    @property
    def models(self) -> Sequence[str]:
        return self._models.keys()


planner = CapacityPlanner()
planner.register_model(
    name="nflx_stateless_java_app",
    capacity_model=estimate_java_app_requirement,
    provision_model=estimate_java_app_region,
)
planner.register_model(
    name="nflx_cassandra",
    capacity_model=estimate_cassandra_requirement,
    provision_model=estimate_cassandra_cluster_zone,
)
