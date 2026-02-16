from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple

from .stateless_java import nflx_java_app_capacity_model
from service_capacity_modeling.interface import AccessConsistency
from service_capacity_modeling.interface import AccessPattern
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import CapacityPlan
from service_capacity_modeling.interface import Consistency
from service_capacity_modeling.interface import DataShape
from service_capacity_modeling.interface import Drive
from service_capacity_modeling.interface import FixedInterval
from service_capacity_modeling.interface import GlobalConsistency
from service_capacity_modeling.interface import Instance
from service_capacity_modeling.interface import Interval
from service_capacity_modeling.interface import QueryPattern
from service_capacity_modeling.interface import RegionContext
from service_capacity_modeling.models import CapacityModel


class NflxSearchCapacityModel(CapacityModel):
    @staticmethod
    def capacity_plan(
        instance: Instance,
        drive: Drive,
        context: RegionContext,
        desires: CapacityDesires,
        extra_model_arguments: Dict[str, Any],
    ) -> Optional[CapacityPlan]:
        extra_model_arguments.setdefault("root_disk_gib", 20)

        search_app = nflx_java_app_capacity_model.capacity_plan(
            instance=instance,
            drive=drive,
            context=context,
            desires=desires,
            extra_model_arguments=extra_model_arguments,
        )
        if search_app is None:
            return None

        for cluster in search_app.candidate_clusters.regional:
            cluster.cluster_type = "dgwsearch"
        return search_app

    @staticmethod
    def description() -> str:
        return "Netflix Streaming Search Abstraction"

    @staticmethod
    def extra_model_arguments_schema() -> Dict[str, Any]:
        return nflx_java_app_capacity_model.extra_model_arguments_schema()

    @staticmethod
    def compose_with(
        user_desires: CapacityDesires, extra_model_arguments: Dict[str, Any]
    ) -> Tuple[Tuple[str, Callable[[CapacityDesires], CapacityDesires]], ...]:
        def _modify_elasticsearch_desires(
            user_desires: CapacityDesires,
        ) -> CapacityDesires:
            relaxed = user_desires.model_copy(deep=True)
            relaxed.query_pattern.access_consistency.same_region.target_consistency = (
                AccessConsistency.eventual
            )
            return relaxed

        return (("org.netflix.elasticsearch", _modify_elasticsearch_desires),)

    @staticmethod
    def default_desires(
        user_desires: CapacityDesires, extra_model_arguments: Dict[str, Any]
    ) -> CapacityDesires:
        if user_desires.query_pattern.access_pattern == AccessPattern.latency:
            return CapacityDesires(
                query_pattern=QueryPattern(
                    access_pattern=AccessPattern.latency,
                    access_consistency=GlobalConsistency(
                        same_region=Consistency(
                            target_consistency=AccessConsistency.eventual,
                        ),
                        cross_region=Consistency(
                            target_consistency=AccessConsistency.eventual,
                        ),
                    ),
                    estimated_mean_read_size_bytes=Interval(
                        low=256, mid=4096, high=65536, confidence=0.95
                    ),
                    estimated_mean_write_size_bytes=Interval(
                        low=128, mid=1024, high=8192, confidence=0.95
                    ),
                    estimated_mean_read_latency_ms=Interval(
                        low=1, mid=5, high=50, confidence=0.98
                    ),
                    estimated_mean_write_latency_ms=Interval(
                        low=0.5, mid=2, high=10, confidence=0.98
                    ),
                    read_latency_slo_ms=FixedInterval(
                        minimum_value=1,
                        maximum_value=100,
                        low=5,
                        mid=20,
                        high=50,
                        confidence=0.98,
                    ),
                    write_latency_slo_ms=FixedInterval(
                        minimum_value=0.5,
                        maximum_value=50,
                        low=2,
                        mid=10,
                        high=20,
                        confidence=0.98,
                    ),
                ),
                data_shape=DataShape(
                    estimated_state_size_gib=Interval(
                        low=10, mid=100, high=1000, confidence=0.98
                    ),
                    reserved_instance_app_mem_gib=8,
                ),
            )
        else:
            return CapacityDesires(
                query_pattern=QueryPattern(
                    access_pattern=AccessPattern.throughput,
                    access_consistency=GlobalConsistency(
                        same_region=Consistency(
                            target_consistency=AccessConsistency.eventual,
                        ),
                        cross_region=Consistency(
                            target_consistency=AccessConsistency.eventual,
                        ),
                    ),
                    estimated_mean_read_size_bytes=Interval(
                        low=256, mid=4096, high=131072, confidence=0.95
                    ),
                    estimated_mean_write_size_bytes=Interval(
                        low=128, mid=4096, high=65536, confidence=0.95
                    ),
                    estimated_mean_read_latency_ms=Interval(
                        low=2, mid=10, high=100, confidence=0.98
                    ),
                    estimated_mean_write_latency_ms=Interval(
                        low=1, mid=5, high=50, confidence=0.98
                    ),
                    read_latency_slo_ms=FixedInterval(
                        minimum_value=5,
                        maximum_value=200,
                        low=10,
                        mid=50,
                        high=100,
                        confidence=0.98,
                    ),
                    write_latency_slo_ms=FixedInterval(
                        minimum_value=1,
                        maximum_value=100,
                        low=5,
                        mid=20,
                        high=50,
                        confidence=0.98,
                    ),
                ),
                data_shape=DataShape(
                    estimated_state_size_gib=Interval(
                        low=50, mid=500, high=5000, confidence=0.98
                    ),
                    reserved_instance_app_mem_gib=8,
                ),
            )


nflx_search_capacity_model = NflxSearchCapacityModel()
