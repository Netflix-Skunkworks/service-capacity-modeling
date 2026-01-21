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
from service_capacity_modeling.interface import certain_int


class NflxControlCapacityModel(CapacityModel):
    @staticmethod
    def capacity_plan(
        instance: Instance,
        drive: Drive,
        context: RegionContext,
        desires: CapacityDesires,
        extra_model_arguments: Dict[str, Any],
    ) -> Optional[CapacityPlan]:
        # Control wants 20GiB root volumes
        extra_model_arguments.setdefault("root_disk_gib", 20)

        # Ensure Java app has enough memory to cache the whole dataset
        modified_desires = desires.model_copy(deep=True)
        if modified_desires.data_shape.estimated_state_size_gib:
            # double buffer the cache
            additional_mem = 2 * desires.data_shape.estimated_state_size_gib.mid
            modified_desires.data_shape.reserved_instance_app_mem_gib += additional_mem

        control_app = nflx_java_app_capacity_model.capacity_plan(
            instance=instance,
            drive=drive,
            context=context,
            desires=modified_desires,
            extra_model_arguments=extra_model_arguments,
        )
        if control_app is None:
            return None

        for cluster in control_app.candidate_clusters.regional:
            cluster.cluster_type = "dgwcontrol"
        return control_app

    @staticmethod
    def description() -> str:
        return "Netflix Control Model"

    @staticmethod
    def extra_model_arguments_schema() -> Dict[str, Any]:
        return nflx_java_app_capacity_model.extra_model_arguments_schema()

    @staticmethod
    def compose_with(
        user_desires: CapacityDesires, extra_model_arguments: Dict[str, Any]
    ) -> Tuple[Tuple[str, Callable[[CapacityDesires], CapacityDesires]], ...]:
        def _modify_postgres_desires(
            user_desires: CapacityDesires,
        ) -> CapacityDesires:
            relaxed = user_desires.model_copy(deep=True)

            # Postgres doesn't support tier 0, so downgrade to tier 1
            if relaxed.service_tier == 0:
                relaxed.service_tier = 1

            # Control caches reads in memory, only writes go to Postgres
            # Set read QPS to minimal since Postgres only handles writes
            if relaxed.query_pattern.estimated_read_per_second:
                relaxed.query_pattern.estimated_read_per_second = certain_int(1)

            return relaxed

        return (("org.netflix.postgres", _modify_postgres_desires),)

    @staticmethod
    def default_desires(
        user_desires: CapacityDesires, extra_model_arguments: Dict[str, Any]
    ) -> CapacityDesires:
        return CapacityDesires(
            query_pattern=QueryPattern(
                access_pattern=AccessPattern.latency,
                access_consistency=GlobalConsistency(
                    same_region=Consistency(
                        target_consistency=AccessConsistency.read_your_writes,
                    ),
                    cross_region=Consistency(
                        target_consistency=AccessConsistency.eventual,
                    ),
                ),
                estimated_mean_read_size_bytes=Interval(
                    low=128, mid=1024, high=65536, confidence=0.95
                ),
                estimated_mean_write_size_bytes=Interval(
                    low=128, mid=1024, high=65536, confidence=0.95
                ),
                estimated_mean_read_latency_ms=Interval(
                    low=0.2, mid=1, high=2, confidence=0.98
                ),
                estimated_mean_write_latency_ms=Interval(
                    low=0.2, mid=1, high=2, confidence=0.98
                ),
                # "Single digit milliseconds SLO"
                read_latency_slo_ms=FixedInterval(
                    minimum_value=0.2,
                    maximum_value=10,
                    low=1,
                    mid=3,
                    high=6,
                    confidence=0.98,
                ),
                write_latency_slo_ms=FixedInterval(
                    minimum_value=0.2,
                    maximum_value=10,
                    low=0.4,
                    mid=2,
                    high=5,
                    confidence=0.98,
                ),
            ),
            # Most Control clusters are small
            data_shape=DataShape(
                estimated_state_size_gib=Interval(
                    low=0.1, mid=1, high=10, confidence=0.98
                ),
                estimated_state_item_count=Interval(
                    low=100000, mid=1000000, high=10000000, confidence=0.98
                ),
                reserved_instance_app_mem_gib=8,
            ),
        )


nflx_control_capacity_model = NflxControlCapacityModel()
