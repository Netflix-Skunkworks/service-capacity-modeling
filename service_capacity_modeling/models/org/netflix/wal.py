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


class NflxWALCapacityModel(CapacityModel):
    @staticmethod
    def capacity_plan(
        instance: Instance,
        drive: Drive,
        context: RegionContext,
        desires: CapacityDesires,
        extra_model_arguments: Dict[str, Any],
    ) -> Optional[CapacityPlan]:
        wal_app = nflx_java_app_capacity_model.capacity_plan(
            instance=instance,
            drive=drive,
            context=context,
            desires=desires,
            extra_model_arguments=extra_model_arguments,
        )
        if wal_app is None:
            return None

        for cluster in wal_app.candidate_clusters.regional:
            cluster.cluster_type = "dgwwal"
        return wal_app

    @staticmethod
    def description() -> str:
        return "Netflix Streaming WAL Model"

    @staticmethod
    def extra_model_arguments_schema() -> Dict[str, Any]:
        return nflx_java_app_capacity_model.extra_model_arguments_schema()

    @staticmethod
    def compose_with(
        user_desires: CapacityDesires, extra_model_arguments: Dict[str, Any]
    ) -> Tuple[Tuple[str, Callable[[CapacityDesires], CapacityDesires]], ...]:
        provision_kv_shard = extra_model_arguments.get("wal.provisionkvshard", False)
        if provision_kv_shard:
            return (("org.netflix.key-value", lambda x: x),)
        else:
            return ()

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
                    low=64, mid=128, high=1024, confidence=0.95
                ),
                estimated_mean_read_latency_ms=Interval(
                    low=0.2, mid=4, high=6, confidence=0.98
                ),
                estimated_mean_write_latency_ms=Interval(
                    low=0.2, mid=1, high=2, confidence=0.98
                ),
                read_latency_slo_ms=FixedInterval(
                    minimum_value=1,
                    maximum_value=100,
                    low=1,
                    mid=8,
                    high=90,
                    confidence=0.98,
                ),
                write_latency_slo_ms=FixedInterval(
                    minimum_value=1,
                    maximum_value=20,
                    low=2,
                    mid=4,
                    high=10,
                    confidence=0.98,
                ),
            ),
            # Most throughput WAL clusters are large
            data_shape=DataShape(
                estimated_state_size_gib=Interval(
                    low=100, mid=1000, high=4000, confidence=0.98
                ),
                reserved_instance_app_mem_gib=8,
            ),
        )


nflx_wal_capacity_model = NflxWALCapacityModel()
