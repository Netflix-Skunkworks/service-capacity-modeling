from typing import Optional
from typing import Sequence
from typing import Tuple

from .stateless_java import nflx_java_app_capacity_model
from service_capacity_modeling.interface import AccessPattern
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import CapacityPlan
from service_capacity_modeling.interface import DataShape
from service_capacity_modeling.interface import Drive
from service_capacity_modeling.interface import FixedInterval
from service_capacity_modeling.interface import Instance
from service_capacity_modeling.interface import Interval
from service_capacity_modeling.interface import QueryPattern
from service_capacity_modeling.interface import RegionContext
from service_capacity_modeling.models import CapacityModel


class NflxKeyValueCapacityModel(CapacityModel):
    @staticmethod
    def capacity_plan(
        instance: Instance,
        drive: Drive,
        context: RegionContext,
        desires: CapacityDesires,
        **kwargs,
    ) -> Optional[CapacityPlan]:
        # KeyValue wants 20GiB root volumes
        java_root_size = kwargs.pop("root_disk_gib", 20)
        desires = desires.merge_with(
            nflx_key_value_capacity_model.default_desires(desires, **kwargs)
        )

        kv_app = nflx_java_app_capacity_model.capacity_plan(
            instance=instance,
            drive=drive,
            context=context,
            desires=desires,
            root_disk_gib=java_root_size,
            **kwargs,
        )
        if kv_app is None:
            return None

        for cluster in kv_app.candidate_clusters.regional:
            cluster.cluster_type = "dgwkv"
        return kv_app

    @staticmethod
    def description():
        return "Netflix Streaming Key-Value Model"

    @staticmethod
    def extra_model_arguments() -> Sequence[Tuple[str, str, str]]:
        return nflx_java_app_capacity_model.extra_model_arguments()

    @staticmethod
    def compose_with(user_desires: CapacityDesires, **kwargs) -> Tuple[str, ...]:
        # In the future depending on the user desire we might need EVCache
        # as well, e.g. if the latency SLO is reduced
        return ("org.netflix.cassandra",)

    @staticmethod
    def default_desires(user_desires, **kwargs):
        if user_desires.query_pattern.access_pattern == AccessPattern.latency:
            return CapacityDesires(
                query_pattern=QueryPattern(
                    access_pattern=AccessPattern.latency,
                    estimated_mean_read_size_bytes=Interval(
                        low=128, mid=1024, high=65536, confidence=0.95
                    ),
                    estimated_mean_write_size_bytes=Interval(
                        low=64, mid=128, high=1024, confidence=0.95
                    ),
                    estimated_mean_read_latency_ms=Interval(
                        low=0.2, mid=1, high=10, confidence=0.98
                    ),
                    estimated_mean_write_latency_ms=Interval(
                        low=0.2, mid=0.6, high=2, confidence=0.98
                    ),
                    # "Single digit milliseconds SLO"
                    read_latency_slo_ms=FixedInterval(
                        low=0.4, mid=2.5, high=10, confidence=0.98
                    ),
                    write_latency_slo_ms=FixedInterval(
                        low=0.4, mid=2, high=10, confidence=0.98
                    ),
                ),
                # Most KeyValue clusters are small
                data_shape=DataShape(
                    estimated_state_size_gib=Interval(
                        low=10, mid=50, high=200, confidence=0.98
                    )
                ),
            )
        else:
            return CapacityDesires(
                query_pattern=QueryPattern(
                    access_pattern=AccessPattern.latency,
                    estimated_mean_read_size_bytes=Interval(
                        low=128, mid=1024, high=65536, confidence=0.95
                    ),
                    estimated_mean_write_size_bytes=Interval(
                        low=64, mid=128, high=1024, confidence=0.95
                    ),
                    # KV scan queries can be slower
                    estimated_mean_read_latency_ms=Interval(
                        low=0.2, mid=4, high=20, confidence=0.98
                    ),
                    estimated_mean_write_latency_ms=Interval(
                        low=0.2, mid=0.6, high=2, confidence=0.98
                    ),
                    # "Single digit milliseconds SLO"
                    read_latency_slo_ms=FixedInterval(
                        low=0.4, mid=4, high=10, confidence=0.98
                    ),
                    write_latency_slo_ms=FixedInterval(
                        low=0.4, mid=4, high=10, confidence=0.98
                    ),
                ),
                # Most throughput KV clusters are large
                data_shape=DataShape(
                    estimated_state_size_gib=Interval(
                        low=100, mid=1000, high=4000, confidence=0.98
                    )
                ),
            )


nflx_key_value_capacity_model = NflxKeyValueCapacityModel()
