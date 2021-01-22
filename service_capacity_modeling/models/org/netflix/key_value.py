from typing import Optional
from typing import Sequence
from typing import Tuple

from .cassandra import NflxCassandraCapacityModel
from .stateless_java import NflxJavaAppCapacityModel
from service_capacity_modeling.interface import AccessPattern
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import CapacityPlan
from service_capacity_modeling.interface import DataShape
from service_capacity_modeling.interface import Drive
from service_capacity_modeling.interface import FixedInterval
from service_capacity_modeling.interface import Instance
from service_capacity_modeling.interface import Interval
from service_capacity_modeling.interface import QueryPattern
from service_capacity_modeling.models import CapacityModel
from service_capacity_modeling.models.common import merge_plan


class NflxKeyValueCapacityModel(CapacityModel):
    @staticmethod
    def capacity_plan(
        instance: Instance,
        drive: Drive,
        desires: CapacityDesires,
        **kwargs,
    ) -> Optional[CapacityPlan]:
        cass_cluster = NflxCassandraCapacityModel().capacity_plan(
            instance=instance, drive=drive, desires=desires, **kwargs
        )
        kv_app = NflxJavaAppCapacityModel().capacity_plan(
            instance=instance, drive=drive, desires=desires, **kwargs
        )
        if cass_cluster is None or kv_app is None:
            return None

        for cluster in kv_app.candidate_clusters.regional:
            cluster.cluster_type = "dgwkv"
        return merge_plan(cass_cluster, kv_app)

    @staticmethod
    def description():
        return "Netflix Streaming Key-Value Model"

    @staticmethod
    def extra_model_arguments() -> Sequence[Tuple[str, str, str]]:
        return tuple(NflxCassandraCapacityModel.extra_model_arguments()) + tuple(
            NflxJavaAppCapacityModel.extra_model_arguments()
        )

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
                    # Cassandra point queries usualy take just around 1ms
                    # of on CPU time for reads and 0.6ms for writes
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
