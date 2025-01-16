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


class NflxKeyValueCapacityModel(CapacityModel):
    @staticmethod
    def capacity_plan(
        instance: Instance,
        drive: Drive,
        context: RegionContext,
        desires: CapacityDesires,
        extra_model_arguments: Dict[str, Any],
    ) -> Optional[CapacityPlan]:
        # KeyValue wants 20GiB root volumes
        extra_model_arguments.setdefault("root_disk_gib", 20)

        kv_app = nflx_java_app_capacity_model.capacity_plan(
            instance=instance,
            drive=drive,
            context=context,
            desires=desires,
            extra_model_arguments=extra_model_arguments,
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
    def extra_model_arguments_schema() -> Dict[str, Any]:
        return nflx_java_app_capacity_model.extra_model_arguments_schema()

    @staticmethod
    def compose_with(
        user_desires: CapacityDesires, extra_model_arguments: Dict[str, Any]
    ) -> Tuple[Tuple[str, Callable[[CapacityDesires], CapacityDesires]], ...]:
        query_pattern = user_desires.query_pattern
        target_consistency = (
            query_pattern.access_consistency.same_region.target_consistency
        )
        rps_interval = query_pattern.estimated_read_per_second
        rps: float = rps_interval.mid
        wps: float = query_pattern.estimated_write_per_second.mid
        read_write_ratio: float = rps / wps

        # Parameterizing this in case we want to configure it to something else later.
        # The read/write ratio should be relatively high to make EVCache effective.
        evcache_rw_ratio_threshold: float = extra_model_arguments.get(
            "kv_evcache_read_write_ratio_threshold", 0.9
        )
        use_evcache = target_consistency in (
            AccessConsistency.eventual,
            AccessConsistency.best_effort,
        ) and (
            rps > 250_000
            or (rps > 100_000 and read_write_ratio > evcache_rw_ratio_threshold)
        )

        if use_evcache:

            def _modify_cassandra_desires(
                desires: CapacityDesires,
            ) -> CapacityDesires:
                relaxed = desires.model_copy(deep=True)

                # This is an initial best guess. Parameterizing in case we want to
                # configure it in the future.
                estimated_kv_cache_hit_rate: float = extra_model_arguments.get(
                    "estimated_kv_cache_hit_rate", 0.8
                )

                # Scale down the Cassandra estimated rps since those reads will be
                # serviced by EVCache.
                relaxed.query_pattern.estimated_read_per_second = rps_interval.scale(
                    1 - estimated_kv_cache_hit_rate
                )
                return relaxed

            def _modify_evcache_desires(
                desires: CapacityDesires,
            ) -> CapacityDesires:
                relaxed = desires.model_copy(deep=True)
                access_consistency = relaxed.query_pattern.access_consistency
                access_consistency.same_region.target_consistency = (
                    AccessConsistency.best_effort
                )
                return relaxed

            return (
                ("org.netflix.cassandra", _modify_cassandra_desires),
                ("org.netflix.evcache", _modify_evcache_desires),
            )
        else:
            return (("org.netflix.cassandra", lambda x: x),)

    @staticmethod
    def default_desires(user_desires, extra_model_arguments):
        if user_desires.query_pattern.access_pattern == AccessPattern.latency:
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
                # Most KeyValue clusters are small
                data_shape=DataShape(
                    estimated_state_size_gib=Interval(
                        low=10, mid=50, high=200, confidence=0.98
                    ),
                    reserved_instance_app_mem_gib=8,
                ),
            )
        else:
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
                    # KV scan queries can be more expensive
                    estimated_mean_read_latency_ms=Interval(
                        low=0.2, mid=4, high=6, confidence=0.98
                    ),
                    estimated_mean_write_latency_ms=Interval(
                        low=0.2, mid=1, high=2, confidence=0.98
                    ),
                    # Assume they're doing GetItems scans -> slow reads
                    read_latency_slo_ms=FixedInterval(
                        minimum_value=1,
                        maximum_value=100,
                        low=1,
                        mid=8,
                        high=90,
                        confidence=0.98,
                    ),
                    # Assume they're doing PutRecords (BATCH)
                    write_latency_slo_ms=FixedInterval(
                        minimum_value=1,
                        maximum_value=20,
                        low=2,
                        mid=4,
                        high=10,
                        confidence=0.98,
                    ),
                ),
                # Most throughput KV clusters are large
                data_shape=DataShape(
                    estimated_state_size_gib=Interval(
                        low=100, mid=1000, high=4000, confidence=0.98
                    ),
                    reserved_instance_app_mem_gib=8,
                ),
            )


nflx_key_value_capacity_model = NflxKeyValueCapacityModel()
