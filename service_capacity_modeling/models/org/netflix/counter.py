from datetime import timedelta
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple

from pydantic import Field

from .stateless_java import nflx_java_app_capacity_model
from .stateless_java import NflxJavaAppArguments
from service_capacity_modeling.enum_utils import StrEnum
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


class NflxCounterCardinality(StrEnum):
    low = "low"
    medium = "medium"
    high = "high"


class NflxCounterMode(StrEnum):
    best_effort = "best-effort"
    eventual = "eventual"
    exact = "exact"


class NflxCounterArguments(NflxJavaAppArguments):
    counter_global: bool = Field(
        alias="counter.global",
        description="Indicate if this use case requires global counts which "
        "is more expensive than regional",
    )
    counter_cardinality: NflxCounterCardinality = Field(
        alias="counter.cardinality",
        description="Low means < 10,000, medium (10,000—1,000,000), high means "
        "> 1,000,000.",
    )
    counter_mode: NflxCounterMode = Field(
        alias="counter.mode",
        description="What mode of counting",
    )


class NflxCounterCapacityModel(CapacityModel):
    @staticmethod
    def capacity_plan(
        instance: Instance,
        drive: Drive,
        context: RegionContext,
        desires: CapacityDesires,
        extra_model_arguments: Dict[str, Any],
    ) -> Optional[CapacityPlan]:
        # Counter wants 20GiB root volumes
        extra_model_arguments.setdefault("root_disk_gib", 20)

        counter_app = nflx_java_app_capacity_model.capacity_plan(
            instance=instance,
            drive=drive,
            context=context,
            desires=desires,
            extra_model_arguments=extra_model_arguments,
        )
        if counter_app is None:
            return None

        for cluster in counter_app.candidate_clusters.regional:
            cluster.cluster_type = "dgwcounter"
        return counter_app

    @staticmethod
    def description() -> str:
        return "Netflix Streaming Counter Model"

    @staticmethod
    def extra_model_arguments_schema() -> Dict[str, Any]:
        return NflxCounterArguments.model_json_schema()

    @staticmethod
    def compose_with(
        user_desires: CapacityDesires, extra_model_arguments: Dict[str, Any]
    ) -> Tuple[Tuple[str, Callable[[CapacityDesires], CapacityDesires]], ...]:
        stores = []

        if extra_model_arguments["counter.mode"] == NflxCounterMode.best_effort:
            stores.append(("org.netflix.evcache", lambda x: x))
        else:
            # Shared evcache cluster is used for eventual and exact counters
            def _modify_cassandra_desires(
                user_desires: CapacityDesires,
            ) -> CapacityDesires:
                modified = user_desires.model_copy(deep=True)
                counter_cardinality = extra_model_arguments["counter.cardinality"]

                counter_deltas_per_second = (
                    user_desires.query_pattern.estimated_write_per_second
                )

                # low cardinality : rollups happen once every 60 seconds
                # medium cardinality : rollups happen once every 30 seconds
                # high cardinality : rollups happen once every 10 seconds
                # TODO: Account for read amplification from time slice configs
                #       for better model accuracy
                if counter_cardinality == NflxCounterCardinality.low:
                    rollups_per_second = counter_deltas_per_second.scale(0.0167)
                elif counter_cardinality == NflxCounterCardinality.medium:
                    rollups_per_second = counter_deltas_per_second.scale(0.0333)
                else:
                    rollups_per_second = counter_deltas_per_second.scale(0.1)

                modified.query_pattern.estimated_read_per_second = rollups_per_second

                # storage size fix
                delta_event_size = 256  # bytes
                rolled_up_count_size = 128  # bytes
                GiB = 1024 * 1024 * 1024

                # Events can be discarded as soon as rollup is complete
                # We default to a 1 day slice with 2 day retention
                retention = timedelta(days=2).total_seconds()

                cardinality = {
                    "low": 10_000,
                    "medium": 100_000,
                    "high": 1_000_000,
                }[extra_model_arguments["counter.cardinality"]]

                event_storage_size = counter_deltas_per_second.scale(
                    delta_event_size * retention / GiB
                )
                rollup_storage_size = rolled_up_count_size * cardinality / GiB
                total_store_size = event_storage_size.offset(rollup_storage_size)
                modified.data_shape.estimated_state_size_gib = total_store_size

                return modified

            stores.append(("org.netflix.cassandra", _modify_cassandra_desires))
        return tuple(stores)

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
                            target_consistency=AccessConsistency.eventual,
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
                    # counter scan queries can be more expensive
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
                # Most throughput counter clusters are large
                data_shape=DataShape(
                    estimated_state_size_gib=Interval(
                        low=100, mid=1000, high=4000, confidence=0.98
                    ),
                    reserved_instance_app_mem_gib=8,
                ),
            )


nflx_counter_capacity_model = NflxCounterCapacityModel()
