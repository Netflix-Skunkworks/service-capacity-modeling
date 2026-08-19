from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple

from .stateless_java import nflx_java_app_capacity_model
from .time_series_config import TimeSeriesConfiguration
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


# These bailouts send the Cassandra tier back to regular local disk planning.
CASSANDRA_EBS_MAX_READ_PER_SECOND = 200_000
CASSANDRA_EBS_MIN_STATE_SIZE_GIB = 1024

# Ordered bailout rules, each a (name, predicate over the Cassandra facing
# desires) pair. The first rule that holds wins and names the reason we stayed
# on local disks.
CASSANDRA_EBS_BAILOUTS: Tuple[Tuple[str, Callable[[CapacityDesires], bool]], ...] = (
    (
        "read_throughput",
        lambda desires: desires.query_pattern.estimated_read_per_second.mid
        > CASSANDRA_EBS_MAX_READ_PER_SECOND,
    ),
    (
        "state_size",
        lambda desires: desires.data_shape.estimated_state_size_gib.mid
        < CASSANDRA_EBS_MIN_STATE_SIZE_GIB,
    ),
)


def cassandra_ebs_bailout(cassandra_desires: CapacityDesires) -> Optional[str]:
    """Name of the first rule keeping the Cassandra tier on local disks.

    Pass the desires Cassandra will actually be planned with, so after
    TimeSeries read amplification: the read rule is about load landing on
    Cassandra, not load landing on the TimeSeries tier. An unstated dataset
    size reads as zero and therefore bails out to local disks. Returns None
    when nothing bails out and the tier should plan on EBS.
    """
    for name, applies in CASSANDRA_EBS_BAILOUTS:
        if applies(cassandra_desires):
            return name
    return None


class NflxTimeSeriesCapacityModel(CapacityModel):
    @staticmethod
    def capacity_plan(
        instance: Instance,
        drive: Drive,
        context: RegionContext,
        desires: CapacityDesires,
        extra_model_arguments: Dict[str, Any],
    ) -> Optional[CapacityPlan]:
        # TimeSeries wants 20GiB root volumes
        extra_model_arguments.setdefault("root_disk_gib", 20)

        ts_app = nflx_java_app_capacity_model.capacity_plan(
            instance=instance,
            drive=drive,
            context=context,
            desires=desires,
            extra_model_arguments=extra_model_arguments,
        )
        if ts_app is None:
            return None

        for cluster in ts_app.candidate_clusters.regional:
            cluster.cluster_type = "dgwts"
        return ts_app

    @staticmethod
    def description() -> str:
        return "Netflix Streaming TimeSeries Model"

    @staticmethod
    def extra_model_arguments_schema() -> Dict[str, Any]:
        return nflx_java_app_capacity_model.extra_model_arguments_schema()

    @staticmethod
    def compose_with(
        user_desires: CapacityDesires, extra_model_arguments: Dict[str, Any]
    ) -> Tuple[Tuple[str, Callable[[CapacityDesires], CapacityDesires]], ...]:
        # In the future depending on the user desire we might need EVCache
        # as well, e.g. if the latency SLO is reduced
        ts_config = TimeSeriesConfiguration(extra_model_arguments)

        def _modify_cassandra_desires(desires: CapacityDesires) -> CapacityDesires:
            modified = desires.model_copy(deep=True)
            modified.query_pattern.estimated_read_per_second = (
                modified.query_pattern.estimated_read_per_second.scale(
                    ts_config.read_amplification
                )
            )
            return modified

        def _modify_elasticsearch_desires(
            user_desires: CapacityDesires,
        ) -> CapacityDesires:
            relaxed = user_desires.model_copy(deep=True)
            relaxed.query_pattern.access_consistency.same_region.target_consistency = (
                AccessConsistency.eventual
            )
            return relaxed

        # require_local_disks defaults true and on its own excuses every EBS-only
        # instance, so asking for EBS has to clear it too.
        if cassandra_ebs_bailout(_modify_cassandra_desires(user_desires)) is None:
            extra_model_arguments["require_local_disks"] = False
            extra_model_arguments["require_attached_disks"] = True

        if ts_config.search_enabled:
            return (
                ("org.netflix.cassandra", _modify_cassandra_desires),
                ("org.netflix.elasticsearch", _modify_elasticsearch_desires),
            )
        else:
            return (("org.netflix.cassandra", _modify_cassandra_desires),)

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
                    # ts scan queries can be more expensive
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
                # Most throughput ts clusters are large
                data_shape=DataShape(
                    estimated_state_size_gib=Interval(
                        low=100, mid=1000, high=4000, confidence=0.98
                    ),
                    reserved_instance_app_mem_gib=8,
                ),
            )


nflx_time_series_capacity_model = NflxTimeSeriesCapacityModel()
