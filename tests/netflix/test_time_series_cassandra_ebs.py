"""TimeSeries composes with Cassandra's attached-storage preference."""

from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.interface import AccessPattern
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import DataShape
from service_capacity_modeling.interface import Interval
from service_capacity_modeling.interface import QueryPattern
from service_capacity_modeling.models.org.netflix.time_series import (
    NflxTimeSeriesCapacityModel,
)
from service_capacity_modeling.models.org.netflix.time_series_config import (
    TimeSeriesConfiguration,
)


# A namespace retaining 30 days of events, read back one day at a time. The
# read interval fits inside a slice, so Cassandra sees one read per TS read.
NAMESPACE = {
    "ts.read-interval": "PT24H",
    "ts.hot.retention-interval": "PT720H",
    "ts.events-per-day-per-ts": "10",
    "ts.event-size": "1024",
}

# A denser namespace whose day of events spills into five buckets per id, so
# every TimeSeries read fans out to five Cassandra reads.
AMPLIFYING_NAMESPACE = {
    "ts.read-interval": "PT24H",
    "ts.hot.retention-interval": "PT96H",
    "ts.events-per-day-per-ts": "1000",
    "ts.event-size": "20000",
}


def _namespace(
    state_gib: int, reads_per_second: int, writes_per_second: int = 50_000
) -> CapacityDesires:
    """A TimeSeries namespace: steady event ingest with range reads over it."""
    return CapacityDesires(
        service_tier=1,
        query_pattern=QueryPattern(
            access_pattern=AccessPattern.throughput,
            estimated_read_per_second=Interval(
                low=reads_per_second // 2,
                mid=reads_per_second,
                high=reads_per_second * 2,
                confidence=0.98,
            ),
            estimated_write_per_second=Interval(
                low=writes_per_second // 2,
                mid=writes_per_second,
                high=writes_per_second * 2,
                confidence=0.98,
            ),
            estimated_mean_read_size_bytes=Interval(
                low=1024, mid=4096, high=65536, confidence=0.95
            ),
            estimated_mean_write_size_bytes=Interval(
                low=128, mid=1024, high=4096, confidence=0.95
            ),
        ),
        data_shape=DataShape(
            estimated_state_size_gib=Interval(
                low=state_gib // 2,
                mid=state_gib,
                high=state_gib * 2,
                confidence=0.98,
            ),
        ),
    )


def _cassandra_tier(desires: CapacityDesires, namespace=None):
    plan = planner.plan_certain(
        model_name="org.netflix.time-series",
        region="us-east-1",
        desires=desires,
        extra_model_arguments=dict(namespace or NAMESPACE),
    )[0]
    clusters = [
        cluster
        for cluster in plan.candidate_clusters.zonal
        if cluster.cluster_type == "cassandra"
    ]
    assert clusters, "planner returned no Cassandra tier at all"
    return clusters


def _assert_on_ebs(clusters):
    for cluster in clusters:
        assert cluster.instance.drive is None, (
            f"{cluster.instance.name} has local disks"
        )
        assert [drive.name for drive in cluster.attached_drives] == ["gp3"]


def test_timeseries_uses_cassandra_default_storage_preference():
    _assert_on_ebs(_cassandra_tier(_namespace(4_000, 10_000)))


def test_timeseries_applies_read_amplification_to_cassandra_desires():
    desires = _namespace(4_000, 40_000)
    ((child_model, modify_child_desires),) = NflxTimeSeriesCapacityModel.compose_with(
        desires, dict(AMPLIFYING_NAMESPACE)
    )
    amplification = TimeSeriesConfiguration(AMPLIFYING_NAMESPACE).read_amplification

    cassandra_desires = modify_child_desires(desires)

    assert child_model == "org.netflix.cassandra"
    assert cassandra_desires.query_pattern.estimated_read_per_second == (
        desires.query_pattern.estimated_read_per_second.scale(amplification)
    )


def test_timeseries_tier_is_unchanged_by_cassandra_storage():
    plan = planner.plan_certain(
        model_name="org.netflix.time-series",
        region="us-east-1",
        desires=_namespace(4_000, 10_000),
        extra_model_arguments=dict(NAMESPACE),
    )[0]

    assert [cluster.cluster_type for cluster in plan.candidate_clusters.regional] == [
        "dgwts"
    ]


def test_timeseries_does_not_store_cassandra_policy_in_caller_arguments():
    extra_model_arguments = dict(NAMESPACE)

    planner.plan_certain(
        model_name="org.netflix.time-series",
        region="us-east-1",
        desires=_namespace(4_000, 10_000),
        extra_model_arguments=extra_model_arguments,
    )

    assert "require_local_disks" not in extra_model_arguments
    assert "require_attached_disks" not in extra_model_arguments


def test_uncertain_timeseries_plan_uses_ebs_for_cassandra():
    plan = planner.plan(
        model_name="org.netflix.time-series",
        region="us-east-1",
        desires=_namespace(4_000, 10_000),
        extra_model_arguments=dict(NAMESPACE),
        simulations=32,
    ).least_regret[0]

    _assert_on_ebs(
        [
            cluster
            for cluster in plan.candidate_clusters.zonal
            if cluster.cluster_type == "cassandra"
        ]
    )
