"""TimeSeries composes with Cassandra's attached-storage preference."""

import pytest

from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.interface import AccessPattern
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import DataShape
from service_capacity_modeling.interface import Interval
from service_capacity_modeling.interface import QueryPattern


NAMESPACE = {
    "ts.read-interval": "PT24H",
    "ts.hot.retention-interval": "PT720H",
    "ts.events-per-day-per-ts": "10",
    "ts.event-size": "1024",
}

AMPLIFYING_NAMESPACE = {
    "ts.read-interval": "PT24H",
    "ts.hot.retention-interval": "PT96H",
    "ts.events-per-day-per-ts": "1000",
    "ts.event-size": "20000",
}


def _namespace(reads_per_second: int, state_gib: int = 4_000) -> CapacityDesires:
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
                low=25_000,
                mid=50_000,
                high=100_000,
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


def _plan(desires: CapacityDesires, extra_model_arguments):
    return planner.plan_certain(
        model_name="org.netflix.time-series",
        region="us-east-1",
        desires=desires,
        extra_model_arguments=extra_model_arguments,
    )[0]


def _cassandra_clusters(plan):
    clusters = [
        cluster
        for cluster in plan.candidate_clusters.zonal
        if cluster.cluster_type == "cassandra"
    ]
    assert clusters
    return clusters


def _assert_on_ebs(clusters):
    assert clusters
    for cluster in clusters:
        assert cluster.instance.drive is None
        assert [drive.name for drive in cluster.attached_drives] == ["gp3"]


@pytest.mark.parametrize(
    "state_gib,reads_per_second,namespace",
    [
        (4_000, 10_000, NAMESPACE),
        (1_024, 10_000, NAMESPACE),
        (4_000, 200_000, NAMESPACE),
        (4_000, 40_000, AMPLIFYING_NAMESPACE),
    ],
    ids=["large_low_read", "one_tib", "read_ceiling", "read_amplified"],
)
def test_original_timeseries_ebs_workloads_remain_on_ebs(
    state_gib, reads_per_second, namespace
):
    plan = _plan(
        _namespace(reads_per_second, state_gib=state_gib),
        dict(namespace),
    )

    _assert_on_ebs(_cassandra_clusters(plan))


def test_timeseries_tier_is_unchanged_by_cassandra_storage():
    plan = _plan(_namespace(10_000), dict(NAMESPACE))

    assert [cluster.cluster_type for cluster in plan.candidate_clusters.regional] == [
        "dgwts"
    ]


def test_timeseries_does_not_store_cassandra_policy_in_caller_arguments():
    extra_model_arguments = dict(NAMESPACE)

    _plan(_namespace(10_000), extra_model_arguments)

    assert "require_local_disks" not in extra_model_arguments
    assert "require_attached_disks" not in extra_model_arguments


def test_uncertain_timeseries_plan_uses_ebs_for_cassandra():
    plan = planner.plan(
        model_name="org.netflix.time-series",
        region="us-east-1",
        desires=_namespace(10_000),
        extra_model_arguments=dict(NAMESPACE),
        simulations=16,
        num_results=1,
    ).least_regret[0]

    _assert_on_ebs(_cassandra_clusters(plan))
