"""TimeSeries planning its Cassandra tier on EBS, and the cases that stay local.

Every case plans a real TimeSeries namespace end to end, so what is asserted is
the disk type the Cassandra tier actually lands on.
"""

import pytest

from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.interface import AccessPattern
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import DataShape
from service_capacity_modeling.interface import Interval
from service_capacity_modeling.interface import QueryPattern
from service_capacity_modeling.models.org.netflix.time_series import (
    CASSANDRA_EBS_MAX_READ_PER_SECOND,
)
from service_capacity_modeling.models.org.netflix.time_series import (
    CASSANDRA_EBS_MIN_STATE_SIZE_GIB,
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
READ_AMPLIFICATION = 5


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


def _assert_on_local_disks(clusters):
    for cluster in clusters:
        assert cluster.instance.drive is not None, (
            f"{cluster.instance.name} is EBS only"
        )
        assert not cluster.attached_drives


def test_large_low_read_namespace_lands_on_ebs():
    _assert_on_ebs(_cassandra_tier(_namespace(4_000, 10_000)))


@pytest.mark.parametrize(
    "state_gib,on_ebs",
    [
        (CASSANDRA_EBS_MIN_STATE_SIZE_GIB, True),
        (CASSANDRA_EBS_MIN_STATE_SIZE_GIB - 1, False),
    ],
    ids=["at_one_tib", "just_under_one_tib"],
)
def test_namespace_smaller_than_one_tib_stays_on_local_disks(state_gib, on_ebs):
    clusters = _cassandra_tier(_namespace(state_gib, 10_000))

    (_assert_on_ebs if on_ebs else _assert_on_local_disks)(clusters)


@pytest.mark.parametrize(
    "reads_per_second,on_ebs",
    [
        (CASSANDRA_EBS_MAX_READ_PER_SECOND, True),
        (CASSANDRA_EBS_MAX_READ_PER_SECOND + 1, False),
    ],
    ids=["at_the_read_ceiling", "just_over_the_read_ceiling"],
)
def test_high_read_namespace_stays_on_local_disks(reads_per_second, on_ebs):
    clusters = _cassandra_tier(_namespace(4_000, reads_per_second))

    (_assert_on_ebs if on_ebs else _assert_on_local_disks)(clusters)


@pytest.mark.parametrize(
    "reads_per_second,on_ebs",
    [
        (CASSANDRA_EBS_MAX_READ_PER_SECOND // READ_AMPLIFICATION, True),
        (CASSANDRA_EBS_MAX_READ_PER_SECOND // READ_AMPLIFICATION + 1, False),
    ],
    ids=["amplifies_to_the_ceiling", "amplifies_over_the_ceiling"],
)
def test_read_amplification_counts_against_the_read_ceiling(reads_per_second, on_ebs):
    # The read ceiling is about load landing on Cassandra, so a namespace well
    # under it at its own front door can still be over it once amplified.
    clusters = _cassandra_tier(
        _namespace(4_000, reads_per_second), AMPLIFYING_NAMESPACE
    )

    (_assert_on_ebs if on_ebs else _assert_on_local_disks)(clusters)


def test_namespace_without_a_stated_size_stays_on_local_disks():
    unsized = _namespace(4_000, 10_000)
    unsized.data_shape = DataShape()

    _assert_on_local_disks(_cassandra_tier(unsized))


def test_uncertain_plan_of_a_large_low_read_namespace_lands_on_ebs():
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


def test_timeseries_tier_is_unchanged_by_the_ebs_choice():
    plan = planner.plan_certain(
        model_name="org.netflix.time-series",
        region="us-east-1",
        desires=_namespace(4_000, 10_000),
        extra_model_arguments=dict(NAMESPACE),
    )[0]

    assert [cluster.cluster_type for cluster in plan.candidate_clusters.regional] == [
        "dgwts"
    ]
