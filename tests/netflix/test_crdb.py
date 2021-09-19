from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import DataShape
from service_capacity_modeling.interface import Interval
from service_capacity_modeling.interface import QueryPattern


def test_crdb_basic():
    basic = CapacityDesires(
        service_tier=1,
        query_pattern=QueryPattern(
            estimated_read_per_second=Interval(
                low=100, mid=1000, high=10000, confidence=0.98
            ),
            estimated_write_per_second=Interval(
                low=100, mid=1000, high=10000, confidence=0.98
            ),
        ),
        data_shape=DataShape(
            estimated_state_size_gib=Interval(
                low=10, mid=100, high=1000, confidence=0.98
            ),
        ),
    )

    plan = planner.plan(
        model_name="org.netflix.cockroachdb",
        region="us-east-1",
        desires=basic,
    )

    lr = plan.least_regret[0]
    lr_cluster = lr.candidate_clusters.zonal[0]

    # Resulting cluster should not be too expensive
    assert 2000 < lr.candidate_clusters.total_annual_cost < 10_000

    # Should have enough disk space for around 80GiB of data in a single
    # replica (compression). Also that drive should be ephemeral
    assert lr_cluster.instance.drive is not None
    assert lr_cluster.count * lr_cluster.instance.drive.size_gib > 80

    # Should have enough CPU to handle 1000 QPS
    assert lr_cluster.count * lr_cluster.instance.cpu > 4


def test_crdb_footprint():
    space = CapacityDesires(
        service_tier=1,
        query_pattern=QueryPattern(
            estimated_read_per_second=Interval(
                low=100, mid=1000, high=10000, confidence=0.98
            ),
            estimated_write_per_second=Interval(
                low=100, mid=1000, high=10000, confidence=0.98
            ),
        ),
        data_shape=DataShape(
            estimated_state_size_gib=Interval(
                low=100, mid=1000, high=10000, confidence=0.98
            ),
        ),
    )

    plan = planner.plan(
        model_name="org.netflix.cockroachdb",
        region="us-east-1",
        desires=space,
    )

    lr = plan.least_regret[0]
    lr_cluster = lr.candidate_clusters.zonal[0]

    # Resulting cluster should not be too expensive
    assert 4000 < lr.candidate_clusters.total_annual_cost < 12_000

    # Should have enough disk space for around 80GiB of data in a single
    # replica (compression). Also that drive should be ephemeral
    assert lr_cluster.instance.drive is not None
    assert lr_cluster.count * lr_cluster.instance.drive.size_gib > 800

    # Should have enough CPU to handle 1000 QPS
    assert lr_cluster.count * lr_cluster.instance.cpu >= 8
