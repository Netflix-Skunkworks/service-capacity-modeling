from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import DataShape
from service_capacity_modeling.interface import Interval
from service_capacity_modeling.interface import QueryPattern


def test_zk_tier_0():
    locking_tier_0 = CapacityDesires(
        service_tier=0,
        query_pattern=QueryPattern(
            estimated_read_per_second=Interval(
                low=1, mid=10, high=100, confidence=0.98
            ),
            estimated_write_per_second=Interval(
                low=10, mid=100, high=1000, confidence=0.98
            ),
        ),
        data_shape=DataShape(
            estimated_state_size_gib=Interval(low=0.1, mid=1, high=10, confidence=0.98),
        ),
    )

    plan = planner.plan(
        model_name="org.netflix.zookeeper",
        region="us-east-1",
        desires=locking_tier_0,
    )

    lr = plan.least_regret[0]
    lr_cluster = lr.candidate_clusters.zonal

    assert sum(c.count for c in lr_cluster) == 5
    assert lr_cluster[0].instance.family in ("m5d", "r5d")


def test_zk_tier_1():
    locking_tier_1 = CapacityDesires(
        service_tier=1,
        query_pattern=QueryPattern(
            estimated_read_per_second=Interval(
                low=1, mid=10, high=100, confidence=0.98
            ),
            estimated_write_per_second=Interval(
                low=10, mid=100, high=1000, confidence=0.98
            ),
        ),
        data_shape=DataShape(
            estimated_state_size_gib=Interval(low=0.1, mid=1, high=10, confidence=0.98),
        ),
    )

    plan = planner.plan(
        model_name="org.netflix.zookeeper",
        region="us-east-1",
        desires=locking_tier_1,
    )

    lr = plan.least_regret[0]
    lr_cluster = lr.candidate_clusters.zonal

    assert sum(c.count for c in lr_cluster) == 3
    assert lr_cluster[0].instance.name in (
        "m5d.large",
        "r5d.large",
    )


def test_zk_tier_1_10gb_state():
    tier_1 = CapacityDesires(
        service_tier=1,
        query_pattern=QueryPattern(
            estimated_read_per_second=Interval(
                low=1, mid=10, high=100, confidence=0.98
            ),
            estimated_write_per_second=Interval(
                low=10, mid=100, high=1000, confidence=0.98
            ),
        ),
        data_shape=DataShape(
            estimated_state_size_gib=Interval(low=1, mid=10, high=100, confidence=0.98),
        ),
    )

    plan = planner.plan(
        model_name="org.netflix.zookeeper",
        region="us-east-1",
        desires=tier_1,
    )

    lr = plan.least_regret[0]
    lr_cluster = lr.candidate_clusters.zonal

    assert sum(c.count for c in lr_cluster) == 3
    assert lr_cluster[0].instance.name in (
        "m5d.xlarge",
        "r5d.large",
    )
