from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import DataShape
from service_capacity_modeling.interface import Interval
from service_capacity_modeling.interface import QueryPattern


uncertain_mid = CapacityDesires(
    service_tier=1,
    query_pattern=QueryPattern(
        estimated_read_per_second=Interval(
            low=1000, mid=10000, high=100000, confidence=0.98
        ),
        estimated_write_per_second=Interval(
            low=1000, mid=10000, high=100000, confidence=0.98
        ),
    ),
    data_shape=DataShape(
        estimated_state_size_gib=Interval(low=100, mid=500, high=1000, confidence=0.98),
    ),
)

uncertain_tiny = CapacityDesires(
    service_tier=2,
    query_pattern=QueryPattern(
        estimated_read_per_second=Interval(low=1, mid=10, high=100, confidence=0.98),
        estimated_write_per_second=Interval(low=1, mid=10, high=100, confidence=0.98),
    ),
    data_shape=DataShape(
        estimated_state_size_gib=Interval(low=1, mid=10, high=30, confidence=0.98),
    ),
)


def test_uncertain_planning():
    mid_plan = planner.plan(
        model_name="org.netflix.cassandra",
        region="us-east-1",
        desires=uncertain_mid,
    )
    lr = mid_plan.least_regret[0]
    lr_cluster = lr.candidate_clusters.zonal[0]
    assert 8 <= lr_cluster.count * lr_cluster.instance.cpu <= 64
    assert 5_000 <= lr.candidate_clusters.total_annual_cost < 40_000

    sr = mid_plan.least_regret[1]
    sr_cluster = sr.candidate_clusters.zonal[0]
    assert 8 <= sr_cluster.count * sr_cluster.instance.cpu <= 64
    assert 5_000 <= sr.candidate_clusters.total_annual_cost < 40_000

    tiny_plan = planner.plan(
        model_name="org.netflix.cassandra",
        region="us-east-1",
        desires=uncertain_tiny,
    )
    lr = tiny_plan.least_regret[0]
    lr_cluster = lr.candidate_clusters.zonal[0]
    assert 2 <= lr_cluster.count * lr_cluster.instance.cpu < 16
    assert 1_000 < lr.candidate_clusters.total_annual_cost < 6_000


def test_increasing_qps_simple():
    qps_values = (100, 1000, 10_000, 100_000)
    result = []
    for qps in qps_values:
        simple = CapacityDesires(
            service_tier=1,
            query_pattern=QueryPattern(
                estimated_read_per_second=Interval(
                    low=qps // 10, mid=qps, high=qps * 10, confidence=0.98
                ),
                estimated_write_per_second=Interval(
                    low=qps // 10, mid=qps, high=qps * 10, confidence=0.98
                ),
            ),
            data_shape=DataShape(
                estimated_state_size_gib=Interval(
                    low=20, mid=200, high=2000, confidence=0.98
                ),
            ),
        )

        cap_plan = planner.plan(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=simple,
            simulations=256,
        )

        lr = cap_plan.least_regret[0].candidate_clusters.zonal[0]
        lr_cpu = lr.count * lr.instance.cpu
        lr_cost = cap_plan.least_regret[0].candidate_clusters.total_annual_cost
        lr_family = lr.instance.family
        if lr.instance.drive is None:
            assert sum(dr.size_gib for dr in lr.attached_drives) >= 200
        else:
            assert lr.instance.drive.size_gib >= 100

        result.append(
            (lr_family, lr_cpu, lr_cost, cap_plan.least_regret[0].requirements.zonal[0])
        )

    # We should generally want cheap CPUs
    assert all(r[0] in ("r5", "m5d", "m5", "i3") for r in result)

    # Should have more capacity as requirement increases
    x = [r[1] for r in result]
    assert x[0] < x[-1]
    assert sorted(x) == x


def test_worn_dataset():
    """Assert that a write once read never (aka tracing) dataset uses
    CPU and GP2 cloud drives to max ability. Paying for fast ephmeral storage
    is silly when we're never reading from it.
    """
    worn_desire = CapacityDesires(
        service_tier=1,
        query_pattern=QueryPattern(
            # Very Very few reads.
            estimated_read_per_second=Interval(
                low=1, mid=10, high=100, confidence=0.98
            ),
            # We think we're going to have around 1 million writes per second
            estimated_write_per_second=Interval(
                low=100_000, mid=1_000_000, high=2_000_000, confidence=0.98
            ),
        ),
        # We think we're going to have around 200 TiB of data
        data_shape=DataShape(
            estimated_state_size_gib=Interval(
                low=104800, mid=204800, high=404800, confidence=0.98
            ),
        ),
    )
    cap_plan = planner.plan(
        model_name="org.netflix.cassandra",
        region="us-east-1",
        desires=worn_desire,
        extra_model_arguments=dict(
            max_regional_size=200,
            copies_per_region=2,
        ),
    )

    lr = cap_plan.least_regret[0]
    lr_cluster = lr.candidate_clusters.zonal[0]
    assert 256 <= lr_cluster.count * lr_cluster.instance.cpu <= 1024
    assert 100_000 <= lr.candidate_clusters.total_annual_cost < 500_000
    assert lr_cluster.instance.name.startswith(
        "m5."
    ) or lr_cluster.instance.name.startswith("r5.")
    assert lr_cluster.attached_drives[0].name == "gp2"
    assert lr_cluster.attached_drives[0].size_gib * lr_cluster.count * 3 > 204800
    # We should have S3 backup cost
    assert lr.candidate_clusters.services[0].annual_cost > 5_000


def test_very_small_has_disk():
    very_small = CapacityDesires(
        service_tier=2,
        query_pattern=QueryPattern(
            estimated_read_per_second=Interval(
                low=1, mid=10, high=100, confidence=0.98
            ),
            estimated_write_per_second=Interval(
                low=1, mid=10, high=100, confidence=0.98
            ),
        ),
        data_shape=DataShape(
            estimated_state_size_gib=Interval(low=1, mid=10, high=30, confidence=0.98),
        ),
    )
    cap_plan = planner.plan(
        model_name="org.netflix.cassandra", region="us-east-1", desires=very_small
    )

    for lr in cap_plan.least_regret:
        lr_cluster = lr.candidate_clusters.zonal[0]
        assert 2 <= lr_cluster.count * lr_cluster.instance.cpu < 16
        assert 1_000 < lr.candidate_clusters.total_annual_cost < 6_000
        if lr_cluster.instance.drive is None:
            assert sum(dr.size_gib for dr in lr_cluster.attached_drives) > 10
        else:
            assert lr_cluster.instance.drive.size_gib > 10
