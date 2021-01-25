from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import DataShape
from service_capacity_modeling.interface import Interval
from service_capacity_modeling.interface import QueryPattern


uncertain_mid = CapacityDesires(
    service_tier=1,
    query_pattern=QueryPattern(
        estimated_read_per_second=Interval(
            low=1000, mid=10000, high=100000, confidence=0.9
        ),
        estimated_write_per_second=Interval(
            low=1000, mid=10000, high=100000, confidence=0.9
        ),
    ),
    data_shape=DataShape(
        estimated_state_size_gib=Interval(low=100, mid=500, high=1000, confidence=0.9),
    ),
)

uncertain_tiny = CapacityDesires(
    service_tier=2,
    query_pattern=QueryPattern(
        estimated_read_per_second=Interval(low=1, mid=10, high=100, confidence=0.9),
        estimated_write_per_second=Interval(low=1, mid=10, high=100, confidence=0.9),
    ),
    data_shape=DataShape(
        estimated_state_size_gib=Interval(low=1, mid=10, high=30, confidence=0.9),
    ),
)


def test_uncertain_planning_ebs():
    # with cProfile.Profile() as pr:
    mid_plan = planner.plan(
        model_name="org.netflix.cassandra",
        region="us-east-1",
        desires=uncertain_mid,
        allow_gp2=True,
    )
    lr = mid_plan.least_regret[0]
    lr_cluster = lr.candidate_clusters.zonal[0]
    assert 12 <= lr_cluster.count * lr_cluster.instance.cpu <= 64
    assert 5_000 <= lr.candidate_clusters.total_annual_cost < 50_000

    tiny_plan = planner.plan(
        model_name="org.netflix.cassandra",
        region="us-east-1",
        desires=uncertain_tiny,
        allow_gp2=True,
    )
    lr = tiny_plan.least_regret[0]
    lr_cluster = lr.candidate_clusters.zonal[0]
    assert 4 < lr_cluster.count * lr_cluster.instance.cpu < 16
    assert 2_000 < lr.candidate_clusters.total_annual_cost < 8_000


def test_increasing_qps_simple():
    qps_values = (100, 1000, 10_000, 100_000)
    result = []
    for qps in qps_values:
        simple = CapacityDesires(
            service_tier=1,
            query_pattern=QueryPattern(
                estimated_read_per_second=Interval(
                    low=qps // 10, mid=qps, high=qps * 10, confidence=0.9
                ),
                estimated_write_per_second=Interval(
                    low=qps // 10, mid=qps, high=qps * 10, confidence=0.9
                ),
            ),
            data_shape=DataShape(
                estimated_state_size_gib=Interval(
                    low=20, mid=200, high=2000, confidence=0.9
                ),
            ),
        )

        cap_plan = planner.plan(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=simple,
            allow_gp2=True,
        )
        # pr.print_stats()

        lr = cap_plan.least_regret[0].candidate_clusters.zonal[0]
        lr_cpu = lr.count * lr.instance.cpu
        lr_cost = cap_plan.least_regret[0].candidate_clusters.total_annual_cost
        lr_family = lr.instance.family
        result.append(
            (lr_family, lr_cpu, lr_cost, cap_plan.least_regret[0].requirement)
        )

    # We should generally want CPU
    assert all([r[0] in ("m5d", "m5") for r in result])

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
            estimated_read_per_second=Interval(low=1, mid=10, high=100, confidence=0.9),
            # We think we're going to have around 1 million writes per second
            estimated_write_per_second=Interval(
                low=100_000, mid=1_000_000, high=2_000_000, confidence=0.9
            ),
        ),
        # We think we're going to have around 200 TiB of data
        data_shape=DataShape(
            estimated_state_size_gib=Interval(
                low=104800, mid=204800, high=404800, confidence=0.9
            ),
        ),
    )
    cap_plan = planner.plan(
        model_name="org.netflix.cassandra",
        region="us-east-1",
        desires=worn_desire,
        max_regional_size=200,
        copies_per_region=2,
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