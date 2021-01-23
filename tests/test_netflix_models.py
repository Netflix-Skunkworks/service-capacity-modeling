from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import certain_float
from service_capacity_modeling.interface import certain_int
from service_capacity_modeling.interface import DataShape
from service_capacity_modeling.interface import Interval
from service_capacity_modeling.interface import QueryPattern


small_but_high_qps = CapacityDesires(
    service_tier=1,
    query_pattern=QueryPattern(
        estimated_read_per_second=certain_int(100000),
        estimated_write_per_second=certain_int(100000),
        estimated_mean_read_latency_ms=certain_float(0.4),
        estimated_mean_write_latency_ms=certain_float(0.4),
    ),
    data_shape=DataShape(
        estimated_state_size_gib=certain_int(10),
    ),
)

high_writes = CapacityDesires(
    service_tier=1,
    query_pattern=QueryPattern(
        estimated_read_per_second=certain_int(10000),
        estimated_write_per_second=certain_int(100000),
        estimated_mean_read_latency_ms=certain_float(0.4),
        estimated_mean_write_latency_ms=certain_float(0.3),
    ),
    data_shape=DataShape(
        estimated_state_size_gib=certain_int(300),
    ),
)

large_footprint = CapacityDesires(
    service_tier=1,
    query_pattern=QueryPattern(
        estimated_read_per_second=certain_int(60000),
        estimated_write_per_second=certain_int(60000),
        estimated_mean_read_latency_ms=certain_float(0.8),
        estimated_mean_write_latency_ms=certain_float(0.5),
    ),
    data_shape=DataShape(
        estimated_state_size_gib=certain_int(4000),
    ),
)


def test_capacity_small_fast():
    for allow_ebs in (True, False):
        cap_plan = planner.plan_certain(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=small_but_high_qps,
            allow_gp2=allow_ebs,
        )[0]
        small_result = cap_plan.candidate_clusters.zonal[0]
        # We really should just pay for CPU here
        assert small_result.instance.name.startswith("m5")

        cores = small_result.count * small_result.instance.cpu
        assert 50 <= cores <= 80
        # Even though it's a small dataset we need IOs so should end up
        # with lots of ebs_gp2 to handle the read IOs
        if small_result.attached_drives:
            assert sum(d.size_gib for d in small_result.attached_drives) > 1000


def test_capacity_high_writes():
    cap_plan = planner.plan_certain(
        model_name="org.netflix.cassandra",
        region="us-east-1",
        desires=high_writes,
        copies_per_region=2,
    )[0]
    high_writes_result = cap_plan.candidate_clusters.zonal[0]
    assert high_writes_result.instance.name == "m5d.2xlarge"
    assert high_writes_result.count == 2
    assert high_writes_result.instance.drive is not None
    assert high_writes_result.instance.drive.size_gib >= 200


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
    assert 5_000 <= lr.candidate_clusters.total_annual_cost.mid < 50_000

    tiny_plan = planner.plan(
        model_name="org.netflix.cassandra",
        region="us-east-1",
        desires=uncertain_tiny,
        allow_gp2=True,
    )
    lr = tiny_plan.least_regret[0]
    lr_cluster = lr.candidate_clusters.zonal[0]
    assert 4 < lr_cluster.count * lr_cluster.instance.cpu < 16
    assert 2_000 < lr.candidate_clusters.total_annual_cost.mid < 8_000


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
    assert all([r[0] in ("m5d", "i3") for r in result])

    # Should have more capacity as requirement increases
    x = [r[1] for r in result]
    assert x[0] < x[-1]
    assert sorted(x) == x


def test_capacity_large_footprint():
    cap_plan = planner.plan_certain(
        model_name="org.netflix.cassandra",
        region="us-east-1",
        desires=large_footprint,
        allow_gp2=False,
        required_cluster_size=4,
    )[0]

    large_footprint_result = cap_plan.candidate_clusters.zonal[0]
    assert large_footprint_result.instance.name == "i3en.3xlarge"
    assert large_footprint_result.count == 4


def test_java_app():
    java_cap_plan = planner.plan_certain(
        model_name="org.netflix.stateless-java",
        region="us-east-1",
        desires=large_footprint,
    )[0]
    java_result = java_cap_plan.candidate_clusters.regional[0]
    cores = java_result.count * java_result.instance.cpu
    assert java_result.instance.name.startswith("m5")
    assert 100 <= cores <= 300

    java_cap_plan = planner.plan(
        model_name="org.netflix.stateless-java",
        region="us-east-1",
        desires=small_but_high_qps,
    ).least_regret[0]
    java_result = java_cap_plan.candidate_clusters.regional[0]
    cores = java_result.count * java_result.instance.cpu
    assert java_result.instance.name.startswith("m5")
    assert 100 <= cores <= 300
