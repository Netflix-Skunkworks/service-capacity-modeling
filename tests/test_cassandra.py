from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.models import CapacityDesires
from service_capacity_modeling.models import certain_float
from service_capacity_modeling.models import certain_int
from service_capacity_modeling.models import DataShape
from service_capacity_modeling.models import QueryPattern


small_but_high_qps = CapacityDesires(
    service_tier=1,
    query_pattern=QueryPattern(
        estimated_read_per_second=certain_int(100000),
        estimated_write_per_second=certain_int(100000),
        estimated_mean_read_latency_ms=certain_float(0.4),
        estimated_mean_write_latency_ms=certain_float(0.4),
    ),
    data_shape=DataShape(
        estimated_state_size_gb=certain_int(10),
        estimated_working_set_percent=certain_float(0.5),
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
        estimated_state_size_gb=certain_int(300),
        estimated_working_set_percent=certain_float(0.1),
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
        estimated_state_size_gb=certain_int(4000),
        estimated_working_set_percent=certain_float(0.05),
    ),
)


def test_capacity_small_fast():
    for allow_ebs in (True, False):
        cap_plan = planner.plan_certain(
            model_name="nflx_cassandra",
            region="us-east-1",
            desires=small_but_high_qps,
            allow_gp2=allow_ebs,
        )
        small_result = cap_plan.candidate_clusters[0].zonal[0]
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
        model_name="nflx_cassandra",
        region="us-east-1",
        desires=high_writes,
        copies_per_region=2,
    )
    high_writes_result = cap_plan.candidate_clusters[0].zonal[0]
    assert high_writes_result.instance.name == "m5.2xlarge"
    assert high_writes_result.count == 4
    assert high_writes_result.attached_drives[0].size_gib >= 500


def test_capacity_large_footprint():
    cap_plan = planner.plan_certain(
        model_name="nflx_cassandra",
        region="us-east-1",
        desires=large_footprint,
        allow_gp2=False,
        required_cluster_size=4,
    )

    large_footprint_result = cap_plan.candidate_clusters[0].zonal[0]
    assert large_footprint_result.instance.name == "i3en.3xlarge"
    assert large_footprint_result.count == 4

    java_cap_plan = planner.plan_certain(
        model_name="nflx_stateless_java_app",
        region="us-east-1",
        desires=large_footprint,
    )
    java_result = java_cap_plan.candidate_clusters[0].regional[0]
    cores = java_result.count * java_result.instance.cpu
    assert java_result.instance.name.startswith("m5")
    assert 100 <= cores <= 200
