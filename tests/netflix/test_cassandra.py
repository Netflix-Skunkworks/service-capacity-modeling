from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import certain_float
from service_capacity_modeling.interface import certain_int
from service_capacity_modeling.interface import DataShape
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
