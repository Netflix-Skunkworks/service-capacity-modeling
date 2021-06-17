from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.interface import AccessConsistency
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import certain_float
from service_capacity_modeling.interface import certain_int
from service_capacity_modeling.interface import Consistency
from service_capacity_modeling.interface import DataShape
from service_capacity_modeling.interface import FixedInterval
from service_capacity_modeling.interface import GlobalConsistency
from service_capacity_modeling.interface import QueryPattern


small_but_high_qps = CapacityDesires(
    service_tier=1,
    query_pattern=QueryPattern(
        estimated_read_per_second=certain_int(100_000),
        estimated_write_per_second=certain_int(100_000),
        estimated_mean_read_latency_ms=certain_float(0.5),
        estimated_mean_write_latency_ms=certain_float(0.4),
    ),
    data_shape=DataShape(
        estimated_state_size_gib=certain_int(10),
    ),
)

high_writes = CapacityDesires(
    service_tier=1,
    query_pattern=QueryPattern(
        estimated_read_per_second=certain_int(10_000),
        estimated_write_per_second=certain_int(100_000),
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
    for require_local_disks in (True, False):
        cap_plan = planner.plan_certain(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=small_but_high_qps,
            extra_model_arguments=dict(require_local_disks=require_local_disks),
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
        extra_model_arguments={"copies_per_region": 2},
    )[0]
    high_writes_result = cap_plan.candidate_clusters.zonal[0]
    assert high_writes_result.instance.family == "m5"
    assert high_writes_result.count > 4

    num_cpus = high_writes_result.instance.cpu * high_writes_result.count
    assert 32 < num_cpus <= 128
    assert high_writes_result.attached_drives[0].size_gib >= 400
    assert cap_plan.candidate_clusters.total_annual_cost < 40_000


def test_capacity_large_footprint():
    cap_plan = planner.plan_certain(
        model_name="org.netflix.cassandra",
        region="us-east-1",
        desires=large_footprint,
        extra_model_arguments=dict(require_local_disks=True, required_cluster_size=16),
    )[0]

    large_footprint_result = cap_plan.candidate_clusters.zonal[0]
    assert large_footprint_result.instance.name.startswith("i3")
    assert large_footprint_result.count == 16


def test_reduced_durability():
    expensive = CapacityDesires(
        service_tier=1,
        query_pattern=QueryPattern(
            estimated_read_per_second=certain_int(1000),
            estimated_write_per_second=certain_int(1_000_000),
        ),
        data_shape=DataShape(
            estimated_state_size_gib=certain_int(100_000),
        ),
    )

    cheaper = CapacityDesires(
        service_tier=1,
        query_pattern=QueryPattern(
            estimated_read_per_second=certain_int(1000),
            estimated_write_per_second=certain_int(1_000_000),
            access_consistency=GlobalConsistency(
                same_region=Consistency(target_consistency=AccessConsistency.eventual)
            ),
        ),
        data_shape=DataShape(
            estimated_state_size_gib=certain_int(100_000),
            durability_slo_order=FixedInterval(low=10, mid=100, high=100000),
        ),
    )

    expensive_plan = planner.plan_certain(
        model_name="org.netflix.cassandra",
        region="us-east-1",
        desires=expensive,
    )[0]

    cheap_plan = planner.plan_certain(
        model_name="org.netflix.cassandra",
        region="us-east-1",
        desires=cheaper,
    )[0]

    assert cheap_plan.candidate_clusters.total_annual_cost < (
        0.7 * float(expensive_plan.candidate_clusters.total_annual_cost)
    )
    # The reduced durability and consistency requirement let's us
    # use less compute
    assert expensive_plan.requirements.zonal[0].context["replication_factor"] == 3
    assert cheap_plan.requirements.zonal[0].context["replication_factor"] == 2

    assert (
        cheap_plan.candidate_clusters.zonal[0].cluster_params["cassandra.keyspace.rf"]
        == 2
    )
