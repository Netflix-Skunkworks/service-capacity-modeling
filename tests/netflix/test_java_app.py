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


def test_java_app():
    java_cap_plan = planner.plan_certain(
        model_name="org.netflix.stateless-java",
        region="us-east-1",
        desires=large_footprint,
    )[0]
    java_result = java_cap_plan.candidate_clusters.regional[0]
    cores = java_result.count * java_result.instance.cpu
    assert java_result.instance.name.startswith("m5.")
    assert 100 <= cores <= 300

    java_cap_plan = planner.plan(
        model_name="org.netflix.stateless-java",
        region="us-east-1",
        desires=small_but_high_qps,
    ).least_regret[0]
    java_result = java_cap_plan.candidate_clusters.regional[0]
    cores = java_result.count * java_result.instance.cpu
    assert java_result.instance.name.startswith("m5.")
    assert 100 <= cores <= 300
