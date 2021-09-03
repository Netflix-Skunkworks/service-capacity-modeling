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


def test_uncertain_java_app():
    uncertain = CapacityDesires(
        service_tier=1,
        query_pattern=QueryPattern(
            estimated_read_per_second=Interval(
                low=2_000, mid=30_000, high=60_000, confidence=0.98
            ),
            estimated_write_per_second=Interval(
                low=2_000, mid=30_000, high=60_000, confidence=0.98
            ),
        ),
        # Should be ignored
        data_shape=DataShape(
            estimated_state_size_gib=Interval(low=50, mid=500, high=1000),
            reserved_instance_app_mem_gib=4,
        ),
    )

    java_cap_plan = planner.plan(
        model_name="org.netflix.stateless-java",
        region="us-east-1",
        desires=uncertain,
    )
    java_least_regret = java_cap_plan.least_regret[0]
    java_result = java_least_regret.candidate_clusters.regional[0]

    cores = java_result.count * java_result.instance.cpu
    assert java_result.instance.name.startswith("m5.")
    assert 100 <= cores <= 300

    # KeyValue regional clusters should match
    kv_cap_plan = planner.plan(
        model_name="org.netflix.key-value",
        region="us-east-1",
        desires=uncertain,
    )
    kv_least_regret = kv_cap_plan.least_regret[0]
    kv_result = kv_least_regret.candidate_clusters.regional[0]

    kv_cores = kv_result.count * kv_result.instance.cpu
    assert kv_result.instance.name.startswith("m5.")
    assert 0.5 <= float(kv_cores) / cores <= 1.5

    assert kv_least_regret.candidate_clusters.zonal[0].count > 0


def test_java_heap_heavy():
    large_heap = CapacityDesires(
        service_tier=1,
        query_pattern=QueryPattern(
            estimated_read_per_second=Interval(
                low=2_000, mid=10_000, high=60_000, confidence=0.98
            ),
            estimated_write_per_second=Interval(
                low=2_000, mid=10_000, high=60_000, confidence=0.98
            ),
        ),
        data_shape=DataShape(
            reserved_instance_app_mem_gib=40,
        ),
    )

    java_cap_plan = planner.plan(
        model_name="org.netflix.stateless-java",
        region="us-east-1",
        desires=large_heap,
    )
    java_least_regret = java_cap_plan.least_regret[0]
    java_result = java_least_regret.candidate_clusters.regional[0]

    cores = java_result.count * java_result.instance.cpu
    assert 20 <= cores <= 100
    assert java_result.instance.ram_gib > 40

    # Should bump the heap due to the traffic
    large_traffic = CapacityDesires(
        service_tier=1,
        query_pattern=QueryPattern(
            estimated_read_per_second=Interval(
                low=2_000, mid=30_000, high=60_000, confidence=0.98
            ),
            estimated_write_per_second=Interval(
                low=2_000, mid=30_000, high=60_000, confidence=0.98
            ),
            estimated_mean_write_size_bytes=Interval(
                low=1024, mid=32768, high=262144, confidence=0.98
            ),
        ),
    )

    java_cap_plan = planner.plan(
        model_name="org.netflix.stateless-java",
        region="us-east-1",
        desires=large_traffic,
    )
    java_least_regret = java_cap_plan.least_regret[0]
    java_result = java_least_regret.candidate_clusters.regional[0]

    cores = java_result.count * java_result.instance.cpu
    assert 100 <= cores <= 300
    # 32 KiB payloads * 30k/second is around 1 GiB per second
    # which should require a decent chunk of heap memory
    memory = java_result.count * java_result.instance.ram_gib
    assert memory > 50
