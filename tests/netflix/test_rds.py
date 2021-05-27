from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.interface import CapacityDesires, Interval
from service_capacity_modeling.interface import DataShape
from service_capacity_modeling.interface import QueryPattern
from service_capacity_modeling.interface import certain_float
from service_capacity_modeling.interface import certain_int

tier_0 = CapacityDesires(
    service_tier=0,
    query_pattern=QueryPattern(
        estimated_read_per_second=certain_int(200),
        estimated_write_per_second=certain_int(100),
        estimated_mean_read_latency_ms=certain_float(20),
        estimated_mean_write_latency_ms=certain_float(20),
    ),
    data_shape=DataShape(
        estimated_state_size_gib=certain_int(200),
    ),
)

small_footprint = CapacityDesires(
    service_tier=1,
    query_pattern=QueryPattern(
        estimated_read_per_second=certain_int(200),
        estimated_write_per_second=certain_int(100),
        estimated_mean_read_latency_ms=certain_float(10),
        estimated_mean_write_latency_ms=certain_float(10),
    ),
    data_shape=DataShape(
        estimated_state_size_gib=certain_int(50),
    ),
)

mid_footprint = CapacityDesires(
    service_tier=1,
    query_pattern=QueryPattern(
        estimated_read_per_second=certain_int(300),
        estimated_write_per_second=certain_int(150),
        estimated_mean_read_latency_ms=certain_float(20),
        estimated_mean_write_latency_ms=certain_float(20),
    ),
    data_shape=DataShape(
        estimated_state_size_gib=certain_int(400),
    ),
)

large_footprint = CapacityDesires(
    service_tier=1,
    query_pattern=QueryPattern(
        estimated_read_per_second=certain_int(1000),
        estimated_write_per_second=certain_int(1000),
        estimated_mean_read_latency_ms=certain_float(20),
        estimated_mean_write_latency_ms=certain_float(20),
    ),
    data_shape=DataShape(
        estimated_state_size_gib=certain_int(800),
        estimated_working_set_percent=Interval(
                    low=0.05,
                    mid=0.30,
                    high=0.50,
                    confidence=0.8
                )
    ),
)

tier_3 = CapacityDesires(
    service_tier=3,
    query_pattern=QueryPattern(
        estimated_read_per_second=certain_int(200),
        estimated_write_per_second=certain_int(100),
        estimated_mean_read_latency_ms=certain_float(20),
        estimated_mean_write_latency_ms=certain_float(20),
    ),
    data_shape=DataShape(
        estimated_state_size_gib=certain_int(200),
    ),
)


def test_tier_0_not_supported():
    cap_plan = planner.plan_certain(
        model_name="org.netflix.rds",
        region="us-east-1",
        desires=tier_0,
    )
    # RDS can't support tier 0 service
    assert len(cap_plan) == 0


def test_small_footprint():
    cap_plan = planner.plan_certain(
        model_name="org.netflix.rds",
        region="us-east-1",
        desires=small_footprint,
    )
    assert cap_plan[0].candidate_clusters.regional[0].instance.name == "r5.xlarge"
    assert len(cap_plan[0].candidate_clusters.regional) == 2  # has replica


def test_medium_footprint():
    cap_plan = planner.plan_certain(
        model_name="org.netflix.rds",
        region="us-east-1",
        desires=mid_footprint
    )
    assert cap_plan[0].candidate_clusters.regional[0].instance.name == "r5.8xlarge"
    assert len(cap_plan[0].candidate_clusters.regional) == 2  # has replica


def test_large_footprint():
    cap_plan = planner.plan_certain(
        model_name="org.netflix.rds",
        region="us-east-1",
        desires=large_footprint,
    )
    print("result", cap_plan, "\n")
    assert cap_plan[0].candidate_clusters.regional[0].instance.name == "r5.8xlarge"
    assert len(cap_plan[0].candidate_clusters.regional) == 2  # has replica


def test_tier_3():
    cap_plan = planner.plan_certain(
        model_name="org.netflix.rds",
        region="us-east-1",
        desires=tier_3,
    )
    assert cap_plan[0].candidate_clusters.regional[0].instance.name == "r5.4xlarge"
    assert len(cap_plan[0].candidate_clusters.regional) == 1  # no replica
    print(cap_plan[1].candidate_clusters.regional[0].instance.name)
