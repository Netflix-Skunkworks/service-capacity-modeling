from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import certain_float
from service_capacity_modeling.interface import certain_int
from service_capacity_modeling.interface import DataShape
from service_capacity_modeling.interface import Interval
from service_capacity_modeling.interface import QueryPattern

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
        estimated_read_per_second=certain_int(100),
        estimated_write_per_second=certain_int(100),
        estimated_mean_read_latency_ms=certain_float(1),
        estimated_mean_write_latency_ms=certain_float(1),
    ),
    data_shape=DataShape(
        estimated_state_size_gib=certain_int(50),
    ),
)

large_footprint = CapacityDesires(
    service_tier=1,
    query_pattern=QueryPattern(
        estimated_read_per_second=certain_int(2000),
        estimated_write_per_second=certain_int(3000),
        estimated_mean_read_latency_ms=certain_float(20),
        estimated_mean_write_latency_ms=certain_float(20),
    ),
    data_shape=DataShape(
        estimated_state_size_gib=certain_int(10000),
        estimated_working_set_percent=Interval(
            low=0.05, mid=0.30, high=0.50, confidence=0.8
        ),
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
        model_name="org.netflix.postgres",
        region="us-east-1",
        desires=tier_0,
        num_regions=1,
    )
    # Aurora can't support tier 0 service
    assert len(cap_plan) == 0


def test_small_footprint():
    cap_plan = planner.plan_certain(
        model_name="org.netflix.postgres",
        region="us-east-1",
        desires=small_footprint,
        num_regions=1,
    )
    assert cap_plan[0].candidate_clusters.regional[0].instance.name == "db.r5.large"

    # two instance plus storage and io
    assert (
        2000
        < cap_plan[0].candidate_clusters.annual_costs[
            "aurora-cluster.regional-clusters"
        ]
        < 4500
    )


def test_small_footprint_multi_region():
    cap_plan = planner.plan_certain(
        model_name="org.netflix.postgres",
        region="us-east-1",
        desires=small_footprint,
        num_regions=3,
    )
    assert cap_plan[0].candidate_clusters.regional[0].instance.name == "db.r5.large"

    assert 2000 < cap_plan[0].candidate_clusters.total_annual_cost < 4000


def test_small_footprint_plan_uncertain():
    cap_plan = planner.plan(
        model_name="org.netflix.postgres",
        region="us-east-1",
        desires=small_footprint,
        num_regions=1,
        simulations=256,
    )
    plan_a = cap_plan.least_regret[0]

    assert plan_a.candidate_clusters.regional[0].instance.name == "db.r5.large"

    assert 2000 < plan_a.candidate_clusters.total_annual_cost < 4000


def test_large_footprint():
    cap_plan = planner.plan_certain(
        model_name="org.netflix.postgres",
        region="us-east-1",
        desires=large_footprint,
        num_regions=1,
    )
    #Aurora cannot handle the scale, so don't recommend anything
    assert cap_plan == []

def test_tier_3():
    cap_plan = planner.plan_certain(
        model_name="org.netflix.postgres",
        region="us-east-1",
        desires=tier_3,
        num_regions=1,
    )
    assert cap_plan[0].candidate_clusters.regional[0].instance.name == "db.r5.2xlarge"
