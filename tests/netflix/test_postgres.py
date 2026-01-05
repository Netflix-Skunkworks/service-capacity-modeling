from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import certain_float
from service_capacity_modeling.interface import certain_int
from service_capacity_modeling.interface import DataShape
from service_capacity_modeling.interface import Interval
from service_capacity_modeling.interface import QueryPattern
from tests.util import assert_similar_compute
from tests.util import shape

# Property test configuration for PostgreSQL model.
# See tests/netflix/PROPERTY_TESTING.md for configuration options and examples.
PROPERTY_TEST_CONFIG = {
    "org.netflix.postgres": {
        # PostgreSQL requires num_regions parameter
        "extra_model_arguments": {"num_regions": 1},
        # PostgreSQL has restrictive limits: max ~500 QPS total, max ~50 GiB
        # These limits are based on Aurora's capabilities as the underlying service
        # Note: Property tests use same QPS for reads+writes, so total = 2x this value
        "qps_range": (50, 250),
        "data_range_gib": (10, 50),
        # PostgreSQL doesn't support tier 0, so test tier 1 vs tier 2 instead
        # (tier 0 support is inferred from tier_range[0] > 0)
        "tier_range": (1, 2),
    },
}

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
    leader = cap_plan[0].candidate_clusters.regional[0].instance
    expected = shape("db.r6g.large")
    assert_similar_compute(expected_shape=expected, actual_shape=leader)

    # two instance plus storage and io
    assert (
        1500
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
    leader = cap_plan[0].candidate_clusters.regional[0].instance
    expected = shape("db.r6g.large")
    assert_similar_compute(expected_shape=expected, actual_shape=leader)

    assert 1500 < cap_plan[0].candidate_clusters.total_annual_cost < 4000


def test_small_footprint_plan_uncertain():
    cap_plan = planner.plan(
        model_name="org.netflix.postgres",
        region="us-east-1",
        desires=small_footprint,
        num_regions=1,
    )
    plan_a = cap_plan.least_regret[0]

    leader = plan_a.candidate_clusters.regional[0].instance
    expected = shape("db.r6g.large")
    assert_similar_compute(expected_shape=expected, actual_shape=leader)

    assert 1500 < plan_a.candidate_clusters.total_annual_cost < 4000


def test_large_footprint():
    cap_plan = planner.plan_certain(
        model_name="org.netflix.postgres",
        region="us-east-1",
        desires=large_footprint,
        num_regions=1,
    )
    # Aurora cannot handle the scale, so don't recommend anything
    assert cap_plan == []


def test_tier_3():
    cap_plan = planner.plan_certain(
        model_name="org.netflix.postgres",
        region="us-east-1",
        desires=tier_3,
        num_regions=1,
    )

    leader = cap_plan[0].candidate_clusters.regional[0].instance
    expected = shape("db.r6g.2xlarge")
    assert_similar_compute(expected_shape=expected, actual_shape=leader)
