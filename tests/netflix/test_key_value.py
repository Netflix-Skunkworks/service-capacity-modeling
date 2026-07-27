"""
Tests for Netflix key-value model.
"""

from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.interface import AccessConsistency
from service_capacity_modeling.interface import AccessPattern
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import Consistency
from service_capacity_modeling.interface import DataShape
from service_capacity_modeling.interface import GlobalConsistency
from service_capacity_modeling.interface import Interval
from service_capacity_modeling.interface import QueryPattern

# Property test configuration for KeyValue model.
# See tests/netflix/PROPERTY_TESTING.md for configuration options and examples.
PROPERTY_TEST_CONFIG = {
    # "org.netflix.key-value": {
    #     "extra_model_arguments": {},
    # },
}

# A KV namespace that reserves 24 GiB for the app. That reserve flows through
# to the composed Cassandra model, where it overruns small shapes.
LARGE_APP_RESERVE_KV = CapacityDesires(
    service_tier=0,
    query_pattern=QueryPattern(
        access_pattern=AccessPattern.latency,
        access_consistency=GlobalConsistency(
            same_region=Consistency(
                target_consistency=AccessConsistency.read_your_writes
            ),
        ),
        estimated_read_per_second=Interval(
            low=287, mid=2870, high=28700, confidence=0.98
        ),
        estimated_write_per_second=Interval(
            low=138.1, mid=1381, high=13810, confidence=0.98
        ),
        estimated_mean_write_size_bytes=Interval(
            low=102.4, mid=1024, high=10240, confidence=0.98
        ),
    ),
    data_shape=DataShape(
        estimated_state_size_gib=Interval(
            low=1520.5, mid=15205.1, high=152050.7, confidence=0.98
        ),
        estimated_state_item_count=Interval(
            low=10_000_000, mid=100_000_000, high=1_000_000_000, confidence=0.98
        ),
        reserved_instance_app_mem_gib=24.0,
    ),
)


def test_large_app_memory_reserve_plans_on_shapes_that_fit():
    """Reserved app memory can overrun a small shape's RAM.

    24 GiB of reserved app memory plus the JVM heap leaves no page cache on
    shapes like i4i.xlarge, which used to divide by zero while sizing the
    Cassandra cluster. Those shapes are now excused and planning continues on
    shapes with RAM to spare.
    """
    plans = planner.plan_certain(
        model_name="org.netflix.key-value",
        region="us-east-1",
        desires=LARGE_APP_RESERVE_KV,
    )

    assert plans, "Expected a plan on shapes large enough for the app reserve"
    for plan in plans:
        for cluster in plan.candidate_clusters.zonal:
            assert cluster.instance.ram_gib > 40


def test_large_app_memory_reserve_survives_simulation():
    """The uncertain path walks the same shapes and must not blow up either."""
    plan = planner.plan(
        model_name="org.netflix.key-value",
        region="us-east-1",
        desires=LARGE_APP_RESERVE_KV,
        simulations=32,
    )

    assert plan.least_regret
