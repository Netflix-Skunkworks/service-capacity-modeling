"""
Tests for Netflix key-value model.
"""

from typing import Dict

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


# 200 GiB / 5000 rps sits near Cassandra's memory bound, so a leaked 24 GiB
# reserve visibly changes the shape it picks. A disk-bound namespace would
# hide the difference.
MEMORY_SENSITIVE_KV = CapacityDesires(
    service_tier=1,
    query_pattern=QueryPattern(
        estimated_read_per_second=Interval(
            low=500, mid=5000, high=50_000, confidence=0.98
        ),
        estimated_write_per_second=Interval(
            low=250, mid=2500, high=25_000, confidence=0.98
        ),
    ),
    data_shape=DataShape(
        estimated_state_size_gib=Interval(low=20, mid=200, high=2000, confidence=0.98),
    ),
)


def _clusters(app_mem_gib: float) -> Dict[str, str]:
    desires = MEMORY_SENSITIVE_KV.model_copy(deep=True)
    desires.data_shape.reserved_instance_app_mem_gib = app_mem_gib
    plan = planner.plan_certain(
        model_name="org.netflix.key-value", region="us-east-1", desires=desires
    )[0]
    clusters = list(plan.candidate_clusters.zonal) + list(
        plan.candidate_clusters.regional
    )
    return {c.cluster_type: f"{c.instance.name}x{c.count}" for c in clusters}


def test_app_memory_reserve_stays_on_the_kv_tier():
    """The dgwkv app's reserve is not Cassandra's reserve.

    reserved_instance_app_mem_gib is memory per instance of the tier it was
    written for. Callers set it for the KV Java app, which runs on its own
    shapes, so raising it must move the dgwkv tier and leave Cassandra alone.
    """
    lean = _clusters(4)
    heavy = _clusters(24)

    assert heavy["cassandra"] == lean["cassandra"]
    assert heavy["dgwkv"] != lean["dgwkv"]


def test_evcache_also_gets_its_own_app_memory_reserve():
    """EVCache runs on its own shapes too, so the KV reserve stops at KV."""
    cached = LARGE_APP_RESERVE_KV.model_copy(deep=True)
    cached.query_pattern.access_consistency.same_region.target_consistency = (
        AccessConsistency.eventual
    )
    cached.query_pattern.estimated_read_per_second = Interval(
        low=30_000, mid=300_000, high=3_000_000, confidence=0.98
    )

    plan = planner.plan_certain(
        model_name="org.netflix.key-value", region="us-east-1", desires=cached
    )[0]
    evcache = [c for c in plan.candidate_clusters.zonal if c.cluster_type == "evcache"]
    assert evcache, "Expected this workload to attach EVCache"

    # EVCache reserves 1 GiB for its own app, so it is free to use small
    # shapes. A node cannot be holding back 24 GiB of dgwkv heap and still
    # fit on a box with less RAM than that.
    for cluster in evcache:
        assert cluster.instance.ram_gib < 24
