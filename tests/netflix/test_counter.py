"""
Tests for Netflix counter model.
"""

from typing import Set

from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import DataShape
from service_capacity_modeling.interface import Interval
from service_capacity_modeling.interface import QueryPattern
from service_capacity_modeling.interface import UncertainCapacityPlan

# Property test configuration for Counter model.
# See tests/netflix/PROPERTY_TESTING.md for configuration options and examples.
PROPERTY_TEST_CONFIG = {
    "org.netflix.counter": {
        "extra_model_arguments": {
            "counter.mode": "exact",
            "counter.cardinality": "high",
        },
    },
}


def test_eventual_counter_storage_targets():
    """Test that counter.mode='eventual' selects Cassandra storage."""
    qps = 1000

    simple = CapacityDesires(
        service_tier=1,
        query_pattern=QueryPattern(
            estimated_read_per_second=Interval(
                low=qps // 10, mid=qps, high=qps * 10, confidence=0.98
            ),
            estimated_write_per_second=Interval(
                low=qps // 10, mid=qps, high=qps * 10, confidence=0.98
            ),
        ),
        data_shape=DataShape(
            estimated_state_size_gib=Interval(
                low=20, mid=200, high=2000, confidence=0.98
            ),
        ),
    )

    cap_plan = planner.plan(
        model_name="org.netflix.counter",
        region="us-east-1",
        desires=simple,
        simulations=256,
        extra_model_arguments={
            "counter.mode": "eventual",
            "counter.cardinality": "high",
        },
    )

    assert extract_storage_types(cap_plan) == {"cassandra", "nflx-java-app"}


def test_best_eff_counter_storage_targets():
    """Test that counter.mode='best-effort' selects EVCache storage."""
    qps = 1000

    simple = CapacityDesires(
        service_tier=1,
        query_pattern=QueryPattern(
            estimated_read_per_second=Interval(
                low=qps // 10, mid=qps, high=qps * 10, confidence=0.98
            ),
            estimated_write_per_second=Interval(
                low=qps // 10, mid=qps, high=qps * 10, confidence=0.98
            ),
        ),
        data_shape=DataShape(
            estimated_state_size_gib=Interval(
                low=20, mid=200, high=2000, confidence=0.98
            ),
        ),
    )

    cap_plan = planner.plan(
        model_name="org.netflix.counter",
        region="us-east-1",
        desires=simple,
        simulations=256,
        extra_model_arguments={
            "counter.mode": "best-effort",
            "counter.cardinality": "high",
        },
    )

    assert extract_storage_types(cap_plan) == {"evcache", "nflx-java-app"}


def extract_storage_types(cap_plan: UncertainCapacityPlan) -> Set[str]:
    """Extract the set of storage types used in a capacity plan."""
    storage_types = set()
    for storage_target in cap_plan.requirements.zonal:
        storage_types.add(storage_target.requirement_type)
    for storage_target in cap_plan.requirements.regional:
        storage_types.add(storage_target.requirement_type)
    return storage_types
