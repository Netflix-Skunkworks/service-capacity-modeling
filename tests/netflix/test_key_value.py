"""
Tests for Netflix key-value model.
"""

from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import DataShape
from service_capacity_modeling.interface import Interval
from service_capacity_modeling.interface import QueryPattern

# Property test configuration for KeyValue model.
# See tests/netflix/PROPERTY_TESTING.md for configuration options and examples.
PROPERTY_TEST_CONFIG = {
    # "org.netflix.key-value": {
    #     "extra_model_arguments": {},
    # },
}


def test_a_namespace_with_no_write_rate_can_be_planned():
    """compose_with sees the request as written, before defaults are filled.

    A namespace that states only a read rate -- or neither rate -- arrives with
    a write rate of zero, and dividing reads by it crashed before planning even
    started. The ratio only gates whether EVCache is worth attaching, so no
    writes is read-only, not undefined.
    """
    for query_pattern in (
        QueryPattern(
            estimated_read_per_second=Interval(
                low=100, mid=1000, high=10_000, confidence=0.98
            )
        ),
        QueryPattern(),
    ):
        desires = CapacityDesires(
            service_tier=2,
            query_pattern=query_pattern,
            data_shape=DataShape(
                estimated_state_size_gib=Interval(
                    low=10, mid=100, high=1000, confidence=0.98
                )
            ),
        )
        plans = planner.plan_certain(
            model_name="org.netflix.key-value", region="us-east-1", desires=desires
        )
        assert plans, "a read-only namespace should still produce a plan"
        assert any(
            c.cluster_type == "cassandra" for c in plans[0].candidate_clusters.zonal
        )
