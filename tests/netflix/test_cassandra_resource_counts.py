"""Tests for cassandra cluster_size excuse resource breakdown
and write_buffer derived buffer integration."""

from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.interface import (
    AccessPattern,
    CapacityDesires,
    DataShape,
    Interval,
    QueryPattern,
)

SMALL_KV = CapacityDesires(
    service_tier=1,
    query_pattern=QueryPattern(
        access_pattern=AccessPattern.latency,
        estimated_read_per_second=Interval(
            low=1000, mid=5000, high=10000, confidence=0.98
        ),
        estimated_write_per_second=Interval(
            low=1000, mid=5000, high=10000, confidence=0.98
        ),
    ),
    data_shape=DataShape(
        estimated_state_size_gib=Interval(low=100, mid=200, high=300, confidence=0.98),
    ),
)


def test_cluster_size_excuse_has_resource_breakdown():
    """cluster_size excuses should include resource_counts and binding_resource."""
    explained = planner.plan_certain_explained(
        model_name="org.netflix.cassandra",
        region="us-east-1",
        desires=SMALL_KV,
        extra_model_arguments={"required_cluster_size": 2, "require_local_disks": True},
        num_results=5,
    )
    excuses = [e for e in explained.excuses if "driven by" in e.reason]
    assert excuses, "Expected cluster_size excuses with binding resource"
    for e in excuses:
        rc = e.context["resource_counts"]
        assert set(rc.keys()) == {"cpu", "memory", "network", "disk", "disk_iops"}
        assert e.context["binding_resource"] in rc
