"""Tests for Cassandra cluster-size excuse explainability."""

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


def test_cluster_size_excuse_has_count_bottleneck_details():
    explained = planner.plan_certain_explained(
        model_name="org.netflix.cassandra",
        region="us-east-1",
        desires=SMALL_KV,
        extra_model_arguments={"required_cluster_size": 2, "require_local_disks": True},
        num_results=5,
    )
    excuses = [e for e in explained.excuses if "count bottleneck:" in e.reason]
    assert excuses, "Expected cluster_size excuses with count bottleneck details"
    for e in excuses:
        counts = e.context["required_nodes_by_type"]
        assert set(counts.keys()) == {
            "cpu",
            "memory",
            "network",
            "disk_capacity",
            "disk_iops",
            "cluster_size",
            "min_count",
        }
        assert e.context["count_bottleneck"] in counts
        assert e.context["count_bottleneck"] in e.reason
