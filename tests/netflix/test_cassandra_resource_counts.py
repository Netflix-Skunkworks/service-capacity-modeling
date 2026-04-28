"""Tests for Cassandra cluster-size excuse explainability."""

from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.hardware import shapes
from service_capacity_modeling.interface import (
    AccessPattern,
    Buffer,
    BufferComponent,
    BufferIntent,
    Buffers,
    CapacityDesires,
    CurrentClusters,
    CurrentZoneClusterCapacity,
    DataShape,
    Interval,
    QueryPattern,
    certain_float,
    certain_int,
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


def test_cluster_size_excuse_has_resource_bottleneck_details():
    explained = planner.plan_certain_explained(
        model_name="org.netflix.cassandra",
        region="us-east-1",
        desires=SMALL_KV,
        extra_model_arguments={"required_cluster_size": 2, "require_local_disks": True},
        num_results=5,
    )
    excuses = [e for e in explained.excuses if "resource bottleneck:" in e.reason]
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
        assert e.context["resource_bottleneck"] in counts
        assert e.context["resource_bottleneck"] in e.reason


def test_count_memory_denominator_capped_at_page_cache():
    """reserve_memory(ram) should yield denom = page_cache_capacity_gib.

    For all RAM sizes above ~61 GiB, memory_layout caps page cache at 28 GiB.
    The reserve_memory lambda should return ram - 28, so compute_stateful_zone
    divides needed_memory by 28 — not by the raw uncapped page cache.
    """
    from service_capacity_modeling.models.org.netflix.cassandra_memory import (
        memory_layout,
    )

    base_mem = 3.0
    max_page_cache_gib = 28.0

    def reserve_fn(ram_gib):
        return (
            ram_gib
            - memory_layout(
                ram_gib=ram_gib,
                base_reserves_gib=base_mem,
                max_page_cache_gib=max_page_cache_gib,
            ).page_cache_capacity_gib
        )

    for ram in [61.04, 122.07, 244.14, 384.0]:
        denom = ram - reserve_fn(ram)
        assert denom == max_page_cache_gib, (
            f"ram={ram}: denominator should be {max_page_cache_gib}, got {denom}"
        )


WRITE_HEAVY_KV = CapacityDesires(
    service_tier=1,
    query_pattern=QueryPattern(
        access_pattern=AccessPattern.latency,
        estimated_read_per_second=Interval(
            low=1000, mid=5000, high=10000, confidence=0.98
        ),
        estimated_write_per_second=Interval(
            low=5000, mid=20000, high=40000, confidence=0.98
        ),
    ),
    data_shape=DataShape(
        estimated_state_size_gib=Interval(low=100, mid=200, high=300, confidence=0.98),
    ),
)


def test_memory_scale_down_caps_write_buffer():
    """Memory scale_down should not exceed current write-buffer capacity."""
    i4i_4xl = shapes.instance("i4i.4xlarge")
    current_cluster = CurrentZoneClusterCapacity(
        cluster_instance=i4i_4xl,
        cluster_instance_name="i4i.4xlarge",
        cluster_instance_count=certain_int(4),
        cpu_utilization=certain_float(10),
        memory_utilization_gib=certain_float(10),
        disk_utilization_gib=certain_float(500),
        network_utilization_mbps=certain_float(100),
    )

    uncapped_plans = planner.plan_certain(
        model_name="org.netflix.cassandra",
        region="us-east-1",
        desires=WRITE_HEAVY_KV,
        extra_model_arguments={"require_local_disks": True},
        num_results=1,
    )
    assert uncapped_plans, "Expected uncapped plan"

    capped_desires = WRITE_HEAVY_KV.model_copy(deep=True)
    capped_desires.current_clusters = CurrentClusters(zonal=[current_cluster])
    capped_desires.buffers = Buffers(
        derived={
            "memory": Buffer(
                intent=BufferIntent.scale_down,
                ratio=1.0,
                components=[BufferComponent.memory],
            ),
        }
    )

    capped_plans = planner.plan_certain(
        model_name="org.netflix.cassandra",
        region="us-east-1",
        desires=capped_desires,
        extra_model_arguments={"require_local_disks": True},
        num_results=1,
    )
    assert capped_plans, "Expected capped plan"

    capped_zone = capped_plans[0].candidate_clusters.zonal[0]
    capped_counts = capped_zone.cluster_params["required_nodes_by_type"]

    assert "memory" in capped_counts
    assert capped_zone.cluster_params["resource_bottleneck"] is not None

    for plan in uncapped_plans:
        zone = plan.candidate_clusters.zonal[0]
        if zone.instance.name == capped_zone.instance.name:
            uncapped_mem = zone.cluster_params["required_nodes_by_type"]["memory"]
            assert capped_counts["memory"] <= uncapped_mem, (
                f"Expected capped memory ({capped_counts['memory']}) "
                f"<= uncapped ({uncapped_mem}) for {capped_zone.instance.name}"
            )
            break
