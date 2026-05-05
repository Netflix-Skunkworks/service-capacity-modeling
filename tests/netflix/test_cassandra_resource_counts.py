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
    Drive,
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

    For all RAM sizes above ~61 GiB, MemoryLayout caps page cache at 28 GiB.
    The reserve_memory lambda should return ram - 28, so compute_stateful_zone
    divides needed_memory by 28 — not by the raw uncapped page cache.
    """
    from service_capacity_modeling.models.org.netflix.cassandra_memory import (
        DEFAULT_MAX_PAGE_CACHE_GIB,
        MemoryLayout,
    )

    base_mem = 3.0

    def reserve_fn(ram_gib):
        return (
            ram_gib
            - MemoryLayout.for_ram(
                ram_gib=ram_gib,
                base_reserves_gib=base_mem,
            ).page_cache_capacity_gib
        )

    for ram in [61.04, 122.07, 244.14, 384.0]:
        denom = ram - reserve_fn(ram)
        assert denom == DEFAULT_MAX_PAGE_CACHE_GIB, (
            f"ram={ram}: denominator should be {DEFAULT_MAX_PAGE_CACHE_GIB}, "
            f"got {denom}"
        )


def test_existing_count_does_not_seed_memory_count():
    """A large current topology is not itself a page-cache memory demand."""
    current_cluster = CurrentZoneClusterCapacity(
        cluster_instance=shapes.instance("r6a.4xlarge"),
        cluster_instance_name="r6a.4xlarge",
        cluster_drive=Drive(
            name="gp3",
            drive_type="attached-ssd",
            size_gib=1000,
        ),
        cluster_instance_count=certain_int(64),
        cpu_utilization=certain_float(1),
        disk_utilization_gib=certain_float(50),
        network_utilization_mbps=certain_float(1),
    )
    desires = CapacityDesires(
        service_tier=1,
        query_pattern=QueryPattern(
            access_pattern=AccessPattern.latency,
            estimated_read_per_second=certain_int(100),
            estimated_write_per_second=certain_int(100),
        ),
        data_shape=DataShape(estimated_state_size_gib=certain_int(100)),
        current_clusters=CurrentClusters(zonal=[current_cluster] * 3),
        buffers=Buffers(
            derived={
                "storage": Buffer(
                    ratio=1.0,
                    intent=BufferIntent.scale_down,
                    components=[BufferComponent.storage],
                )
            }
        ),
    )

    plans = planner.plan_certain(
        model_name="org.netflix.cassandra",
        region="us-east-1",
        desires=desires,
        extra_model_arguments={
            "require_local_disks": False,
        },
        instance_families=["m7a"],
        num_results=1,
    )

    params = plans[0].candidate_clusters.zonal[0].cluster_params
    counts = params["required_nodes_by_type"]

    assert counts["memory"] == 1
    assert counts["memory"] < current_cluster.cluster_instance_count.mid
    assert counts["memory"] <= max(
        counts["cpu"],
        counts["disk_capacity"],
        counts["disk_iops"],
        counts["network"],
    )


def test_memory_scale_up_floors_existing_page_cache_capacity():
    current_instance = shapes.instance("r6a.4xlarge")
    current_cluster = CurrentZoneClusterCapacity(
        cluster_instance=current_instance,
        cluster_instance_name=current_instance.name,
        cluster_drive=Drive(
            name="gp3",
            drive_type="attached-ssd",
            size_gib=1000,
        ),
        cluster_instance_count=certain_int(2),
        cluster_type="cassandra",
        cpu_utilization=certain_float(1),
        disk_utilization_gib=certain_float(50),
        network_utilization_mbps=certain_float(1),
    )
    desires = CapacityDesires(
        service_tier=1,
        query_pattern=QueryPattern(
            access_pattern=AccessPattern.latency,
            estimated_read_per_second=certain_int(100),
            estimated_write_per_second=certain_int(100),
        ),
        data_shape=DataShape(estimated_state_size_gib=certain_int(100)),
        current_clusters=CurrentClusters(zonal=[current_cluster]),
        buffers=Buffers(
            derived={
                "memory": Buffer(
                    intent=BufferIntent.scale_up,
                    ratio=1.0,
                    components=[BufferComponent.memory],
                )
            }
        ),
    )

    plans = planner.plan_certain(
        model_name="org.netflix.cassandra",
        region="us-east-1",
        desires=desires,
        extra_model_arguments={
            "require_local_disks": False,
            "require_attached_disks": True,
        },
        instance_families=["m7a"],
        num_results=1,
    )

    requirement = plans[0].requirements.zonal[0]
    counts = (
        plans[0].candidate_clusters.zonal[0].cluster_params["required_nodes_by_type"]
    )

    assert requirement.mem_gib.mid > 1
    assert counts["memory"] > max(
        counts["cpu"],
        counts["disk_capacity"],
        counts["disk_iops"],
        counts["network"],
    )


def test_normal_page_cache_not_dumped_in_requirement_context():
    desires = CapacityDesires(
        service_tier=1,
        query_pattern=QueryPattern(
            access_pattern=AccessPattern.latency,
            estimated_read_per_second=certain_int(100),
            estimated_write_per_second=certain_int(100),
        ),
        data_shape=DataShape(estimated_state_size_gib=certain_int(10_000)),
    )

    plans = planner.plan_certain(
        model_name="org.netflix.cassandra",
        region="us-east-1",
        desires=desires,
        extra_model_arguments={"require_local_disks": True},
        instance_families=["m6id"],
        num_results=1,
    )

    ctx = plans[0].requirements.zonal[0].context
    assert not [key for key in ctx if key.startswith("page_cache")]


def test_write_buffer_is_hard_memory_pressure():
    desires = CapacityDesires(
        service_tier=1,
        query_pattern=QueryPattern(
            access_pattern=AccessPattern.latency,
            estimated_read_per_second=certain_int(10),
            estimated_write_per_second=certain_int(100),
            estimated_mean_write_size_bytes=certain_int(1 << 20),
        ),
        data_shape=DataShape(estimated_state_size_gib=certain_int(100)),
    )

    plans = planner.plan_certain(
        model_name="org.netflix.cassandra",
        region="us-east-1",
        desires=desires,
        extra_model_arguments={"require_local_disks": True},
        instance_families=["m6id"],
        num_results=1,
    )

    params = plans[0].candidate_clusters.zonal[0].cluster_params
    counts = params["required_nodes_by_type"]

    assert counts["memory"] > max(
        counts["cpu"],
        counts["disk_capacity"],
        counts["disk_iops"],
        counts["network"],
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
