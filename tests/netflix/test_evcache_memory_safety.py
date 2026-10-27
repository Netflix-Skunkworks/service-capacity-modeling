import math

import pytest

from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.interface import (
    Buffer,
    Buffers,
    CapacityDesires,
    CurrentClusters,
    CurrentZoneClusterCapacity,
    DataShape,
    QueryPattern,
    certain_float,
)


def workload(state=2048, items=4_000_000_000, reads=1, writes=1):
    return CapacityDesires(
        query_pattern=QueryPattern(
            estimated_read_per_second=certain_float(reads),
            estimated_write_per_second=certain_float(writes),
            estimated_mean_read_size_bytes=certain_float(400),
            estimated_mean_write_size_bytes=certain_float(400),
        ),
        data_shape=DataShape(
            estimated_state_size_gib=certain_float(state),
            estimated_state_item_count=certain_float(items),
            estimated_working_set_percent=certain_float(0.01),
        ),
    )


def plans(desires, **extra):
    return planner.plan_certain(
        model_name="org.netflix.evcache",
        region="us-east-1",
        instance_families=["i4i.large"],
        desires=desires,
        extra_model_arguments=extra,
    )


def test_configured_memory_reserve_is_used_for_sizing():
    candidates = plans(workload(state=100, items=1_000_000))
    assert candidates
    for plan in candidates:
        required = plan.requirements.zonal[0].mem_gib.mid
        cluster = plan.candidate_clusters.zonal[0]
        reserve = plan.requirements.zonal[0].context["osmem"]
        assert cluster.count * (cluster.instance.ram_gib - reserve) >= required
        assert (cluster.count - 1) * (cluster.instance.ram_gib - reserve) < required


def test_reserves_cannot_consume_all_instance_memory():
    desire = workload()
    desire.data_shape.reserved_instance_app_mem_gib = 16
    assert not plans(desire)


def test_zero_traffic_flash_workload():
    candidates = plans(workload(reads=0, writes=0))
    assert candidates
    assert math.isfinite(candidates[0].candidate_clusters.total_annual_cost)


def test_single_current_cluster_of_another_datastore_is_not_evcache_usage():
    desire = workload()
    expected = plans(desire)[0].requirements.zonal[0].mem_gib.mid
    desire.current_clusters = CurrentClusters(
        zonal=[
            CurrentZoneClusterCapacity(
                cluster_type="cassandra",
                cluster_instance_name="i4i.large",
                cluster_instance_count=certain_float(1000),
                memory_utilization_gib=certain_float(12),
                cpu_utilization=certain_float(99),
            )
        ]
    )
    candidates = plans(desire)
    assert candidates
    assert candidates[0].requirements.zonal[0].mem_gib.mid == expected


def test_unused_reserve_budget_cannot_reduce_measured_memory_demand():
    desire = workload(state=1, items=1000)
    desire.current_clusters = CurrentClusters(
        zonal=[
            CurrentZoneClusterCapacity(
                cluster_instance_name="i4i.large",
                cluster_instance_count=certain_float(24),
                memory_utilization_gib=certain_float(10),
                cpu_utilization=certain_float(1),
            )
        ]
    )
    candidates = plans(desire)
    assert candidates
    assert candidates[0].requirements.zonal[0].mem_gib.mid >= 24 * 10


def test_new_cluster_explicit_memory_and_network_buffers():
    desire = workload()
    desire.buffers = Buffers(
        desired={
            "memory": Buffer(ratio=1, components=["memory"]),
            "network": Buffer(ratio=1, components=["network"]),
        }
    )
    baseline = plans(desire)[0].requirements.zonal[0]
    desire.buffers.desired["memory"].ratio = 3
    desire.buffers.desired["network"].ratio = 3
    buffered = plans(desire)[0].requirements.zonal[0]
    assert buffered.mem_gib.mid == pytest.approx(3 * baseline.mem_gib.mid)
    assert buffered.network_mbps.mid == pytest.approx(3 * baseline.network_mbps.mid)


@pytest.mark.parametrize(
    "extra",
    [
        {"copies_per_region": 0},
        {"copies_per_region": float("nan")},
        {"max_local_data_per_node_gib": 0},
        {"max_local_data_per_node_gib": -1},
        {"max_local_data_per_node_gib": 0.5},
        {"max_local_data_per_node_gib": float("inf")},
    ],
)
def test_invalid_sizing_constraints(extra):
    with pytest.raises(ValueError):
        plans(workload(), **extra)


def test_legacy_disk_limit_keeps_precedence():
    expected = plans(workload(), max_local_disk_gib=10)
    legacy = plans(workload(), max_local_data_per_node_gib=10, max_local_disk_gib=20)
    assert expected and legacy
    assert legacy[0].candidate_clusters == expected[0].candidate_clusters


def test_preserve_storage_uses_usable_capacity_without_readding_reserves():
    from service_capacity_modeling.hardware import shapes

    desire = workload(items=1_000_000)
    desire.current_clusters = CurrentClusters(
        zonal=[
            CurrentZoneClusterCapacity(
                cluster_instance_name="i4i.large",
                cluster_instance_count=certain_float(24),
                cluster_drive=shapes.instance("i4i.large").drive,
                memory_utilization_gib=certain_float(6),
                cpu_utilization=certain_float(1),
                disk_utilization_gib=certain_float(64),
            )
        ]
    )
    desire.buffers = Buffers(
        derived={"keep": Buffer(intent="preserve", components=["storage"])}
    )
    candidates = plans(desire)
    assert candidates
    assert candidates[0].candidate_clusters.zonal[0].count == 24


def test_current_disk_buffer_is_applied_once():
    from service_capacity_modeling.hardware import shapes
    from service_capacity_modeling.models.common import get_effective_disk_per_node_gib

    desire = workload(items=1_000_000)
    desire.current_clusters = CurrentClusters(
        zonal=[
            CurrentZoneClusterCapacity(
                cluster_instance_name="i4i.large",
                cluster_instance_count=certain_float(24),
                cluster_drive=shapes.instance("i4i.large").drive,
                memory_utilization_gib=certain_float(1),
                cpu_utilization=certain_float(1),
                disk_utilization_gib=certain_float(64),
            )
        ]
    )
    desire.buffers = Buffers(
        desired={
            "storage": Buffer(ratio=1, components=["storage"]),
            "disk": Buffer(ratio=2, components=["disk"]),
        }
    )
    candidates = plans(desire, max_local_data_per_node_gib=100)
    assert candidates
    requirement = candidates[0].requirements.zonal[0]
    cluster = candidates[0].candidate_clusters.zonal[0]
    assert requirement.disk_gib.mid == 2 * 64 * 24
    # The current path buffers demand, so per-node capacity remains unbuffered.
    assert cluster.cluster_params["effective_disk_per_node_gib"] == (
        get_effective_disk_per_node_gib(
            cluster.instance, cluster.instance.drive, 1, max_local_data_per_node_gib=100
        )
    )


def test_fractional_memory_demand_cannot_drop_a_required_node():
    # Eight i4i.large nodes hold 90.08 GiB after the configured reserve.
    desire = workload(state=90.1, items=1_000_000)
    desire.buffers = Buffers(
        desired={
            "storage": Buffer(ratio=1, components=["storage"]),
            "memory": Buffer(ratio=1, components=["memory"]),
        }
    )
    candidates = plans(desire)
    assert candidates
    requirement = candidates[0].requirements.zonal[0]
    assert requirement.mem_gib.mid == pytest.approx(90.1)
    cluster = candidates[0].candidate_clusters.zonal[0]
    reserve = requirement.context["osmem"]
    assert (
        cluster.count * (cluster.instance.ram_gib - reserve) >= requirement.mem_gib.mid
    )


def test_fractional_disk_demand_cannot_drop_a_required_node():
    desire = workload(state=200.1, items=1_000_000)
    desire.buffers = Buffers(
        desired={
            "storage": Buffer(ratio=1, components=["storage"]),
            "disk": Buffer(ratio=1, components=["disk"]),
        }
    )
    candidates = plans(desire, max_local_data_per_node_gib=100)
    assert candidates
    requirement = candidates[0].requirements.zonal[0]
    cluster = candidates[0].candidate_clusters.zonal[0]
    assert requirement.disk_gib.mid == pytest.approx(200.1)
    assert (
        cluster.count * cluster.cluster_params["effective_disk_per_node_gib"] >= 200.1
    )
