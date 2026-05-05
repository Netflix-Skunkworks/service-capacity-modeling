"""Tests for cassandra_memory.py memory estimation."""

import pytest

from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.hardware import shapes
from service_capacity_modeling.interface import (
    AccessPattern,
    CapacityDesires,
    CurrentClusters,
    CurrentZoneClusterCapacity,
    DataShape,
    Drive,
    DriveType,
    QueryPattern,
    certain_float,
    certain_int,
)
from service_capacity_modeling.models.org.netflix.cassandra_memory import (
    DEFAULT_MAX_PAGE_CACHE_GIB,
    DEFAULT_MAX_HEAP_GIB,
    MemoryInputs,
    MemoryLayout,
    base_memory_gib,
    estimate_memory,
)


def _make_desires(**overrides) -> CapacityDesires:
    defaults = {
        "service_tier": 1,
        "query_pattern": QueryPattern(
            estimated_read_per_second=certain_int(10_000),
            estimated_write_per_second=certain_int(1_000),
        ),
        "data_shape": DataShape(
            estimated_state_size_gib=certain_int(100),
        ),
    }
    defaults.update(overrides)
    return CapacityDesires(**defaults)


def _write_heavy_existing_4xlarge_desires() -> CapacityDesires:
    current_instance = shapes.instance("r7a.4xlarge")
    return CapacityDesires(
        service_tier=1,
        query_pattern=QueryPattern(
            access_pattern=AccessPattern.latency,
            estimated_read_per_second=certain_int(3_000),
            estimated_write_per_second=certain_int(90_000),
        ),
        data_shape=DataShape(
            estimated_state_size_gib=certain_int(3_000),
        ),
        current_clusters=CurrentClusters(
            zonal=[
                CurrentZoneClusterCapacity(
                    cluster_instance_name=current_instance.name,
                    cluster_instance=current_instance,
                    cluster_drive=Drive(
                        name="gp3",
                        drive_type=DriveType.attached_ssd,
                        size_gib=2_500,
                    ),
                    cluster_instance_count=certain_int(8),
                    cluster_type="cassandra",
                    cpu_utilization=certain_float(20),
                    network_utilization_mbps=certain_float(250),
                    disk_utilization_gib=certain_float(400),
                )
            ],
        ),
    )


def _inputs(
    *,
    disk_used_gib: float,
    write_buffer_gib: float,
    disk_slo_working_set: float = 0.5,
    rps_working_set: float = 0.4,
    effective_page_cache_gib_per_node: float = DEFAULT_MAX_PAGE_CACHE_GIB,
) -> MemoryInputs:
    return MemoryInputs(
        disk_used_gib=disk_used_gib,
        write_buffer_gib=write_buffer_gib,
        disk_slo_working_set=disk_slo_working_set,
        rps_working_set=rps_working_set,
        effective_page_cache_gib_per_node=effective_page_cache_gib_per_node,
    )


@pytest.mark.parametrize(
    "ram_gib, cap, expected_page_cache",
    [
        (128.0, 32.0, 32.0),
        (32.0, 32.0, 13.0),
        (128.0, 64.0, 64.0),
        (128.0, 1e9, 95.0),
    ],
    ids=["capped", "below_cap", "custom_cap", "high_cap"],
)
def test_page_cache_cap(ram_gib, cap, expected_page_cache):
    desires = _make_desires()
    layout = MemoryLayout.for_ram(
        ram_gib=ram_gib,
        base_reserves_gib=base_memory_gib(desires),
        max_page_cache_gib=cap,
    )

    assert layout.page_cache_capacity_gib == pytest.approx(expected_page_cache)


def test_m6id_2xlarge_memory_layout_fits_heap_base_and_page_cache():
    desires = _make_desires()
    instance = shapes.instance("m6id.2xlarge")
    layout = MemoryLayout.for_ram(
        ram_gib=instance.ram_gib,
        base_reserves_gib=base_memory_gib(desires),
    )

    assert layout.total_gib <= instance.ram_gib
    assert layout.page_cache_capacity_gib == pytest.approx(
        instance.ram_gib - layout.heap_gib - layout.base_reserves_gib
    )
    assert layout.page_cache_capacity_gib <= DEFAULT_MAX_PAGE_CACHE_GIB


def test_4xlarge_memory_layout_keeps_page_cache_within_ram():
    desires = _make_desires()
    instance = shapes.instance("m7a.4xlarge")
    layout = MemoryLayout.for_ram(
        ram_gib=instance.ram_gib,
        base_reserves_gib=base_memory_gib(desires),
    )

    assert layout.heap_gib == DEFAULT_MAX_HEAP_GIB
    assert layout.page_cache_capacity_gib == DEFAULT_MAX_PAGE_CACHE_GIB
    assert layout.total_gib <= instance.ram_gib


def test_page_cache_does_not_force_r_family_for_4xlarge_candidates():
    plans = planner.plan_certain(
        model_name="org.netflix.cassandra",
        region="us-east-1",
        desires=_write_heavy_existing_4xlarge_desires(),
        instance_families=["m7a", "r7a"],
        extra_model_arguments={
            "require_local_disks": False,
            "require_attached_disks": True,
        },
        max_results_per_family=20,
        num_results=100,
    )
    by_instance = {p.candidate_clusters.zonal[0].instance.name: p for p in plans}

    m7a = by_instance["m7a.4xlarge"].candidate_clusters.zonal[0]
    r7a = by_instance["r7a.4xlarge"].candidate_clusters.zonal[0]
    m7a_counts = m7a.cluster_params["required_nodes_by_type"]

    assert m7a.count == r7a.count == 8
    assert m7a_counts["memory"] <= max(
        m7a_counts["cpu"],
        m7a_counts["network"],
        m7a_counts["disk_capacity"],
        m7a_counts["disk_iops"],
    )


def test_no_current_capacity_uses_theoretical():
    result = estimate_memory(
        _inputs(
            disk_used_gib=500.0,
            write_buffer_gib=1.0,
        )
    )

    assert result.effective_ws_fraction == pytest.approx(0.4)
    assert result.page_cache_demand_gib == 200
    assert result.page_cache_capped_demand_gib == DEFAULT_MAX_PAGE_CACHE_GIB


def test_page_cache_capped_demand_keeps_small_datasets_uncapped():
    result = estimate_memory(
        _inputs(
            disk_used_gib=20.0,
            write_buffer_gib=1.0,
            disk_slo_working_set=1.0,
            rps_working_set=1.0,
        )
    )

    assert result.page_cache_capped_demand_gib == 20


def test_page_cache_capped_demand_caps_large_datasets_by_per_node_budget():
    result = estimate_memory(
        _inputs(
            disk_used_gib=10_000.0,
            write_buffer_gib=1.0,
            disk_slo_working_set=0.5,
            rps_working_set=0.5,
            effective_page_cache_gib_per_node=26.0,
        )
    )

    assert result.page_cache_demand_gib == 5_000
    assert result.page_cache_capped_demand_gib == 26


def test_write_buffer_passthrough():
    result = estimate_memory(
        _inputs(
            disk_used_gib=1000.0,
            write_buffer_gib=5.0,
        )
    )

    assert result.write_buffer_gib == 5.0
