"""Tests for cassandra_memory.py — experimental memory estimation."""

import pytest

from service_capacity_modeling.interface import (
    Buffer,
    BufferComponent,
    BufferIntent,
    Buffers,
    CapacityDesires,
    CurrentZoneClusterCapacity,
    DataShape,
    Instance,
    QueryPattern,
    certain_float,
    certain_int,
)
from service_capacity_modeling.models.org.netflix.cassandra_memory import (
    MemoryInputs,
    MemoryPolicy,
    _cass_heap,
    _get_base_memory,
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


def _make_instance(
    *,
    name: str = "m5d.8xlarge",
    ram_gib: float = 128.0,
    cpu: int = 32,
):
    return Instance(
        name=name,
        cpu=cpu,
        cpu_ghz=2.5,
        ram_gib=ram_gib,
        net_mbps=10000,
        drive=None,
    )


def _make_current_capacity(
    *,
    instance_name: str = "m5d.8xlarge",
    ram_gib: float = 128.0,
    instance_count: int = 12,
    disk_util_gib: float = 200.0,
    cpu: int = 32,
) -> CurrentZoneClusterCapacity:
    return CurrentZoneClusterCapacity(
        cluster_instance_name=instance_name,
        cluster_instance=_make_instance(
            name=instance_name,
            ram_gib=ram_gib,
            cpu=cpu,
        ),
        cluster_instance_count=certain_float(instance_count),
        cpu_utilization=certain_float(0.3),
        disk_utilization_gib=certain_float(disk_util_gib),
    )


def _inputs(
    *,
    current_capacity,
    desires,
    disk_used_gib: float,
    write_buffer_gib: float,
    ws_slo_bound: float = 0.5,
    ws_rps_bound: float = 0.4,
    zones_per_region: int = 3,
    effective_page_cache_gib_per_node: float = 28.0,
    planned_page_cache_nodes: int = 1,
) -> MemoryInputs:
    return MemoryInputs(
        current_capacity=current_capacity,
        desires=desires,
        disk_used_gib=disk_used_gib,
        write_buffer_gib=write_buffer_gib,
        zones_per_region=zones_per_region,
        ws_slo_bound=ws_slo_bound,
        ws_rps_bound=ws_rps_bound,
        effective_page_cache_gib_per_node=effective_page_cache_gib_per_node,
        planned_page_cache_nodes=planned_page_cache_nodes,
    )


# 128 GiB: heap=30, base=3, raw_page_cache=95
# 32 GiB: heap=16, base=3, raw_page_cache=13
@pytest.mark.parametrize(
    "ram_gib, disk_util, cap, expected_ws",
    [
        (128.0, 200.0, 32.0, 32.0 / 200.0),  # capped: 95 → 32
        (32.0, 100.0, 32.0, 13.0 / 100.0),  # below cap: stays 13
        (128.0, 200.0, 64.0, 64.0 / 200.0),  # custom cap: 95 → 64
        (128.0, 200.0, 1e9, 95.0 / 200.0),  # high cap effectively disables
    ],
    ids=["capped", "below_cap", "custom_cap", "high_cap"],
)
def test_page_cache_cap(ram_gib, disk_util, cap, expected_ws):
    desires = _make_desires()
    current = _make_current_capacity(ram_gib=ram_gib, disk_util_gib=disk_util)

    result = estimate_memory(
        _inputs(
            current_capacity=current,
            desires=desires,
            disk_used_gib=1000.0,
            write_buffer_gib=2.0,
            ws_slo_bound=1.0,
            ws_rps_bound=1.0,
        ),
        MemoryPolicy(max_page_cache_gib=cap),
    )

    assert result.effective_ws_fraction == pytest.approx(expected_ws)


def test_no_current_capacity_uses_theoretical():
    desires = _make_desires()

    result = estimate_memory(
        _inputs(
            current_capacity=None,
            desires=desires,
            disk_used_gib=500.0,
            write_buffer_gib=1.0,
        ),
        MemoryPolicy(),
    )

    assert result.effective_ws_fraction == pytest.approx(0.4)
    assert result.page_cache_linear_demand_gib == 200
    assert result.page_cache_demand_gib == 28


def test_page_cache_demand_keeps_small_datasets_linear():
    desires = _make_desires()

    result = estimate_memory(
        _inputs(
            current_capacity=None,
            desires=desires,
            disk_used_gib=20.0,
            write_buffer_gib=1.0,
            ws_slo_bound=1.0,
            ws_rps_bound=1.0,
            planned_page_cache_nodes=4,
        ),
        MemoryPolicy(),
    )

    assert result.page_cache_demand_gib == 20


def test_page_cache_demand_caps_large_datasets_by_node_budget():
    desires = _make_desires()

    result = estimate_memory(
        _inputs(
            current_capacity=None,
            desires=desires,
            disk_used_gib=10_000.0,
            write_buffer_gib=1.0,
            ws_slo_bound=0.5,
            ws_rps_bound=0.5,
            effective_page_cache_gib_per_node=26.0,
            planned_page_cache_nodes=8,
        ),
        MemoryPolicy(),
    )

    assert result.page_cache_linear_demand_gib == 5_000
    assert result.effective_page_cache_budget_gib == 208
    assert result.page_cache_demand_gib == 208


def test_write_buffer_passthrough():
    desires = _make_desires()
    current = _make_current_capacity()

    result = estimate_memory(
        _inputs(
            current_capacity=current,
            desires=desires,
            disk_used_gib=1000.0,
            write_buffer_gib=5.0,
        ),
        MemoryPolicy(),
    )

    assert result.write_buffer_gib == 5.0


def test_preserve_buffer_keeps_existing_memory():
    """Memory preserve buffer returns current cluster's total page cache."""
    desires = _make_desires(
        buffers=Buffers(
            derived={
                "memory": Buffer(
                    intent=BufferIntent.preserve,
                    components=[BufferComponent.memory],
                )
            }
        ),
    )
    # 128 GiB RAM, heap=30, base=3 → page_cache_per_node=95
    # 12 instances → total page cache = 95 * 12 = 1140
    current = _make_current_capacity()

    result = estimate_memory(
        _inputs(
            current_capacity=current,
            desires=desires,
            disk_used_gib=1000.0,
            write_buffer_gib=5.0,
        ),
        MemoryPolicy(),
    )

    base = _get_base_memory(desires)
    heap = _cass_heap(128.0)
    expected = (128.0 - heap - base) * 12
    assert result.page_cache_demand_gib == pytest.approx(expected)
    assert result.write_buffer_gib == 0
