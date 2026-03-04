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
    _cass_heap,
    _get_base_memory,
    estimate_memory_experimental,
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


# 128 GiB: heap=30, base=3, raw_page_cache=95
# 32 GiB: heap=16, base=3, raw_page_cache=13
@pytest.mark.parametrize(
    "ram_gib, disk_util, cap, expected_ws",
    [
        (128.0, 200.0, 32.0, 32.0 / 200.0),  # capped: 95 → 32
        (32.0, 100.0, 32.0, 13.0 / 100.0),  # below cap: stays 13
        (128.0, 200.0, 64.0, 64.0 / 200.0),  # custom cap: 95 → 64
        (128.0, 200.0, 0, 95.0 / 200.0),  # cap=0 disables: full 95
    ],
    ids=["capped", "below_cap", "custom_cap", "cap_disabled"],
)
def test_page_cache_cap(ram_gib, disk_util, cap, expected_ws):
    desires = _make_desires()
    current = _make_current_capacity(ram_gib=ram_gib, disk_util_gib=disk_util)

    result = estimate_memory_experimental(
        current_capacity=current,
        working_set=0.5,
        rps_working_set=0.4,
        disk_used_gib=1000.0,
        desires=desires,
        write_buffer_gib=2.0,
        max_page_cache_gib=cap,
    )

    assert result.effective_working_set == pytest.approx(expected_ws)


def test_no_current_capacity_uses_theoretical():
    desires = _make_desires()

    result = estimate_memory_experimental(
        current_capacity=None,
        working_set=0.5,
        rps_working_set=0.4,
        disk_used_gib=500.0,
        desires=desires,
        write_buffer_gib=1.0,
    )

    assert result.effective_working_set == pytest.approx(0.4)
    assert result.needed_memory_gib == 200


def test_write_buffer_passthrough():
    desires = _make_desires()
    current = _make_current_capacity()

    result = estimate_memory_experimental(
        current_capacity=current,
        working_set=0.5,
        rps_working_set=0.4,
        disk_used_gib=1000.0,
        desires=desires,
        write_buffer_gib=5.0,
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

    result = estimate_memory_experimental(
        current_capacity=current,
        working_set=0.5,
        rps_working_set=0.4,
        disk_used_gib=1000.0,
        desires=desires,
        write_buffer_gib=5.0,
    )

    base = _get_base_memory(desires)
    heap = _cass_heap(128.0)
    expected = (128.0 - heap - base) * 12
    assert result.needed_memory_gib == pytest.approx(expected)
    assert result.write_buffer_gib == 0
