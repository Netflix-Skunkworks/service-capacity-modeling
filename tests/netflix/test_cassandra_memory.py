"""Tests for cassandra_memory.py — experimental memory estimation."""

import pytest

from service_capacity_modeling.interface import (
    CapacityDesires,
    CurrentZoneClusterCapacity,
    DataShape,
    Instance,
    QueryPattern,
    certain_float,
    certain_int,
)
from service_capacity_modeling.models.org.netflix.cassandra import (
    NflxCassandraArguments,
)
from service_capacity_modeling.models.org.netflix.cassandra_memory import (
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
    memory_util_gib: float = 0.0,
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
        memory_utilization_gib=certain_float(memory_util_gib),
    )


class TestCurrentCapacityMemoryUtil:
    """When current_capacity has memory_utilization_gib, use it directly."""

    def test_no_args_uses_current_capacity(self):
        """args=None → fallback to current_capacity."""
        desires = _make_desires()
        current = _make_current_capacity(memory_util_gib=20.0)

        result = estimate_memory_experimental(
            current_capacity=current,
            working_set=0.5,
            rps_working_set=0.4,
            disk_used_gib=1000.0,
            desires=desires,
            write_buffer_gib=2.0,
            args=None,
        )

        assert result.memory_utilization_gib == pytest.approx(20.0)

    def test_args_present_uses_current_capacity(self):
        """Args present → still uses current_capacity.memory_utilization_gib."""
        args = NflxCassandraArguments(experimental_memory_model=True)
        desires = _make_desires()
        current = _make_current_capacity(memory_util_gib=22.0)

        result = estimate_memory_experimental(
            current_capacity=current,
            working_set=0.5,
            rps_working_set=0.4,
            disk_used_gib=1000.0,
            desires=desires,
            write_buffer_gib=2.0,
            args=args,
        )

        assert result.memory_utilization_gib == pytest.approx(22.0)


class TestTheoreticalFallback:
    """When current_capacity memory_util is unavailable → theoretical working set."""

    def test_no_current_capacity(self):
        """No cluster data → theoretical working set."""
        args = NflxCassandraArguments(experimental_memory_model=True)
        desires = _make_desires()

        result = estimate_memory_experimental(
            current_capacity=None,
            working_set=0.5,
            rps_working_set=0.4,
            disk_used_gib=500.0,
            desires=desires,
            write_buffer_gib=1.0,
            args=args,
        )

        # memory_utilization_gib is None (no data to derive it)
        assert result.memory_utilization_gib is None
        # effective_working_set = min(0.5, 0.4) = 0.4
        assert result.effective_working_set == pytest.approx(0.4)

    def test_current_capacity_zero_memory_util(self):
        """Current capacity exists but memory_util=0 → conservative."""
        args = NflxCassandraArguments(experimental_memory_model=True)
        desires = _make_desires()
        current = _make_current_capacity(memory_util_gib=0.0)

        result = estimate_memory_experimental(
            current_capacity=current,
            working_set=0.5,
            rps_working_set=0.4,
            disk_used_gib=1000.0,
            desires=desires,
            write_buffer_gib=2.0,
            args=args,
        )

        # memory_utilization_gib is None (was 0, treated as unavailable)
        assert result.memory_utilization_gib is None
