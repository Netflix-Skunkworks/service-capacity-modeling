"""Memory estimation for Cassandra capacity planning.

Provides two strategies for estimating memory requirements:
- Legacy: theoretical working set from drive latency and read SLO
- Experimental: page-cache-capped working set with configurable max
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from service_capacity_modeling.interface import (
    BufferComponent,
    CapacityDesires,
    CurrentClusterCapacity,
)
from service_capacity_modeling.models.common import DerivedBuffers


def _get_base_memory(desires: CapacityDesires) -> float:
    return (
        desires.data_shape.reserved_instance_app_mem_gib
        + desires.data_shape.reserved_instance_system_mem_gib
    )


def _cass_heap(node_memory_gib: float, max_heap_gib: float = 30) -> float:
    # Netflix Cassandra heap formula
    return min(max(4, node_memory_gib // 2), max_heap_gib)


class MemoryEstimate(BaseModel):
    """Result of memory sizing: clear contract between old/new paths."""

    effective_working_set: float
    needed_memory_gib: float
    write_buffer_gib: float


def estimate_memory_legacy(
    working_set: float,
    rps_working_set: float,
    disk_used_gib: float,
    zones_per_region: int,
    write_buffer_gib: float,
) -> MemoryEstimate:
    """Legacy: theoretical working set, simple memory calculation."""
    effective_ws = min(working_set, rps_working_set)
    needed = effective_ws * disk_used_gib * zones_per_region
    needed = max(1, int(needed // zones_per_region))
    return MemoryEstimate(
        effective_working_set=effective_ws,
        needed_memory_gib=needed,
        write_buffer_gib=write_buffer_gib,
    )


def estimate_memory_experimental(  # pylint: disable=too-many-positional-arguments
    current_capacity: Optional[CurrentClusterCapacity],
    working_set: float,
    rps_working_set: float,
    disk_used_gib: float,
    desires: CapacityDesires,
    write_buffer_gib: float,
    max_page_cache_gib: float = 32.0,
) -> MemoryEstimate:
    """Experimental: page-cache-capped working set.

    Computes page cache per node as RAM - heap - base_reserves, then caps it
    at max_page_cache_gib (0 or negative disables the cap).  The cap prevents
    large-RAM instances from inflating the working-set estimate for workloads
    that don't benefit from extra cache (e.g. write-heavy timeseries).

    Honors memory preserve buffers: when set, keeps the current cluster's
    total page cache as the memory requirement and zeros write_buffer_gib.

    When no current capacity is available, falls back to the theoretical
    working set (same as legacy).
    """
    if current_capacity and current_capacity.cluster_instance:
        reserve = _get_base_memory(desires) + _cass_heap(
            current_capacity.cluster_instance.ram_gib
        )
        page_cache_per_node = max(
            0, current_capacity.cluster_instance.ram_gib - reserve
        )
        # Apply the cap (0 or negative disables it)
        if max_page_cache_gib > 0:
            page_cache_per_node = min(page_cache_per_node, max_page_cache_gib)

        # Derive working set from capped page cache / disk ratio
        raw_disk_per_node = current_capacity.disk_utilization_gib.mid
        if raw_disk_per_node <= 0:
            raw_disk_per_node = (
                disk_used_gib / current_capacity.cluster_instance_count.mid
            )
        raw_disk_per_node = max(1, raw_disk_per_node)
        effective_ws = min(1.0, page_cache_per_node / raw_disk_per_node)
    else:
        # Theoretical: from drive latency vs read SLO
        effective_ws = min(working_set, rps_working_set)

    needed_memory: float = max(1, int(effective_ws * disk_used_gib))

    # Honor memory preserve buffer: keep current cluster's total page cache
    memory_derived = DerivedBuffers.for_components(
        desires.buffers.derived, [BufferComponent.memory]
    )
    if (
        memory_derived.is_preserve
        and current_capacity
        and current_capacity.cluster_instance
    ):
        reserve_memory = _get_base_memory(desires) + _cass_heap(
            current_capacity.cluster_instance.ram_gib
        )
        needed_memory = (
            current_capacity.cluster_instance.ram_gib - reserve_memory
        ) * current_capacity.cluster_instance_count.mid
        write_buffer_gib = 0

    return MemoryEstimate(
        effective_working_set=effective_ws,
        needed_memory_gib=needed_memory,
        write_buffer_gib=write_buffer_gib,
    )
