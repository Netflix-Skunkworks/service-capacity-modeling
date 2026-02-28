"""Memory estimation for Cassandra capacity planning.

Provides two strategies for estimating memory requirements:
- Legacy: theoretical working set from drive latency and read SLO
- Experimental: observed working set from memory_utilization_gib metrics
"""

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
    memory_utilization_gib: Optional[float] = None
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
        memory_utilization_gib=None,
        write_buffer_gib=write_buffer_gib,
    )


def estimate_memory_experimental(  # pylint: disable=too-many-positional-arguments
    current_capacity: Optional[CurrentClusterCapacity],
    working_set: float,
    rps_working_set: float,
    disk_used_gib: float,
    desires: CapacityDesires,
    write_buffer_gib: float,
) -> MemoryEstimate:
    """Experimental: observed working set + DerivedBuffers for memory.

    memory_utilization_gib represents non-page-cache memory per node
    (JVM heap + OS buffers + write buffers). Source: antigravity-cass.
    page_cache = instance_RAM - memory_utilization_gib
    """
    memory_utilization_gib = (
        current_capacity.memory_utilization_gib.mid
        if current_capacity
        and current_capacity.cluster_instance
        and current_capacity.memory_utilization_gib.mid > 0
        else None
    )

    if (
        memory_utilization_gib is not None
        and current_capacity is not None
        and current_capacity.cluster_instance
    ):
        # Observed: derive working set from actual cluster memory usage.
        # Prefer raw disk utilization (not buffer-scaled) so the observed
        # page cache ratio isn't distorted by buffer policy. Fall back to
        # estimated disk per node when disk utilization isn't reported.
        page_cache_per_node = max(
            0, current_capacity.cluster_instance.ram_gib - memory_utilization_gib
        )
        raw_disk_per_node = current_capacity.disk_utilization_gib.mid
        if raw_disk_per_node <= 0:
            raw_disk_per_node = (
                disk_used_gib / current_capacity.cluster_instance_count.mid
            )
        raw_disk_per_node = max(1, raw_disk_per_node)
        effective_ws = min(1.0, page_cache_per_node / raw_disk_per_node)
    elif current_capacity and current_capacity.cluster_instance:
        # Conservative: no memory metrics, but we know the instance type.
        # Estimate page cache as RAM minus heap and base memory reserves.
        reserve = _get_base_memory(desires) + _cass_heap(
            current_capacity.cluster_instance.ram_gib
        )
        page_cache_per_node = max(
            0, current_capacity.cluster_instance.ram_gib - reserve
        )
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

    # Base memory need from working set (per-zone)
    base_needed_memory = max(1, int(effective_ws * disk_used_gib))

    # Apply memory buffer policy via DerivedBuffers (handles preserve,
    # scale_up, scale_down) — same pattern as CPU and disk.
    memory_derived = DerivedBuffers.for_components(
        desires.buffers.derived, [BufferComponent.memory]
    )
    if current_capacity and current_capacity.cluster_instance:
        reserve_memory = _get_base_memory(desires) + _cass_heap(
            current_capacity.cluster_instance.ram_gib
        )
        existing_page_cache = (
            current_capacity.cluster_instance.ram_gib - reserve_memory
        ) * current_capacity.cluster_instance_count.mid
        needed_memory = memory_derived.calculate_requirement(
            current_usage=base_needed_memory,
            existing_capacity=existing_page_cache,
        )
        if memory_derived.preserve:
            write_buffer_gib = 0
    else:
        needed_memory = base_needed_memory

    return MemoryEstimate(
        effective_working_set=effective_ws,
        needed_memory_gib=needed_memory,
        memory_utilization_gib=memory_utilization_gib,
        write_buffer_gib=write_buffer_gib,
    )
