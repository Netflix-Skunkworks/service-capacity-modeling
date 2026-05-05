"""Memory estimation for Cassandra capacity planning.

Single ``estimate_memory`` pipeline: combine the Cassandra working-set bounds,
cap normal page cache to the candidate node's RAM layout, and emit the raw
memory demand before current-cluster derived-buffer policy is applied.

Glossary (used throughout this module):
- ``ram_gib``: total instance RAM
- ``heap_gib``: JVM heap portion from ``MemoryLayout.for_ram``
- ``base_reserves_gib``: app + system reserves (from ``base_memory_gib``)
- ``page_cache_capacity_gib``: per-node ``ram - heap - base_reserves_gib``,
  optionally capped at ``max_page_cache_gib``
- ``page_cache_demand_gib``: uncapped page-cache demand from
  ``effective_ws_fraction * disk_used_gib``
- ``page_cache_capped_demand_gib``: per-node soft page-cache target capped to
  this instance shape's RAM layout. Normal page cache is not a zone-level
  node-count requirement.
- ``effective_ws_fraction``: final working-set fraction after applying the
  disk-SLO and RPS bounds
- ``effective_page_cache_gib_per_node``: candidate shape's per-node page-cache
  capacity from ``MemoryLayout.for_ram``
- ``disk_slo_working_set``: working-set fraction from drive latency vs read
  SLO. This pushes cache up when disk misses are too slow for the request
  latency target.
- ``rps_working_set``: working-set fraction from read volume vs disk-read
  budget. This pulls cache down when read traffic is low enough that disk can
  absorb more cache misses.
- ``write_buffer_gib``: raw zone-level memtable/write-buffer memory demand
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from service_capacity_modeling.interface import CapacityDesires


DEFAULT_MAX_PAGE_CACHE_GIB = 28.0
DEFAULT_MAX_HEAP_GIB = 30.0
MIN_WORKING_SET = 0.01


def base_memory_gib(desires: CapacityDesires) -> float:
    return (
        desires.data_shape.reserved_instance_app_mem_gib
        + desires.data_shape.reserved_instance_system_mem_gib
    )


class MemoryLayout(BaseModel):
    """Per-node RAM partition: heap + reserves + page cache <= ram_gib."""

    model_config = ConfigDict(frozen=True)

    heap_gib: float
    base_reserves_gib: float
    page_cache_capacity_gib: float

    @property
    def total_gib(self) -> float:
        return self.heap_gib + self.base_reserves_gib + self.page_cache_capacity_gib

    @classmethod
    def for_ram(
        cls,
        *,
        ram_gib: float,
        base_reserves_gib: float,
        max_page_cache_gib: float = DEFAULT_MAX_PAGE_CACHE_GIB,
        max_heap_gib: float = DEFAULT_MAX_HEAP_GIB,
    ) -> "MemoryLayout":
        """Partition node RAM among heap, reserves, and page cache.

        Heap follows the Cassandra rule: ``min(ram//2, max_heap_gib)``. Cache
        is the residual ``ram - heap - base``, soft-capped by
        ``max_page_cache_gib``. The default cache cap (28) is chosen so that
        ``28 + 3 base + 30 heap = 61 <= 64 GiB`` boxes - a 4xlarge fit budget.

        Invariant: ``heap + base + page_cache_capacity <= ram_gib`` whenever
        ``ram_gib >= max(4, base) + 4``.
        """
        heap = float(min(max(4, ram_gib // 2), max_heap_gib))
        cache_capacity = max(
            0.0, min(max_page_cache_gib, ram_gib - heap - base_reserves_gib)
        )
        return cls(
            heap_gib=heap,
            base_reserves_gib=base_reserves_gib,
            page_cache_capacity_gib=cache_capacity,
        )


class MemoryEstimate(BaseModel):
    """Raw memory sizing result before current-capacity derived buffers."""

    effective_ws_fraction: float
    page_cache_demand_gib: float
    page_cache_capped_demand_gib: float
    write_buffer_gib: float


class MemoryInputs(BaseModel):
    """Inputs to raw Cassandra memory sizing."""

    model_config = ConfigDict(frozen=True)

    disk_used_gib: float
    write_buffer_gib: float
    disk_slo_working_set: float
    rps_working_set: float
    effective_page_cache_gib_per_node: float = DEFAULT_MAX_PAGE_CACHE_GIB


def estimate_memory(inputs: MemoryInputs) -> MemoryEstimate:
    """Estimate raw Cassandra memory demand before derived-buffer policy."""
    effective_ws = max(
        MIN_WORKING_SET,
        min(
            1.0,
            inputs.disk_slo_working_set,
            inputs.rps_working_set,
        ),
    )
    page_cache_demand = max(1.0, effective_ws * inputs.disk_used_gib)
    page_cache_capped_demand: float = max(
        1, int(min(page_cache_demand, inputs.effective_page_cache_gib_per_node))
    )
    return MemoryEstimate(
        effective_ws_fraction=effective_ws,
        page_cache_demand_gib=page_cache_demand,
        page_cache_capped_demand_gib=page_cache_capped_demand,
        write_buffer_gib=inputs.write_buffer_gib,
    )
