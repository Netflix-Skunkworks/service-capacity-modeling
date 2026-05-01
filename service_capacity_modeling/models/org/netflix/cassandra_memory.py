"""Memory estimation for Cassandra capacity planning.

Single ``estimate_memory`` pipeline: compute candidate bounds on the
effective working-set fraction, take the min, floor at
``policy.min_working_set``, emit a ``MemoryEstimate``.

The ``use_observational_anchor`` policy flag (was
``experimental_memory_model``) adds one more bound derived from the
current cluster's page-cache-per-node vs disk-per-node ratio. The
other bounds (SLO-derived, RPS-derived) always apply.

Glossary (used throughout this module):
- ``ram_gib``: total instance RAM
- ``heap_gib``: JVM heap portion (capped by ``_cass_heap``)
- ``physical_reserves_gib``: app + system reserves (from ``_get_base_memory``)
- ``page_cache_capacity_gib``: ``ram - heap - reserves``, optionally capped at
  ``max_page_cache_gib``
- ``page_cache_demand_gib``: zone-level page-cache requirement, capped by
  the useful page-cache budget for the expected node count
- ``effective_ws_fraction``: fraction of data that must be hot after all
  bounds apply
- ``ws_slo_bound``: theoretical working-set bound from drive latency vs read SLO
- ``ws_rps_bound``: working-set bound from read-rate vs disk IOPS budget
- ``write_buffer_gib``: zone-level memtable/write-buffer memory demand
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict

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
    """Netflix Cassandra heap formula (RAM-fit aware).

    Standalone heap sizing for callers that don't need the full layout.
    For partitioning under tight RAM (small instances), prefer
    ``memory_layout()`` which yields heap, reserves, and cache capacity
    that sum to ``<= ram_gib`` by construction.
    """
    return min(max(4, node_memory_gib // 2), max_heap_gib)


class MemoryLayout(BaseModel):
    """Per-node RAM partition: heap + reserves + page cache <= ram_gib."""

    model_config = ConfigDict(frozen=True)

    heap_gib: float
    base_reserves_gib: float
    page_cache_capacity_gib: float

    @property
    def total_gib(self) -> float:
        return self.heap_gib + self.base_reserves_gib + self.page_cache_capacity_gib


def memory_layout(
    ram_gib: float,
    base_reserves_gib: float,
    max_page_cache_gib: float = 28.0,
    max_heap_gib: float = 30.0,
) -> MemoryLayout:
    """Partition node RAM among heap, reserves, and page cache.

    Heap follows the Cassandra rule: ``min(ram//2, max_heap_gib)``. Cache
    is the residual ``ram - heap - base``, soft-capped by
    ``max_page_cache_gib``. The default cache cap (28) is chosen so that
    ``28 + 3 base + 30 heap = 61 <= 64 GiB`` boxes — a 4xlarge fit budget.

    Invariant: ``heap + base + page_cache_capacity <= ram_gib`` whenever
    ``ram_gib >= max(4, base) + 4``.
    """
    heap = float(min(max(4, int(ram_gib // 2)), int(max_heap_gib)))
    cache_capacity = max(
        0.0, min(max_page_cache_gib, ram_gib - heap - base_reserves_gib)
    )
    return MemoryLayout(
        heap_gib=heap,
        base_reserves_gib=base_reserves_gib,
        page_cache_capacity_gib=cache_capacity,
    )


class MemoryEstimate(BaseModel):
    """Result of memory sizing."""

    effective_ws_fraction: float
    page_cache_demand_gib: float
    page_cache_linear_demand_gib: float
    effective_page_cache_budget_gib: float
    write_buffer_gib: float


class MemoryInputs(BaseModel):
    """Shared inputs to the memory-sizing pipeline."""

    model_config = ConfigDict(frozen=True)

    current_capacity: Optional[CurrentClusterCapacity]
    desires: CapacityDesires
    disk_used_gib: float
    write_buffer_gib: float
    zones_per_region: int
    ws_slo_bound: float
    ws_rps_bound: float
    effective_page_cache_gib_per_node: float = 28.0
    planned_page_cache_nodes: int = 1


class MemoryPolicy(BaseModel):
    """Cassandra-specific memory-sizing knobs."""

    model_config = ConfigDict(frozen=True)

    max_page_cache_gib: float = 28.0
    use_observational_anchor: bool = True  # was `experimental_memory_model`
    min_working_set: float = 0.01


def estimate_memory(inputs: MemoryInputs, policy: MemoryPolicy) -> MemoryEstimate:
    """Take min of candidate working-set bounds, emit a ``MemoryEstimate``.

    The ``min_working_set`` floor only applies when no observational anchor
    is available.  When a running cluster provides real page-cache-to-disk
    ratios, that signal is more trustworthy than a static percentage —
    especially for large clusters where 1% of disk dwarfs actual page cache.
    """
    bounds: List[float] = [1.0, inputs.ws_slo_bound, inputs.ws_rps_bound]
    has_anchor = policy.use_observational_anchor and _has_usable_current(inputs)

    if has_anchor:
        bounds.append(_observational_ws_fraction(inputs, policy))

    floor = 0.0 if has_anchor else policy.min_working_set
    effective_ws = max(floor, min(bounds))
    return _emit(inputs, effective_ws)


def _has_usable_current(inputs: MemoryInputs) -> bool:
    return bool(inputs.current_capacity and inputs.current_capacity.cluster_instance)


def _observational_ws_fraction(inputs: MemoryInputs, policy: MemoryPolicy) -> float:
    """Derive working-set from current cluster's page-cache-to-disk ratio.

    Page cache headroom is taken from ``memory_layout`` so the value is
    consistent with how heap and reserves are budgeted on the same node.
    """
    cc = inputs.current_capacity
    assert cc and cc.cluster_instance  # guaranteed by _has_usable_current

    layout = memory_layout(
        ram_gib=cc.cluster_instance.ram_gib,
        base_reserves_gib=_get_base_memory(inputs.desires),
        max_page_cache_gib=policy.max_page_cache_gib,
    )

    raw_disk_per_node = cc.disk_utilization_gib.mid
    if raw_disk_per_node <= 0:
        raw_disk_per_node = inputs.disk_used_gib / cc.cluster_instance_count.mid
    raw_disk_per_node = max(1, raw_disk_per_node)

    return min(1.0, layout.page_cache_capacity_gib / raw_disk_per_node)


def _emit(inputs: MemoryInputs, effective_ws: float) -> MemoryEstimate:
    """Build ``MemoryEstimate``, honoring the memory-preserve buffer override.

    When memory preserve is set, keep the current cluster's total page cache
    as the requirement and zero out write_buffer_gib (memtables live in heap,
    separate from page cache — preserve handles heap sizing separately).
    """
    page_cache_linear_demand = max(1.0, effective_ws * inputs.disk_used_gib)
    effective_page_cache_budget = max(
        1.0,
        inputs.effective_page_cache_gib_per_node
        * max(1, inputs.planned_page_cache_nodes),
    )
    page_cache_demand: float = max(
        1, int(min(page_cache_linear_demand, effective_page_cache_budget))
    )
    write_buffer_gib = inputs.write_buffer_gib

    cc = inputs.current_capacity
    memory_derived = DerivedBuffers.for_components(
        inputs.desires.buffers.derived, [BufferComponent.memory]
    )
    if memory_derived.preserve and cc and cc.cluster_instance:
        layout = memory_layout(
            ram_gib=cc.cluster_instance.ram_gib,
            base_reserves_gib=_get_base_memory(inputs.desires),
            max_page_cache_gib=cc.cluster_instance.ram_gib,
        )
        page_cache_demand = (
            layout.page_cache_capacity_gib * cc.cluster_instance_count.mid
        )
        write_buffer_gib = 0

    return MemoryEstimate(
        effective_ws_fraction=effective_ws,
        page_cache_demand_gib=page_cache_demand,
        page_cache_linear_demand_gib=page_cache_linear_demand,
        effective_page_cache_budget_gib=effective_page_cache_budget,
        write_buffer_gib=write_buffer_gib,
    )
