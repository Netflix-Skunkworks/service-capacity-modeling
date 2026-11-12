"""
Partition-Aware Capacity Planning Algorithm

This module contains the core algorithm for partition-aware capacity planning.
The algorithm minimizes node count and prefers higher replication factors (RF)
when multiple valid configurations have the same node count.

Key principle: Minimize cost first, then prefer fault tolerance among plans with
the same cost.

Besides the search, a caller can place partitions at a fixed layout
(``plan_at_layout``) or on exactly some number of nodes (``plan_at_count``),
and ask which resource blocks a layout that does not fit.
"""

import math
from typing import Optional
from typing import Tuple

from pydantic import BaseModel
from pydantic import Field
from pydantic import model_validator

from service_capacity_modeling.interface import Bottleneck


class CapacityProblem(BaseModel):
    """Input parameters for the partition-aware capacity algorithm."""

    n_partitions: int  # Total number of partitions
    partition_size_gib: float  # Size of one partition (with buffer)
    disk_per_node_gib: float  # Effective disk capacity per node
    memory_per_partition_gib: Optional[float] = Field(default=None, gt=0)
    memory_per_node_gib: Optional[float] = Field(default=None, ge=0)
    cpu_per_node: int  # CPU cores per node
    cpu_needed: int  # Total CPU cores needed
    min_rf: int  # Minimum replication factor
    max_nodes: int  # Maximum allowed nodes in cluster
    # Most partitions any node may hold, on top of disk and memory limits.
    max_partitions_per_node: Optional[int] = Field(default=None, ge=1)

    @model_validator(mode="after")
    def memory_constraints_are_complete(self) -> "CapacityProblem":
        """Require both memory values when partition placement uses memory."""
        if (self.memory_per_partition_gib is None) != (
            self.memory_per_node_gib is None
        ):
            raise ValueError(
                "memory_per_partition_gib and memory_per_node_gib must be set together"
            )
        return self


class CapacityResult(BaseModel):
    """Output of the partition-aware capacity algorithm."""

    node_count: int  # Total nodes in cluster
    replica_count: int  # Replication factor
    partitions_per_node: int  # Partitions placed on each node
    nodes_for_one_copy: int  # Nodes needed for one complete copy of data
    max_partitions_per_node_by_disk: int
    max_partitions_per_node_by_memory: Optional[int] = None


def _partitions_that_fit(capacity_gib: float, partition_size_gib: float) -> int:
    """Floor a capacity ratio without losing an exact fit to float error."""
    fit = capacity_gib / partition_size_gib
    return math.floor(math.nextafter(fit, math.inf))


def partition_limits(problem: CapacityProblem) -> Tuple[int, Optional[int]]:
    """The most partitions one node holds by disk and, when placement uses
    memory, by memory."""
    disk_ppn = _partitions_that_fit(
        problem.disk_per_node_gib, problem.partition_size_gib
    )
    memory_ppn = None
    # CapacityProblem's validator requires both memory values or neither.
    if (
        problem.memory_per_partition_gib is not None
        and problem.memory_per_node_gib is not None
    ):
        memory_ppn = _partitions_that_fit(
            problem.memory_per_node_gib, problem.memory_per_partition_gib
        )
    return disk_ppn, memory_ppn


def _max_partitions_per_node(problem: CapacityProblem) -> int:
    """The most partitions a node may hold under every limit."""
    disk_ppn, memory_ppn = partition_limits(problem)
    limits = [disk_ppn, memory_ppn, problem.max_partitions_per_node]
    return min(limit for limit in limits if limit is not None)


def replicas_needed(problem: CapacityProblem, nodes_for_one_copy: int) -> int:
    """The fewest replicas that meet the minimum RF and the CPU requirement."""
    return max(
        problem.min_rf,
        math.ceil(problem.cpu_needed / (nodes_for_one_copy * problem.cpu_per_node)),
    )


def plan_at_layout(
    problem: CapacityProblem, partitions_per_node: int, replica_count: int
) -> Optional[CapacityResult]:
    """The configuration at exactly this layout, or None if it does not fit."""
    if not 1 <= partitions_per_node <= _max_partitions_per_node(problem):
        return None
    nodes_for_one_copy = math.ceil(problem.n_partitions / partitions_per_node)
    if replica_count < replicas_needed(problem, nodes_for_one_copy):
        return None
    # Ensure at least 2 nodes for availability
    node_count = max(nodes_for_one_copy * replica_count, 2)
    if node_count > problem.max_nodes:
        return None
    disk_ppn, memory_ppn = partition_limits(problem)
    return CapacityResult(
        node_count=node_count,
        replica_count=replica_count,
        partitions_per_node=partitions_per_node,
        nodes_for_one_copy=nodes_for_one_copy,
        max_partitions_per_node_by_disk=disk_ppn,
        max_partitions_per_node_by_memory=memory_ppn,
    )


def _layouts_at_count(count: int, n_partitions: int) -> list[Tuple[int, int]]:
    """(partitions per node, replicas) for each way to split ``count`` nodes
    into equal copies, densest first."""
    layouts = []
    for nodes_for_one_copy in range(1, count + 1):
        if count % nodes_for_one_copy == 0:
            ppn = math.ceil(n_partitions / nodes_for_one_copy)
            layouts.append((ppn, count // nodes_for_one_copy))
    return layouts


def plan_at_count(problem: CapacityProblem, count: int) -> Optional[CapacityResult]:
    """The densest layout on exactly ``count`` nodes, or None if none fits.
    Densest first leaves the most replicas for CPU and availability."""
    for ppn, rf in _layouts_at_count(count, problem.n_partitions):
        result = plan_at_layout(problem, ppn, rf)
        if result is not None and result.node_count == count:
            return result
    return None


def layout_shortfall(
    problem: CapacityProblem, partitions_per_node: int, replica_count: int
) -> Optional[Bottleneck]:
    """The resource that keeps a node from hosting this layout, or None when
    the limit is the node count or the partition cap instead."""
    disk_ppn, memory_ppn = partition_limits(problem)
    if partitions_per_node > disk_ppn:
        return Bottleneck.disk_capacity
    if memory_ppn is not None and partitions_per_node > memory_ppn:
        return Bottleneck.memory
    nodes_for_one_copy = math.ceil(problem.n_partitions / partitions_per_node)
    if replica_count < replicas_needed(problem, nodes_for_one_copy):
        return Bottleneck.cpu
    return None


def count_shortfall(problem: CapacityProblem, count: int) -> Optional[Bottleneck]:
    """Why no layout fits on exactly ``count`` nodes: the resource that blocks
    the densest layout whose partitions fit on disk and in memory, else the
    sparsest layout's resource."""
    shortfall = None
    for ppn, rf in _layouts_at_count(count, problem.n_partitions):
        shortfall = layout_shortfall(problem, ppn, rf)
        if shortfall not in (Bottleneck.disk_capacity, Bottleneck.memory):
            return shortfall
    return shortfall


def search_for_min_nodes(
    problem: CapacityProblem,
) -> Optional[CapacityResult]:
    """
    Find the configuration with the fewest nodes that fits within max_nodes.

    The algorithm evaluates every valid partition density. It prefers higher RF
    for configurations with the same node count.

    Args:
        problem: The capacity planning problem parameters

    Returns:
        CapacityResult with the lowest node count, or None if no valid
        configuration exists within the constraints.
    """
    disk_ppn, memory_ppn = partition_limits(problem)
    max_ppn = _max_partitions_per_node(problem)
    if max_ppn < 1:
        return None

    best_candidate: Optional[tuple[int, int, int, int]] = None
    ppn = max_ppn
    while ppn > 0:
        nodes_for_one_copy = math.ceil(problem.n_partitions / ppn)
        rf = replicas_needed(problem, nodes_for_one_copy)
        # Ensure at least 2 nodes for availability
        total_nodes = max(nodes_for_one_copy * rf, 2)

        if total_nodes <= problem.max_nodes:
            candidate = (total_nodes, -rf, -ppn, nodes_for_one_copy)
            if best_candidate is None or candidate[:3] < best_candidate[:3]:
                best_candidate = candidate

        # All skipped densities have the same nodes_for_one_copy and therefore
        # the same RF and node count. The current ppn wins their tie-break.
        ppn = (problem.n_partitions - 1) // nodes_for_one_copy

    if best_candidate is None:
        return None

    node_count, negative_rf, negative_ppn, nodes_for_one_copy = best_candidate
    return CapacityResult(
        node_count=node_count,
        replica_count=-negative_rf,
        partitions_per_node=-negative_ppn,
        nodes_for_one_copy=nodes_for_one_copy,
        max_partitions_per_node_by_disk=disk_ppn,
        max_partitions_per_node_by_memory=memory_ppn,
    )
