"""
Partition-Aware Capacity Planning Algorithm

This module contains the core algorithm for partition-aware capacity planning.
The algorithm minimizes node count and prefers higher replication factors (RF)
when multiple valid configurations have the same node count.

Key principle: Minimize cost first, then prefer fault tolerance among plans with
the same cost.
"""

import math
from typing import Optional

from pydantic import BaseModel
from pydantic import Field
from pydantic import model_validator


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
    disk_ppn = _partitions_that_fit(
        problem.disk_per_node_gib, problem.partition_size_gib
    )
    memory_ppn = None
    if problem.memory_per_partition_gib is not None:
        assert problem.memory_per_node_gib is not None
        memory_ppn = _partitions_that_fit(
            problem.memory_per_node_gib, problem.memory_per_partition_gib
        )

    max_ppn = min(disk_ppn, memory_ppn) if memory_ppn is not None else disk_ppn
    if max_ppn < 1:
        return None

    best_candidate: Optional[tuple[int, int, int, int]] = None
    ppn = max_ppn
    while ppn > 0:
        nodes_for_one_copy = math.ceil(problem.n_partitions / ppn)

        # Calculate minimum RF for CPU
        rf = max(
            problem.min_rf,
            math.ceil(problem.cpu_needed / (nodes_for_one_copy * problem.cpu_per_node)),
        )
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
