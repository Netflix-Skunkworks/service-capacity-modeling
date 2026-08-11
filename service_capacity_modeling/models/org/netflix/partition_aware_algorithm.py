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
    max_ppn = int(problem.disk_per_node_gib / problem.partition_size_gib)
    if problem.memory_per_partition_gib is not None:
        assert problem.memory_per_node_gib is not None
        max_ppn = min(
            max_ppn,
            int(problem.memory_per_node_gib / problem.memory_per_partition_gib),
        )
    if max_ppn < 1:
        return None

    best_result: Optional[CapacityResult] = None
    best_rank: Optional[tuple[int, int, int]] = None
    for ppn in range(min(max_ppn, problem.n_partitions), 0, -1):
        nodes_for_one_copy = math.ceil(problem.n_partitions / ppn)
        reported_ppn = max_ppn if nodes_for_one_copy == 1 else ppn

        # Calculate minimum RF for CPU
        rf = max(
            problem.min_rf,
            math.ceil(problem.cpu_needed / (nodes_for_one_copy * problem.cpu_per_node)),
        )
        # Ensure at least 2 nodes for availability
        total_nodes = max(nodes_for_one_copy * rf, 2)

        if total_nodes <= problem.max_nodes:
            candidate = CapacityResult(
                node_count=total_nodes,
                replica_count=rf,
                partitions_per_node=reported_ppn,
                nodes_for_one_copy=nodes_for_one_copy,
            )
            candidate_rank = (total_nodes, -rf, -reported_ppn)
            if best_rank is None or candidate_rank < best_rank:
                best_result = candidate
                best_rank = candidate_rank

    return best_result
