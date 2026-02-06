"""
Partition-Aware Capacity Planning Algorithm

This module contains the core algorithm for partition-aware capacity planning.
The algorithm optimizes for fault tolerance by preferring higher replication
factors (RF) when multiple valid configurations exist.

Key principle: Higher PPn → fewer base nodes → higher RF for same CPU → better
fault tolerance.
"""

import math
from typing import Optional

from pydantic import BaseModel


class CapacityProblem(BaseModel):
    """Input parameters for the partition-aware capacity algorithm."""

    n_partitions: int  # Total number of partitions
    partition_size_gib: float  # Size of one partition (with buffer)
    disk_per_node_gib: float  # Effective disk capacity per node
    cpu_per_node: int  # CPU cores per node
    cpu_needed: int  # Total CPU cores needed
    min_rf: int  # Minimum replication factor
    max_nodes: int  # Maximum allowed nodes in cluster


class CapacityResult(BaseModel):
    """Output of the partition-aware capacity algorithm."""

    node_count: int  # Total nodes in cluster
    replica_count: int  # Replication factor
    partitions_per_node: int  # Partitions placed on each node
    nodes_for_one_copy: int  # Nodes needed for one complete copy of data


def search_for_max_rf(
    problem: CapacityProblem,
) -> Optional[CapacityResult]:
    """
    Find the configuration with the highest RF (replication factor) that fits
    within max_nodes.

    Higher RF = better fault tolerance. The algorithm searches from max PPn
    down to 1, returning the first valid configuration found.

    Why higher PPn gives higher RF:
        Higher PPn → fewer nodes_for_one_copy → less CPU per copy →
        need more copies (higher RF) to meet CPU requirements.

    Args:
        problem: The capacity planning problem parameters

    Returns:
        CapacityResult with the highest RF configuration, or None if no valid
        configuration exists within the constraints.
    """
    max_ppn = int(problem.disk_per_node_gib / problem.partition_size_gib)
    if max_ppn < 1:
        return None

    for ppn in range(max_ppn, 0, -1):
        nodes_for_one_copy = math.ceil(problem.n_partitions / ppn)

        # Calculate minimum RF for CPU
        rf = max(
            problem.min_rf,
            math.ceil(problem.cpu_needed / (nodes_for_one_copy * problem.cpu_per_node)),
        )
        # Ensure at least 2 nodes for availability
        total_nodes = max(nodes_for_one_copy * rf, 2)

        if total_nodes <= problem.max_nodes:
            return CapacityResult(
                node_count=total_nodes,
                replica_count=rf,
                partitions_per_node=ppn,
                nodes_for_one_copy=nodes_for_one_copy,
            )

    return None
