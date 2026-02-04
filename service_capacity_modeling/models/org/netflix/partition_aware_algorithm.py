"""
Partition-Aware Capacity Planning Algorithm

This module contains the core algorithm for partition-aware capacity planning.
The algorithm optimizes for fault tolerance by preferring higher replication
factors (RF) when multiple valid configurations exist.

Key principle: Higher PPn → fewer base nodes → higher RF for same CPU → better
fault tolerance.
"""

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class CapacityProblem:
    """Input parameters for the partition-aware capacity algorithm."""

    n_partitions: int  # Total number of partitions
    partition_size_gib: float  # Size of one partition (with buffer)
    disk_per_node_gib: float  # Effective disk capacity per node
    cpu_per_node: int  # CPU cores per node
    cpu_needed: int  # Total CPU cores needed
    min_rf: int  # Minimum replication factor
    max_nodes: int  # Maximum allowed nodes in cluster


@dataclass
class CapacityResult:
    """Output of the partition-aware capacity algorithm."""

    node_count: int  # Total nodes in cluster
    replica_count: int  # Replication factor
    partitions_per_node: int  # Partitions placed on each node
    nodes_for_one_copy: int  # Nodes needed for one complete copy of data


def find_first_valid_configuration(
    problem: CapacityProblem,
) -> Optional[CapacityResult]:
    """
    Search algorithm: tries PPn from max down to 1, returns first valid config.

    Strategy: Start with max PPn (highest RF, best fault tolerance). If that
    exceeds max_nodes, try lower PPn values until one fits.

    Why start from max PPn?
        Higher PPn → fewer base nodes → higher RF for same CPU → better fault
        tolerance. This is the key optimization: we ALWAYS prefer configurations
        with higher RF when they fit within max_nodes.

    Why linear search?
        node_count(ppn) is NON-MONOTONIC due to ceiling effects:
            PPn=10: count=12
            PPn=5:  count=10  ← drops!
            PPn=4:  count=12  ← jumps back!
        Binary search doesn't work. Linear is O(max_ppn) ≈ O(100) in practice.


    Args:
        problem: The capacity planning problem parameters

    Returns:
        CapacityResult with the optimal configuration, or None if no valid
        configuration exists within the constraints.
    """
    max_ppn = int(problem.disk_per_node_gib / problem.partition_size_gib)
    if max_ppn < 1:
        return None

    for ppn in range(max_ppn, 0, -1):
        base = math.ceil(problem.n_partitions / ppn)

        # Calculate minimum RF for CPU (same math as closed_form)
        if base >= 2:
            rf = max(
                problem.min_rf,
                math.ceil(problem.cpu_needed / (base * problem.cpu_per_node)),
            )
            total = base * rf
        else:
            # base == 1: special case - need at least 2 nodes for availability
            if 2 * problem.cpu_per_node >= problem.cpu_needed:
                rf = problem.min_rf
                total = max(2, rf)
            else:
                rf = max(
                    problem.min_rf,
                    math.ceil(problem.cpu_needed / problem.cpu_per_node),
                )
                total = rf

        if total <= problem.max_nodes:
            return CapacityResult(
                node_count=total,
                replica_count=rf,
                partitions_per_node=ppn,
                nodes_for_one_copy=base,
            )

    return None
