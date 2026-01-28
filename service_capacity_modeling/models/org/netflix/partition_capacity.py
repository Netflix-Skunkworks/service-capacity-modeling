"""
Partition-aware capacity planning algorithms.

This module contains the core algorithms for read-only KV capacity planning,
extracted for clarity and testability. The algorithms solve:

    Given P partitions of size S, find (node_count, replica_factor) such that:
    - DISK: All partitions × replicas fit on cluster
    - CPU:  node_count × cpu_per_node ≥ cpu_needed
    - RF:   replica_factor ≥ min_rf
    - SIZE: node_count ≤ max_nodes

Three algorithm variants are provided:
1. ORIGINAL: While-loop from legacy code (greedy, max PPn only)
2. CLOSED_FORM: O(1) mathematical equivalent to original
3. SEARCH: Searches PPn from max to 1, finds solutions original misses

The SEARCH algorithm is strictly more capable than ORIGINAL/CLOSED_FORM.
"""

import math
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class CapacityProblem:
    """Input specification for partition-aware capacity planning.

    All fields are pure numbers - no model objects. This makes the algorithm
    testable without constructing Instance/Requirement/etc.
    """

    n_partitions: int  # Total partitions to distribute
    partition_size_gib: float  # Size of each partition
    disk_per_node_gib: float  # Usable disk per node
    min_rf: int  # Minimum replication factor
    cpu_needed: int  # Total CPU cores required
    cpu_per_node: int  # CPU cores per node
    max_nodes: int = 10000  # Maximum cluster size


@dataclass(frozen=True)
class CapacityResult:
    """Output of capacity planning algorithm."""

    node_count: int  # Total nodes in cluster
    rf: int  # Replication factor
    partitions_per_node: int  # Partitions stored per node
    base_nodes: int  # Nodes for one copy (before replication)


# =============================================================================
# ALGORITHM 1: ORIGINAL (while loop, greedy max PPn)
# =============================================================================


def original_algorithm(p: CapacityProblem) -> Optional[CapacityResult]:
    """
    Original while-loop algorithm from read_only_kv.py.

    Strategy: Use maximum partitions-per-node (greedy), increment RF until
    CPU is satisfied.

    Limitation: Only considers max PPn. If that exceeds max_nodes, returns None
    even when a lower PPn would work.
    """
    partitions_per_node = int(p.disk_per_node_gib / p.partition_size_gib)
    if partitions_per_node < 1:
        return None

    nodes_one_copy = math.ceil(p.n_partitions / partitions_per_node)

    replica_count = p.min_rf
    while True:
        count = nodes_one_copy * replica_count
        count = max(2, count)

        if count > p.max_nodes:
            return None

        if (count * p.cpu_per_node) >= p.cpu_needed:
            break

        replica_count += 1

    return CapacityResult(count, replica_count, partitions_per_node, nodes_one_copy)


# =============================================================================
# ALGORITHM 2: CLOSED-FORM (O(1), mathematically equivalent to original)
# =============================================================================


def closed_form_algorithm(p: CapacityProblem) -> Optional[CapacityResult]:
    """
    Closed-form replacement for the while loop. O(1), same results as original.

    Mathematical derivation:
        Need: max(2, base × rf) × cpu_per_node ≥ cpu_needed

        Case base ≥ 2:
            max(2, base×rf) = base×rf, so rf = ⌈cpu_needed / (base × cpu_per_node)⌉

        Case base = 1:
            rf=1 gives max(2,1)=2 nodes. If 2×cpu_per_node ≥ cpu_needed, done.
            Otherwise rf = ⌈cpu_needed / cpu_per_node⌉
    """
    ppn = int(p.disk_per_node_gib / p.partition_size_gib)
    if ppn < 1:
        return None

    base = math.ceil(p.n_partitions / ppn)

    if base >= 2:
        # Normal case: max(2, base×rf) = base×rf
        rf = max(p.min_rf, math.ceil(p.cpu_needed / (base * p.cpu_per_node)))
        total = base * rf
    else:
        # base == 1: 2-node minimum may satisfy CPU at min_rf
        if 2 * p.cpu_per_node >= p.cpu_needed:
            rf = p.min_rf
            total = max(2, rf)
        else:
            rf = max(p.min_rf, math.ceil(p.cpu_needed / p.cpu_per_node))
            total = rf  # rf ≥ 2 guaranteed, so max(2, 1×rf) = rf

    if total > p.max_nodes:
        return None

    return CapacityResult(total, rf, ppn, base)


# =============================================================================
# ALGORITHM 3: SEARCH (searches PPn from max to 1)
# =============================================================================


def search_algorithm(p: CapacityProblem) -> Optional[CapacityResult]:
    """
    Search algorithm: tries PPn from max down to 1, returns first valid config.

    Strategy: Start with max PPn (highest RF, best fault tolerance). If that
    exceeds max_nodes, try lower PPn values until one fits.

    Why start from max PPn?
        Higher PPn → fewer base nodes → higher RF for same CPU → better fault tolerance

    Why linear search?
        node_count(ppn) is NON-MONOTONIC due to ceiling effects:
            PPn=10: count=12
            PPn=5:  count=10  ← drops!
            PPn=4:  count=12  ← jumps back!
        Binary search doesn't work. Linear is O(max_ppn) ≈ O(100) in practice.

    This algorithm is strictly more capable than original/closed_form:
        - Returns same results when max_nodes is relaxed
        - Finds solutions when greedy (max PPn) exceeds max_nodes
    """
    max_ppn = int(p.disk_per_node_gib / p.partition_size_gib)
    if max_ppn < 1:
        return None

    for ppn in range(max_ppn, 0, -1):
        base = math.ceil(p.n_partitions / ppn)

        # Calculate minimum RF for CPU (same math as closed_form)
        if base >= 2:
            rf = max(p.min_rf, math.ceil(p.cpu_needed / (base * p.cpu_per_node)))
            total = base * rf
        else:
            if 2 * p.cpu_per_node >= p.cpu_needed:
                rf = p.min_rf
                total = max(2, rf)
            else:
                rf = max(p.min_rf, math.ceil(p.cpu_needed / p.cpu_per_node))
                total = rf

        if total <= p.max_nodes:
            return CapacityResult(total, rf, ppn, base)

    return None
