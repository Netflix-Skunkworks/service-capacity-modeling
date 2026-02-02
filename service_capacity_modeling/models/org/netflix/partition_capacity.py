"""
Partition capacity planning for read-only KV stores with fault tolerance.

Uses a closed-form algorithm to calculate optimal (node_count, RF) configurations
based on CPU, disk, and availability constraints. The utility-based optimizer
balances availability against cost for different service tiers.

Example:
    problem = CapacityProblem(n_partitions=100, partition_size_gib=100,
                               disk_per_node_gib=2048, min_rf=2,
                               cpu_needed=800, cpu_per_node=16)
    result = search_with_fault_tolerance(problem, tier=2, cost_per_node=100)
"""

import math
from dataclasses import dataclass
from math import comb
from typing import Optional

__all__ = [
    # Entry points
    "search_with_fault_tolerance",
    "find_capacity_config",
    "closed_form_algorithm",
    # Data structures
    "CapacityProblem",
    "CapacityResult",
    "FaultTolerantResult",
    # Configuration
    "TIER_DEFAULTS",
    "FaultToleranceConfig",
    "get_tier_config",
    # Probability functions
    "per_partition_unavailability",
    "system_availability",
    "system_unavailability",
    "utility",
]


# =============================================================================
# PUBLIC API: Data Structures
# =============================================================================


@dataclass(frozen=True)
class CapacityProblem:
    """Input: What capacity do you need?

    Example:
        problem = CapacityProblem(
            n_partitions=100,       # 100 data partitions
            partition_size_gib=100, # 100 GiB each
            disk_per_node_gib=2048, # 2 TiB usable per node
            min_rf=2,               # At least 2 replicas
            cpu_needed=800,         # 800 CPU cores total
            cpu_per_node=16,        # 16 cores per node
        )
    """

    n_partitions: int  # Total partitions to distribute
    partition_size_gib: float  # Size of each partition
    disk_per_node_gib: float  # Usable disk per node
    min_rf: int  # Minimum replication factor
    cpu_needed: int  # Total CPU cores required
    cpu_per_node: int  # CPU cores per node
    max_nodes: int = 10000  # Maximum cluster size


@dataclass(frozen=True)
class FaultTolerantResult:
    """Output: What to provision.

    Attributes:
        node_count: Total nodes to deploy
        rf: Replication factor to use
        system_availability: P(all partitions available | 1 AZ fails)
        cost: Total cost (node_count * cost_per_node)
        zone_aware_savings: How much we'd save with zone-aware placement
    """

    node_count: int
    rf: int
    partitions_per_node: int
    base_nodes: int
    system_availability: float
    per_partition_unavail: float
    cost: float
    zone_aware_cost: float
    zone_aware_savings: float
    utility_score: float
    tier: int


# =============================================================================
# PUBLIC API: Configuration
# =============================================================================


@dataclass(frozen=True)
class FaultToleranceConfig:
    """Configuration for a service tier."""

    min_rf: int  # Minimum replication factor
    target_availability: float  # Target availability (e.g., 0.999)
    cost_sensitivity: float  # How much to penalize cost (higher = more cost-focused)
    max_cost_multiplier: float  # Max cost as multiple of baseline


# Tier 0: Critical (99.9%), Tier 1: Important (99%), Tier 2: Standard (95%)
# Tier 3: Dev (80%)
TIER_DEFAULTS: dict[int, FaultToleranceConfig] = {
    0: FaultToleranceConfig(
        min_rf=3,
        target_availability=0.999,
        cost_sensitivity=0.3,
        max_cost_multiplier=3.0,
    ),
    1: FaultToleranceConfig(
        min_rf=3,
        target_availability=0.99,
        cost_sensitivity=0.5,
        max_cost_multiplier=2.5,
    ),
    2: FaultToleranceConfig(
        min_rf=2,
        target_availability=0.95,
        cost_sensitivity=1.0,
        max_cost_multiplier=2.0,
    ),
    3: FaultToleranceConfig(
        min_rf=2,
        target_availability=0.80,
        cost_sensitivity=2.0,
        max_cost_multiplier=1.5,
    ),
}


def get_tier_config(tier: int) -> FaultToleranceConfig:
    """Get config for tier (0-3). Defaults to tier 2 if out of range."""
    return TIER_DEFAULTS.get(tier, TIER_DEFAULTS[2])


# =============================================================================
# PUBLIC API: Entry Points
# =============================================================================


def search_with_fault_tolerance(
    problem: CapacityProblem,
    tier: int,
    cost_per_node: float,
    n_zones: int = 3,
    zone_aware: bool = False,
) -> Optional[FaultTolerantResult]:
    """Find optimal configuration balancing availability and cost.

    This is the PRIMARY entry point. It enumerates configurations,
    scores each with the utility function, and returns the best.

    Args:
        problem: Capacity requirements
        tier: Service tier (0=critical, 3=dev)
        cost_per_node: Cost per node for total cost calculation
        n_zones: Number of availability zones (default 3)
        zone_aware: If True, assume zone-aware placement where RF>=2
                    guarantees replicas span multiple AZs (100% availability).
                    If False (default), assume random placement.

    Returns:
        FaultTolerantResult with optimal config, or None if impossible.

    Example:
        result = search_with_fault_tolerance(problem, tier=2, cost_per_node=100)
        print(f"Deploy {result.node_count} nodes with RF={result.rf}")
        print(f"Availability: {result.system_availability:.2%}")

        # With zone-aware placement (if your infrastructure supports it):
        result = search_with_fault_tolerance(problem, tier=2, cost_per_node=100,
                                              zone_aware=True)
    """
    config = get_tier_config(tier)

    max_ppn = int(problem.disk_per_node_gib / problem.partition_size_gib)
    if max_ppn < 1:
        return None

    # Baseline cost for utility comparison
    baseline_result = find_capacity_config(problem)
    if baseline_result is None:
        return None
    base_cost = baseline_result.node_count * cost_per_node
    max_cost = base_cost * config.max_cost_multiplier

    # Pre-compute zone-aware cost for comparison
    zone_aware_cost = _calculate_zone_aware_cost(problem, tier, cost_per_node, n_zones)

    best_result: Optional[FaultTolerantResult] = None
    best_utility = float("-inf")

    # Effective min_rf: respect both tier requirement and problem constraint
    effective_min_rf = max(config.min_rf, problem.min_rf)

    # Enumerate all (PPn, RF) configurations
    for ppn in range(max_ppn, 0, -1):
        base_nodes = math.ceil(problem.n_partitions / ppn)

        # Minimum RF to meet CPU (also respects effective_min_rf)
        min_rf_for_cpu = _min_rf_for_cpu(
            base_nodes, effective_min_rf, problem.cpu_needed, problem.cpu_per_node
        )

        # Maximum RF limited by cost
        max_rf_for_cost = int(max_cost / (base_nodes * cost_per_node))
        max_rf = max(min_rf_for_cpu, min(10, max_rf_for_cost))

        for rf in range(min_rf_for_cpu, max_rf + 1):
            node_count = max(2, base_nodes * rf) if base_nodes >= 2 else max(2, rf)

            if node_count > problem.max_nodes or node_count > max_cost / cost_per_node:
                continue

            # Calculate availability based on placement strategy
            avail = _system_availability_for_placement(
                node_count, n_zones, rf, problem.n_partitions, zone_aware
            )
            cost = node_count * cost_per_node

            u = utility(avail, cost, tier, base_cost)

            if u > best_utility:
                best_utility = u
                best_result = FaultTolerantResult(
                    node_count=node_count,
                    rf=rf,
                    partitions_per_node=ppn,
                    base_nodes=base_nodes,
                    system_availability=avail,
                    per_partition_unavail=_per_partition_unavail_for_placement(
                        node_count, n_zones, rf, zone_aware
                    ),
                    cost=cost,
                    zone_aware_cost=zone_aware_cost,
                    zone_aware_savings=cost - zone_aware_cost,
                    utility_score=u,
                    tier=tier,
                )

    return best_result


# =============================================================================
# Capacity Algorithms
# =============================================================================


@dataclass(frozen=True)
class CapacityResult:
    """Result type for capacity algorithms."""

    node_count: int
    rf: int
    partitions_per_node: int
    base_nodes: int


def closed_form_algorithm(
    p: CapacityProblem, ppn: Optional[int] = None
) -> Optional[CapacityResult]:
    """O(1) capacity calculation for a given PPn.

    When ppn=None, uses max PPn (equivalent to original_algorithm).
    When ppn is specified, calculates for that specific PPn value.
    """
    if ppn is None:
        ppn = int(p.disk_per_node_gib / p.partition_size_gib)
    if ppn < 1:
        return None

    base = math.ceil(p.n_partitions / ppn)
    rf = _min_rf_for_cpu(base, p.min_rf, p.cpu_needed, p.cpu_per_node)
    total = max(2, base * rf) if base >= 2 else max(2, rf)

    if total > p.max_nodes:
        return None
    return CapacityResult(total, rf, ppn, base)


def find_capacity_config(p: CapacityProblem) -> Optional[CapacityResult]:
    """Find a valid capacity configuration by searching PPn values.

    Searches partitions-per-node (PPn) from max to 1, returning the first
    valid configuration. Uses closed_form_algorithm for O(1) calculation
    at each PPn value.

    This is the recommended algorithm - it finds solutions that a greedy
    max-PPn approach would miss when max_nodes is tight.
    """
    max_ppn = int(p.disk_per_node_gib / p.partition_size_gib)
    for ppn in range(max_ppn, 0, -1):
        result = closed_form_algorithm(p, ppn=ppn)
        if result is not None:
            return result
    return None


def _min_rf_for_cpu(
    base_nodes: int, min_rf: int, cpu_needed: int, cpu_per_node: int
) -> int:
    """Calculate minimum RF to satisfy CPU requirement."""
    if base_nodes >= 2:
        return max(min_rf, math.ceil(cpu_needed / (base_nodes * cpu_per_node)))
    else:
        if 2 * cpu_per_node >= cpu_needed:
            return min_rf
        return max(min_rf, math.ceil(cpu_needed / cpu_per_node))


def _calculate_zone_aware_cost(
    problem: CapacityProblem,
    tier: int,
    cost_per_node: float,
    n_zones: int,
) -> float:
    """Calculate optimal cost using zone-aware placement.

    Runs the same utility-based optimization but assuming zone-aware placement
    where RF>=2 guarantees 100% availability. This gives a fair apples-to-apples
    comparison of what you'd pay with zone-aware infrastructure.
    """
    config = get_tier_config(tier)

    max_ppn = int(problem.disk_per_node_gib / problem.partition_size_gib)
    if max_ppn < 1:
        return float("inf")

    # Baseline cost for utility comparison (same as main search)
    baseline_result = find_capacity_config(problem)
    if baseline_result is None:
        return float("inf")
    base_cost = baseline_result.node_count * cost_per_node
    max_cost = base_cost * config.max_cost_multiplier

    best_cost = float("inf")
    best_utility = float("-inf")

    # Effective min_rf: respect both tier requirement and problem constraint
    effective_min_rf = max(config.min_rf, problem.min_rf)

    # Same search loop, but using zone-aware availability
    for ppn in range(max_ppn, 0, -1):
        base_nodes = math.ceil(problem.n_partitions / ppn)
        min_rf_for_cpu = _min_rf_for_cpu(
            base_nodes, effective_min_rf, problem.cpu_needed, problem.cpu_per_node
        )
        max_rf_for_cost = int(max_cost / (base_nodes * cost_per_node))
        max_rf = max(min_rf_for_cpu, min(10, max_rf_for_cost))

        for rf in range(min_rf_for_cpu, max_rf + 1):
            node_count = max(2, base_nodes * rf) if base_nodes >= 2 else max(2, rf)

            if node_count > problem.max_nodes or node_count > max_cost / cost_per_node:
                continue

            # Zone-aware: RF>=2 guarantees 100% availability
            avail = _system_availability_for_placement(
                node_count, n_zones, rf, problem.n_partitions, zone_aware=True
            )
            cost = node_count * cost_per_node

            u = utility(avail, cost, tier, base_cost)

            if u > best_utility:
                best_utility = u
                best_cost = cost

    return best_cost


# =============================================================================
# Probability Functions
# =============================================================================


def per_partition_unavailability(n_nodes: int, n_zones: int, rf: int) -> float:
    """P(partition unavailable | 1 AZ fails) for random placement.

    Formula: (1/n_zones) * sum over z of: C(nodes_in_z, rf) / C(n_nodes, rf)
    """
    if n_zones <= 0 or n_nodes <= 0 or rf <= 0 or rf > n_nodes:
        return 0.0

    base_per_zone = n_nodes // n_zones
    remainder = n_nodes % n_zones
    denominator = comb(n_nodes, rf)
    if denominator == 0:
        return 0.0

    total_prob = 0.0
    for z in range(n_zones):
        nodes_in_zone = base_per_zone + (1 if z < remainder else 0)
        if rf <= nodes_in_zone:
            total_prob += comb(nodes_in_zone, rf) / denominator

    return total_prob / n_zones


def system_availability(
    n_nodes: int, n_zones: int, rf: int, n_partitions: int
) -> float:
    """P(all partitions available | 1 AZ fails).

    Formula: (1/n_zones) * sum over z of: (1 - p_z)^n_partitions
    """
    if n_zones <= 0 or n_nodes <= 0 or rf <= 0 or n_partitions <= 0:
        return 1.0 if n_partitions <= 0 else 0.0
    if rf > n_nodes:
        return 0.0

    base_per_zone = n_nodes // n_zones
    remainder = n_nodes % n_zones
    denominator = comb(n_nodes, rf)
    if denominator == 0:
        return 0.0

    total_avail = 0.0
    for z in range(n_zones):
        nodes_in_zone = base_per_zone + (1 if z < remainder else 0)
        p_unavail = (
            comb(nodes_in_zone, rf) / denominator if rf <= nodes_in_zone else 0.0
        )
        total_avail += (1.0 - p_unavail) ** n_partitions if p_unavail < 1.0 else 0.0

    return total_avail / n_zones


# =============================================================================
# Utility Function (AIMA Ch 16)
# =============================================================================


def utility(availability: float, cost: float, tier: int, base_cost: float) -> float:
    """Score a config: U = availability_value - cost_penalty.

    Above target: diminishing returns via log1p
    Below target: linear penalty
    """
    config = get_tier_config(tier)
    avail_above_target = availability - config.target_availability

    if avail_above_target >= 0:
        availability_value = math.log1p(avail_above_target * 100)
    else:
        availability_value = avail_above_target * 100

    cost_penalty = (
        config.cost_sensitivity * (cost / base_cost - 1.0) if base_cost > 0 else 0.0
    )
    return availability_value - cost_penalty


# Convenience function: system unavailability = 1 - system availability
def system_unavailability(
    n_nodes: int, n_zones: int, rf: int, n_partitions: int
) -> float:
    """P(at least one partition unavailable | 1 AZ fails)."""
    return 1.0 - system_availability(n_nodes, n_zones, rf, n_partitions)


# =============================================================================
# Placement Strategy Helpers
# =============================================================================


def _system_availability_for_placement(
    n_nodes: int, n_zones: int, rf: int, n_partitions: int, zone_aware: bool
) -> float:
    """Calculate system availability based on placement strategy.

    Args:
        zone_aware: If True, RF>=2 guarantees 100% availability (replicas
                    are placed across AZs). If False, uses random placement.
    """
    if zone_aware:
        # Zone-aware: RF>=2 guarantees cross-AZ spread
        if rf >= 2:
            return 1.0
        if n_zones <= 0 or n_partitions <= 0:
            return 1.0
        return (1.0 - 1.0 / n_zones) ** n_partitions
    else:
        # Random placement: use binomial failure model
        return system_availability(n_nodes, n_zones, rf, n_partitions)


def _per_partition_unavail_for_placement(
    n_nodes: int, n_zones: int, rf: int, zone_aware: bool
) -> float:
    """Calculate per-partition unavailability based on placement strategy."""
    if zone_aware:
        # Zone-aware: RF>=2 means 0 unavailability
        return 0.0 if rf >= 2 else (1.0 / n_zones if n_zones > 0 else 1.0)
    else:
        # Random placement: use binomial failure model
        return per_partition_unavailability(n_nodes, n_zones, rf)
