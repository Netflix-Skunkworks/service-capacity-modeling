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
from math import comb
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


# =============================================================================
# PLACEMENT MODELS: Different strategies for placing replicas across nodes
# =============================================================================


class PlacementModel:
    """Base class for placement strategies.

    Placement models define how replicas are distributed across nodes and zones,
    which affects fault tolerance under AZ failure. Override methods as needed.

    Following the codebase pattern: base class with default implementations,
    not ABC. This allows partial implementation and gradual adoption.
    """

    def per_partition_unavailability(
        self, n_nodes: int, n_zones: int, rf: int
    ) -> float:
        """P(partition unavailable | 1 AZ fails). Default: random placement."""
        return per_partition_unavailability(n_nodes, n_zones, rf)

    def system_availability(
        self, n_nodes: int, n_zones: int, rf: int, n_partitions: int
    ) -> float:
        """P(all partitions available | 1 AZ fails). Default: random placement."""
        return system_availability(n_nodes, n_zones, rf, n_partitions)

    def name(self) -> str:
        """Return the name of this placement model."""
        return self.__class__.__name__


class UniformRandomPlacement(PlacementModel):
    """Uniform random placement - the current default.

    Replicas are placed on randomly selected nodes without considering
    zone boundaries. This is how the current system works (region-aware
    only, not zone-aware).
    """


class ZoneAwarePlacement(PlacementModel):
    """Zone-aware placement - future optimization.

    With zone-aware placement, replicas are guaranteed to span multiple
    availability zones. This means RF>=2 gives 100% availability under
    single AZ failure.

    This is a placeholder for future optimization. The current system
    does NOT use zone-aware placement, but we calculate what it would
    cost to show stakeholders the savings opportunity.
    """

    def per_partition_unavailability(
        self, n_nodes: int, n_zones: int, rf: int
    ) -> float:
        """With zone-aware, RF>=2 guarantees cross-AZ spread."""
        if rf >= 2:
            return 0.0  # At least 2 replicas -> guaranteed cross-AZ
        # RF=1 still has 1/n_zones probability of being in failed zone
        return 1.0 / n_zones if n_zones > 0 else 1.0

    def system_availability(
        self, n_nodes: int, n_zones: int, rf: int, n_partitions: int
    ) -> float:
        """With zone-aware, RF>=2 gives 100% availability."""
        if rf >= 2:
            return 1.0
        # RF=1: P(system avail) = (1 - 1/n_zones)^n_partitions
        if n_zones <= 0 or n_partitions <= 0:
            return 1.0
        p = 1.0 / n_zones
        return (1.0 - p) ** n_partitions


# Default placement model (current system behavior)
DEFAULT_PLACEMENT = UniformRandomPlacement()


# =============================================================================
# FAULT TOLERANCE: Availability under AZ failure
# =============================================================================


def per_partition_unavailability(n_nodes: int, n_zones: int, rf: int) -> float:
    """Calculate P(partition unavailable | 1 AZ fails) for random placement.

    With uniform random placement, a partition is unavailable when an AZ fails
    if ALL its RF replicas happen to be in that failed AZ.

    When zones have uneven node counts (n_nodes % n_zones != 0), we must
    account for the varying probability of all replicas landing in each zone:

    P = (1/n_zones) * sum over zones z of: C(nodes_in_zone_z, rf) / C(n_nodes, rf)

    This averages across all possible zones that could fail.

    Args:
        n_nodes: Total nodes in cluster
        n_zones: Number of availability zones (typically 3)
        rf: Replication factor

    Returns:
        Probability in [0, 1]. Returns 0 if RF > max_nodes_per_zone.
    """
    if n_zones <= 0 or n_nodes <= 0 or rf <= 0:
        return 0.0

    if rf > n_nodes:
        return 0.0

    # Calculate nodes per zone (may be uneven)
    base_per_zone = n_nodes // n_zones
    remainder = n_nodes % n_zones

    # zones 0..remainder-1 get base+1 nodes, rest get base nodes
    # nodes_in_zone[z] = base + 1 if z < remainder else base

    denominator = comb(n_nodes, rf)
    if denominator == 0:
        return 0.0

    # Sum P(all rf in zone z) for each zone z, then average
    total_prob = 0.0
    for z in range(n_zones):
        nodes_in_zone = base_per_zone + (1 if z < remainder else 0)
        if rf <= nodes_in_zone:
            total_prob += comb(nodes_in_zone, rf) / denominator

    # Average across zones (each zone equally likely to fail)
    return total_prob / n_zones


def system_unavailability(
    n_nodes: int, n_zones: int, rf: int, n_partitions: int
) -> float:
    """Calculate P(system unavailable | 1 AZ fails).

    System is unavailable if ANY partition is unavailable. The key insight is
    that we must compute per-zone system availability FIRST, then average:

    P(system avail) = (1/n_zones) * sum over z of: (1 - p_z)^P

    where p_z = P(partition unavailable | zone z fails) = C(nodes_in_z, rf) / C(n, rf)

    This is NOT equivalent to (1 - avg_p)^P because exponentiation is non-linear!
    The correct formula accounts for uneven zone sizes by computing system
    availability under each zone's failure, then averaging.

    Args:
        n_nodes: Total nodes in cluster
        n_zones: Number of availability zones
        rf: Replication factor
        n_partitions: Number of partitions

    Returns:
        Probability in [0, 1]
    """
    if n_zones <= 0 or n_nodes <= 0 or rf <= 0:
        return 0.0

    if n_partitions <= 0:
        return 0.0  # No partitions = system is trivially available

    if rf > n_nodes:
        return 1.0  # Can't place replicas

    # Calculate nodes per zone (may be uneven)
    base_per_zone = n_nodes // n_zones
    remainder = n_nodes % n_zones

    denominator = comb(n_nodes, rf)
    if denominator == 0:
        return 1.0

    # For each zone z, compute P(system avail | zone z fails)
    # Then average across zones
    total_system_avail = 0.0
    for z in range(n_zones):
        nodes_in_zone = base_per_zone + (1 if z < remainder else 0)

        if rf <= nodes_in_zone:
            p_unavail_z = comb(nodes_in_zone, rf) / denominator
        else:
            p_unavail_z = 0.0  # Can't fit all replicas in this zone

        if p_unavail_z >= 1.0:
            sys_avail_z = 0.0
        elif p_unavail_z <= 0.0:
            sys_avail_z = 1.0
        else:
            sys_avail_z = (1.0 - p_unavail_z) ** n_partitions

        total_system_avail += sys_avail_z

    avg_system_avail = total_system_avail / n_zones
    return 1.0 - avg_system_avail


def system_availability(
    n_nodes: int, n_zones: int, rf: int, n_partitions: int
) -> float:
    """Calculate P(system available | 1 AZ fails).

    This computes the average system availability across all possible single-zone
    failures. For each zone z:
        P(system avail | zone z fails) = (1 - P(partition unavail | z fails))^P

    The overall availability is the average across zones:
        P(system avail) = (1/n_zones) * sum over z of: (1 - p_z)^P

    Args:
        n_nodes: Total nodes in cluster
        n_zones: Number of availability zones
        rf: Replication factor
        n_partitions: Number of partitions

    Returns:
        P(all partitions available | 1 AZ fails) in [0, 1]
    """
    return 1.0 - system_unavailability(n_nodes, n_zones, rf, n_partitions)


# =============================================================================
# TIER CONFIGURATION: Service level requirements
# =============================================================================


@dataclass(frozen=True)
class FaultToleranceConfig:
    """Configuration for fault tolerance requirements by service tier.

    Each tier specifies:
    - min_rf: Minimum acceptable replication factor
    - target_availability: Target system availability under AZ failure
    - cost_sensitivity: How much to penalize cost vs availability
    - max_cost_multiplier: Maximum acceptable cost increase over baseline
    """

    min_rf: int
    target_availability: float
    cost_sensitivity: float
    max_cost_multiplier: float


# Default tier configurations:
# Tier 0: Critical production services (99.9% availability target)
# Tier 1: Important production services (99% availability target)
# Tier 2: Standard services (95% availability target)
# Tier 3: Test/development services (80% availability target)
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
    """Get fault tolerance configuration for a service tier.

    Args:
        tier: Service tier (0-3, where 0 is most critical)

    Returns:
        FaultToleranceConfig for the specified tier.
        Defaults to tier 2 if tier is out of range.
    """
    return TIER_DEFAULTS.get(tier, TIER_DEFAULTS[2])


# =============================================================================
# UTILITY FUNCTION: Balancing availability and cost
# =============================================================================


def utility(
    availability: float,
    cost: float,
    *,
    tier: int,
    base_cost: float,
) -> float:
    """Calculate utility score balancing fault tolerance and cost.

    The utility function has two components:
    1. Availability value: Positive value for exceeding target, with
       diminishing returns (log1p) to avoid over-optimizing availability.
    2. Cost penalty: Negative value proportional to cost increase over baseline.

    U = availability_weight × log1p(availability_above_target × 100)
        - cost_sensitivity × (cost / base_cost - 1)

    The log1p provides diminishing returns: going from 99% to 99.9% is more
    valuable than going from 99.9% to 99.99% for the same cost increase.

    Args:
        availability: System availability (0-1)
        cost: Actual cost of the configuration
        tier: Service tier (0-3)
        base_cost: Baseline cost for comparison (typically min viable config)

    Returns:
        Utility score. Higher is better. Negative if below target availability.
    """
    config = get_tier_config(tier)

    # Availability component: positive for above target, negative for below
    avail_above_target = availability - config.target_availability

    if avail_above_target >= 0:
        # Diminishing returns for exceeding target
        # Scale by 100 so 1% above → log1p(1) ≈ 0.69
        availability_value = math.log1p(avail_above_target * 100)
    else:
        # Strong penalty for not meeting target
        # Use linear scaling to clearly signal inadequacy
        availability_value = avail_above_target * 100  # -1 per 1% below target

    # Cost component: penalize cost increase over baseline
    if base_cost > 0:
        cost_ratio = cost / base_cost
        cost_penalty = config.cost_sensitivity * (cost_ratio - 1.0)
    else:
        cost_penalty = 0.0

    return availability_value - cost_penalty


# =============================================================================
# FAULT-TOLERANT SEARCH: Find optimal (node_count, RF) configuration
# =============================================================================


@dataclass(frozen=True)
class FaultTolerantResult:
    """Result of fault-tolerant capacity planning.

    Extends basic CapacityResult with fault tolerance metrics and
    zone-aware comparison for cost optimization insights.
    """

    # Basic capacity result
    node_count: int
    rf: int
    partitions_per_node: int
    base_nodes: int

    # Fault tolerance metrics
    system_availability: float  # P(all partitions available | 1 AZ fails)
    per_partition_unavail: float  # P(partition unavailable | 1 AZ fails)

    # Cost analysis
    cost: float  # Actual cost with random placement
    zone_aware_cost: float  # What we'd pay if zone-aware
    zone_aware_savings: float  # Money left on table (cost - zone_aware_cost)

    # Configuration
    utility_score: float
    tier: int


def search_with_fault_tolerance(  # noqa: C901
    problem: CapacityProblem,
    tier: int,
    cost_per_node: float,
    n_zones: int = 3,
) -> Optional[FaultTolerantResult]:
    """Search for optimal configuration balancing fault tolerance and cost.

    This algorithm enumerates all valid (PPn, RF) configurations and scores
    them using the utility function. It returns the configuration with the
    highest utility that meets constraints.

    Algorithm:
    1. For each PPn from max down to 1:
       a. Calculate base_nodes = ceil(n_partitions / PPn)
       b. For each RF from tier.min_rf to max feasible:
          - Calculate node_count, cost, availability
          - Score with utility function
          - Track best configuration
    2. Return best configuration, or None if none meets constraints.

    The zone-aware comparison cost shows what we'd pay if the system had
    zone-aware placement (RF=2 gives 100% availability with zone-aware).

    Args:
        problem: Capacity planning problem specification
        tier: Service tier (0-3)
        cost_per_node: Cost per node for total cost calculation
        n_zones: Number of availability zones (default 3)

    Returns:
        FaultTolerantResult with optimal configuration, or None if no valid
        configuration exists within constraints.
    """
    config = get_tier_config(tier)

    max_ppn = int(problem.disk_per_node_gib / problem.partition_size_gib)
    if max_ppn < 1:
        return None

    # Calculate baseline cost (minimum viable config) for utility comparison
    baseline_result = search_algorithm(problem)
    if baseline_result is None:
        return None
    base_cost = baseline_result.node_count * cost_per_node

    # Maximum cost we're willing to pay
    max_cost = base_cost * config.max_cost_multiplier

    best_result: Optional[FaultTolerantResult] = None
    best_utility = float("-inf")

    # Enumerate all (PPn, RF) configurations
    for ppn in range(max_ppn, 0, -1):
        base_nodes = math.ceil(problem.n_partitions / ppn)

        # Minimum RF to meet CPU requirement
        if base_nodes >= 2:
            min_rf_for_cpu = max(
                config.min_rf,
                math.ceil(problem.cpu_needed / (base_nodes * problem.cpu_per_node)),
            )
        else:
            if 2 * problem.cpu_per_node >= problem.cpu_needed:
                min_rf_for_cpu = config.min_rf
            else:
                min_rf_for_cpu = max(
                    config.min_rf, math.ceil(problem.cpu_needed / problem.cpu_per_node)
                )

        # Maximum RF is limited by cost ceiling
        max_rf_for_cost = int(max_cost / (base_nodes * cost_per_node))
        max_rf = max(min_rf_for_cpu, min(10, max_rf_for_cost))  # Cap at 10

        for rf in range(min_rf_for_cpu, max_rf + 1):
            # Calculate node count
            if base_nodes >= 2:
                node_count = base_nodes * rf
            else:
                node_count = max(2, rf)

            # Check constraints
            if node_count > problem.max_nodes:
                continue
            if node_count > max_cost / cost_per_node:
                continue

            # Calculate fault tolerance metrics
            avail = system_availability(node_count, n_zones, rf, problem.n_partitions)
            per_part_unavail = per_partition_unavailability(node_count, n_zones, rf)

            # Calculate cost
            cost = node_count * cost_per_node

            # Calculate zone-aware comparison cost
            # With zone-aware placement, RF=2 gives 100% availability
            # So we use the minimum config that meets CPU with RF=2
            zone_aware_cost = _calculate_zone_aware_cost(
                problem, cost_per_node, config.min_rf
            )

            # Score with utility function
            u = utility(
                availability=avail,
                cost=cost,
                tier=tier,
                base_cost=base_cost,
            )

            if u > best_utility:
                best_utility = u
                best_result = FaultTolerantResult(
                    node_count=node_count,
                    rf=rf,
                    partitions_per_node=ppn,
                    base_nodes=base_nodes,
                    system_availability=avail,
                    per_partition_unavail=per_part_unavail,
                    cost=cost,
                    zone_aware_cost=zone_aware_cost,
                    zone_aware_savings=cost - zone_aware_cost,
                    utility_score=u,
                    tier=tier,
                )

    return best_result


def _calculate_zone_aware_cost(
    problem: CapacityProblem,
    cost_per_node: float,
    min_rf: int = 2,
) -> float:
    """Calculate what we'd pay if placement were zone-aware.

    With zone-aware placement, RF>=2 guarantees replicas span AZs,
    so per_partition_unavailability = 0 and system_availability = 100%.

    This means we only need the minimum RF that meets CPU requirements
    to achieve any availability target.
    """
    # Create a problem with min_rf (zone-aware needs at least 2)
    za_problem = CapacityProblem(
        n_partitions=problem.n_partitions,
        partition_size_gib=problem.partition_size_gib,
        disk_per_node_gib=problem.disk_per_node_gib,
        min_rf=max(2, min_rf),
        cpu_needed=problem.cpu_needed,
        cpu_per_node=problem.cpu_per_node,
        max_nodes=problem.max_nodes,
    )

    result = search_algorithm(za_problem)
    if result is None:
        return float("inf")

    return result.node_count * cost_per_node
