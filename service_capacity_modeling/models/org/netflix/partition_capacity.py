"""
Partition-aware capacity planning with fault tolerance optimization.

This module finds optimal (node_count, replication_factor) configurations
for partition-based KV stores, balancing availability against cost.

QUICK START
-----------
    from partition_capacity import CapacityProblem, search_with_fault_tolerance

    problem = CapacityProblem(
        n_partitions=100,
        partition_size_gib=100,
        disk_per_node_gib=2048,
        min_rf=2,
        cpu_needed=800,
        cpu_per_node=16,
    )

    result = search_with_fault_tolerance(problem, tier=2, cost_per_node=100)
    print(f"Use {result.node_count} nodes with RF={result.rf}")

PUBLIC API
----------
    Entry Points:
        search_with_fault_tolerance()  - Get optimal config for a tier
        compute_pareto_frontier()      - See all efficient tradeoffs

    Data Structures:
        CapacityProblem      - Input: what you need
        FaultTolerantResult  - Output: what to provision
        ParetoPoint          - One point on the efficiency frontier

    Configuration:
        TIER_DEFAULTS        - Tier 0-3 configurations
        get_tier_config()    - Lookup helper

THEORETICAL FOUNDATIONS
-----------------------
    - Utility Theory (AIMA Ch 16): Multi-attribute utility functions
    - Constraint Satisfaction (AIMA Ch 6): CPU, disk, RF constraints
    - Probability (AIMA Ch 12): Availability under AZ failure
    - Pareto Optimality (Operations Research): Efficient tradeoffs
"""

import math
from dataclasses import dataclass
from math import comb
from typing import Optional

__all__ = [
    # Entry points
    "search_with_fault_tolerance",
    "compute_pareto_frontier",
    # Data structures
    "CapacityProblem",
    "FaultTolerantResult",
    "ParetoPoint",
    # Configuration
    "TIER_DEFAULTS",
    "FaultToleranceConfig",
    "get_tier_config",
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


@dataclass(frozen=True)
class ParetoPoint:
    """One point on the Pareto frontier (for exploring tradeoffs)."""

    availability: float
    cost: float
    rf: int
    node_count: int
    ppn: int


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
    placement: "PlacementModel | None" = None,
) -> Optional[FaultTolerantResult]:
    """Find optimal configuration balancing availability and cost.

    This is the PRIMARY entry point. It enumerates configurations,
    scores each with the utility function, and returns the best.

    Args:
        problem: Capacity requirements
        tier: Service tier (0=critical, 3=dev)
        cost_per_node: Cost per node for total cost calculation
        n_zones: Number of availability zones (default 3)
        placement: Placement strategy (default: UniformRandomPlacement).
                   Use ZoneAwarePlacement() for zone-aware optimization.

    Returns:
        FaultTolerantResult with optimal config, or None if impossible.

    Example:
        result = search_with_fault_tolerance(problem, tier=2, cost_per_node=100)
        print(f"Deploy {result.node_count} nodes with RF={result.rf}")
        print(f"Availability: {result.system_availability:.2%}")

        # With zone-aware placement (if your infrastructure supports it):
        from partition_capacity import ZoneAwarePlacement
        result = search_with_fault_tolerance(problem, tier=2, cost_per_node=100,
                                              placement=ZoneAwarePlacement())
    """
    # Use default placement if none specified (avoids circular import at module load)
    if placement is None:
        placement = UniformRandomPlacement()

    config = get_tier_config(tier)

    max_ppn = int(problem.disk_per_node_gib / problem.partition_size_gib)
    if max_ppn < 1:
        return None

    # Baseline cost for utility comparison
    baseline_result = search_algorithm(problem)
    if baseline_result is None:
        return None
    base_cost = baseline_result.node_count * cost_per_node
    max_cost = base_cost * config.max_cost_multiplier

    # Pre-compute zone-aware cost for comparison (uses ZoneAwarePlacement)
    zone_aware_cost = _calculate_zone_aware_cost(problem, tier, cost_per_node, n_zones)

    best_result: Optional[FaultTolerantResult] = None
    best_utility = float("-inf")

    # Enumerate all (PPn, RF) configurations
    for ppn in range(max_ppn, 0, -1):
        base_nodes = math.ceil(problem.n_partitions / ppn)

        # Minimum RF to meet CPU
        min_rf_for_cpu = _min_rf_for_cpu(
            base_nodes, config.min_rf, problem.cpu_needed, problem.cpu_per_node
        )

        # Maximum RF limited by cost
        max_rf_for_cost = int(max_cost / (base_nodes * cost_per_node))
        max_rf = max(min_rf_for_cpu, min(10, max_rf_for_cost))

        for rf in range(min_rf_for_cpu, max_rf + 1):
            node_count = max(2, base_nodes * rf) if base_nodes >= 2 else max(2, rf)

            if node_count > problem.max_nodes or node_count > max_cost / cost_per_node:
                continue

            # Use placement model for availability calculation
            avail = placement.system_availability(
                node_count, n_zones, rf, problem.n_partitions
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
                    per_partition_unavail=placement.per_partition_unavailability(
                        node_count, n_zones, rf
                    ),
                    cost=cost,
                    zone_aware_cost=zone_aware_cost,
                    zone_aware_savings=cost - zone_aware_cost,
                    utility_score=u,
                    tier=tier,
                )

    return best_result


def compute_pareto_frontier(
    problem: CapacityProblem,
    tier: int,
    cost_per_node: float,
    n_zones: int = 3,
    placement: "PlacementModel | None" = None,
) -> list[ParetoPoint]:
    """Find all Pareto-optimal (availability, cost) configurations.

    Use this to explore the tradeoff space or validate that
    search_with_fault_tolerance returns efficient solutions.

    Args:
        problem: Capacity requirements
        tier: Service tier (0=critical, 3=dev)
        cost_per_node: Cost per node for total cost calculation
        n_zones: Number of availability zones (default 3)
        placement: Placement strategy (default: UniformRandomPlacement)

    Returns:
        List of ParetoPoint sorted by cost (cheapest first).
    """
    # Use default placement if none specified
    if placement is None:
        placement = UniformRandomPlacement()

    config = get_tier_config(tier)
    tier_problem = CapacityProblem(
        n_partitions=problem.n_partitions,
        partition_size_gib=problem.partition_size_gib,
        disk_per_node_gib=problem.disk_per_node_gib,
        min_rf=config.min_rf,
        cpu_needed=problem.cpu_needed,
        cpu_per_node=problem.cpu_per_node,
        max_nodes=problem.max_nodes,
    )

    max_ppn = int(problem.disk_per_node_gib / problem.partition_size_gib)
    if max_ppn < 1:
        return []

    # Collect configs
    all_configs: list[ParetoPoint] = []
    for ppn in range(max_ppn, 0, -1):
        result = closed_form_algorithm(tier_problem, ppn=ppn)
        if result is None:
            continue

        # Use placement model for availability calculation
        avail = placement.system_availability(
            result.node_count, n_zones, result.rf, problem.n_partitions
        )
        all_configs.append(
            ParetoPoint(
                availability=avail,
                cost=result.node_count * cost_per_node,
                rf=result.rf,
                node_count=result.node_count,
                ppn=ppn,
            )
        )

    # Filter to Pareto frontier
    frontier = [
        c
        for c in all_configs
        if not any(
            o.availability >= c.availability
            and o.cost <= c.cost
            and (o.availability > c.availability or o.cost < c.cost)
            for o in all_configs
            if o is not c
        )
    ]

    # Deduplicate and sort
    seen: set[tuple[float, float]] = set()
    unique: list[ParetoPoint] = []
    for p in sorted(frontier, key=lambda x: x.cost):
        key = (round(p.availability, 6), round(p.cost, 2))
        if key not in seen:
            seen.add(key)
            unique.append(p)

    return unique


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


def original_algorithm(p: CapacityProblem) -> Optional[CapacityResult]:
    """Original while-loop from read_only_kv.py. Greedy max PPn, increment RF.

    This algorithm only tries the maximum possible partitions-per-node (PPn).
    If that doesn't fit within max_nodes, it fails — even if a lower PPn would work.

    Use closed_form_algorithm for O(1) equivalent, or search_algorithm to try
    all PPn values.
    """
    ppn = int(p.disk_per_node_gib / p.partition_size_gib)
    if ppn < 1:
        return None
    base = math.ceil(p.n_partitions / ppn)

    rf = p.min_rf
    while True:
        total = max(2, base * rf) if base >= 2 else max(2, rf)
        if total > p.max_nodes:
            return None
        if total * p.cpu_per_node >= p.cpu_needed:
            break
        rf += 1

    return CapacityResult(total, rf, ppn, base)


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


def search_algorithm(p: CapacityProblem) -> Optional[CapacityResult]:
    """Search PPn from max to 1, return first valid config.

    More capable than original/closed_form — finds solutions they miss when
    max_nodes is tight.
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

    Runs the same utility-based optimization but with ZoneAwarePlacement,
    where RF>=2 guarantees 100% availability. This gives a fair apples-to-apples
    comparison of what you'd pay with zone-aware infrastructure.
    """
    config = get_tier_config(tier)
    zone_aware = ZoneAwarePlacement()

    max_ppn = int(problem.disk_per_node_gib / problem.partition_size_gib)
    if max_ppn < 1:
        return float("inf")

    # Baseline cost for utility comparison (same as main search)
    baseline_result = search_algorithm(problem)
    if baseline_result is None:
        return float("inf")
    base_cost = baseline_result.node_count * cost_per_node
    max_cost = base_cost * config.max_cost_multiplier

    best_cost = float("inf")
    best_utility = float("-inf")

    # Same search loop, but using ZoneAwarePlacement for availability
    for ppn in range(max_ppn, 0, -1):
        base_nodes = math.ceil(problem.n_partitions / ppn)
        min_rf_for_cpu = _min_rf_for_cpu(
            base_nodes, config.min_rf, problem.cpu_needed, problem.cpu_per_node
        )
        max_rf_for_cost = int(max_cost / (base_nodes * cost_per_node))
        max_rf = max(min_rf_for_cpu, min(10, max_rf_for_cost))

        for rf in range(min_rf_for_cpu, max_rf + 1):
            node_count = max(2, base_nodes * rf) if base_nodes >= 2 else max(2, rf)

            if node_count > problem.max_nodes or node_count > max_cost / cost_per_node:
                continue

            # Use ZoneAwarePlacement for availability
            avail = zone_aware.system_availability(
                node_count, n_zones, rf, problem.n_partitions
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
# EXPERIMENTAL: Placement Models (future zone-aware optimization)
# =============================================================================


class PlacementModel:
    """Base class for placement strategies."""

    def per_partition_unavailability(
        self, n_nodes: int, n_zones: int, rf: int
    ) -> float:
        return per_partition_unavailability(n_nodes, n_zones, rf)

    def system_availability(
        self, n_nodes: int, n_zones: int, rf: int, n_partitions: int
    ) -> float:
        return system_availability(n_nodes, n_zones, rf, n_partitions)

    def name(self) -> str:
        return self.__class__.__name__


class UniformRandomPlacement(PlacementModel):
    """Current default: random placement without zone awareness."""


class ZoneAwarePlacement(PlacementModel):
    """Future: RF>=2 guarantees cross-AZ spread."""

    def per_partition_unavailability(
        self, n_nodes: int, n_zones: int, rf: int
    ) -> float:
        return 0.0 if rf >= 2 else (1.0 / n_zones if n_zones > 0 else 1.0)

    def system_availability(
        self, n_nodes: int, n_zones: int, rf: int, n_partitions: int
    ) -> float:
        if rf >= 2:
            return 1.0
        if n_zones <= 0 or n_partitions <= 0:
            return 1.0
        return (1.0 - 1.0 / n_zones) ** n_partitions


DEFAULT_PLACEMENT = UniformRandomPlacement()
