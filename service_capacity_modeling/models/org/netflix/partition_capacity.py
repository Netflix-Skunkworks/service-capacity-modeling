"""
Partition capacity planning for read-only KV stores with fault tolerance.

PROBLEM
-------
Given a read-only KV store with N partitions, find the optimal (node_count, RF)
that balances availability against cost. Higher RF = better fault tolerance
but more nodes/cost. The challenge: random partition placement means even RF=3
might have all replicas land in the same AZ (then fail together).

ALGORITHM OVERVIEW
------------------
1. Enumerate all valid (base_nodes, RF) configurations
2. For each config, calculate:
   - node_count = max(2, base_nodes × RF)
   - availability = P(all partitions survive 1 AZ failure)
   - cost = node_count × cost_per_node
3. Score each with: utility = availability_value - cost_penalty
4. Return the config with highest utility

KEY INSIGHT: The PPn Optimization
---------------------------------
Naively, we'd iterate over all PPn (partitions-per-node) values from 1 to
disk_size/partition_size (could be 20,000+). But:

    base_nodes = ceil(n_partitions / ppn)

Many PPn values give the SAME base_nodes. Since utility depends only on
(base_nodes, RF), not PPn itself, we skip redundant PPn values.

For n_partitions=12, instead of checking 20,000 PPn values, we check just 6:
    [(12,1), (6,2), (4,3), (3,4), (2,6), (1,12)]  # (ppn, base_nodes)

This reduces search from O(max_ppn × RF) to O(n_partitions × RF).

PROBABILITY MODEL
-----------------
For random partition placement with RF replicas across N nodes in Z zones:

    P(partition unavailable | zone z fails) = C(nodes_in_z, RF) / C(N, RF)
    P(system unavailable | zone z fails) = 1 - (1 - p_z)^n_partitions
    P(system available) = avg across zones of (1 - p_z)^n_partitions

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

# Hours per year (365.25 days × 24 hours) - used for annual availability calculations
HOURS_PER_YEAR = 8760.0

__all__ = [
    # Entry points
    "search_with_fault_tolerance",
    "find_capacity_config",
    "closed_form_algorithm",
    # Data structures
    "CapacityProblem",
    "CapacityResult",
    "FaultTolerantResult",
    "FailureModel",
    # Configuration
    "TIER_DEFAULTS",
    "FaultToleranceConfig",
    "get_tier_config",
    "HOURS_PER_YEAR",
    # Probability functions
    "per_partition_unavailability",
    "system_availability",
    "system_unavailability",
    "expected_annual_availability",
    "recovery_hours",
    "nines",
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
class FailureModel:
    """Tunable parameters for failure probabilities.

    These parameters convert "conditional availability given AZ failure" into
    "expected annual availability" - a much more meaningful metric since AZ
    failures are rare events (~1-2 per year across all of AWS).

    Formula:
        expected_avail = 1 - (az_rate × p_loss × recovery_hours / HOURS_PER_YEAR)

    Recovery time for read-only KV is dominated by S3 download:
        recovery_hours = data_per_node_gib / s3_download_gib_per_hour

    IMPORTANT: Recovery time is controlled by max_data_per_node_gib in the model
    arguments. This hard limit caps data per node, which directly limits recovery
    time. For example:
        - max_data_per_node_gib=500  → ~1 hour recovery
        - max_data_per_node_gib=2048 → ~4 hour recovery

    Attributes:
        az_failure_rate_per_year: Probability of an AZ failure per year (default 2%)
        s3_download_gib_per_hour: S3 download throughput in GiB/hour
            - Conservative: ~200 GiB/hour (single stream, ~55 MB/s)
            - Typical: ~500 GiB/hour (multi-stream, ~140 MB/s)
            - Optimized: ~1000+ GiB/hour (parallel downloads)
        min_recovery_hours: Minimum recovery time (instance spinup, warming, etc.)
    """

    az_failure_rate_per_year: float = 0.02  # ~2% chance of AZ issue per year
    s3_download_gib_per_hour: float = 500.0  # ~140 MB/s, typical multi-stream
    min_recovery_hours: float = 0.1  # 6 min minimum for instance spinup


def recovery_hours(data_per_node_gib: float, failure_model: FailureModel) -> float:
    """Calculate recovery time based on data size and S3 download speed.

    This is the key link between max_data_per_node_gib (a hard limit users set)
    and expected availability. Lower max_data_per_node → faster recovery →
    higher expected availability.

    Args:
        data_per_node_gib: Amount of data each node needs to download.
            Capped by max_data_per_node_gib in model arguments.
        failure_model: Contains S3 download speed parameters

    Returns:
        Estimated hours to recover (download data + spinup overhead)
    """
    download_hours = data_per_node_gib / failure_model.s3_download_gib_per_hour
    return max(failure_model.min_recovery_hours, download_hours)


# Default failure model with typical parameters
DEFAULT_FAILURE_MODEL = FailureModel()


@dataclass(frozen=True)
class FaultTolerantResult:
    """Output: What to provision.

    Attributes:
        node_count: Total nodes to deploy
        rf: Replication factor to use
        expected_availability: Expected annual availability
        az_failure_availability: P(survive | AZ fails)
        cost: Total cost (node_count * cost_per_node)
        zone_aware_savings: Savings with zone-aware placement
    """

    node_count: int
    rf: int
    partitions_per_node: int
    base_nodes: int
    expected_availability: float  # Expected annual availability
    az_failure_availability: float  # P(survive | AZ fails)
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
    """Configuration for a service tier.

    The utility function scores configurations as:
        utility = (excess_nines × value_per_nine) - cost

    Where excess_nines = nines(expected_availability) - nines(target_availability)

    This means:
        - value_per_nine = $10000: Willing to pay $10k/year for one extra nine
        - value_per_nine = $1000: More cost-conscious, only $1k/year per nine
    """

    min_rf: int  # Minimum replication factor
    target_availability: float  # Target expected annual availability (e.g., 0.9999)
    value_per_nine: float  # Dollar value per additional nine of availability
    max_cost_multiplier: float  # Max cost as multiple of baseline


# Tier 0: Critical (99.99%), Tier 1: Important (99.9%), Tier 2: Standard (99%)
# Tier 3: Dev (90%)
#
# Note: These are EXPECTED ANNUAL availability targets, not conditional.
# With 2% AZ failure rate, even 90% conditional → 99.9%+ expected.
TIER_DEFAULTS: dict[int, FaultToleranceConfig] = {
    0: FaultToleranceConfig(
        min_rf=3,
        target_availability=0.9999,  # 4 nines expected annual
        value_per_nine=10000.0,  # Pay $10k/year for extra nine
        max_cost_multiplier=3.0,
    ),
    1: FaultToleranceConfig(
        min_rf=3,
        target_availability=0.999,  # 3 nines expected annual
        value_per_nine=5000.0,  # Pay $5k/year for extra nine
        max_cost_multiplier=2.5,
    ),
    2: FaultToleranceConfig(
        min_rf=2,
        target_availability=0.99,  # 2 nines expected annual
        value_per_nine=1000.0,  # Pay $1k/year for extra nine
        max_cost_multiplier=2.0,
    ),
    3: FaultToleranceConfig(
        min_rf=2,
        target_availability=0.9,  # 1 nine expected annual
        value_per_nine=100.0,  # Only $100/year for extra nine
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
    # Pre-compute zone-aware cost for comparison (only needed for random placement)
    if zone_aware:
        zone_aware_cost = 0.0  # Will be set to actual cost in result
    else:
        zone_aware_cost = _calculate_zone_aware_cost(
            problem, tier, cost_per_node, n_zones
        )

    return _search_with_fault_tolerance_impl(
        problem, tier, cost_per_node, n_zones, zone_aware, zone_aware_cost
    )


def _interesting_ppn_values(n_partitions: int, max_ppn: int) -> list[tuple[int, int]]:
    """Compute PPn values where base_nodes changes.

    WHY THIS OPTIMIZATION IS VALID
    ------------------------------
    The utility function depends on (node_count, availability, cost) where:
        node_count = max(2, base_nodes × rf)
        availability = f(node_count, rf, n_partitions)
        cost = node_count × cost_per_node

    Notice: utility depends on base_nodes, not PPn directly. So if two PPn
    values give the same base_nodes, they produce identical utility scores.

    We only need to check one PPn per unique base_nodes. We pick the MAXIMUM
    PPn for each base_nodes since that's the most space-efficient disk usage.

    HOW IT WORKS
    ------------
    Given: base_nodes = ceil(n_partitions / ppn)

    For each target base_nodes from 1 to n_partitions, find the range of PPn
    values that produce it:
        ceil(n/ppn) = b  ⟹  b-1 < n/ppn ≤ b  ⟹  n/b ≤ ppn < n/(b-1)

    We take the maximum PPn in that range (capped by max_ppn).

    EXAMPLE
    -------
    For n_partitions=12, max_ppn=100:
        [(12, 1), (6, 2), (4, 3), (3, 4), (2, 6), (1, 12)]

    Only 6 values to check instead of 100. For n_partitions=12, max_ppn=20000,
    we still only check 6 values - a 3333x reduction in search space.
    """
    results = []
    seen_base_nodes = set()

    # For each possible base_nodes, find the max ppn that gives it
    # ceil(n/ppn) = b  means  b-1 < n/ppn <= b  means  n/b <= ppn < n/(b-1)
    # So max ppn for base_nodes=b is: min(max_ppn, floor(n/(b-1)) - 1) if b > 1
    #                                 or max_ppn if b == 1
    for base_nodes in range(1, n_partitions + 1):
        if base_nodes == 1:
            # ppn >= n_partitions gives base_nodes=1
            # Use min to cap at n_partitions (exact fit is most efficient)
            ppn = min(max_ppn, n_partitions)
        else:
            # Max ppn where ceil(n/ppn) = base_nodes
            # ppn must satisfy: n/base_nodes <= ppn < n/(base_nodes-1)
            min_ppn_for_base = math.ceil(n_partitions / base_nodes)
            max_ppn_for_base = math.ceil(n_partitions / (base_nodes - 1)) - 1

            if max_ppn_for_base < min_ppn_for_base:
                continue  # No valid ppn for this base_nodes
            if min_ppn_for_base > max_ppn:
                continue  # All valid ppn values exceed max_ppn

            # Use the max valid ppn (capped by max_ppn)
            ppn = min(max_ppn_for_base, max_ppn)

        # Verify (sanity check)
        actual_base = math.ceil(n_partitions / ppn)
        if actual_base != base_nodes:
            continue

        if base_nodes in seen_base_nodes:
            continue
        seen_base_nodes.add(base_nodes)
        results.append((ppn, base_nodes))

    # Sort by descending ppn (highest ppn = lowest base_nodes first)
    results.sort(key=lambda x: -x[0])
    return results


# pylint: disable=too-many-positional-arguments,too-many-locals
def _search_with_fault_tolerance_impl(
    problem: CapacityProblem,
    tier: int,
    cost_per_node: float,
    n_zones: int,
    zone_aware: bool,
    zone_aware_cost: float,
    failure_model: FailureModel = DEFAULT_FAILURE_MODEL,
) -> Optional[FaultTolerantResult]:
    """Core implementation of fault-tolerance search.

    Pseudocode:
        baseline = find_minimum_viable_config(problem)
        max_cost = baseline.cost × tier.max_cost_multiplier

        for each (ppn, base_nodes) in interesting_ppn_values:
            for rf in range(min_rf_for_cpu, max_rf_for_cost):
                node_count = max(2, base_nodes × rf)
                if node_count exceeds constraints: skip

                expected_avail = expected_annual_availability(...)
                cost = node_count × cost_per_node
                score = utility(expected_avail, cost, tier)

                if score > best_score (or equal score with lower cost):
                    best = this config

        return best
    """
    config = get_tier_config(tier)

    max_ppn = int(problem.disk_per_node_gib / problem.partition_size_gib)
    if max_ppn < 1:
        return None

    # Baseline cost for max_cost calculation
    baseline_result = find_capacity_config(problem)
    if baseline_result is None:
        return None
    base_cost = baseline_result.node_count * cost_per_node
    max_cost = base_cost * config.max_cost_multiplier

    best_result: Optional[FaultTolerantResult] = None
    best_utility = float("-inf")
    best_cost = float("inf")  # For tie-breaking

    # Effective min_rf: respect both tier requirement and problem constraint
    effective_min_rf = max(config.min_rf, problem.min_rf)

    # Only enumerate PPn values where base_nodes changes (huge optimization!)
    # Instead of O(max_ppn) iterations, we get O(n_partitions) iterations
    for ppn, base_nodes in _interesting_ppn_values(problem.n_partitions, max_ppn):
        # Minimum RF to meet CPU (also respects effective_min_rf)
        min_rf_for_cpu = _min_rf_for_cpu(
            base_nodes, effective_min_rf, problem.cpu_needed, problem.cpu_per_node
        )

        # Maximum RF limited by cost
        max_rf_for_cost = int(max_cost / (base_nodes * cost_per_node))
        max_rf = max(min_rf_for_cpu, min(10, max_rf_for_cost))

        for rf in range(min_rf_for_cpu, max_rf + 1):
            node_count = max(2, base_nodes * rf)

            if node_count > problem.max_nodes or node_count > max_cost / cost_per_node:
                continue

            cost = node_count * cost_per_node

            # Data per node = partitions_per_node × partition_size
            # Used for recovery time calculation (S3 download time)
            data_per_node_gib = ppn * problem.partition_size_gib

            # Calculate conditional availability (P(survive | AZ fails))
            conditional_avail = _system_availability_for_placement(
                node_count, n_zones, rf, problem.n_partitions, zone_aware
            )

            # Convert to expected annual availability
            if zone_aware and rf >= 2:
                # Zone-aware with RF>=2: perfect availability
                expected_avail = 1.0
            else:
                expected_avail = _expected_avail_for_placement(
                    node_count,
                    n_zones,
                    rf,
                    problem.n_partitions,
                    zone_aware,
                    data_per_node_gib,
                    failure_model,
                )

            u = utility(expected_avail, cost, tier)

            # Tie-breaking: prefer lower cost when utility is equal
            if u > best_utility or (u == best_utility and cost < best_cost):
                best_utility = u
                best_cost = cost
                # For zone-aware search, zone_aware_cost equals the result cost
                actual_zone_aware_cost = cost if zone_aware else zone_aware_cost
                best_result = FaultTolerantResult(
                    node_count=node_count,
                    rf=rf,
                    partitions_per_node=ppn,
                    base_nodes=base_nodes,
                    expected_availability=expected_avail,
                    az_failure_availability=conditional_avail,
                    per_partition_unavail=_per_partition_unavail_for_placement(
                        node_count, n_zones, rf, zone_aware
                    ),
                    cost=cost,
                    zone_aware_cost=actual_zone_aware_cost,
                    zone_aware_savings=cost - actual_zone_aware_cost,
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
    total = max(2, base * rf)

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
    """Calculate minimum RF to satisfy CPU requirement.

    Formula: RF = ceil(cpu_needed / (base_nodes * cpu_per_node))
    With minimum node count of 2, this ensures enough CPU capacity.
    """
    return max(min_rf, math.ceil(cpu_needed / (base_nodes * cpu_per_node)))


def _calculate_zone_aware_cost(
    problem: CapacityProblem,
    tier: int,
    cost_per_node: float,
    n_zones: int,
) -> float:
    """Calculate optimal cost using zone-aware placement.

    Reuses search_with_fault_tolerance with zone_aware=True to get the cost
    that would result from zone-aware infrastructure (where RF>=2 guarantees
    100% availability).
    """
    result = _search_with_fault_tolerance_impl(
        problem, tier, cost_per_node, n_zones, zone_aware=True, zone_aware_cost=0.0
    )
    return result.cost if result else float("inf")


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


def utility(
    expected_avail: float, cost: float, tier: int, base_cost: float = 0.0
) -> float:
    """Score a configuration using nines-based value.

    Formula:
        utility = (excess_nines × value_per_nine) - cost

    Where:
        excess_nines = nines(expected_availability) - nines(target_availability)

    DESIGN RATIONALE
    ----------------
    - Uses expected annual availability (not conditional)
    - Value is linear in nines (each nine worth same dollar amount)
    - This makes trade-offs explicit: "Is one more nine worth $X?"

    Args:
        expected_avail: Expected annual availability (not conditional)
        cost: Annual cost of this configuration
        tier: Service tier (0=critical, 3=dev)
        base_cost: Unused (kept for API compatibility during transition)

    Returns:
        Utility score (higher is better)
    """
    _ = base_cost  # Unused - kept for API compatibility
    config = get_tier_config(tier)

    # Convert to nines and calculate excess
    avail_nines = nines(expected_avail)
    target_nines = nines(config.target_availability)
    excess_nines = avail_nines - target_nines

    # Value from excess availability (can be negative if below target)
    availability_value = excess_nines * config.value_per_nine

    return availability_value - cost


# Convenience function: system unavailability = 1 - system availability
def system_unavailability(
    n_nodes: int, n_zones: int, rf: int, n_partitions: int
) -> float:
    """P(at least one partition unavailable | 1 AZ fails)."""
    return 1.0 - system_availability(n_nodes, n_zones, rf, n_partitions)


def nines(availability: float) -> float:
    """Convert availability to "nines" (e.g., 0.999 → 3.0 nines).

    Formula: nines = -log10(1 - availability)

    Examples:
        0.9    → 1.0 nine
        0.99   → 2.0 nines
        0.999  → 3.0 nines
        0.9999 → 4.0 nines
    """
    if availability >= 1.0:
        return float("inf")
    if availability <= 0.0:
        return 0.0
    return -math.log10(1.0 - availability)


def expected_annual_availability(  # pylint: disable=too-many-positional-arguments
    n_nodes: int,
    n_zones: int,
    rf: int,
    n_partitions: int,
    data_per_node_gib: float = 0.0,
    failure_model: FailureModel = DEFAULT_FAILURE_MODEL,
) -> float:
    """Compute expected annual availability incorporating failure probabilities.

    This converts "conditional availability given AZ failure" into actual expected
    availability by accounting for the rarity of AZ failures.

    Formula:
        P(loss | year) = P(AZ_fails) × P(loss | AZ_fails) × recovery / HOURS_PER_YEAR
        expected_availability = 1 - P(system loss | year)

    Recovery time is based on S3 download speed:
        recovery_hours = data_per_node_gib / s3_download_gib_per_hour

    Why this matters:
        A service showing 90% conditional availability (bad!) is actually 99.9%+
        expected availability (good!) because AZ failures are rare (~2%/year).

    Args:
        n_nodes: Total nodes in cluster
        n_zones: Number of availability zones
        rf: Replication factor
        n_partitions: Number of data partitions
        data_per_node_gib: Data per node (for recovery time calculation)
        failure_model: Tunable failure parameters

    Returns:
        Expected annual availability (0.0 to 1.0)
    """
    # Get conditional availability (P(survive | AZ fails))
    conditional = system_availability(n_nodes, n_zones, rf, n_partitions)

    # P(any partition lost | AZ fails)
    p_loss_given_az_fail = 1.0 - conditional

    # Calculate recovery time based on data size
    recover_time = recovery_hours(data_per_node_gib, failure_model)

    # Expected downtime from AZ failures (as fraction of year)
    hours_per_year = HOURS_PER_YEAR
    expected_downtime_fraction = (
        failure_model.az_failure_rate_per_year
        * p_loss_given_az_fail
        * recover_time
        / hours_per_year
    )

    return 1.0 - expected_downtime_fraction


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


def _expected_avail_for_placement(  # pylint: disable=too-many-positional-arguments
    n_nodes: int,
    n_zones: int,
    rf: int,
    n_partitions: int,
    zone_aware: bool,
    data_per_node_gib: float,
    failure_model: FailureModel,
) -> float:
    """Calculate expected annual availability based on placement strategy.

    This wraps expected_annual_availability() with zone-aware semantics.
    """
    if zone_aware and rf >= 2:
        # Zone-aware with RF>=2: perfect conditional availability → perfect expected
        return 1.0
    elif zone_aware:
        # Zone-aware with RF=1: still have AZ failure risk
        conditional = (1.0 - 1.0 / n_zones) ** n_partitions if n_zones > 0 else 0.0
        p_loss = 1.0 - conditional
        hours_per_year = HOURS_PER_YEAR
        recover_time = recovery_hours(data_per_node_gib, failure_model)
        downtime_fraction = (
            failure_model.az_failure_rate_per_year
            * p_loss
            * recover_time
            / hours_per_year
        )
        return 1.0 - downtime_fraction
    else:
        # Random placement: use the standard expected availability function
        return expected_annual_availability(
            n_nodes, n_zones, rf, n_partitions, data_per_node_gib, failure_model
        )
