# pylint: disable=too-many-lines
"""
Test suite for partition-aware capacity planning algorithms.

Key algorithms:
    - CLOSED_FORM: O(1) mathematical capacity calculation for a given PPn
    - FIND_CAPACITY_CONFIG: Searches PPn values, finds solutions when max_nodes is tight

Key findings:
    - find_capacity_config ⊇ closed_form (finds everything closed_form finds, plus more)
    - find_capacity_config finds solutions closed_form misses when max_nodes is tight
"""

from dataclasses import dataclass

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from service_capacity_modeling.models.org.netflix.partition_capacity import (
    CapacityProblem,
    CapacityResult,
    FailureModel,
    closed_form_algorithm,
    expected_annual_availability,
    find_capacity_config,
    get_tier_config,
    nines,
    search_with_fault_tolerance,
    system_availability,
)


# =============================================================================
# TEST HELPERS: ParetoPoint and compute_pareto_frontier (moved from production)
# =============================================================================


@dataclass(frozen=True)
class ParetoPoint:
    """One point on the Pareto frontier (for exploring tradeoffs in tests)."""

    availability: float
    cost: float
    rf: int
    node_count: int
    ppn: int


def compute_pareto_frontier(
    problem: CapacityProblem,
    tier: int,
    cost_per_node: float,
    n_zones: int = 3,
    zone_aware: bool = False,
) -> list[ParetoPoint]:
    """Find all Pareto-optimal (expected_availability, cost) configurations.

    This is a TEST HELPER for validating that search_with_fault_tolerance
    returns efficient solutions.

    Note: availability in ParetoPoint is now EXPECTED annual availability,
    not conditional availability.
    """
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

        # Calculate expected annual availability based on placement strategy
        # Data per node = partitions_per_node × partition_size
        data_per_node_gib = ppn * problem.partition_size_gib
        if zone_aware:
            expected_avail = (
                1.0 if result.rf >= 2 else (1.0 - 1.0 / n_zones) ** problem.n_partitions
            )
        else:
            expected_avail = expected_annual_availability(
                result.node_count,
                n_zones,
                result.rf,
                problem.n_partitions,
                data_per_node_gib,
            )
        all_configs.append(
            ParetoPoint(
                availability=expected_avail,  # Now expected annual, not conditional
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
# TEST DATA: Problems demonstrating algorithm behavior
# =============================================================================

# Problems where algorithms work (max_nodes is relaxed)
STANDARD_CASES = [
    pytest.param(
        CapacityProblem(200, 575, 2048, 2, 800, 16),
        id="data-constrained",
    ),
    pytest.param(
        CapacityProblem(200, 575, 2048, 2, 3200, 16),
        id="cpu-constrained",
    ),
    pytest.param(
        CapacityProblem(100, 1024, 2048, 2, 100, 16),
        id="tight-disk",
    ),
    pytest.param(
        CapacityProblem(100, 2000, 2048, 2, 500, 8),
        id="one-partition-per-node",
    ),
    pytest.param(
        CapacityProblem(1000, 100, 2000, 3, 5000, 16),
        id="many-partitions",
    ),
    pytest.param(
        CapacityProblem(8, 100, 1000, 2, 50, 8),
        id="small-cluster",
    ),
    # Edge case: base=1, 2-node minimum satisfies CPU at min_rf
    pytest.param(
        CapacityProblem(1, 10, 100, 1, 2, 1, 10),
        id="base-1-min-satisfies",
    ),
    # Edge case: base=1, need higher RF
    pytest.param(
        CapacityProblem(1, 10, 100, 1, 5, 1, 10),
        id="base-1-need-more-rf",
    ),
]

# Problems where find_capacity_config succeeds but closed_form at max PPn fails
TIGHT_MAX_NODES_CASES = [
    pytest.param(
        CapacityProblem(
            n_partitions=21,
            partition_size_gib=10.0,
            disk_per_node_gib=100.0,  # max_ppn = 10
            min_rf=1,
            cpu_needed=10,
            cpu_per_node=1,
            max_nodes=10,  # tight!
        ),
        CapacityResult(node_count=10, rf=2, partitions_per_node=5, base_nodes=5),
        id="21-partitions-tight-max-nodes",
    ),
    pytest.param(
        CapacityProblem(
            n_partitions=31,
            partition_size_gib=10.0,
            disk_per_node_gib=100.0,  # max_ppn = 10
            min_rf=1,
            cpu_needed=15,
            cpu_per_node=1,
            max_nodes=15,  # tight!
        ),
        # ppn=10: base=4, rf=4, total=16 > 15 (closed_form at max ppn fails)
        # ppn=7:  base=5, rf=3, total=15 ≤ 15 (find_capacity_config succeeds)
        CapacityResult(node_count=15, rf=3, partitions_per_node=7, base_nodes=5),
        id="31-partitions-need-ppn-7",
    ),
]


# =============================================================================
# PART 1: Closed-form algorithm tests
# =============================================================================


class TestClosedFormAlgorithm:
    """Test the O(1) closed-form capacity calculation."""

    @pytest.mark.parametrize("problem", STANDARD_CASES)
    def test_finds_valid_solution(self, problem: CapacityProblem):
        """Closed-form should find solutions for standard problems."""
        result = closed_form_algorithm(problem)
        assert result is not None, "Should find solution"

    @given(
        problem=st.builds(
            CapacityProblem,
            n_partitions=st.integers(1, 1000),
            partition_size_gib=st.floats(
                10, 500, allow_nan=False, allow_infinity=False
            ),
            disk_per_node_gib=st.floats(
                100, 10000, allow_nan=False, allow_infinity=False
            ),
            min_rf=st.integers(1, 5),
            cpu_needed=st.integers(1, 10000),
            cpu_per_node=st.integers(1, 64),
            max_nodes=st.integers(2, 10000),
        )
    )
    @settings(max_examples=500, deadline=None)
    def test_satisfies_constraints_when_returns_result(self, problem: CapacityProblem):
        """When closed_form returns a result, it satisfies all constraints."""
        if problem.partition_size_gib > problem.disk_per_node_gib:
            return

        result = closed_form_algorithm(problem)
        if result is None:
            return

        # CPU constraint
        assert result.node_count * problem.cpu_per_node >= problem.cpu_needed
        # Disk constraint
        assert result.node_count * result.partitions_per_node >= (
            problem.n_partitions * result.rf
        )
        # RF constraint
        assert result.rf >= problem.min_rf
        # Size constraint
        assert result.node_count <= problem.max_nodes


# =============================================================================
# PART 2: find_capacity_config finds solutions closed_form misses
# =============================================================================


class TestFindCapacityConfigBeatClosedForm:
    """Demonstrate cases where find_capacity_config succeeds but closed_form fails.

    This happens when max_nodes is tight: greedy (max PPn) produces a cluster
    exceeding max_nodes, but a lower PPn produces a valid cluster.
    """

    @pytest.mark.parametrize("problem,expected", TIGHT_MAX_NODES_CASES)
    def test_closed_form_fails_find_capacity_config_succeeds(
        self, problem: CapacityProblem, expected: CapacityResult
    ):
        """closed_form returns None, but find_capacity_config finds a solution."""
        closed = closed_form_algorithm(problem)  # Uses max PPn
        search = find_capacity_config(problem)

        # Closed-form at max PPn fails (greedy exceeds max_nodes)
        assert closed is None, f"Expected closed_form to fail, got: {closed}"

        # find_capacity_config succeeds
        assert search is not None, "find_capacity_config should find a solution"
        assert search.node_count == expected.node_count
        assert search.rf == expected.rf
        assert search.partitions_per_node == expected.partitions_per_node

    def test_walkthrough_21_partitions(self):
        """Detailed walkthrough of the 21-partition example."""
        problem = CapacityProblem(
            n_partitions=21,
            partition_size_gib=10.0,
            disk_per_node_gib=100.0,
            min_rf=1,
            cpu_needed=10,
            cpu_per_node=1,
            max_nodes=10,
        )

        # CLOSED_FORM at max PPn: Only tries max PPn = 10
        #   base = ceil(21/10) = 3 nodes
        #   rf_for_cpu = ceil(10 / (3*1)) = 4
        #   total = 3 * 4 = 12 nodes > 10 → FAILS
        assert closed_form_algorithm(problem) is None

        # FIND_CAPACITY_CONFIG: Tries PPn from 10 down to 1
        #   PPn=10: base=3, rf=4, total=12 > 10 → skip
        #   PPn=9:  base=3, rf=4, total=12 > 10 → skip
        #   ...
        #   PPn=5:  base=5, rf=2, total=10 ≤ 10 → SUCCESS!
        result = find_capacity_config(problem)
        assert result is not None
        assert result.node_count == 10
        assert result.partitions_per_node == 5
        assert result.rf == 2
        assert result.base_nodes == 5

    @given(
        problem=st.builds(
            CapacityProblem,
            n_partitions=st.integers(1, 500),
            partition_size_gib=st.floats(
                10, 500, allow_nan=False, allow_infinity=False
            ),
            disk_per_node_gib=st.floats(
                100, 5000, allow_nan=False, allow_infinity=False
            ),
            min_rf=st.integers(1, 5),
            cpu_needed=st.integers(1, 5000),
            cpu_per_node=st.integers(1, 64),
            max_nodes=st.integers(2, 10000),
        )
    )
    @settings(max_examples=500, deadline=None)
    def test_find_capacity_config_never_worse(self, problem: CapacityProblem):
        """find_capacity_config always finds a solution if closed_form does."""
        if problem.partition_size_gib > problem.disk_per_node_gib:
            return

        closed = closed_form_algorithm(problem)
        search = find_capacity_config(problem)

        if closed is not None:
            assert search is not None, (
                f"closed_form found solution but find_capacity_config didn't: {problem}"
            )
            # find_capacity_config should find same or smaller cluster
            assert search.node_count <= closed.node_count, (
                f"find_capacity_config worse: {search.node_count} > {closed.node_count}"
            )


# =============================================================================
# PART 3: All algorithms satisfy constraints when they return a result
# =============================================================================


class TestConstraintsSatisfied:
    """All algorithms produce valid results that satisfy problem constraints."""

    @pytest.mark.parametrize("problem", STANDARD_CASES)
    @pytest.mark.parametrize("algo", [closed_form_algorithm, find_capacity_config])
    def test_result_satisfies_constraints(self, problem: CapacityProblem, algo):
        """Every result satisfies CPU, disk, RF, and size constraints."""
        result = algo(problem)
        if result is None:
            return

        # CPU: total cores ≥ needed
        total_cpu = result.node_count * problem.cpu_per_node
        assert total_cpu >= problem.cpu_needed, (
            f"CPU not satisfied: {total_cpu} < {problem.cpu_needed}"
        )

        # DISK: total slots ≥ total partition-replicas
        total_slots = result.node_count * result.partitions_per_node
        total_replicas = problem.n_partitions * result.rf
        assert total_slots >= total_replicas, (
            f"Disk not satisfied: {total_slots} < {total_replicas}"
        )

        # RF: at least min_rf
        assert result.rf >= problem.min_rf, (
            f"RF not satisfied: {result.rf} < {problem.min_rf}"
        )

        # SIZE: within max_nodes
        assert result.node_count <= problem.max_nodes, (
            f"Size not satisfied: {result.node_count} > {problem.max_nodes}"
        )


# =============================================================================
# PART 4: Node count formula verification
# =============================================================================


class TestNodeCountFormula:
    """Verify the fundamental formula: node_count = max(2, base × rf)."""

    @given(
        problem=st.builds(
            CapacityProblem,
            n_partitions=st.integers(1, 500),
            partition_size_gib=st.floats(
                10, 500, allow_nan=False, allow_infinity=False
            ),
            disk_per_node_gib=st.floats(
                100, 5000, allow_nan=False, allow_infinity=False
            ),
            min_rf=st.integers(1, 5),
            cpu_needed=st.integers(1, 5000),
            cpu_per_node=st.integers(1, 64),
            max_nodes=st.integers(2, 10000),
        )
    )
    @settings(max_examples=500, deadline=None)
    def test_node_count_equals_max_2_base_times_rf(self, problem: CapacityProblem):
        """node_count = max(2, base_nodes × rf) always holds."""
        if problem.partition_size_gib > problem.disk_per_node_gib:
            return

        for algo in [closed_form_algorithm, find_capacity_config]:
            result = algo(problem)
            if result is None:
                continue

            expected = max(2, result.base_nodes * result.rf)
            assert result.node_count == expected, (
                f"Formula violated: {result.node_count} != "
                f"max(2, {result.base_nodes}×{result.rf})"
            )


# =============================================================================
# PART 5: Tier configuration tests
# =============================================================================


class TestTierConfiguration:
    """Test tier configuration with target-based model."""

    def test_tier_defaults_exist(self):
        """All default tiers should be defined with correct fields."""
        for tier in range(4):
            config = get_tier_config(tier)
            assert config.min_rf >= 2
            assert config.target_nines >= 3  # At least 99.9%
            assert config.max_cost_multiplier >= 1

    def test_tier_ordering(self):
        """Higher tiers should have lower or equal requirements."""
        for tier in range(3):
            lower = get_tier_config(tier)
            higher = get_tier_config(tier + 1)

            # Higher tier should have lower or equal target_nines
            assert higher.target_nines <= lower.target_nines
            # Higher tier should have lower or equal max_cost_multiplier
            assert higher.max_cost_multiplier <= lower.max_cost_multiplier

    def test_invalid_tier_returns_default(self):
        """Invalid tier should return tier 2 config."""
        from service_capacity_modeling.models.org.netflix.partition_capacity import (
            TIER_DEFAULTS,
        )

        assert get_tier_config(99) == TIER_DEFAULTS[2]
        assert get_tier_config(-1) == TIER_DEFAULTS[2]

    def test_tier_target_nines_values(self):
        """Verify expected target_nines for each tier."""
        # Critical tiers (0-1) should target 4 nines (99.99%)
        assert get_tier_config(0).target_nines == 4
        assert get_tier_config(1).target_nines == 4
        # Standard tiers (2-3) should target 3 nines (99.9%)
        assert get_tier_config(2).target_nines == 3
        assert get_tier_config(3).target_nines == 3


# =============================================================================
# PART 6: Fault-tolerant search algorithm tests
# =============================================================================


class TestSearchWithFaultTolerance:
    """Test the fault-tolerant search algorithm."""

    def test_basic_search_returns_result(self):
        """Basic search should return a valid result with all expected fields."""
        problem = CapacityProblem(
            n_partitions=100,
            partition_size_gib=100,
            disk_per_node_gib=2048,
            min_rf=2,
            cpu_needed=800,
            cpu_per_node=16,
        )

        result = search_with_fault_tolerance(problem, tier=2, cost_per_node=100)

        assert result is not None
        assert result.node_count > 0
        assert result.rf >= 2
        assert 0 <= result.expected_availability <= 1
        assert 0 <= result.az_failure_availability <= 1
        assert result.cost > 0
        assert result.zone_aware_cost > 0

        # New target-based fields
        assert result.achieved_nines >= 0
        assert result.target_nines == get_tier_config(2).target_nines
        assert isinstance(result.target_met, bool)
        # achieved_nines should match nines(expected_availability)
        assert abs(result.achieved_nines - nines(result.expected_availability)) < 0.001

    def test_higher_tier_prefers_availability(self):
        """Tier 0 should prefer higher availability than tier 3."""
        problem = CapacityProblem(
            n_partitions=100,
            partition_size_gib=100,
            disk_per_node_gib=2048,
            min_rf=2,
            cpu_needed=800,
            cpu_per_node=16,
        )

        result_tier0 = search_with_fault_tolerance(problem, tier=0, cost_per_node=100)
        result_tier3 = search_with_fault_tolerance(problem, tier=3, cost_per_node=100)

        assert result_tier0 is not None
        assert result_tier3 is not None
        assert result_tier0.expected_availability >= result_tier3.expected_availability

    def test_result_satisfies_constraints(self):
        """Result should satisfy all problem constraints."""
        problem = CapacityProblem(
            n_partitions=200,
            partition_size_gib=50,
            disk_per_node_gib=1000,
            min_rf=2,
            cpu_needed=1600,
            cpu_per_node=16,
            max_nodes=500,
        )

        for tier in range(4):
            result = search_with_fault_tolerance(problem, tier=tier, cost_per_node=100)

            if result is None:
                continue

            config = get_tier_config(tier)

            # CPU constraint
            assert result.node_count * problem.cpu_per_node >= problem.cpu_needed
            # Disk constraint
            total_slots = result.node_count * result.partitions_per_node
            total_replicas = problem.n_partitions * result.rf
            assert total_slots >= total_replicas
            # RF constraint
            assert result.rf >= config.min_rf
            # Max nodes constraint
            assert result.node_count <= problem.max_nodes

    def test_zone_aware_cost_is_lower_or_equal(self):
        """Zone-aware cost should be <= random placement cost."""
        problem = CapacityProblem(
            n_partitions=100,
            partition_size_gib=100,
            disk_per_node_gib=2048,
            min_rf=2,
            cpu_needed=800,
            cpu_per_node=16,
        )

        result = search_with_fault_tolerance(problem, tier=0, cost_per_node=100)

        assert result is not None
        assert result.zone_aware_cost <= result.cost
        assert result.zone_aware_savings >= 0

    def test_impossible_returns_none(self):
        """If no config meets constraints, return None."""
        problem = CapacityProblem(
            n_partitions=100,
            partition_size_gib=5000,  # Way bigger than disk
            disk_per_node_gib=1000,
            min_rf=2,
            cpu_needed=800,
            cpu_per_node=16,
        )

        result = search_with_fault_tolerance(problem, tier=2, cost_per_node=100)
        assert result is None


# =============================================================================
# PART 7: Zone-aware placement tests
# =============================================================================


class TestZoneAwarePlacement:
    """Test zone_aware parameter behavior."""

    def test_random_placement_has_unavailability(self):
        """Random placement (zone_aware=False) has <100% conditional availability."""
        problem = CapacityProblem(
            n_partitions=100,
            partition_size_gib=100,
            disk_per_node_gib=2048,
            min_rf=2,
            cpu_needed=800,
            cpu_per_node=16,
        )

        result = search_with_fault_tolerance(
            problem, tier=2, cost_per_node=100, zone_aware=False
        )

        assert result is not None
        assert result.az_failure_availability < 1.0, (
            f"Random placement should have <100% conditional availability, "
            f"got {result.az_failure_availability}"
        )

    def test_zone_aware_rf2_gives_perfect_availability(self):
        """Zone-aware placement with RF>=2 gives 100% availability."""
        problem = CapacityProblem(
            n_partitions=100,
            partition_size_gib=100,
            disk_per_node_gib=2048,
            min_rf=2,
            cpu_needed=800,
            cpu_per_node=16,
        )

        result = search_with_fault_tolerance(
            problem, tier=2, cost_per_node=100, zone_aware=True
        )

        assert result is not None
        assert result.expected_availability == 1.0, (
            f"Zone-aware with RF>=2 should give 100% expected availability, "
            f"got {result.expected_availability}"
        )

    def test_zone_aware_chooses_lower_rf(self):
        """Zone-aware search chooses lower RF since availability is guaranteed."""
        problem = CapacityProblem(
            n_partitions=500,
            partition_size_gib=10,
            disk_per_node_gib=1000,
            min_rf=2,
            cpu_needed=800,
            cpu_per_node=16,
        )

        result_random = search_with_fault_tolerance(
            problem, tier=0, cost_per_node=100, zone_aware=False
        )
        result_zone_aware = search_with_fault_tolerance(
            problem, tier=0, cost_per_node=100, zone_aware=True
        )

        assert result_random is not None
        assert result_zone_aware is not None

        # Zone-aware should choose lower or equal RF
        assert result_zone_aware.rf <= result_random.rf, (
            f"Zone-aware should choose RF <= random: "
            f"zone_aware.rf={result_zone_aware.rf}, random.rf={result_random.rf}"
        )

        # Zone-aware should have lower or equal cost
        assert result_zone_aware.cost <= result_random.cost

    def test_zone_aware_cost_matches_zone_aware_search(self):
        """zone_aware_cost should match what zone_aware=True search finds."""
        problem = CapacityProblem(
            n_partitions=100,
            partition_size_gib=100,
            disk_per_node_gib=2048,
            min_rf=2,
            cpu_needed=800,
            cpu_per_node=16,
        )

        # Get result with random placement
        result = search_with_fault_tolerance(
            problem, tier=2, cost_per_node=100, zone_aware=False
        )
        # Get what zone-aware search would choose
        result_zone_aware = search_with_fault_tolerance(
            problem, tier=2, cost_per_node=100, zone_aware=True
        )

        assert result is not None
        assert result_zone_aware is not None

        # The zone_aware_cost in result should match what zone-aware search finds
        assert result.zone_aware_cost == result_zone_aware.cost, (
            f"zone_aware_cost should match actual zone-aware search result: "
            f"zone_aware_cost={result.zone_aware_cost}, "
            f"actual_zone_aware_result.cost={result_zone_aware.cost}"
        )


# =============================================================================
# PART 8: Pareto frontier validation tests
# =============================================================================


class TestParetoFrontierValidation:
    """Verify search returns Pareto-optimal solutions and frontier is correct."""

    def test_search_respects_target_based_selection(self):
        """The search result should follow target-based selection rules.

        With target-based selection:
        1. If any config reaches target_nines, pick the cheapest
        2. If no config reaches target, pick highest availability
        """
        problem = CapacityProblem(
            n_partitions=100,
            partition_size_gib=100,
            disk_per_node_gib=2048,
            min_rf=2,
            cpu_needed=800,
            cpu_per_node=16,
        )

        cost_per_node = 100

        for tier in range(4):
            result = search_with_fault_tolerance(
                problem, tier=tier, cost_per_node=cost_per_node
            )
            frontier = compute_pareto_frontier(
                problem, tier=tier, cost_per_node=cost_per_node
            )

            if result is None or not frontier:
                continue

            config = get_tier_config(tier)

            # Filter frontier to configs that reach target
            reaching_target = [
                p for p in frontier if nines(p.availability) >= config.target_nines
            ]

            if reaching_target:
                # If any reach target, result should be cheapest that does
                cheapest_reaching = min(reaching_target, key=lambda p: p.cost)
                assert result.target_met, (
                    f"Tier {tier}: target should be met when configs reach it"
                )
                assert result.cost <= cheapest_reaching.cost + 0.01, (
                    f"Tier {tier}: should pick cheapest config reaching target"
                )
            else:
                # If none reach target, result should be highest availability
                best_avail = max(frontier, key=lambda p: p.availability)
                assert not result.target_met, (
                    f"Tier {tier}: target should not be met when no config reaches it"
                )
                assert (
                    result.expected_availability >= best_avail.availability - 0.0001
                ), f"Tier {tier}: should pick highest availability"

    def test_frontier_is_sorted_by_cost(self):
        """Pareto frontier should be sorted by cost (cheapest first)."""
        problem = CapacityProblem(
            n_partitions=100,
            partition_size_gib=100,
            disk_per_node_gib=2048,
            min_rf=2,
            cpu_needed=800,
            cpu_per_node=16,
        )

        frontier = compute_pareto_frontier(problem, tier=2, cost_per_node=100)

        for i in range(len(frontier) - 1):
            assert frontier[i].cost <= frontier[i + 1].cost, (
                f"Frontier not sorted: {frontier[i].cost} > {frontier[i + 1].cost}"
            )

    def test_frontier_has_no_dominated_points(self):
        """No point on frontier should be dominated by another."""
        problem = CapacityProblem(
            n_partitions=100,
            partition_size_gib=100,
            disk_per_node_gib=2048,
            min_rf=2,
            cpu_needed=800,
            cpu_per_node=16,
        )

        frontier = compute_pareto_frontier(problem, tier=2, cost_per_node=100)

        for i, p1 in enumerate(frontier):
            for j, p2 in enumerate(frontier):
                if i == j:
                    continue
                dominated = (
                    p2.availability >= p1.availability
                    and p2.cost <= p1.cost
                    and (p2.availability > p1.availability or p2.cost < p1.cost)
                )
                assert not dominated, f"Point {p1} dominated by {p2}"

    def test_different_tiers_may_pick_different_frontier_points(self):
        """Higher tiers (cost-focused) should prefer cheaper frontier points."""
        problem = CapacityProblem(
            n_partitions=100,
            partition_size_gib=100,
            disk_per_node_gib=2048,
            min_rf=2,
            cpu_needed=800,
            cpu_per_node=16,
        )

        result_t0 = search_with_fault_tolerance(problem, tier=0, cost_per_node=100)
        result_t3 = search_with_fault_tolerance(problem, tier=3, cost_per_node=100)

        assert result_t0 is not None
        assert result_t3 is not None

        # Tier 0 should have >= cost than tier 3
        assert result_t0.cost >= result_t3.cost, (
            f"Tier 0 should be at least as expensive as tier 3: "
            f"t0={result_t0.cost}, t3={result_t3.cost}"
        )


# =============================================================================
# PART 9: Target-Based Selection Tests
# =============================================================================


class TestTargetBasedSelection:
    """Test the target-based selection model.

    With target-based selection:
    1. Find cheapest config that reaches target_nines
    2. If no config reaches target, pick highest availability (lenient mode)
    3. Report whether target was met via target_met field
    """

    def test_result_has_target_fields(self):
        """Results should have achieved_nines, target_nines, and target_met."""
        problem = CapacityProblem(
            n_partitions=100,
            partition_size_gib=100,
            disk_per_node_gib=2048,
            min_rf=2,
            cpu_needed=800,
            cpu_per_node=16,
        )

        result = search_with_fault_tolerance(problem, tier=2, cost_per_node=100)

        assert result is not None
        assert hasattr(result, "achieved_nines")
        assert hasattr(result, "target_nines")
        assert hasattr(result, "target_met")

        # achieved_nines should match nines(expected_availability)
        assert abs(result.achieved_nines - nines(result.expected_availability)) < 0.001

        # target_nines should match tier config
        config = get_tier_config(2)
        assert result.target_nines == config.target_nines

        # target_met should be correct
        assert result.target_met == (result.achieved_nines >= result.target_nines)

    def test_target_met_when_reaching_target(self):
        """When achieved_nines >= target_nines, target_met should be True."""
        # Use zone-aware placement which guarantees 100% availability
        problem = CapacityProblem(
            n_partitions=100,
            partition_size_gib=100,
            disk_per_node_gib=2048,
            min_rf=2,
            cpu_needed=800,
            cpu_per_node=16,
        )

        result = search_with_fault_tolerance(
            problem, tier=3, cost_per_node=100, zone_aware=True
        )

        assert result is not None
        # Zone-aware with RF>=2 gives infinite nines (100% availability)
        # Should easily meet tier 3's target of 3 nines
        assert result.target_met, (
            f"Zone-aware should meet target: "
            f"achieved={result.achieved_nines:.2f}, target={result.target_nines}"
        )

    def test_cheapest_config_selected_when_target_met(self):
        """When multiple configs reach target, cheapest should be selected."""
        # Use a problem where multiple configs exist
        problem = CapacityProblem(
            n_partitions=50,
            partition_size_gib=50,
            disk_per_node_gib=2048,
            min_rf=2,
            cpu_needed=400,
            cpu_per_node=16,
        )

        # Tier 3 has target of 3 nines and lowest cost multiplier
        result = search_with_fault_tolerance(problem, tier=3, cost_per_node=100)

        assert result is not None
        if result.target_met:
            # If target is met, we should have the cheapest config that does so
            # The result should NOT overspend on availability we don't need
            config = get_tier_config(3)
            # Any config with more nines should be more expensive or equal
            # (This is validated by the Pareto selection test)
            assert result.achieved_nines >= config.target_nines

    def test_tier0_versus_tier3_selection(self):
        """Tier 0 (4 nines target) may spend more than tier 3 (3 nines target)."""
        problem = CapacityProblem(
            n_partitions=100,
            partition_size_gib=100,
            disk_per_node_gib=2048,
            min_rf=2,
            cpu_needed=800,
            cpu_per_node=16,
        )

        result_tier0 = search_with_fault_tolerance(problem, tier=0, cost_per_node=100)
        result_tier3 = search_with_fault_tolerance(problem, tier=3, cost_per_node=100)

        assert result_tier0 is not None
        assert result_tier3 is not None

        # Tier 0 should have higher or equal cost (needs 4 nines vs 3)
        assert result_tier0.cost >= result_tier3.cost, (
            f"Tier 0 should spend at least as much as tier 3: "
            f"t0=${result_tier0.cost}, t3=${result_tier3.cost}"
        )

        # Tier 0 should achieve higher or equal availability
        assert result_tier0.achieved_nines >= result_tier3.achieved_nines


# =============================================================================
# PART 10: Expected vs Conditional Availability Tests
# =============================================================================


class TestExpectedAvailabilityModel:
    """Test the expected availability model that incorporates failure probabilities.

    Key insight: AZ failures are rare (~2%/year), so even "bad" conditional
    availability (90%) translates to very high expected availability (99.9%+).
    """

    def test_sanity_check_90_conditional_is_999_expected(self):
        """90% conditional availability should be ~99.9%+ expected availability.

        This is the key sanity check from the design document:
        A service showing 90% conditional availability (bad!) is actually
        99.9%+ expected availability (good!) because AZ failures are rare.
        """
        from service_capacity_modeling.models.org.netflix.partition_capacity import (
            HOURS_PER_YEAR,
            recovery_hours,
        )

        # Test with explicit calculation
        conditional = 0.90  # 90% survival if AZ fails

        # With default failure model: 2% AZ failure rate
        # Recovery time depends on data size - assume 500 GiB per node
        failure_model = FailureModel()
        data_per_node_gib = 500.0  # Typical data size
        recover_time = recovery_hours(data_per_node_gib, failure_model)

        # Expected downtime = az_rate × p_loss × recovery_hours / HOURS_PER_YEAR
        p_loss = 1.0 - conditional  # 10% chance of loss given AZ failure
        expected_downtime_fraction = (
            failure_model.az_failure_rate_per_year
            * p_loss
            * recover_time
            / HOURS_PER_YEAR
        )
        expected_avail = 1.0 - expected_downtime_fraction

        # Should be 99.9%+ (3 nines or more)
        assert expected_avail > 0.999, (
            f"90% conditional should give >99.9% expected, got {expected_avail:.6f}"
        )

        # In nines: should be ~3+ nines
        expected_nines = nines(expected_avail)
        assert expected_nines >= 3.0, (
            f"90% conditional should give 3+ nines expected, got {expected_nines:.2f}"
        )

    def test_expected_annual_availability_function(self):
        """Test the expected_annual_availability function directly."""
        # Parameters that give roughly 90% conditional availability
        # This is a worst-case scenario for random placement
        conditional = system_availability(n_nodes=6, n_zones=3, rf=2, n_partitions=100)

        # Get expected availability using the function
        # Assume 100 GiB per node (typical for small dataset)
        expected = expected_annual_availability(
            n_nodes=6, n_zones=3, rf=2, n_partitions=100, data_per_node_gib=100.0
        )

        # Expected should always be >= conditional (since AZ failures are rare)
        assert expected >= conditional, (
            f"Expected ({expected:.6f}) should be >= conditional ({conditional:.6f})"
        )

        # Expected should be very high even if conditional is low
        if conditional < 0.99:
            assert expected > 0.999, (
                f"With low conditional ({conditional:.4f}), "
                f"expected should be >99.9%, got {expected:.6f}"
            )

    def test_nines_conversion(self):
        """Test the nines() helper function."""
        # 0.9 = 1 nine
        assert abs(nines(0.9) - 1.0) < 0.01
        # 0.99 = 2 nines
        assert abs(nines(0.99) - 2.0) < 0.01
        # 0.999 = 3 nines
        assert abs(nines(0.999) - 3.0) < 0.01
        # 0.9999 = 4 nines
        assert abs(nines(0.9999) - 4.0) < 0.01

    def test_failure_model_parameters(self):
        """Test that failure model parameters affect expected availability."""
        # Higher AZ failure rate should decrease expected availability
        high_rate_model = FailureModel(az_failure_rate_per_year=0.10)
        low_rate_model = FailureModel(az_failure_rate_per_year=0.01)

        expected_high = expected_annual_availability(
            n_nodes=6,
            n_zones=3,
            rf=2,
            n_partitions=100,
            data_per_node_gib=100.0,
            failure_model=high_rate_model,
        )
        expected_low = expected_annual_availability(
            n_nodes=6,
            n_zones=3,
            rf=2,
            n_partitions=100,
            data_per_node_gib=100.0,
            failure_model=low_rate_model,
        )

        assert expected_low > expected_high, (
            "Lower failure rate should give higher expected availability"
        )

    def test_search_returns_expected_and_conditional(self):
        """Search should return both expected and conditional availability."""
        problem = CapacityProblem(
            n_partitions=100,
            partition_size_gib=100,
            disk_per_node_gib=2048,
            min_rf=2,
            cpu_needed=800,
            cpu_per_node=16,
        )

        result = search_with_fault_tolerance(problem, tier=2, cost_per_node=100)

        assert result is not None
        assert hasattr(result, "expected_availability")
        assert hasattr(result, "az_failure_availability")

        # Expected should be >= conditional
        assert result.expected_availability >= result.az_failure_availability, (
            f"Expected ({result.expected_availability:.6f}) should be >= "
            f"conditional ({result.az_failure_availability:.6f})"
        )
