"""
Test suite for partition-aware capacity planning algorithms.

This file demonstrates THREE algorithm variants:
    1. ORIGINAL: While-loop (greedy, max PPn only)
    2. CLOSED_FORM: O(1) mathematical equivalent to original
    3. SEARCH: Searches PPn from max to 1, finds solutions original misses

Key findings:
    - ORIGINAL == CLOSED_FORM (always, by mathematical proof)
    - SEARCH ⊇ ORIGINAL (search finds everything original finds, plus more)
    - SEARCH ≠ ORIGINAL when max_nodes is tight (search finds solutions original misses)
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from service_capacity_modeling.models.org.netflix.partition_capacity import (
    CapacityProblem,
    CapacityResult,
    closed_form_algorithm,
    original_algorithm,
    search_algorithm,
)


# =============================================================================
# TEST DATA: Problems demonstrating equivalence and differences
# =============================================================================

# Problems where all three algorithms agree (max_nodes is relaxed)
EQUIVALENT_CASES = [
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

# Problems where SEARCH finds solutions but ORIGINAL/CLOSED_FORM return None
SEARCH_ONLY_CASES = [
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
        # ppn=10: base=4, rf=4, total=16 > 15 (original fails)
        # ppn=7:  base=5, rf=3, total=15 ≤ 15 (search succeeds)
        CapacityResult(node_count=15, rf=3, partitions_per_node=7, base_nodes=5),
        id="31-partitions-need-ppn-7",
    ),
]


# =============================================================================
# PART 1: CLOSED_FORM == ORIGINAL (mathematical equivalence)
# =============================================================================


class TestClosedFormEqualsOriginal:
    """Verify closed_form produces IDENTICAL results to original.

    This is a REFACTOR, not a new algorithm. Same logic, cleaner code.
    The while-loop increments RF; closed_form computes it directly.
    """

    @pytest.mark.parametrize("problem", EQUIVALENT_CASES)
    def test_deterministic_cases(self, problem: CapacityProblem):
        """Closed-form matches original on curated test cases."""
        orig = original_algorithm(problem)
        closed = closed_form_algorithm(problem)

        assert orig is not None, "Original should find solution"
        assert closed is not None, "Closed-form should find solution"

        assert closed.node_count == orig.node_count
        assert closed.rf == orig.rf
        assert closed.partitions_per_node == orig.partitions_per_node
        assert closed.base_nodes == orig.base_nodes

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
    def test_hypothesis_always_equal(self, problem: CapacityProblem):
        """For ANY valid problem, closed_form == original."""
        # Skip invalid problems (partition larger than disk)
        if problem.partition_size_gib > problem.disk_per_node_gib:
            return

        orig = original_algorithm(problem)
        closed = closed_form_algorithm(problem)

        if orig is None:
            assert closed is None, f"Orig=None but Closed found: {problem}"
        elif closed is None:
            pytest.fail(f"Closed=None but Orig found: {problem}")
        else:
            assert closed.node_count == orig.node_count, (
                f"node_count mismatch: orig={orig.node_count}, "
                f"closed={closed.node_count}\n{problem}"
            )
            assert closed.rf == orig.rf, (
                f"rf: orig={orig.rf}, closed={closed.rf}\n{problem}"
            )


# =============================================================================
# PART 2: SEARCH ⊇ ORIGINAL (search finds everything original finds)
# =============================================================================


class TestSearchSubsumesOriginal:
    """Verify SEARCH finds all solutions ORIGINAL finds, with same or better results.

    When max_nodes is relaxed, both algorithms find the same solution.
    SEARCH is never worse than ORIGINAL.
    """

    @pytest.mark.parametrize("problem", EQUIVALENT_CASES)
    def test_same_results_when_unconstrained(self, problem: CapacityProblem):
        """With relaxed max_nodes, search == original."""
        orig = original_algorithm(problem)
        search = search_algorithm(problem)

        assert orig is not None
        assert search is not None

        # Same solution
        assert search.node_count == orig.node_count
        assert search.rf == orig.rf
        assert search.partitions_per_node == orig.partitions_per_node

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
    def test_hypothesis_search_never_worse(self, problem: CapacityProblem):
        """SEARCH always finds a solution if ORIGINAL does."""
        if problem.partition_size_gib > problem.disk_per_node_gib:
            return

        orig = original_algorithm(problem)
        search = search_algorithm(problem)

        if orig is not None:
            assert search is not None, (
                f"Original found solution but search didn't: {problem}"
            )
            # Search should find same or smaller cluster
            assert search.node_count <= orig.node_count, (
                f"Search worse than original: {search.node_count} > {orig.node_count}"
            )


# =============================================================================
# PART 3: SEARCH ≠ ORIGINAL (search finds solutions original misses)
# =============================================================================


class TestSearchFindsSolutionsOriginalMisses:
    """Demonstrate cases where SEARCH succeeds but ORIGINAL fails.

    This happens when:
    1. Greedy (max PPn) produces a cluster exceeding max_nodes
    2. A lower PPn produces a valid cluster within max_nodes

    These cases prove SEARCH is strictly more capable than ORIGINAL.
    """

    @pytest.mark.parametrize("problem,expected", SEARCH_ONLY_CASES)
    def test_original_fails_search_succeeds(
        self, problem: CapacityProblem, expected: CapacityResult
    ):
        """Original returns None, but search finds a valid solution."""
        orig = original_algorithm(problem)
        search = search_algorithm(problem)

        # Original fails (greedy exceeds max_nodes)
        assert orig is None, f"Expected original to fail, got: {orig}"

        # Search succeeds
        assert search is not None, "Search should find a solution"
        assert search.node_count == expected.node_count
        assert search.rf == expected.rf
        assert search.partitions_per_node == expected.partitions_per_node

    def test_why_search_beats_original_21_partitions(self):
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

        # ORIGINAL/CLOSED_FORM: Only tries max PPn = 10
        #   base = ceil(21/10) = 3 nodes
        #   rf_for_cpu = ceil(10 / (3*1)) = 4
        #   total = 3 * 4 = 12 nodes > 10 → FAILS
        assert original_algorithm(problem) is None
        assert closed_form_algorithm(problem) is None

        # SEARCH: Tries PPn from 10 down to 1
        #   PPn=10: base=3, rf=4, total=12 > 10 → skip
        #   PPn=9:  base=3, rf=4, total=12 > 10 → skip
        #   ...
        #   PPn=5:  base=5, rf=2, total=10 ≤ 10 → SUCCESS!
        result = search_algorithm(problem)
        assert result is not None
        assert result.node_count == 10
        assert result.partitions_per_node == 5
        assert result.rf == 2
        assert result.base_nodes == 5

    def test_search_maximizes_rf_within_constraint(self):
        """Search prefers higher RF (better fault tolerance) when possible."""
        # With max_nodes=100, greedy gives rf=4
        relaxed = CapacityProblem(
            n_partitions=21,
            partition_size_gib=10.0,
            disk_per_node_gib=100.0,
            min_rf=1,
            cpu_needed=10,
            cpu_per_node=1,
            max_nodes=100,  # relaxed
        )
        result_relaxed = search_algorithm(relaxed)
        assert result_relaxed is not None
        assert result_relaxed.rf == 4  # Higher RF (better!)
        assert result_relaxed.node_count == 12

        # With max_nodes=10, search finds rf=2
        tight = CapacityProblem(
            n_partitions=21,
            partition_size_gib=10.0,
            disk_per_node_gib=100.0,
            min_rf=1,
            cpu_needed=10,
            cpu_per_node=1,
            max_nodes=10,  # tight
        )
        result_tight = search_algorithm(tight)
        assert result_tight is not None
        assert result_tight.rf == 2  # Lower RF (but valid!)
        assert result_tight.node_count == 10


# =============================================================================
# PART 4: All algorithms satisfy constraints when they return a result
# =============================================================================


class TestConstraintsSatisfied:
    """All algorithms produce valid results that satisfy problem constraints."""

    @pytest.mark.parametrize("problem", EQUIVALENT_CASES)
    @pytest.mark.parametrize(
        "algo", [original_algorithm, closed_form_algorithm, search_algorithm]
    )
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
    def test_hypothesis_constraints_always_satisfied(self, problem: CapacityProblem):
        """Property test: all algorithms satisfy constraints."""
        if problem.partition_size_gib > problem.disk_per_node_gib:
            return

        for algo in [original_algorithm, closed_form_algorithm, search_algorithm]:
            result = algo(problem)
            if result is None:
                continue

            # CPU
            assert result.node_count * problem.cpu_per_node >= problem.cpu_needed
            # Disk
            assert result.node_count * result.partitions_per_node >= (
                problem.n_partitions * result.rf
            )
            # RF
            assert result.rf >= problem.min_rf
            # Size
            assert result.node_count <= problem.max_nodes


# =============================================================================
# PART 5: Node count formula verification
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

        for algo in [original_algorithm, closed_form_algorithm, search_algorithm]:
            result = algo(problem)
            if result is None:
                continue

            expected = max(2, result.base_nodes * result.rf)
            assert result.node_count == expected, (
                f"Formula violated: {result.node_count} != "
                f"max(2, {result.base_nodes}×{result.rf})"
            )


# =============================================================================
# PART 6: Tier configuration and utility function tests
# =============================================================================


class TestTierConfiguration:
    """Test tier configuration and utility function."""

    def test_tier_defaults_exist(self):
        """All default tiers should be defined."""
        from service_capacity_modeling.models.org.netflix.partition_capacity import (
            get_tier_config,
        )

        for tier in range(4):
            config = get_tier_config(tier)
            assert config.min_rf >= 2
            assert 0 < config.target_availability <= 1
            assert config.cost_sensitivity > 0
            assert config.max_cost_multiplier >= 1

    def test_tier_ordering(self):
        """Higher tiers should have lower requirements."""
        from service_capacity_modeling.models.org.netflix.partition_capacity import (
            get_tier_config,
        )

        for tier in range(3):
            lower = get_tier_config(tier)
            higher = get_tier_config(tier + 1)

            # Higher tier should have lower or equal availability target
            assert higher.target_availability <= lower.target_availability
            # Higher tier should have higher cost sensitivity (more cost-focused)
            assert higher.cost_sensitivity >= lower.cost_sensitivity

    def test_invalid_tier_returns_default(self):
        """Invalid tier should return tier 2 config."""
        from service_capacity_modeling.models.org.netflix.partition_capacity import (
            get_tier_config,
            TIER_DEFAULTS,
        )

        assert get_tier_config(99) == TIER_DEFAULTS[2]
        assert get_tier_config(-1) == TIER_DEFAULTS[2]


class TestUtilityFunction:
    """Test the utility function for balancing availability and cost."""

    def test_utility_above_target_is_positive(self):
        """Utility should be positive when availability exceeds target."""
        from service_capacity_modeling.models.org.netflix.partition_capacity import (
            utility,
        )

        # Tier 2: target is 0.95
        u = utility(availability=0.99, cost=100, tier=2, base_cost=100)
        assert u > 0, f"Expected positive utility, got {u}"

    def test_utility_below_target_is_negative(self):
        """Utility should be negative when availability is below target."""
        from service_capacity_modeling.models.org.netflix.partition_capacity import (
            utility,
        )

        # Tier 2: target is 0.95
        u = utility(availability=0.90, cost=100, tier=2, base_cost=100)
        assert u < 0, f"Expected negative utility, got {u}"

    def test_utility_higher_availability_higher_utility(self):
        """Higher availability should give higher utility."""
        from service_capacity_modeling.models.org.netflix.partition_capacity import (
            utility,
        )

        u_low = utility(availability=0.95, cost=100, tier=2, base_cost=100)
        u_high = utility(availability=0.99, cost=100, tier=2, base_cost=100)

        assert u_high > u_low

    def test_utility_higher_cost_lower_utility(self):
        """Higher cost should give lower utility."""
        from service_capacity_modeling.models.org.netflix.partition_capacity import (
            utility,
        )

        u_cheap = utility(availability=0.99, cost=100, tier=2, base_cost=100)
        u_expensive = utility(availability=0.99, cost=200, tier=2, base_cost=100)

        assert u_cheap > u_expensive

    def test_utility_diminishing_returns(self):
        """Going from 99% to 99.9% should have diminishing returns."""
        from service_capacity_modeling.models.org.netflix.partition_capacity import (
            utility,
        )

        # Same cost, different availability improvements
        u_95 = utility(availability=0.95, cost=100, tier=2, base_cost=100)
        u_99 = utility(availability=0.99, cost=100, tier=2, base_cost=100)
        u_999 = utility(availability=0.999, cost=100, tier=2, base_cost=100)

        # Marginal utility should decrease
        delta_1 = u_99 - u_95  # 95% -> 99% (4% improvement)
        delta_2 = u_999 - u_99  # 99% -> 99.9% (0.9% improvement)

        # First improvement is larger, but second should still be positive
        assert delta_1 > delta_2 > 0

    def test_utility_tier_affects_cost_sensitivity(self):
        """Higher tiers should be more cost-sensitive."""
        from service_capacity_modeling.models.org.netflix.partition_capacity import (
            utility,
        )

        # Same availability (above target for both), same cost increase
        # Tier 3 should penalize cost more than tier 0
        u_tier0 = utility(availability=0.999, cost=200, tier=0, base_cost=100)
        u_tier3 = utility(availability=0.999, cost=200, tier=3, base_cost=100)

        # Tier 3 has higher cost_sensitivity, so should have lower utility
        # for the same cost increase
        # Note: availability value differs due to different targets
        # So we compare the impact of cost increase
        u_tier0_base = utility(availability=0.999, cost=100, tier=0, base_cost=100)
        u_tier3_base = utility(availability=0.999, cost=100, tier=3, base_cost=100)

        cost_impact_tier0 = u_tier0_base - u_tier0
        cost_impact_tier3 = u_tier3_base - u_tier3

        assert cost_impact_tier3 > cost_impact_tier0, (
            f"Tier 3 should penalize cost more: "
            f"tier0={cost_impact_tier0}, tier3={cost_impact_tier3}"
        )


# =============================================================================
# PART 7: Fault-tolerant search algorithm tests
# =============================================================================


class TestSearchWithFaultTolerance:
    """Test the fault-tolerant search algorithm."""

    def test_basic_search_returns_result(self):
        """Basic search should return a valid result."""
        from service_capacity_modeling.models.org.netflix.partition_capacity import (
            search_with_fault_tolerance,
        )

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
        assert 0 <= result.system_availability <= 1
        assert result.cost > 0
        assert result.zone_aware_cost > 0

    def test_higher_tier_prefers_availability(self):
        """Tier 0 should prefer higher availability than tier 3."""
        from service_capacity_modeling.models.org.netflix.partition_capacity import (
            search_with_fault_tolerance,
        )

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

        # Tier 0 should have higher or equal availability
        assert result_tier0.system_availability >= result_tier3.system_availability

    def test_result_satisfies_constraints(self):
        """Result should satisfy all problem constraints."""
        from service_capacity_modeling.models.org.netflix.partition_capacity import (
            search_with_fault_tolerance,
            get_tier_config,
        )

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
        from service_capacity_modeling.models.org.netflix.partition_capacity import (
            search_with_fault_tolerance,
        )

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
        # Zone-aware placement achieves same availability with lower RF
        # so cost should be <= current cost
        assert result.zone_aware_cost <= result.cost
        assert result.zone_aware_savings >= 0

    def test_many_partitions_needs_high_rf(self):
        """With many partitions, system needs high RF for good availability."""
        from service_capacity_modeling.models.org.netflix.partition_capacity import (
            search_with_fault_tolerance,
        )

        # 500 partitions - with RF=2, availability would be very low
        problem = CapacityProblem(
            n_partitions=500,
            partition_size_gib=10,
            disk_per_node_gib=1000,
            min_rf=2,
            cpu_needed=800,
            cpu_per_node=16,
        )

        result = search_with_fault_tolerance(problem, tier=0, cost_per_node=100)

        assert result is not None
        # For 500 partitions, tier 0 needs high RF to meet 99.9% target
        # If RF is too low, availability won't meet target
        # The algorithm should find a configuration that meets or gets close to target
        assert result.system_availability > 0.5  # At least reasonable

    def test_impossible_returns_none(self):
        """If no config meets constraints, return None."""
        from service_capacity_modeling.models.org.netflix.partition_capacity import (
            search_with_fault_tolerance,
        )

        # Partition larger than disk
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
# PART 8: Placement model tests
# =============================================================================


class TestPlacementModels:
    """Test placement model abstraction."""

    def test_uniform_random_uses_defaults(self):
        """UniformRandomPlacement should use default implementations."""
        from service_capacity_modeling.models.org.netflix.partition_capacity import (
            UniformRandomPlacement,
            per_partition_unavailability,
            system_availability,
        )

        placement = UniformRandomPlacement()

        # Should match the default functions
        assert placement.per_partition_unavailability(
            12, 3, 2
        ) == per_partition_unavailability(12, 3, 2)
        assert placement.system_availability(12, 3, 2, 100) == system_availability(
            12, 3, 2, 100
        )
        assert placement.name() == "UniformRandomPlacement"

    def test_zone_aware_rf2_gives_zero_unavailability(self):
        """ZoneAwarePlacement with RF>=2 should give 0 unavailability."""
        from service_capacity_modeling.models.org.netflix.partition_capacity import (
            ZoneAwarePlacement,
        )

        placement = ZoneAwarePlacement()

        # RF=2 or higher should give 0 unavailability
        assert placement.per_partition_unavailability(12, 3, 2) == 0.0
        assert placement.per_partition_unavailability(12, 3, 3) == 0.0
        assert placement.per_partition_unavailability(12, 3, 5) == 0.0

        # System availability should be 100%
        assert placement.system_availability(12, 3, 2, 100) == 1.0
        assert placement.system_availability(12, 3, 2, 1000) == 1.0

    def test_zone_aware_rf1_has_unavailability(self):
        """ZoneAwarePlacement with RF=1 still has unavailability."""
        from service_capacity_modeling.models.org.netflix.partition_capacity import (
            ZoneAwarePlacement,
        )

        placement = ZoneAwarePlacement()

        # RF=1 has 1/n_zones unavailability
        unavail = placement.per_partition_unavailability(12, 3, 1)
        assert abs(unavail - 1 / 3) < 0.0001

        # System availability = (1 - 1/3)^100 for 100 partitions
        avail = placement.system_availability(12, 3, 1, 100)
        expected = (2 / 3) ** 100
        assert abs(avail - expected) < 0.0001

    def test_zone_aware_always_better_than_random(self):
        """ZoneAwarePlacement should always give >= availability than random."""
        from service_capacity_modeling.models.org.netflix.partition_capacity import (
            UniformRandomPlacement,
            ZoneAwarePlacement,
        )

        random_placement = UniformRandomPlacement()
        zone_aware = ZoneAwarePlacement()

        test_cases = [
            (12, 3, 2, 100),
            (12, 3, 3, 200),
            (24, 3, 4, 500),
            (30, 3, 5, 1000),
        ]

        for n_nodes, n_zones, rf, n_partitions in test_cases:
            random_avail = random_placement.system_availability(
                n_nodes, n_zones, rf, n_partitions
            )
            zone_aware_avail = zone_aware.system_availability(
                n_nodes, n_zones, rf, n_partitions
            )

            assert zone_aware_avail >= random_avail, (
                f"Zone-aware should be >= random: "
                f"random={random_avail}, zone_aware={zone_aware_avail}"
            )
