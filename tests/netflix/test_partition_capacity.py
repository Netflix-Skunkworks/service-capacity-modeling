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
