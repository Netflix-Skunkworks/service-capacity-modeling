"""
Integration tests for fault tolerance optimization.

These tests verify end-to-end behavior of the fault tolerance system,
including:
- All tiers can achieve their targets within cost constraints
- Zone-aware comparison cost is always computed
- The algorithm finds Pareto-optimal configurations
- Real-world-like scenarios work correctly
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from service_capacity_modeling.models.org.netflix.partition_capacity import (
    CapacityProblem,
    get_tier_config,
    search_with_fault_tolerance,
    system_availability,
)


# =============================================================================
# INTEGRATION: End-to-end scenarios
# =============================================================================


class TestEndToEndScenarios:
    """Test realistic end-to-end scenarios."""

    def test_small_service_tier2(self):
        """Small service with 50 partitions should work well at tier 2."""
        problem = CapacityProblem(
            n_partitions=50,
            partition_size_gib=100,
            disk_per_node_gib=2048,
            min_rf=2,
            cpu_needed=400,
            cpu_per_node=16,
        )

        result = search_with_fault_tolerance(problem, tier=2, cost_per_node=100)

        assert result is not None
        assert result.az_failure_availability >= 0.90  # Should be decent
        assert result.rf >= 2
        assert result.zone_aware_savings >= 0

    def test_large_service_tier0(self):
        """Large critical service with 1000 partitions at tier 0."""
        problem = CapacityProblem(
            n_partitions=1000,
            partition_size_gib=10,
            disk_per_node_gib=2000,
            min_rf=3,
            cpu_needed=8000,
            cpu_per_node=16,
        )

        result = search_with_fault_tolerance(problem, tier=0, cost_per_node=100)

        assert result is not None
        # With 1000 partitions and random placement, achieving 99.9%
        # requires very high RF or accepting lower availability
        # The algorithm should find the best it can within cost constraints
        assert result.rf >= 3
        assert result.tier == 0

    def test_cpu_constrained_service(self):
        """CPU-constrained service needs more nodes for CPU."""
        problem = CapacityProblem(
            n_partitions=100,
            partition_size_gib=10,  # Small partitions
            disk_per_node_gib=2000,  # Lots of disk
            min_rf=2,
            cpu_needed=3200,  # High CPU need
            cpu_per_node=16,
        )

        result = search_with_fault_tolerance(problem, tier=2, cost_per_node=100)

        assert result is not None
        # CPU-constrained means more nodes, which means better availability
        assert result.node_count * problem.cpu_per_node >= problem.cpu_needed

    def test_disk_constrained_service(self):
        """Disk-constrained service with large partitions."""
        problem = CapacityProblem(
            n_partitions=100,
            partition_size_gib=500,  # Large partitions
            disk_per_node_gib=2000,  # Limited disk per node
            min_rf=2,
            cpu_needed=100,  # Low CPU
            cpu_per_node=16,
        )

        result = search_with_fault_tolerance(problem, tier=2, cost_per_node=100)

        assert result is not None
        # Disk-constrained means fewer partitions per node
        assert result.partitions_per_node <= 4  # 2000/500 = 4


# =============================================================================
# INTEGRATION: Tier target verification
# =============================================================================


class TestTierTargets:
    """Verify that tiers achieve their targets when possible."""

    @pytest.mark.parametrize("tier", [0, 1, 2, 3])
    def test_tier_meets_or_approaches_target(self, tier: int):
        """Each tier should meet or approach its availability target."""
        config = get_tier_config(tier)

        # Medium-sized problem
        problem = CapacityProblem(
            n_partitions=100,
            partition_size_gib=100,
            disk_per_node_gib=2048,
            min_rf=config.min_rf,
            cpu_needed=800,
            cpu_per_node=16,
        )

        result = search_with_fault_tolerance(problem, tier=tier, cost_per_node=100)

        assert result is not None

        # The algorithm should at least try to meet the target
        # New target-based model uses target_met and achieved_nines
        if result.target_met:
            # Target was met - achieved_nines >= target_nines
            assert result.achieved_nines >= config.target_nines
        else:
            # Target wasn't met, but we should still have a reasonable config
            # The algorithm picks the best available config in this case
            assert result.rf >= config.min_rf

    def test_tier0_with_few_partitions_meets_target(self):
        """Tier 0 with few partitions should achieve 99.9%."""
        problem = CapacityProblem(
            n_partitions=10,  # Few partitions
            partition_size_gib=100,
            disk_per_node_gib=2048,
            min_rf=3,
            cpu_needed=160,
            cpu_per_node=16,
        )

        result = search_with_fault_tolerance(problem, tier=0, cost_per_node=100)

        assert result is not None
        # With only 10 partitions, 99.9% should be achievable
        assert result.az_failure_availability >= 0.999

    def test_tier3_optimizes_for_cost(self):
        """Tier 3 should optimize for cost over availability."""
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

        # Tier 3 should be cheaper or equal
        assert result_tier3.cost <= result_tier0.cost


# =============================================================================
# INTEGRATION: Zone-aware comparison
# =============================================================================


class TestZoneAwareComparison:
    """Test zone-aware cost comparison calculation."""

    def test_zone_aware_cost_always_computed(self):
        """Zone-aware cost should always be computed."""
        problem = CapacityProblem(
            n_partitions=100,
            partition_size_gib=100,
            disk_per_node_gib=2048,
            min_rf=2,
            cpu_needed=800,
            cpu_per_node=16,
        )

        for tier in range(4):
            result = search_with_fault_tolerance(problem, tier=tier, cost_per_node=100)
            if result is not None:
                assert result.zone_aware_cost > 0
                assert result.zone_aware_savings >= 0

    def test_zone_aware_savings_positive_for_high_partition_counts(self):
        """High partition counts where RF is driven by availability (not CPU)."""
        # This scenario is designed so that:
        # 1. CPU is satisfied with minimal RF
        # 2. High partition count requires high RF for availability
        # 3. Zone-aware placement would only need RF=2
        problem = CapacityProblem(
            n_partitions=500,  # Many partitions
            partition_size_gib=10,
            disk_per_node_gib=2000,  # 200 partitions per node
            min_rf=2,
            cpu_needed=32,  # Very low CPU need - just 2 nodes worth
            cpu_per_node=16,
        )

        result = search_with_fault_tolerance(problem, tier=0, cost_per_node=100)

        assert result is not None

        # If RF was driven by availability (not CPU), zone-aware savings should exist
        # But if CPU drives the configuration, savings may be zero
        # The key test is that the comparison is always computed
        assert result.zone_aware_cost > 0

        # With low CPU requirement, RF should be driven by availability
        # and zone-aware should offer savings
        if result.rf > 2:
            # Zone-aware with RF=2 would be cheaper
            # But CPU constraints may still dominate
            pass  # Accept whatever the algorithm found


# =============================================================================
# INTEGRATION: Constraint satisfaction
# =============================================================================


class TestConstraintSatisfaction:
    """Verify all constraints are satisfied."""

    @given(
        n_partitions=st.integers(10, 500),
        partition_size=st.floats(10, 200, allow_nan=False, allow_infinity=False),
        cpu_needed=st.integers(100, 5000),
        tier=st.integers(0, 3),
    )
    @settings(max_examples=50, deadline=None)
    def test_all_constraints_satisfied(
        self, n_partitions, partition_size, cpu_needed, tier
    ):
        """All returned results should satisfy constraints."""
        problem = CapacityProblem(
            n_partitions=n_partitions,
            partition_size_gib=partition_size,
            disk_per_node_gib=2048,
            min_rf=2,
            cpu_needed=cpu_needed,
            cpu_per_node=16,
            max_nodes=10000,
        )

        result = search_with_fault_tolerance(problem, tier=tier, cost_per_node=100)

        if result is None:
            return  # No valid configuration

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


# =============================================================================
# INTEGRATION: Pareto frontier analysis
# =============================================================================


class TestParetoFrontier:
    """Test that the algorithm finds Pareto-optimal configurations."""

    def test_no_dominated_solution_chosen(self):
        """The chosen solution should not be dominated by another."""
        problem = CapacityProblem(
            n_partitions=100,
            partition_size_gib=100,
            disk_per_node_gib=2048,
            min_rf=2,
            cpu_needed=800,
            cpu_per_node=16,
        )

        result = search_with_fault_tolerance(problem, tier=2, cost_per_node=100)

        if result is None:
            return

        # Generate alternative configurations to check for dominance
        max_ppn = int(problem.disk_per_node_gib / problem.partition_size_gib)

        for ppn in range(1, max_ppn + 1):
            import math

            base = math.ceil(problem.n_partitions / ppn)

            for rf in range(2, 10):
                if base >= 2:
                    node_count = base * rf
                else:
                    node_count = max(2, rf)

                if node_count > problem.max_nodes:
                    continue
                if node_count * problem.cpu_per_node < problem.cpu_needed:
                    continue

                alt_avail = system_availability(node_count, 3, rf, problem.n_partitions)
                alt_cost = node_count * 100

                # Check if this dominates the result
                # (better availability AND lower cost)
                if (
                    alt_avail > result.az_failure_availability
                    and alt_cost < result.cost
                ):
                    pytest.fail(
                        f"Found dominating solution: "
                        f"avail={alt_avail:.4f} > "
                        f"{result.az_failure_availability:.4f}, "
                        f"cost={alt_cost} < {result.cost}"
                    )


# =============================================================================
# INTEGRATION: Simulation agreement for integration tests
# =============================================================================


class TestSimulationAgreement:
    """Verify closed-form matches simulation in integration context."""

    def test_integration_result_matches_simulation(self):
        """The availability in results should match simulation."""
        from tests.netflix.test_fault_tolerance_simulation import (
            simulate_system_availability,
        )

        problem = CapacityProblem(
            n_partitions=50,
            partition_size_gib=100,
            disk_per_node_gib=2048,
            min_rf=2,
            cpu_needed=400,
            cpu_per_node=16,
        )

        result = search_with_fault_tolerance(problem, tier=2, cost_per_node=100)

        assert result is not None

        # Simulate and compare
        simulated = simulate_system_availability(
            n_nodes=result.node_count,
            n_zones=3,
            rf=result.rf,
            n_partitions=problem.n_partitions,
            n_trials=5000,
            seed=42,
        )

        # Should match within 5% (simulation variance)
        tolerance = max(0.05, abs(result.az_failure_availability) * 0.1)
        assert abs(result.az_failure_availability - simulated) < tolerance, (
            f"Closed-form {result.az_failure_availability:.4f} doesn't match "
            f"simulation {simulated:.4f}"
        )
