"""
Monte Carlo simulation to validate closed-form fault tolerance math.

This test file follows the simulation-first approach: we PROVE the closed-form
math is correct by comparing it against Monte Carlo simulation. If they disagree,
we know something is wrong.

The closed-form uses:
    P(partition unavailable) = C(nodes_per_zone, rf) / C(n_nodes, rf)
    P(system unavailable) = 1 - (1 - p)^n_partitions

This simulation empirically computes the same probabilities and verifies they match.
"""

import random
from typing import List, Set

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from service_capacity_modeling.models.org.netflix.partition_capacity import (
    per_partition_unavailability,
    system_availability,
    system_unavailability,
)


def simulate_system_availability(  # noqa: C901  # pylint: disable=R0917
    n_nodes: int,
    n_zones: int,
    rf: int,
    n_partitions: int,
    n_trials: int = 10000,
    seed: int = None,
) -> float:
    """
    Monte Carlo simulation of system availability under AZ failure.

    Algorithm:
    1. Distribute n_nodes uniformly across n_zones
    2. For each partition, randomly select RF nodes as replicas
    3. For each of n_zones (one failing at a time):
       - Check if ANY partition has all replicas in failed zone
       - Record unavailability
    4. Average across all (trials × zones) scenarios

    Returns: P(system available | 1 random AZ fails)
    """
    if seed is not None:
        random.seed(seed)

    if n_nodes <= 0 or n_zones <= 0 or rf <= 0 or n_partitions <= 0:
        return 1.0

    if rf > n_nodes:
        return 0.0  # Can't even place replicas

    # Assign nodes to zones uniformly
    # nodes_per_zone[z] = count of nodes in zone z
    base_per_zone = n_nodes // n_zones
    remainder = n_nodes % n_zones
    nodes_per_zone = [
        base_per_zone + (1 if z < remainder else 0) for z in range(n_zones)
    ]

    # Create zone membership for each node: node_zones[i] = zone of node i
    node_zones: List[int] = []
    for zone, count in enumerate(nodes_per_zone):
        node_zones.extend([zone] * count)

    unavailable_count = 0
    total_scenarios = n_trials * n_zones

    for _ in range(n_trials):
        # For each partition, randomly assign RF replicas
        # partition_replica_zones[p] = set of zones containing partition p's replicas
        partition_replica_zones: List[Set[int]] = []
        for _ in range(n_partitions):
            replicas = random.sample(range(n_nodes), rf)
            replica_zones = {node_zones[r] for r in replicas}
            partition_replica_zones.append(replica_zones)

        # Simulate each AZ failing
        for failed_zone in range(n_zones):
            system_unavailable_flag = False
            for replica_zones in partition_replica_zones:
                if replica_zones == {failed_zone}:
                    # All replicas in failed zone - partition is down!
                    system_unavailable_flag = True
                    break
            if system_unavailable_flag:
                unavailable_count += 1

    return 1.0 - (unavailable_count / total_scenarios)


def simulate_per_partition_unavailability(
    n_nodes: int,
    n_zones: int,
    rf: int,
    n_trials: int = 100000,
    seed: int = None,
) -> float:
    """
    Simulate P(partition unavailable | 1 AZ fails) for a single partition.

    This simulates placing RF replicas randomly and checking if all land in
    the same zone (which would then fail).
    """
    if seed is not None:
        random.seed(seed)

    if n_nodes <= 0 or n_zones <= 0 or rf <= 0:
        return 0.0

    if rf > n_nodes:
        return 1.0

    # Assign nodes to zones
    base_per_zone = n_nodes // n_zones
    remainder = n_nodes % n_zones
    nodes_per_zone = [
        base_per_zone + (1 if z < remainder else 0) for z in range(n_zones)
    ]

    node_zones: List[int] = []
    for zone, count in enumerate(nodes_per_zone):
        node_zones.extend([zone] * count)

    unavailable_count = 0

    for _ in range(n_trials):
        replicas = random.sample(range(n_nodes), rf)
        replica_zones = {node_zones[r] for r in replicas}

        # All replicas in one zone? That zone could fail.
        if len(replica_zones) == 1:
            unavailable_count += 1

    # Probability all replicas in same zone (which could be the one that fails)
    # But we want P(unavailable | specific zone fails), which is:
    # P(all in zone z) for any specific z = unavailable_count / (n_trials * n_zones)
    # Actually: P(all in one zone) = unavailable_count / n_trials
    # P(that zone fails) = 1/n_zones
    # But since we pick a random zone to fail,
    # P = unavailable_count / n_trials / n_zones?
    #
    # No, think about it differently:
    # The closed form gives P(all RF in one specific zone | random placement)
    # which equals C(npz, rf) / C(n, rf).
    # When that zone fails, partition is unavailable.
    # The "per partition unavailability" is this same value.
    #
    # In simulation: we're computing P(all in same zone) which should equal
    # n_zones * C(npz, rf) / C(n, rf) (since any of n_zones could be "the one")
    #
    # So to match per_partition_unavailability, divide by n_zones:
    return unavailable_count / n_trials / n_zones


# =============================================================================
# TEST: Closed-form matches simulation
# =============================================================================


class TestSimulationMatchesClosedForm:
    """Validate closed-form math against Monte Carlo simulation."""

    @pytest.mark.parametrize(
        "n_nodes,n_zones,rf,n_partitions",
        [
            (12, 3, 2, 10),  # Basic case, even distribution
            (12, 3, 3, 50),  # Higher RF, more partitions
            (12, 3, 4, 100),  # Even higher RF
            (24, 3, 3, 200),  # Larger cluster
            (30, 3, 5, 100),  # RF > nodes_per_zone -> 100% available
            (9, 3, 2, 20),  # Small cluster, even distribution
            (15, 3, 3, 75),  # Even distribution (15/3=5)
            # Uneven distributions (critical for correctness)
            (7, 3, 2, 10),  # 7 nodes, 3 zones -> [3, 2, 2]
            (10, 3, 2, 20),  # 10 nodes, 3 zones -> [4, 3, 3]
            (11, 3, 3, 30),  # 11 nodes, 3 zones -> [4, 4, 3]
        ],
    )
    def test_system_availability_matches_simulation(
        self, n_nodes, n_zones, rf, n_partitions
    ):
        """Closed-form system availability should match simulation within 3%."""
        closed = system_availability(n_nodes, n_zones, rf, n_partitions)
        simulated = simulate_system_availability(
            n_nodes, n_zones, rf, n_partitions, n_trials=10000, seed=42
        )

        # Allow 3% tolerance due to simulation variance
        # For low availability values, use absolute tolerance
        tolerance = max(0.03, abs(closed) * 0.1)
        assert abs(closed - simulated) < tolerance, (
            f"System availability mismatch: "
            f"closed={closed:.4f}, simulated={simulated:.4f}\n"
            f"Params: n_nodes={n_nodes}, n_zones={n_zones}, "
            f"rf={rf}, n_partitions={n_partitions}"
        )

    @pytest.mark.parametrize(
        "n_nodes,n_zones,rf",
        [
            (12, 3, 2),  # Basic: 4 nodes/zone, RF=2
            (12, 3, 3),  # Higher RF
            (12, 3, 4),  # RF=nodes_per_zone
            (15, 3, 2),  # 5 nodes/zone
            (15, 3, 5),  # RF=nodes_per_zone
            (30, 3, 5),  # Larger cluster
            # Uneven distributions
            (7, 3, 2),  # [3, 2, 2] nodes per zone
            (10, 3, 2),  # [4, 3, 3] nodes per zone
            (11, 3, 3),  # [4, 4, 3] nodes per zone
        ],
    )
    def test_per_partition_unavailability_matches_simulation(
        self, n_nodes, n_zones, rf
    ):
        """Closed-form per-partition unavailability should match simulation."""
        closed = per_partition_unavailability(n_nodes, n_zones, rf)
        simulated = simulate_per_partition_unavailability(
            n_nodes, n_zones, rf, n_trials=100000, seed=42
        )

        # Tighter tolerance for this simpler calculation
        tolerance = max(0.015, abs(closed) * 0.15)
        assert abs(closed - simulated) < tolerance, (
            f"Per-partition unavailability mismatch: closed={closed:.4f}, "
            f"simulated={simulated:.4f}\n"
            f"Params: n_nodes={n_nodes}, n_zones={n_zones}, rf={rf}"
        )


# =============================================================================
# TEST: Property-based testing with Hypothesis
# =============================================================================
#
# NOTE ON TOOL CHOICE:
# - Monte Carlo (above): Validates closed-form math for SPECIFIC representative cases
# - Hypothesis (below): Tests PROPERTIES of closed-form functions across input space
#
# We do NOT use Hypothesis to generate inputs for Monte Carlo because:
# 1. It's slow (double randomization)
# 2. It's redundant (if Monte Carlo validates specific cases, the math is proven)
# 3. Hypothesis is for finding implementation bugs, not validating math
# =============================================================================


class TestFaultToleranceProperties:
    """Property tests for fault tolerance functions.

    These tests verify mathematical properties of the closed-form functions
    WITHOUT running Monte Carlo simulation. Hypothesis explores the input
    space to find edge cases in the implementation.
    """

    @given(
        rf=st.integers(2, 6),
        n_partitions=st.integers(10, 500),
    )
    @settings(max_examples=100, deadline=None)
    def test_higher_rf_gives_better_availability(self, rf, n_partitions):
        """Property: higher RF -> higher or equal availability."""
        n_nodes = 12
        n_zones = 3

        if rf > n_nodes or rf + 1 > n_nodes:
            return

        avail_low = system_availability(n_nodes, n_zones, rf, n_partitions)
        avail_high = system_availability(n_nodes, n_zones, rf + 1, n_partitions)

        assert avail_high >= avail_low - 1e-9, (
            f"Higher RF should give better availability: "
            f"RF={rf} gives {avail_low}, RF={rf + 1} gives {avail_high}"
        )

    @given(n_partitions=st.integers(1, 1000))
    @settings(max_examples=50, deadline=None)
    def test_zero_risk_rf_gives_perfect_availability(self, n_partitions):
        """Property: RF > nodes_per_zone -> 100% availability."""
        # 12 nodes, 3 zones -> 4 nodes/zone. RF=5 can't fit in one zone.
        avail = system_availability(12, 3, 5, n_partitions)
        assert avail == 1.0, (
            f"RF>nodes_per_zone should give 100% availability, got {avail}"
        )

    @given(
        n_partitions_low=st.integers(1, 100),
        n_partitions_high=st.integers(101, 500),
    )
    @settings(max_examples=50, deadline=None)
    def test_more_partitions_lower_availability(
        self, n_partitions_low, n_partitions_high
    ):
        """More partitions -> lower or equal availability."""
        n_nodes, n_zones, rf = 12, 3, 3

        avail_few = system_availability(n_nodes, n_zones, rf, n_partitions_low)
        avail_many = system_availability(n_nodes, n_zones, rf, n_partitions_high)

        assert avail_many <= avail_few + 1e-9, (
            f"More partitions should mean lower availability: "
            f"P={n_partitions_low} gives {avail_few}, "
            f"P={n_partitions_high} gives {avail_many}"
        )

    @given(
        n_nodes=st.integers(6, 100),
        n_zones=st.integers(1, 5),
        rf=st.integers(1, 10),
        n_partitions=st.integers(1, 1000),
    )
    @settings(max_examples=200, deadline=None)
    def test_availability_always_in_bounds(self, n_nodes, n_zones, rf, n_partitions):
        """Property: availability is always in [0, 1]."""
        if rf > n_nodes:
            return  # Invalid input

        avail = system_availability(n_nodes, n_zones, rf, n_partitions)
        assert 0.0 <= avail <= 1.0, f"Availability {avail} out of bounds"

        unavail = per_partition_unavailability(n_nodes, n_zones, rf)
        assert 0.0 <= unavail <= 1.0, f"Unavailability {unavail} out of bounds"

    @given(
        nodes_per_zone=st.integers(2, 10),
        n_zones=st.integers(2, 5),
        rf=st.integers(2, 5),
        n_partitions=st.integers(10, 200),
    )
    @settings(max_examples=100, deadline=None)
    def test_more_zones_better_availability(
        self, nodes_per_zone, n_zones, rf, n_partitions
    ):
        """Property: more zones (with fixed nodes/zone) -> better availability.

        Note: More TOTAL nodes doesn't always help! With more nodes per zone,
        there are more ways to place all replicas in the same zone.
        The correct property is: more ZONES (with fixed nodes/zone) helps.
        """
        n_nodes = nodes_per_zone * n_zones

        if rf > nodes_per_zone:
            return  # RF exceeds single zone - always 100% available

        avail_fewer_zones = system_availability(n_nodes, n_zones, rf, n_partitions)
        avail_more_zones = system_availability(
            nodes_per_zone * (n_zones + 1), n_zones + 1, rf, n_partitions
        )

        assert avail_more_zones >= avail_fewer_zones - 1e-9, (
            f"More zones should give better availability: "
            f"zones={n_zones} gives {avail_fewer_zones}, "
            f"zones={n_zones + 1} gives {avail_more_zones}"
        )


# =============================================================================
# TEST: Edge cases and boundary conditions
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_partition_single_zone(self):
        """Single partition, all nodes in one zone."""
        # If all nodes are in one zone, any RF still has all replicas there
        # Actually, per_partition_unavailability for n_zones=1 should be 1.0
        # because the only zone can fail
        unavail = per_partition_unavailability(10, 1, 3)
        assert unavail == 1.0, (
            f"Single zone should have 100% unavailability, got {unavail}"
        )

    def test_rf_equals_one(self):
        """RF=1 means no replication - always unavailable when AZ fails."""
        # With RF=1, single replica in one zone. If that zone fails, down.
        # For even distribution (12 nodes, 3 zones = 4 per zone):
        # P(single replica in any specific zone) = (nodes_in_zone)/n_nodes
        # Average across zones = sum(nodes_in_zone/n_nodes) / n_zones = 1/n_zones
        unavail = per_partition_unavailability(12, 3, 1)
        # P = (1/n_zones) * sum over z of: C(npz, 1)/C(n, 1) = 1/3
        expected = 1 / 3
        assert abs(unavail - expected) < 0.001, f"RF=1 unavail wrong: {unavail}"

    def test_rf_equals_nodes(self):
        """RF equals total nodes - replicas span all zones."""
        # RF=12, n_nodes=12, n_zones=3 -> RF=12 > 4 (nodes_per_zone)
        # So unavailability should be 0 (can't fit all in one zone)
        unavail = per_partition_unavailability(12, 3, 12)
        assert unavail == 0.0, f"RF=n_nodes should span all zones: {unavail}"

    def test_zero_partitions(self):
        """Zero partitions means nothing can fail."""
        # This is a degenerate case
        avail = system_availability(12, 3, 3, 0)
        # (1-p)^0 = 1 regardless of p
        assert avail == 1.0


# =============================================================================
# TEST: Validate specific known values (hand-calculated)
# =============================================================================


class TestKnownValues:
    """Test against hand-calculated values to catch formula errors."""

    def test_per_partition_12_nodes_3_zones_rf2(self):
        """Hand-calculated: C(4,2)/C(12,2) = 6/66 = 1/11."""
        expected = 6 / 66  # = 0.0909...
        actual = per_partition_unavailability(12, 3, 2)
        assert abs(actual - expected) < 0.0001, f"Expected {expected}, got {actual}"

    def test_per_partition_12_nodes_3_zones_rf3(self):
        """Hand-calculated: C(4,3)/C(12,3) = 4/220 = 1/55."""
        expected = 4 / 220  # = 0.0182...
        actual = per_partition_unavailability(12, 3, 3)
        assert abs(actual - expected) < 0.0001, f"Expected {expected}, got {actual}"

    def test_per_partition_12_nodes_3_zones_rf4(self):
        """Hand-calculated: C(4,4)/C(12,4) = 1/495."""
        expected = 1 / 495  # = 0.00202...
        actual = per_partition_unavailability(12, 3, 4)
        assert abs(actual - expected) < 0.0001, f"Expected {expected}, got {actual}"

    def test_system_unavailability_100_partitions_rf2(self):
        """For even distribution (12 nodes, 3 zones = 4 per zone):
        p_z = C(4,2)/C(12,2) = 6/66 = 1/11 for all zones
        sys_avail = (1-1/11)^100 for each zone, average = same
        sys_unavail = 1 - (1-1/11)^100
        """
        p = 1 / 11
        expected = 1 - (1 - p) ** 100  # ~0.99987
        actual = system_unavailability(12, 3, 2, 100)
        assert abs(actual - expected) < 0.0001, f"Expected {expected}, got {actual}"

    def test_system_availability_10_partitions_rf3(self):
        """For even distribution (12 nodes, 3 zones = 4 per zone):
        p_z = C(4,3)/C(12,3) = 4/220 = 1/55 for all zones
        sys_avail = (1-1/55)^10 for each zone, average = same
        """
        p = 1 / 55
        expected = (1 - p) ** 10  # ~0.834
        actual = system_availability(12, 3, 3, 10)
        assert abs(actual - expected) < 0.0001, f"Expected {expected}, got {actual}"
