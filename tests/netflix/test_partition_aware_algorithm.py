"""
Tests for the Partition-Aware Capacity Planning Algorithm.

These tests verify the core algorithm behavior, especially minimizing node count
while preferring higher replication factors (RF) for cost-equivalent plans.
"""

import math

from hypothesis import given, settings
from hypothesis import strategies as st
import pytest

from service_capacity_modeling.models.org.netflix.partition_aware_algorithm import (
    CapacityProblem,
    search_for_min_nodes,
)


class TestAlgorithmBasics:
    """Basic functionality tests for the algorithm."""

    def test_returns_none_when_partition_too_large(self):
        """Algorithm returns None when a single partition doesn't fit on disk."""
        problem = CapacityProblem(
            n_partitions=10,
            partition_size_gib=1000,  # 1TB partition
            disk_per_node_gib=500,  # Only 500GB disk
            cpu_per_node=16,
            cpu_needed=32,
            min_rf=2,
            max_nodes=100,
        )
        result = search_for_min_nodes(problem)
        assert result is None

    def test_returns_none_when_exceeds_max_nodes(self):
        """Algorithm returns None when no configuration fits within max_nodes."""
        problem = CapacityProblem(
            n_partitions=100,
            partition_size_gib=500,
            disk_per_node_gib=1000,  # 2 partitions per node
            cpu_per_node=8,
            cpu_needed=1000,  # Needs many nodes for CPU
            min_rf=2,
            max_nodes=10,  # Very restrictive
        )
        result = search_for_min_nodes(problem)
        assert result is None

    def test_simple_case_returns_valid_result(self):
        """Algorithm returns a valid result for a simple case."""
        problem = CapacityProblem(
            n_partitions=10,
            partition_size_gib=100,
            disk_per_node_gib=500,  # 5 partitions per node
            cpu_per_node=16,
            cpu_needed=32,  # 2 nodes worth of CPU
            min_rf=2,
            max_nodes=100,
        )
        result = search_for_min_nodes(problem)

        assert result is not None
        assert result.node_count <= problem.max_nodes
        assert result.replica_count >= problem.min_rf

    def test_memory_limits_partitions_per_node_independently_from_disk(self):
        """Memory can limit partition placement while disk remains unchanged."""
        problem = CapacityProblem(
            n_partitions=10,
            partition_size_gib=100,
            disk_per_node_gib=500,
            memory_per_partition_gib=25,
            memory_per_node_gib=50,
            cpu_per_node=16,
            cpu_needed=32,
            min_rf=2,
            max_nodes=100,
        )

        result = search_for_min_nodes(problem)

        assert result is not None
        assert result.partitions_per_node == 2
        assert result.nodes_for_one_copy == 5

    def test_returns_none_when_partition_exceeds_node_memory(self):
        """Algorithm rejects a partition that cannot fit in node memory."""
        problem = CapacityProblem(
            n_partitions=10,
            partition_size_gib=100,
            disk_per_node_gib=500,
            memory_per_partition_gib=60,
            memory_per_node_gib=50,
            cpu_per_node=16,
            cpu_needed=32,
            min_rf=2,
            max_nodes=100,
        )

        assert search_for_min_nodes(problem) is None

    def test_memory_driven_nodes_reduce_cpu_replica_count(self):
        """Memory-driven nodes replace replicas that were needed only for CPU."""
        problem = CapacityProblem(
            n_partitions=8,
            partition_size_gib=100,
            disk_per_node_gib=500,
            memory_per_partition_gib=25,
            memory_per_node_gib=50,
            cpu_per_node=1,
            cpu_needed=12,
            min_rf=2,
            max_nodes=100,
        )

        result = search_for_min_nodes(problem)

        assert result is not None
        assert result.nodes_for_one_copy == 4
        assert result.replica_count == 3
        assert result.node_count == 12

    def test_requires_complete_memory_constraint(self):
        """Memory placement requires both the partition need and node capacity."""
        with pytest.raises(ValueError, match="must be set together"):
            CapacityProblem(
                n_partitions=10,
                partition_size_gib=100,
                disk_per_node_gib=500,
                memory_per_partition_gib=25,
                cpu_per_node=16,
                cpu_needed=32,
                min_rf=2,
                max_nodes=100,
            )

    def test_tiny_partitions_preserve_maximum_packing(self):
        """Packing above the partition count is one equivalent search candidate."""
        problem = CapacityProblem(
            n_partitions=12,
            partition_size_gib=0.1,
            disk_per_node_gib=14_000,
            cpu_per_node=16,
            cpu_needed=16,
            min_rf=2,
            max_nodes=100,
        )

        result = search_for_min_nodes(problem)

        assert result is not None
        assert result.node_count == 2
        assert result.nodes_for_one_copy == 1
        assert result.partitions_per_node == 140_000


class TestPlanSelection:
    """Tests that verify cost and replication factor plan selection."""

    def test_chooses_max_ppn_when_it_minimizes_nodes(self):
        """Dense packing minimizes nodes when the minimum RF dominates."""
        problem = CapacityProblem(
            n_partitions=100,
            partition_size_gib=100,
            disk_per_node_gib=1000,  # max_ppn = 10
            cpu_per_node=16,
            cpu_needed=64,  # 4 nodes worth
            min_rf=2,
            max_nodes=1000,  # Relaxed constraint
        )
        result = search_for_min_nodes(problem)

        assert result is not None
        # With max_ppn=10, base=ceil(100/10)=10, cpu_per_copy=160 >= 64
        # So RF=min_rf=2, which fits easily
        assert result.partitions_per_node == 10  # Max PPn
        assert result.nodes_for_one_copy == 10
        assert result.replica_count == 2

    def test_prefers_fewer_nodes_over_higher_rf(self):
        """For CPU-constrained workloads, prefer fewer nodes over higher RF.

        Example: 200 partitions, 575 GiB each, 2048 GiB disk, need 3200 cores
        - PPn=3: base=67, needs RF=3 for CPU → 201 nodes
        - PPn=2: base=100, needs RF=2 for CPU → 200 nodes

        The lower RF plan saves one node while satisfying the minimum RF.
        """
        problem = CapacityProblem(
            n_partitions=200,
            partition_size_gib=575,
            disk_per_node_gib=2048,  # max_ppn = 3
            cpu_per_node=16,
            cpu_needed=3200,
            min_rf=2,
            max_nodes=10000,
        )
        result = search_for_min_nodes(problem)

        assert result is not None
        assert result.partitions_per_node == 2
        assert result.replica_count == 2
        assert result.node_count == 200

    def test_returns_none_when_every_ppn_exceeds_limit(self):
        """Algorithm returns None when no partition density fits the node limit."""
        problem = CapacityProblem(
            n_partitions=21,
            partition_size_gib=200,
            disk_per_node_gib=2000,  # max_ppn = 10
            cpu_per_node=8,
            cpu_needed=160,  # Need RF=4 at PPn=10
            min_rf=2,
            max_nodes=10,  # Restrictive
        )
        # PPn=10: base=3, cpu_per_copy=24, needs RF=ceil(160/24)=7, nodes=21 > 10 ❌
        # PPn=5: base=5, cpu_per_copy=40, needs RF=4, nodes=20 > 10 ❌
        # PPn=4: base=6, cpu_per_copy=48, needs RF=4, nodes=24 > 10 ❌
        # PPn=3: base=7, cpu_per_copy=56, needs RF=3, nodes=21 > 10 ❌
        # PPn=2: base=11, cpu_per_copy=88, needs RF=2, nodes=22 > 10 ❌

        result = search_for_min_nodes(problem)
        # All configurations exceed max_nodes
        assert result is None

    def test_selects_cheapest_valid_configuration(self):
        """Algorithm evaluates valid configurations instead of returning the first."""
        problem = CapacityProblem(
            n_partitions=100,
            partition_size_gib=100,
            disk_per_node_gib=500,  # max_ppn = 5
            cpu_per_node=16,
            cpu_needed=800,  # Need 50 nodes
            min_rf=2,
            max_nodes=60,
        )
        # PPn=5: base=20, cpu_per_copy=320, needs RF=3, nodes=60
        # PPn=4: base=25, cpu_per_copy=400, needs RF=2, nodes=50

        result = search_for_min_nodes(problem)

        assert result is not None
        assert result.partitions_per_node == 4
        assert result.replica_count == 2
        assert result.node_count == 50

    def test_prefers_higher_rf_for_equal_node_count(self):
        """Prefer higher RF when two configurations use the same node count."""
        problem = CapacityProblem(
            n_partitions=30,
            partition_size_gib=100,
            disk_per_node_gib=1500,  # max_ppn = 15
            cpu_per_node=16,
            cpu_needed=80,  # 5 nodes worth
            min_rf=2,
            max_nodes=100,
        )
        # PPn=15: base=2, needs RF=3, nodes=6
        # PPn=10: base=3, needs RF=2, nodes=6

        result = search_for_min_nodes(problem)

        assert result is not None
        assert result.node_count == 6
        assert result.partitions_per_node == 15
        assert result.replica_count == 3


class TestAlgorithmProperties:
    """Property-based tests using Hypothesis."""

    @staticmethod
    def valid_problems():
        """Generate valid capacity problems."""
        return st.builds(
            CapacityProblem,
            n_partitions=st.integers(min_value=1, max_value=1000),
            partition_size_gib=st.floats(min_value=1, max_value=500),
            disk_per_node_gib=st.floats(min_value=100, max_value=4000),
            cpu_per_node=st.integers(min_value=2, max_value=128),
            cpu_needed=st.integers(min_value=1, max_value=10000),
            min_rf=st.integers(min_value=1, max_value=5),
            max_nodes=st.integers(min_value=2, max_value=1000),
        ).filter(lambda p: p.disk_per_node_gib >= p.partition_size_gib)

    @given(problem=valid_problems())
    @settings(max_examples=500, deadline=None)
    def test_result_satisfies_all_constraints(self, problem: CapacityProblem):
        """PROPERTY: Any result returned satisfies all constraints."""
        result = search_for_min_nodes(problem)
        if result is None:
            return

        # Node count within limit
        assert result.node_count <= problem.max_nodes

        # RF at least min_rf
        assert result.replica_count >= problem.min_rf

        # PPn is valid
        max_ppn = int(problem.disk_per_node_gib / problem.partition_size_gib)
        assert 1 <= result.partitions_per_node <= max_ppn

        # CPU constraint satisfied
        total_cpu = result.node_count * problem.cpu_per_node
        assert total_cpu >= problem.cpu_needed

    @given(problem=valid_problems())
    @settings(max_examples=500, deadline=None)
    def test_result_is_best_valid_configuration(self, problem: CapacityProblem):
        """PROPERTY: No valid configuration has a better selection rank."""
        result = search_for_min_nodes(problem)
        if result is None:
            return

        max_ppn = int(problem.disk_per_node_gib / problem.partition_size_gib)

        result_rank = (
            result.node_count,
            -result.replica_count,
            -result.partitions_per_node,
        )
        for ppn in range(max_ppn, 0, -1):
            base = math.ceil(problem.n_partitions / ppn)
            rf = max(
                problem.min_rf,
                math.ceil(problem.cpu_needed / (base * problem.cpu_per_node)),
            )
            nodes = max(base * rf, 2)
            if nodes <= problem.max_nodes:
                assert result_rank <= (nodes, -rf, -ppn)

    @given(problem=valid_problems())
    @settings(max_examples=500, deadline=None)
    def test_node_count_formula_is_correct(self, problem: CapacityProblem):
        """PROPERTY: node_count = nodes_for_one_copy * replica_count."""
        result = search_for_min_nodes(problem)
        if result is None:
            return

        if result.nodes_for_one_copy >= 2:
            expected = result.nodes_for_one_copy * result.replica_count
            assert result.node_count == expected
        else:
            # Special case: base=1, node_count = max(2, rf)
            assert result.node_count == max(2, result.replica_count)


class TestEdgeCases:
    """Edge case tests."""

    def test_single_partition(self):
        """Algorithm handles single partition correctly.

        With 1 partition and max_ppn=5, algorithm starts from ppn=5 (max).
        base=ceil(1/5)=1, which triggers special case.
        """
        problem = CapacityProblem(
            n_partitions=1,
            partition_size_gib=100,
            disk_per_node_gib=500,  # max_ppn = 5
            cpu_per_node=16,
            cpu_needed=32,
            min_rf=2,
            max_nodes=100,
        )
        result = search_for_min_nodes(problem)

        assert result is not None
        # Equal-cost plans retain the densest partition packing.
        assert result.partitions_per_node == 5
        assert result.nodes_for_one_copy == 1
        assert result.replica_count == 2  # min_rf (2*16=32 >= 32 cpu_needed)
        assert result.node_count == 2  # max(2, rf)

    def test_min_rf_one(self):
        """Algorithm works with min_rf=1."""
        problem = CapacityProblem(
            n_partitions=10,
            partition_size_gib=100,
            disk_per_node_gib=1000,
            cpu_per_node=16,
            cpu_needed=16,  # 1 node worth
            min_rf=1,
            max_nodes=100,
        )
        result = search_for_min_nodes(problem)

        assert result is not None
        assert result.replica_count >= 1

    def test_exact_fit(self):
        """Algorithm handles exact fit scenarios."""
        problem = CapacityProblem(
            n_partitions=10,
            partition_size_gib=100,
            disk_per_node_gib=500,  # Exactly 5 partitions per node
            cpu_per_node=16,
            cpu_needed=32,
            min_rf=2,
            max_nodes=4,  # Exactly fits 2 nodes * 2 RF
        )
        result = search_for_min_nodes(problem)

        assert result is not None
        assert result.node_count == 4
        assert result.partitions_per_node == 5
        assert result.nodes_for_one_copy == 2
        assert result.replica_count == 2
