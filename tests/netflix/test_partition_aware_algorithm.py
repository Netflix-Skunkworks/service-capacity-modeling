"""
Tests for placing a partitioned dataset on nodes.

These tests verify the core algorithm behavior, especially minimizing node count
while preferring higher replication factors (RF) for cost-equivalent plans.
"""

import math
from typing import Optional
from typing import Tuple

from hypothesis import given, settings
from hypothesis import strategies as st

from service_capacity_modeling.interface import Bottleneck
from service_capacity_modeling.models.org.netflix.partition_aware_algorithm import (
    Layout,
    NodeSupply,
    PartitionedDemand,
    place,
    Placement,
    Shortfall,
)

Problem = Tuple[PartitionedDemand, NodeSupply]


def _fewest(demand: PartitionedDemand, node: NodeSupply) -> Optional[Placement]:
    """The unpinned placement, or None when nothing fits."""
    placed = place(demand, node)
    return placed if isinstance(placed, Placement) else None


@st.composite
def valid_problems(draw):
    """Generate disk-only and memory-constrained capacity problems."""
    n_partitions = draw(st.integers(min_value=1, max_value=1000))
    partition_size_gib = draw(st.floats(min_value=1, max_value=500, allow_nan=False))
    disk_per_node_gib = partition_size_gib * draw(
        st.integers(min_value=1, max_value=40)
    )
    memory_per_partition_gib = None
    memory_per_node_gib = 0.0
    if draw(st.booleans()):
        memory_per_partition_gib = draw(
            st.floats(min_value=0.1, max_value=100, allow_nan=False)
        )
        memory_per_node_gib = memory_per_partition_gib * draw(
            st.integers(min_value=1, max_value=40)
        )

    cpu_per_node = draw(st.integers(min_value=2, max_value=128))
    cpu_needed = draw(st.integers(min_value=1, max_value=10000))
    min_rf = draw(st.integers(min_value=1, max_value=5))
    max_nodes = n_partitions * max(
        min_rf, math.ceil(cpu_needed / (n_partitions * cpu_per_node))
    )

    demand = PartitionedDemand(
        partitions=n_partitions,
        disk_gib_per_partition=partition_size_gib,
        memory_gib_per_partition=memory_per_partition_gib,
        cpu_cores=cpu_needed,
        min_replicas=min_rf,
        max_nodes=max_nodes,
    )
    node = NodeSupply(
        cpu_cores=cpu_per_node,
        disk_gib=disk_per_node_gib,
        memory_gib=memory_per_node_gib,
    )
    return demand, node


class TestAlgorithmBasics:
    """Basic functionality tests for the algorithm."""

    def test_returns_none_when_partition_too_large(self):
        """Algorithm returns None when a single partition doesn't fit on disk."""
        demand = PartitionedDemand(
            partitions=10,
            disk_gib_per_partition=1000,  # 1TB partition
            cpu_cores=32,
            min_replicas=2,
            max_nodes=100,
        )
        node = NodeSupply(
            disk_gib=500,  # Only 500GB disk
            cpu_cores=16,
        )
        result = _fewest(demand, node)
        assert result is None

    def test_returns_none_when_exceeds_max_nodes(self):
        """Algorithm returns None when no configuration fits within max_nodes."""
        demand = PartitionedDemand(
            partitions=100,
            disk_gib_per_partition=500,
            cpu_cores=1000,  # Needs many nodes for CPU
            min_replicas=2,
            max_nodes=10,  # Very restrictive
        )
        node = NodeSupply(
            disk_gib=1000,  # 2 partitions per node
            cpu_cores=8,
        )
        result = _fewest(demand, node)
        assert result is None

    def test_simple_case_returns_valid_result(self):
        """Algorithm returns a valid result for a simple case."""
        demand = PartitionedDemand(
            partitions=10,
            disk_gib_per_partition=100,
            cpu_cores=32,  # 2 nodes worth of CPU
            min_replicas=2,
            max_nodes=100,
        )
        node = NodeSupply(
            disk_gib=500,  # 5 partitions per node
            cpu_cores=16,
        )
        result = _fewest(demand, node)

        assert result is not None
        assert result.node_count <= demand.max_nodes
        assert result.replica_count >= demand.min_replicas

    def test_memory_limits_partitions_per_node_independently_from_disk(self):
        """Memory can limit partition placement while disk remains unchanged."""
        demand = PartitionedDemand(
            partitions=10,
            disk_gib_per_partition=100,
            memory_gib_per_partition=25,
            cpu_cores=32,
            min_replicas=2,
            max_nodes=100,
        )
        node = NodeSupply(
            disk_gib=500,
            memory_gib=50,
            cpu_cores=16,
        )

        result = _fewest(demand, node)

        assert result is not None
        assert result.partitions_per_node == 2
        assert result.nodes_for_one_copy == 5
        assert result.max_partitions_per_node_by_disk == 5
        assert result.max_partitions_per_node_by_memory == 2

    def test_exact_float_boundary_keeps_valid_partition_fit(self):
        demand = PartitionedDemand(
            partitions=188,
            disk_gib_per_partition=1,
            memory_gib_per_partition=53.39 / 47,
            cpu_cores=1,
            min_replicas=2,
            max_nodes=8,
        )
        node = NodeSupply(
            disk_gib=10_000,
            memory_gib=53.39,
            cpu_cores=16,
        )

        result = _fewest(demand, node)

        assert result is not None
        assert result.partitions_per_node == 47
        assert result.node_count == 8

    def test_returns_none_when_partition_exceeds_node_memory(self):
        """Algorithm rejects a partition that cannot fit in node memory."""
        demand = PartitionedDemand(
            partitions=10,
            disk_gib_per_partition=100,
            memory_gib_per_partition=60,
            cpu_cores=32,
            min_replicas=2,
            max_nodes=100,
        )
        node = NodeSupply(
            disk_gib=500,
            memory_gib=50,
            cpu_cores=16,
        )

        assert _fewest(demand, node) is None

    def test_memory_driven_nodes_reduce_cpu_replica_count(self):
        """Memory-driven nodes replace replicas that were needed only for CPU."""
        demand = PartitionedDemand(
            partitions=8,
            disk_gib_per_partition=100,
            memory_gib_per_partition=25,
            cpu_cores=12,
            min_replicas=2,
            max_nodes=100,
        )
        node = NodeSupply(
            disk_gib=500,
            memory_gib=50,
            cpu_cores=1,
        )

        result = _fewest(demand, node)

        assert result is not None
        assert result.nodes_for_one_copy == 4
        assert result.replica_count == 3
        assert result.node_count == 12

    def test_memory_demand_on_a_node_without_memory_names_memory(self):
        """A node that offers no memory cannot host memory-placed partitions."""
        demand = PartitionedDemand(
            partitions=10,
            disk_gib_per_partition=100,
            memory_gib_per_partition=25,
            cpu_cores=32,
            min_replicas=2,
            max_nodes=100,
        )
        node = NodeSupply(cpu_cores=16, disk_gib=500)

        placed = place(demand, node)

        assert isinstance(placed, Shortfall)
        assert placed.bottleneck == Bottleneck.memory

    def test_partitions_per_node_never_exceeds_the_partition_count(self):
        """A node with room for 140,000 tiny partitions still holds all 12."""
        demand = PartitionedDemand(
            partitions=12,
            disk_gib_per_partition=0.1,
            cpu_cores=16,
            min_replicas=2,
            max_nodes=100,
        )
        node = NodeSupply(
            disk_gib=14_000,
            cpu_cores=16,
        )

        result = _fewest(demand, node)

        assert result is not None
        assert result.node_count == 2
        assert result.nodes_for_one_copy == 1
        assert result.partitions_per_node == 12


class TestPlanSelection:
    """Tests that verify cost and replication factor plan selection."""

    def test_chooses_max_ppn_when_it_minimizes_nodes(self):
        """Dense packing minimizes nodes when the minimum RF dominates."""
        demand = PartitionedDemand(
            partitions=100,
            disk_gib_per_partition=100,
            cpu_cores=64,  # 4 nodes worth
            min_replicas=2,
            max_nodes=1000,  # Relaxed constraint
        )
        node = NodeSupply(
            disk_gib=1000,  # max_ppn = 10
            cpu_cores=16,
        )
        result = _fewest(demand, node)

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
        demand = PartitionedDemand(
            partitions=200,
            disk_gib_per_partition=575,
            cpu_cores=3200,
            min_replicas=2,
            max_nodes=10000,
        )
        node = NodeSupply(
            disk_gib=2048,  # max_ppn = 3
            cpu_cores=16,
        )
        result = _fewest(demand, node)

        assert result is not None
        assert result.partitions_per_node == 2
        assert result.replica_count == 2
        assert result.node_count == 200

    def test_returns_none_when_every_ppn_exceeds_limit(self):
        """Algorithm returns None when no partition density fits the node limit."""
        demand = PartitionedDemand(
            partitions=21,
            disk_gib_per_partition=200,
            cpu_cores=160,  # Need RF=4 at PPn=10
            min_replicas=2,
            max_nodes=10,  # Restrictive
        )
        node = NodeSupply(
            disk_gib=2000,  # max_ppn = 10
            cpu_cores=8,
        )
        # PPn=10: base=3, cpu_per_copy=24, needs RF=ceil(160/24)=7, nodes=21 > 10 ❌
        # PPn=5: base=5, cpu_per_copy=40, needs RF=4, nodes=20 > 10 ❌
        # PPn=4: base=6, cpu_per_copy=48, needs RF=4, nodes=24 > 10 ❌
        # PPn=3: base=7, cpu_per_copy=56, needs RF=3, nodes=21 > 10 ❌
        # PPn=2: base=11, cpu_per_copy=88, needs RF=2, nodes=22 > 10 ❌

        result = _fewest(demand, node)
        # All configurations exceed max_nodes
        assert result is None

    def test_selects_cheapest_valid_configuration(self):
        """Algorithm evaluates valid configurations instead of returning the first."""
        demand = PartitionedDemand(
            partitions=100,
            disk_gib_per_partition=100,
            cpu_cores=800,  # Need 50 nodes
            min_replicas=2,
            max_nodes=60,
        )
        node = NodeSupply(
            disk_gib=500,  # max_ppn = 5
            cpu_cores=16,
        )
        # PPn=5: base=20, cpu_per_copy=320, needs RF=3, nodes=60
        # PPn=4: base=25, cpu_per_copy=400, needs RF=2, nodes=50

        result = _fewest(demand, node)

        assert result is not None
        assert result.partitions_per_node == 4
        assert result.replica_count == 2
        assert result.node_count == 50

    def test_prefers_higher_rf_for_equal_node_count(self):
        """Prefer higher RF when two configurations use the same node count."""
        demand = PartitionedDemand(
            partitions=30,
            disk_gib_per_partition=100,
            cpu_cores=80,  # 5 nodes worth
            min_replicas=2,
            max_nodes=100,
        )
        node = NodeSupply(
            disk_gib=1500,  # max_ppn = 15
            cpu_cores=16,
        )
        # PPn=15: base=2, needs RF=3, nodes=6
        # PPn=10: base=3, needs RF=2, nodes=6

        result = _fewest(demand, node)

        assert result is not None
        assert result.node_count == 6
        assert result.partitions_per_node == 15
        assert result.replica_count == 3

    def test_skips_equivalent_partition_densities(self, monkeypatch):
        real_ceil = math.ceil
        ceil_calls = 0

        def counting_ceil(value):
            nonlocal ceil_calls
            ceil_calls += 1
            assert ceil_calls <= 5_000
            return real_ceil(value)

        monkeypatch.setattr(
            "service_capacity_modeling.models.org.netflix."
            "partition_aware_algorithm.math.ceil",
            counting_ceil,
        )
        demand = PartitionedDemand(
            partitions=1_000_000,
            disk_gib_per_partition=1,
            cpu_cores=10_000,
            min_replicas=2,
            max_nodes=1_000_000_000,
        )
        node = NodeSupply(
            disk_gib=100_000,
            cpu_cores=16,
        )

        result = _fewest(demand, node)

        assert result is not None
        assert result.node_count == 625
        assert result.replica_count == 25
        assert result.partitions_per_node == 41_666


class TestAlgorithmProperties:
    """Property-based tests using Hypothesis."""

    @given(problem=valid_problems())
    @settings(max_examples=500, deadline=None)
    def test_result_satisfies_all_constraints(self, problem: Problem):
        """PROPERTY: Any result returned satisfies all constraints."""
        demand, node = problem
        result = _fewest(demand, node)
        if result is None:
            return

        # Node count within limit
        assert result.node_count <= demand.max_nodes

        # RF at least min_rf
        assert result.replica_count >= demand.min_replicas

        # PPn is valid
        assert 1 <= result.partitions_per_node <= result.max_partitions_per_node_by_disk
        used_disk_gib = result.partitions_per_node * demand.disk_gib_per_partition
        assert used_disk_gib <= math.nextafter(node.disk_gib, math.inf)
        if demand.memory_gib_per_partition is not None:
            assert result.max_partitions_per_node_by_memory is not None
            assert (
                result.partitions_per_node <= result.max_partitions_per_node_by_memory
            )
            used_memory_gib = (
                result.partitions_per_node * demand.memory_gib_per_partition
            )
            assert used_memory_gib <= math.nextafter(node.memory_gib, math.inf)

        # CPU constraint satisfied
        total_cpu = result.node_count * node.cpu_cores
        assert total_cpu >= demand.cpu_cores

        # Reported per-node use fits what the node offers
        assert result.disk_gib_per_node <= math.nextafter(node.disk_gib, math.inf)
        assert result.memory_gib_per_node <= math.nextafter(node.memory_gib, math.inf)
        assert result.cpu_utilization <= 1

    @given(problem=valid_problems())
    @settings(max_examples=500, deadline=None)
    def test_result_is_best_valid_configuration(self, problem: Problem):
        """PROPERTY: No valid configuration has a better selection rank."""
        demand, node = problem
        result = _fewest(demand, node)
        if result is None:
            return

        disk_ppn = math.floor(
            math.nextafter(node.disk_gib / demand.disk_gib_per_partition, math.inf)
        )
        max_ppn = disk_ppn
        if demand.memory_gib_per_partition is not None:
            memory_ppn = math.floor(
                math.nextafter(
                    node.memory_gib / demand.memory_gib_per_partition,
                    math.inf,
                )
            )
            max_ppn = min(max_ppn, memory_ppn)

        result_rank = (
            result.node_count,
            -result.replica_count,
            -result.partitions_per_node,
        )
        for ppn in range(min(max_ppn, demand.partitions), 0, -1):
            base = math.ceil(demand.partitions / ppn)
            rf = max(
                demand.min_replicas,
                math.ceil(demand.cpu_cores / (base * node.cpu_cores)),
            )
            nodes = max(base * rf, 2)
            if nodes <= demand.max_nodes:
                assert result_rank <= (nodes, -rf, -ppn)

    @given(problem=valid_problems())
    @settings(max_examples=500, deadline=None)
    def test_node_count_formula_is_correct(self, problem: Problem):
        """PROPERTY: node_count = nodes_for_one_copy * replica_count."""
        demand, node = problem
        result = _fewest(demand, node)
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

        With 1 partition and max_ppn=5, the node holds the one partition.
        base=ceil(1/5)=1, which triggers special case.
        """
        demand = PartitionedDemand(
            partitions=1,
            disk_gib_per_partition=100,
            cpu_cores=32,
            min_replicas=2,
            max_nodes=100,
        )
        node = NodeSupply(
            disk_gib=500,  # max_ppn = 5
            cpu_cores=16,
        )
        result = _fewest(demand, node)

        assert result is not None
        assert result.partitions_per_node == 1
        assert result.nodes_for_one_copy == 1
        assert result.replica_count == 2  # min_rf (2*16=32 >= 32 cpu_needed)
        assert result.node_count == 2  # max(2, rf)

    def test_min_rf_one(self):
        """Algorithm works with min_rf=1."""
        demand = PartitionedDemand(
            partitions=10,
            disk_gib_per_partition=100,
            cpu_cores=16,  # 1 node worth
            min_replicas=1,
            max_nodes=100,
        )
        node = NodeSupply(
            disk_gib=1000,
            cpu_cores=16,
        )
        result = _fewest(demand, node)

        assert result is not None
        assert result.replica_count >= 1

    def test_exact_fit(self):
        """Algorithm handles exact fit scenarios."""
        demand = PartitionedDemand(
            partitions=10,
            disk_gib_per_partition=100,
            memory_gib_per_partition=10,
            cpu_cores=32,
            min_replicas=2,
            max_nodes=4,  # Exactly fits 2 nodes * 2 RF
        )
        node = NodeSupply(
            disk_gib=500,  # Exactly 5 partitions per node
            memory_gib=50,
            cpu_cores=16,
        )
        result = _fewest(demand, node)

        assert result is not None
        assert result.node_count == 4
        assert result.partitions_per_node == 5
        assert result.nodes_for_one_copy == 2
        assert result.replica_count == 2
        assert result.max_partitions_per_node_by_disk == 5
        assert result.max_partitions_per_node_by_memory == 5


class TestPins:
    """Placing at a pinned layout or node count."""

    DEMAND = PartitionedDemand(
        partitions=12,
        disk_gib_per_partition=100,
        cpu_cores=64,
        min_replicas=2,
        max_nodes=100,
    )

    def test_a_layout_that_fits_is_placed_as_pinned(self):
        node = NodeSupply(cpu_cores=16, disk_gib=400)
        layout = Layout(partitions_per_node=3, replica_count=2)

        placed = place(self.DEMAND, node, layout=layout)

        assert isinstance(placed, Placement)
        assert (placed.node_count, placed.partitions_per_node) == (8, 3)

    def test_a_layout_that_overfills_disk_names_disk(self):
        node = NodeSupply(cpu_cores=16, disk_gib=250)
        layout = Layout(partitions_per_node=3, replica_count=2)

        placed = place(self.DEMAND, node, layout=layout)

        assert isinstance(placed, Shortfall)
        assert placed.bottleneck == Bottleneck.disk_capacity

    def test_a_count_uses_the_densest_layout_on_exactly_that_many_nodes(self):
        # All 12 partitions on one node would not fit, so the densest split
        # of 6 nodes is 2 nodes per copy, 3 copies.
        node = NodeSupply(cpu_cores=16, disk_gib=700)

        placed = place(self.DEMAND, node, count=6)

        assert isinstance(placed, Placement)
        assert (placed.node_count, placed.partitions_per_node) == (6, 6)
        assert placed.replica_count == 3

    def test_a_count_too_small_for_the_cpu_names_cpu(self):
        node = NodeSupply(cpu_cores=16, disk_gib=1_200)

        placed = place(self.DEMAND, node, count=2)

        assert isinstance(placed, Shortfall)
        assert placed.bottleneck == Bottleneck.cpu


def test_a_cpu_need_on_a_whole_number_of_cores_adds_no_node():
    """966 cores on nodes worth 14 each is exactly 69 nodes. Converting a
    node's cores between shapes can leave it an ulp short of 14, which a
    plain ceil turns into a 70th node."""
    node_cores = math.nextafter(14, 0)
    demand = PartitionedDemand(
        partitions=1,
        disk_gib_per_partition=1,
        cpu_cores=966,
        min_replicas=1,
        max_nodes=1_000,
    )

    placed = place(demand, NodeSupply(cpu_cores=node_cores, disk_gib=100))

    assert isinstance(placed, Placement)
    assert placed.node_count == 69
