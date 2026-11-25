"""
Tests for placing a partitioned requirement on nodes.

These tests verify the core algorithm behavior, especially minimizing node count
while preferring higher replication factors (RF) for cost-equivalent plans.
"""

import math
from typing import Optional
from typing import Tuple

from hypothesis import given, settings
from hypothesis import strategies as st

from service_capacity_modeling.interface import Bottleneck
from service_capacity_modeling.interface import default_reference_shape
from service_capacity_modeling.models.placement import CPURequirement
from service_capacity_modeling.models.placement import Layout
from service_capacity_modeling.models.placement import LimitNotMet
from service_capacity_modeling.models.placement import NodeCannotServe
from service_capacity_modeling.models.placement import NodeCPU
from service_capacity_modeling.models.placement import NodeDisk
from service_capacity_modeling.models.placement import NodeMemory
from service_capacity_modeling.models.placement import NodeResources
from service_capacity_modeling.models.placement import Partitions
from service_capacity_modeling.models.placement import place
from service_capacity_modeling.models.placement import Placement
from service_capacity_modeling.models.placement import Requirement

# (requirement, node, max_nodes)
Problem = Tuple[Requirement, NodeResources, int]


def _cpu(count: float) -> CPURequirement:
    """CPU in vCPUs of default_reference_shape, where one vCPU is one NCU."""
    return CPURequirement(count=count, reference_shape=default_reference_shape)


def _node(cpu: float, disk_gib: float = 0, memory_gib: float = 0) -> NodeResources:
    return NodeResources(
        shape=default_reference_shape,
        cpu=NodeCPU(count=cpu),
        memory=NodeMemory(gib=memory_gib),
        disk=NodeDisk(gib=disk_gib),
    )


def _fewest(
    requirement: Requirement, node: NodeResources, max_nodes: int
) -> Optional[Placement]:
    """The unpinned placement on at least 2 nodes, or None when nothing fits."""
    placed = place(requirement, node, min_count=2, max_count=max_nodes)
    return placed if isinstance(placed, Placement) else None


def _nodes_for_one_copy(requirement: Requirement, placed: Placement) -> int:
    assert requirement.partitions is not None and placed.layout is not None
    return math.ceil(requirement.partitions.count / placed.layout.partitions_per_node)


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

    requirement = Requirement(
        cpu=_cpu(cpu_needed),
        partitions=Partitions(
            count=n_partitions,
            disk_gib_per_partition=partition_size_gib,
            memory_gib_per_partition=memory_per_partition_gib,
            min_replicas=min_rf,
        ),
    )
    node = _node(
        cpu=cpu_per_node,
        disk_gib=disk_per_node_gib,
        memory_gib=memory_per_node_gib,
    )
    return requirement, node, max_nodes


class TestAlgorithmBasics:
    """Basic functionality tests for the algorithm."""

    def test_returns_none_when_partition_too_large(self):
        """Algorithm returns None when a single partition doesn't fit on disk."""
        max_nodes = 100
        requirement = Requirement(
            cpu=_cpu(32),
            partitions=Partitions(
                count=10,
                disk_gib_per_partition=1000,  # 1TB partition
                min_replicas=2,
            ),
        )
        node = _node(
            disk_gib=500,  # Only 500GB disk
            cpu=16,
        )
        result = _fewest(requirement, node, max_nodes)
        assert result is None

    def test_returns_none_when_exceeds_max_nodes(self):
        """Algorithm returns None when no configuration fits within max_nodes."""
        max_nodes = 10  # Very restrictive
        requirement = Requirement(
            cpu=_cpu(1000),  # Needs many nodes for CPU
            partitions=Partitions(
                count=100,
                disk_gib_per_partition=500,
                min_replicas=2,
            ),
        )
        node = _node(
            disk_gib=1000,  # 2 partitions per node
            cpu=8,
        )
        result = _fewest(requirement, node, max_nodes)
        assert result is None

    def test_simple_case_returns_valid_result(self):
        """Algorithm returns a valid result for a simple case."""
        max_nodes = 100
        requirement = Requirement(
            cpu=_cpu(32),  # 2 nodes worth of CPU
            partitions=Partitions(
                count=10,
                disk_gib_per_partition=100,
                min_replicas=2,
            ),
        )
        node = _node(
            disk_gib=500,  # 5 partitions per node
            cpu=16,
        )
        result = _fewest(requirement, node, max_nodes)

        assert result is not None
        assert result.node_count <= max_nodes
        assert result.layout.replica_count >= requirement.partitions.min_replicas

    def test_memory_limits_partitions_per_node_independently_from_disk(self):
        """Memory can limit partition placement while disk remains unchanged."""
        max_nodes = 100
        requirement = Requirement(
            cpu=_cpu(32),
            partitions=Partitions(
                count=10,
                disk_gib_per_partition=100,
                memory_gib_per_partition=25,
                min_replicas=2,
            ),
        )
        node = _node(
            disk_gib=500,
            memory_gib=50,
            cpu=16,
        )

        result = _fewest(requirement, node, max_nodes)

        assert result is not None
        assert result.layout.partitions_per_node == 2
        assert _nodes_for_one_copy(requirement, result) == 5

    def test_exact_float_boundary_keeps_valid_partition_fit(self):
        max_nodes = 8
        requirement = Requirement(
            cpu=_cpu(1),
            partitions=Partitions(
                count=188,
                disk_gib_per_partition=1,
                memory_gib_per_partition=53.39 / 47,
                min_replicas=2,
            ),
        )
        node = _node(
            disk_gib=10_000,
            memory_gib=53.39,
            cpu=16,
        )

        result = _fewest(requirement, node, max_nodes)

        assert result is not None
        assert result.layout.partitions_per_node == 47
        assert result.node_count == 8

    def test_returns_none_when_partition_exceeds_node_memory(self):
        """Algorithm rejects a partition that cannot fit in node memory."""
        max_nodes = 100
        requirement = Requirement(
            cpu=_cpu(32),
            partitions=Partitions(
                count=10,
                disk_gib_per_partition=100,
                memory_gib_per_partition=60,
                min_replicas=2,
            ),
        )
        node = _node(
            disk_gib=500,
            memory_gib=50,
            cpu=16,
        )

        assert _fewest(requirement, node, max_nodes) is None

    def test_memory_driven_nodes_reduce_cpu_replica_count(self):
        """Memory-driven nodes replace replicas that were needed only for CPU."""
        max_nodes = 100
        requirement = Requirement(
            cpu=_cpu(12),
            partitions=Partitions(
                count=8,
                disk_gib_per_partition=100,
                memory_gib_per_partition=25,
                min_replicas=2,
            ),
        )
        node = _node(
            disk_gib=500,
            memory_gib=50,
            cpu=1,
        )

        result = _fewest(requirement, node, max_nodes)

        assert result is not None
        assert _nodes_for_one_copy(requirement, result) == 4
        assert result.layout.replica_count == 3
        assert result.node_count == 12

    def test_memory_demand_on_a_node_without_memory_names_memory(self):
        """A node that offers no memory cannot host memory-placed partitions."""
        max_nodes = 100
        requirement = Requirement(
            cpu=_cpu(32),
            partitions=Partitions(
                count=10,
                disk_gib_per_partition=100,
                memory_gib_per_partition=25,
                min_replicas=2,
            ),
        )
        node = _node(cpu=16, disk_gib=500)

        placed = place(requirement, node, min_count=2, max_count=max_nodes)

        assert isinstance(placed, NodeCannotServe)
        assert placed.bottlenecks == [Bottleneck.memory]

    def test_a_partition_too_big_for_disk_and_memory_names_both(self):
        max_nodes = 100
        requirement = Requirement(
            cpu=_cpu(32),
            partitions=Partitions(
                count=10,
                disk_gib_per_partition=100,
                memory_gib_per_partition=25,
                min_replicas=2,
            ),
        )
        node = _node(cpu=16, disk_gib=50, memory_gib=10)

        placed = place(requirement, node, min_count=2, max_count=max_nodes)

        assert isinstance(placed, NodeCannotServe)
        assert placed.bottlenecks == [Bottleneck.disk_capacity, Bottleneck.memory]

    def test_partitions_per_node_never_exceeds_the_partition_count(self):
        """A node with room for 140,000 tiny partitions still holds all 12."""
        max_nodes = 100
        requirement = Requirement(
            cpu=_cpu(16),
            partitions=Partitions(
                count=12,
                disk_gib_per_partition=0.1,
                min_replicas=2,
            ),
        )
        node = _node(
            disk_gib=14_000,
            cpu=16,
        )

        result = _fewest(requirement, node, max_nodes)

        assert result is not None
        assert result.node_count == 2
        assert _nodes_for_one_copy(requirement, result) == 1
        assert result.layout.partitions_per_node == 12


class TestPlanSelection:
    """Tests that verify cost and replication factor plan selection."""

    def test_chooses_max_ppn_when_it_minimizes_nodes(self):
        """Dense packing minimizes nodes when the minimum RF dominates."""
        max_nodes = 1000  # Relaxed constraint
        requirement = Requirement(
            cpu=_cpu(64),  # 4 nodes worth
            partitions=Partitions(
                count=100,
                disk_gib_per_partition=100,
                min_replicas=2,
            ),
        )
        node = _node(
            disk_gib=1000,  # max_ppn = 10
            cpu=16,
        )
        result = _fewest(requirement, node, max_nodes)

        assert result is not None
        # With max_ppn=10, base=ceil(100/10)=10, cpu_per_copy=160 >= 64
        # So RF=min_rf=2, which fits easily
        assert result.layout.partitions_per_node == 10  # Max PPn
        assert _nodes_for_one_copy(requirement, result) == 10
        assert result.layout.replica_count == 2

    def test_prefers_fewer_nodes_over_higher_rf(self):
        """For CPU-constrained workloads, prefer fewer nodes over higher RF.

        Example: 200 partitions, 575 GiB each, 2048 GiB disk, need 3200 cores
        - PPn=3: base=67, needs RF=3 for CPU → 201 nodes
        - PPn=2: base=100, needs RF=2 for CPU → 200 nodes

        The lower RF plan saves one node while satisfying the minimum RF.
        """
        max_nodes = 10000
        requirement = Requirement(
            cpu=_cpu(3200),
            partitions=Partitions(
                count=200,
                disk_gib_per_partition=575,
                min_replicas=2,
            ),
        )
        node = _node(
            disk_gib=2048,  # max_ppn = 3
            cpu=16,
        )
        result = _fewest(requirement, node, max_nodes)

        assert result is not None
        assert result.layout.partitions_per_node == 2
        assert result.layout.replica_count == 2
        assert result.node_count == 200

    def test_returns_none_when_every_ppn_exceeds_limit(self):
        """Algorithm returns None when no partition density fits the node limit."""
        max_nodes = 10  # Restrictive
        requirement = Requirement(
            cpu=_cpu(160),  # Need RF=4 at PPn=10
            partitions=Partitions(
                count=21,
                disk_gib_per_partition=200,
                min_replicas=2,
            ),
        )
        node = _node(
            disk_gib=2000,  # max_ppn = 10
            cpu=8,
        )
        # PPn=10: base=3, cpu_per_copy=24, needs RF=ceil(160/24)=7, nodes=21 > 10 ❌
        # PPn=5: base=5, cpu_per_copy=40, needs RF=4, nodes=20 > 10 ❌
        # PPn=4: base=6, cpu_per_copy=48, needs RF=4, nodes=24 > 10 ❌
        # PPn=3: base=7, cpu_per_copy=56, needs RF=3, nodes=21 > 10 ❌
        # PPn=2: base=11, cpu_per_copy=88, needs RF=2, nodes=22 > 10 ❌

        result = _fewest(requirement, node, max_nodes)
        # All configurations exceed max_nodes
        assert result is None

    def test_selects_cheapest_valid_configuration(self):
        """Algorithm evaluates valid configurations instead of returning the first."""
        max_nodes = 60
        requirement = Requirement(
            cpu=_cpu(800),  # Need 50 nodes
            partitions=Partitions(
                count=100,
                disk_gib_per_partition=100,
                min_replicas=2,
            ),
        )
        node = _node(
            disk_gib=500,  # max_ppn = 5
            cpu=16,
        )
        # PPn=5: base=20, cpu_per_copy=320, needs RF=3, nodes=60
        # PPn=4: base=25, cpu_per_copy=400, needs RF=2, nodes=50

        result = _fewest(requirement, node, max_nodes)

        assert result is not None
        assert result.layout.partitions_per_node == 4
        assert result.layout.replica_count == 2
        assert result.node_count == 50

    def test_prefers_higher_rf_for_equal_node_count(self):
        """Prefer higher RF when two configurations use the same node count."""
        max_nodes = 100
        requirement = Requirement(
            cpu=_cpu(80),  # 5 nodes worth
            partitions=Partitions(
                count=30,
                disk_gib_per_partition=100,
                min_replicas=2,
            ),
        )
        node = _node(
            disk_gib=1500,  # max_ppn = 15
            cpu=16,
        )
        # PPn=15: base=2, needs RF=3, nodes=6
        # PPn=10: base=3, needs RF=2, nodes=6

        result = _fewest(requirement, node, max_nodes)

        assert result is not None
        assert result.node_count == 6
        assert result.layout.partitions_per_node == 15
        assert result.layout.replica_count == 3

    def test_skips_equivalent_partition_densities(self, monkeypatch):
        real_ceil = math.ceil
        ceil_calls = 0

        def counting_ceil(value):
            nonlocal ceil_calls
            ceil_calls += 1
            assert ceil_calls <= 5_000
            return real_ceil(value)

        monkeypatch.setattr(
            "service_capacity_modeling.models.placement.math.ceil",
            counting_ceil,
        )
        max_nodes = 1_000_000_000
        requirement = Requirement(
            cpu=_cpu(10_000),
            partitions=Partitions(
                count=1_000_000,
                disk_gib_per_partition=1,
                min_replicas=2,
            ),
        )
        node = _node(
            disk_gib=100_000,
            cpu=16,
        )

        result = _fewest(requirement, node, max_nodes)

        assert result is not None
        assert result.node_count == 625
        assert result.layout.replica_count == 25
        assert result.layout.partitions_per_node == 41_666


class TestAlgorithmProperties:
    """Property-based tests using Hypothesis."""

    @given(problem=valid_problems())
    @settings(max_examples=500, deadline=None)
    def test_result_satisfies_all_constraints(self, problem: Problem):
        """PROPERTY: Any result returned satisfies all constraints."""
        requirement, node, max_nodes = problem
        result = _fewest(requirement, node, max_nodes)
        if result is None:
            return

        # Node count within limit
        assert result.node_count <= max_nodes

        # RF at least min_rf, and at least one partition per node
        assert result.layout.partitions_per_node >= 1
        assert result.layout.replica_count >= requirement.partitions.min_replicas

        used_disk_gib = (
            result.layout.partitions_per_node
            * requirement.partitions.disk_gib_per_partition
        )
        assert used_disk_gib <= math.nextafter(node.disk.gib, math.inf)
        if requirement.partitions.memory_gib_per_partition is not None:
            used_memory_gib = (
                result.layout.partitions_per_node
                * requirement.partitions.memory_gib_per_partition
            )
            assert used_memory_gib <= math.nextafter(node.memory.gib, math.inf)

        # CPU constraint satisfied
        total_cpu = result.node_count * node.cpu.count
        assert total_cpu >= requirement.cpu.count

    @given(problem=valid_problems())
    @settings(max_examples=500, deadline=None)
    def test_result_is_best_valid_configuration(self, problem: Problem):
        """PROPERTY: No valid configuration has a better selection rank."""
        requirement, node, max_nodes = problem
        result = _fewest(requirement, node, max_nodes)
        if result is None:
            return

        disk_ppn = math.floor(
            math.nextafter(
                node.disk.gib / requirement.partitions.disk_gib_per_partition, math.inf
            )
        )
        max_ppn = disk_ppn
        if requirement.partitions.memory_gib_per_partition is not None:
            memory_ppn = math.floor(
                math.nextafter(
                    node.memory.gib / requirement.partitions.memory_gib_per_partition,
                    math.inf,
                )
            )
            max_ppn = min(max_ppn, memory_ppn)

        result_rank = (
            result.node_count,
            -result.layout.replica_count,
            -result.layout.partitions_per_node,
        )
        for ppn in range(min(max_ppn, requirement.partitions.count), 0, -1):
            base = math.ceil(requirement.partitions.count / ppn)
            rf = max(
                requirement.partitions.min_replicas,
                math.ceil(requirement.cpu.count / (base * node.cpu.count)),
            )
            nodes = max(base * rf, 2)
            if nodes <= max_nodes:
                assert result_rank <= (nodes, -rf, -ppn)

    @given(problem=valid_problems())
    @settings(max_examples=500, deadline=None)
    def test_node_count_formula_is_correct(self, problem: Problem):
        """PROPERTY: node_count = nodes_for_one_copy * replica_count."""
        requirement, node, max_nodes = problem
        result = _fewest(requirement, node, max_nodes)
        if result is None:
            return

        if _nodes_for_one_copy(requirement, result) >= 2:
            expected = (
                _nodes_for_one_copy(requirement, result) * result.layout.replica_count
            )
            assert result.node_count == expected
        else:
            # Special case: base=1, node_count = max(2, rf)
            assert result.node_count == max(2, result.layout.replica_count)


class TestEdgeCases:
    """Edge case tests."""

    def test_single_partition(self):
        """Algorithm handles single partition correctly.

        With 1 partition and max_ppn=5, the node holds the one partition.
        base=ceil(1/5)=1, which triggers special case.
        """
        max_nodes = 100
        requirement = Requirement(
            cpu=_cpu(32),
            partitions=Partitions(
                count=1,
                disk_gib_per_partition=100,
                min_replicas=2,
            ),
        )
        node = _node(
            disk_gib=500,  # max_ppn = 5
            cpu=16,
        )
        result = _fewest(requirement, node, max_nodes)

        assert result is not None
        assert result.layout.partitions_per_node == 1
        assert _nodes_for_one_copy(requirement, result) == 1
        assert result.layout.replica_count == 2  # min_rf (2*16=32 >= 32 cpu_needed)
        assert result.node_count == 2  # max(2, rf)

    def test_min_rf_one(self):
        """Algorithm works with min_rf=1."""
        max_nodes = 100
        requirement = Requirement(
            cpu=_cpu(16),  # 1 node worth
            partitions=Partitions(
                count=10,
                disk_gib_per_partition=100,
                min_replicas=1,
            ),
        )
        node = _node(
            disk_gib=1000,
            cpu=16,
        )
        result = _fewest(requirement, node, max_nodes)

        assert result is not None
        assert result.layout.replica_count >= 1

    def test_exact_fit(self):
        """Algorithm handles exact fit scenarios."""
        max_nodes = 4  # Exactly fits 2 nodes * 2 RF
        requirement = Requirement(
            cpu=_cpu(32),
            partitions=Partitions(
                count=10,
                disk_gib_per_partition=100,
                memory_gib_per_partition=10,
                min_replicas=2,
            ),
        )
        node = _node(
            disk_gib=500,  # Exactly 5 partitions per node
            memory_gib=50,
            cpu=16,
        )
        result = _fewest(requirement, node, max_nodes)

        assert result is not None
        assert result.node_count == 4
        assert result.layout.partitions_per_node == 5
        assert _nodes_for_one_copy(requirement, result) == 2
        assert result.layout.replica_count == 2


class TestPins:
    """Placing at a pinned layout or node count."""

    MAX_NODES = 100
    DEMAND = Requirement(
        cpu=_cpu(64),
        partitions=Partitions(
            count=12,
            disk_gib_per_partition=100,
            min_replicas=2,
        ),
    )

    def test_a_layout_that_fits_is_placed_as_pinned(self):
        node = _node(cpu=16, disk_gib=400)
        layout = Layout(partitions_per_node=3, replica_count=2)

        placed = place(
            self.DEMAND, node, min_count=2, max_count=self.MAX_NODES, layout=layout
        )

        assert isinstance(placed, Placement)
        assert (placed.node_count, placed.layout.partitions_per_node) == (8, 3)

    def test_a_layout_that_overfills_disk_names_disk(self):
        node = _node(cpu=16, disk_gib=250)
        layout = Layout(partitions_per_node=3, replica_count=2)

        placed = place(
            self.DEMAND, node, min_count=2, max_count=self.MAX_NODES, layout=layout
        )

        assert isinstance(placed, LimitNotMet)
        assert placed.bottlenecks == [Bottleneck.disk_capacity]

    def test_a_layout_short_on_every_resource_names_each(self):
        assert self.DEMAND.partitions is not None
        partitions = self.DEMAND.partitions.model_copy(
            update={"memory_gib_per_partition": 25}
        )
        requirement = self.DEMAND.model_copy(update={"partitions": partitions})
        node = _node(cpu=16, disk_gib=250, memory_gib=50)
        layout = Layout(partitions_per_node=3, replica_count=1)

        placed = place(
            requirement, node, min_count=2, max_count=self.MAX_NODES, layout=layout
        )

        assert isinstance(placed, LimitNotMet)
        assert placed.bottlenecks == [
            Bottleneck.disk_capacity,
            Bottleneck.memory,
            Bottleneck.cpu,
        ]

    def test_a_count_uses_the_densest_layout_on_exactly_that_many_nodes(self):
        # All 12 partitions on one node would not fit, so the densest split
        # of 6 nodes is 2 nodes per copy, 3 copies.
        node = _node(cpu=16, disk_gib=700)

        placed = place(
            self.DEMAND, node, min_count=2, max_count=self.MAX_NODES, count=6
        )

        assert isinstance(placed, Placement)
        assert (placed.node_count, placed.layout.partitions_per_node) == (6, 6)
        assert placed.layout.replica_count == 3

    def test_a_count_too_small_for_the_cpu_names_cpu(self):
        node = _node(cpu=16, disk_gib=1_200)

        placed = place(
            self.DEMAND, node, min_count=2, max_count=self.MAX_NODES, count=2
        )

        assert isinstance(placed, LimitNotMet)
        assert placed.bottlenecks == [Bottleneck.cpu]

    def test_a_count_no_split_fits_names_each_resource_of_the_sparsest(self):
        # On 2 nodes, even the sparser split (6 partitions per node, one copy)
        # overfills 250 GiB and has too few copies for the CPU need.
        node = _node(cpu=16, disk_gib=250)

        placed = place(
            self.DEMAND, node, min_count=2, max_count=self.MAX_NODES, count=2
        )

        assert isinstance(placed, LimitNotMet)
        assert placed.bottlenecks == [Bottleneck.disk_capacity, Bottleneck.cpu]


def test_a_cpu_need_on_a_whole_number_of_cores_adds_no_node():
    """966 cores on nodes worth 14 each is exactly 69 nodes. Converting a
    node's cores between shapes can leave it an ulp short of 14, which a
    plain ceil turns into a 70th node."""
    node_cores = math.nextafter(14, 0)
    max_nodes = 1_000
    requirement = Requirement(
        cpu=_cpu(966),
        partitions=Partitions(
            count=1,
            disk_gib_per_partition=1,
            min_replicas=1,
        ),
    )

    placed = place(
        requirement,
        _node(cpu=node_cores, disk_gib=100),
        min_count=2,
        max_count=max_nodes,
    )

    assert isinstance(placed, Placement)
    assert placed.node_count == 69
