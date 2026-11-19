import pytest

from service_capacity_modeling.hardware import shapes
from service_capacity_modeling.interface import Bottleneck
from service_capacity_modeling.interface import NodeCountConstraint
from service_capacity_modeling.models.common import compute_stateful_zone
from service_capacity_modeling.models.common import cpu_to_ncu
from service_capacity_modeling.models.common import stateful_node_resources
from service_capacity_modeling.models.placement import NodeResources
from service_capacity_modeling.models.placement import Partitions
from service_capacity_modeling.models.placement import place
from service_capacity_modeling.models.placement import DoesNotFit
from service_capacity_modeling.models.placement import Placement
from service_capacity_modeling.models.placement import Requirement

CONTAINER = NodeResources(ncu=8, memory_gib=32, network_mbps=10_000, disk_gib=500)


def test_each_resource_asks_for_its_own_node_count():
    placement = place(
        Requirement(ncu=20, memory_gib=100, network_mbps=25_000, disk_gib=1_600),
        CONTAINER,
    )

    assert isinstance(placement, Placement)
    by_type = placement.node_counts.required_nodes_by_type
    assert by_type[NodeCountConstraint.cpu] == 3
    assert by_type[NodeCountConstraint.memory] == 4
    assert by_type[NodeCountConstraint.network] == 3
    assert by_type[NodeCountConstraint.disk_capacity] == 4
    assert placement.node_count == 4


def test_cluster_size_rounds_before_the_minimum_applies():
    def pow2(n: int) -> int:
        return 1 << max(0, n - 1).bit_length()

    requirement = Requirement(ncu=40)  # 5 nodes of 8 NCUs

    rounded = place(requirement, CONTAINER, cluster_size=pow2)
    floored = place(requirement, CONTAINER, cluster_size=pow2, min_count=12)

    assert isinstance(rounded, Placement) and rounded.node_count == 8
    assert isinstance(floored, Placement) and floored.node_count == 12


def test_write_buffer_is_a_per_node_floor():
    node = CONTAINER.model_copy(update={"write_buffer_gib": 2})

    placement = place(Requirement(ncu=1, write_buffer_gib=9), node)

    assert isinstance(placement, Placement) and placement.node_count == 5


def test_disk_on_a_node_without_disk_does_not_fit():
    diskless = CONTAINER.model_copy(update={"disk_gib": 0})

    placement = place(Requirement(ncu=1, disk_gib=10), diskless)

    assert isinstance(placement, DoesNotFit)
    assert placement.bottlenecks == [Bottleneck.disk_capacity]


def test_a_node_missing_several_resources_names_each():
    bare = NodeResources(ncu=8)

    placement = place(
        Requirement(ncu=1, memory_gib=4, network_mbps=100, disk_gib=10), bare
    )

    assert isinstance(placement, DoesNotFit)
    assert placement.bottlenecks == [
        Bottleneck.memory,
        Bottleneck.network,
        Bottleneck.disk_capacity,
    ]
    assert placement.reason == (
        "The node has no memory or network or disk_capacity for the requirement"
    )


def test_ec2_node_resources_match_compute_stateful_zone():
    """Placing one NCU requirement on an EC2 node gives the zone the existing
    core-based entry point sizes."""
    i4i = shapes.region("us-east-1").instances["i4i.2xlarge"]
    assert i4i.drive is not None

    placement = place(
        Requirement(ncu=cpu_to_ncu(552, i4i), memory_gib=300, disk_gib=20_000),
        stateful_node_resources(i4i),
    )
    zone = compute_stateful_zone(
        instance=i4i,
        drive=shapes.region("us-east-1").drives["gp3"],
        needed_cores=552,
        needed_disk_gib=20_000,
        needed_memory_gib=300,
        needed_network_mbps=0,
    )

    assert isinstance(placement, Placement)
    assert placement.node_count == zone.count == 69
    assert placement.node_counts.model_dump(mode="json") == zone.cluster_params


def test_whole_partitions_need_more_nodes_than_dividing_evenly():
    """10 partitions of 300 GiB with 2 replicas on 500 GiB nodes: one
    partition fits per node, so 20 nodes, where dividing the same 6,000 GiB
    evenly asks for 12."""
    node = NodeResources(ncu=16, disk_gib=500)

    evenly = place(Requirement(ncu=1, disk_gib=6_000), node)
    whole = place(
        Requirement(
            ncu=1,
            partitions=Partitions(count=10, disk_gib_per_partition=300, min_replicas=2),
        ),
        node,
    )

    assert isinstance(evenly, Placement) and evenly.node_count == 12
    assert isinstance(whole, Placement) and whole.node_count == 20
    assert whole.layout is not None and whole.layout.partitions_per_node == 1


def test_one_partition_puts_the_whole_dataset_on_every_node():
    """40 GiB held in memory with 3 replicas: each replica needs its own node
    with room for all of it."""
    requirement = Requirement(
        ncu=1,
        partitions=Partitions(
            count=1,
            disk_gib_per_partition=40,
            memory_gib_per_partition=40,
            min_replicas=3,
        ),
    )

    roomy = place(requirement, NodeResources(ncu=16, disk_gib=100, memory_gib=64))
    small = place(requirement, NodeResources(ncu=16, disk_gib=100, memory_gib=32))

    assert isinstance(roomy, Placement) and roomy.node_count == 3
    assert isinstance(small, DoesNotFit)
    assert small.bottlenecks == [Bottleneck.memory]


def test_partitions_bring_their_own_disk_and_memory():
    with pytest.raises(ValueError, match="disk_gib cannot be set with partitions"):
        Requirement(
            ncu=1,
            disk_gib=10,
            partitions=Partitions(count=1, disk_gib_per_partition=10, min_replicas=1),
        )


def test_pins_need_partitions():
    with pytest.raises(ValueError, match="need partitions"):
        place(Requirement(ncu=1), CONTAINER, count=3)
