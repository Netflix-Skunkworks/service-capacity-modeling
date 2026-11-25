import math
from typing import Optional

import pytest

from service_capacity_modeling.hardware import shapes
from service_capacity_modeling.interface import Bottleneck
from service_capacity_modeling.interface import Buffer
from service_capacity_modeling.interface import BufferIntent
from service_capacity_modeling.interface import Buffers
from service_capacity_modeling.interface import certain_float
from service_capacity_modeling.interface import certain_int
from service_capacity_modeling.interface import CurrentZoneClusterCapacity
from service_capacity_modeling.interface import default_reference_shape
from service_capacity_modeling.interface import NodeCountConstraint
from service_capacity_modeling.models.common import compute_stateful_zone
from service_capacity_modeling.models.common import normalize_cores_float
from service_capacity_modeling.models.common import RequirementFromCurrentCapacity
from service_capacity_modeling.models.common import size_stateful_zone
from service_capacity_modeling.models.common import stateful_node_resources
from service_capacity_modeling.models.placement import CPURequirement
from service_capacity_modeling.models.placement import DiskRequirement
from service_capacity_modeling.models.placement import DoesNotFit
from service_capacity_modeling.models.placement import Isolation
from service_capacity_modeling.models.placement import Layout
from service_capacity_modeling.models.placement import LimitNotMet
from service_capacity_modeling.models.placement import MemoryRequirement
from service_capacity_modeling.models.placement import NodeCannotServe
from service_capacity_modeling.models.placement import NodeCPU
from service_capacity_modeling.models.placement import NodeDisk
from service_capacity_modeling.models.placement import NodeMemory
from service_capacity_modeling.models.placement import NodeResources
from service_capacity_modeling.models.placement import Partitions
from service_capacity_modeling.models.placement import place
from service_capacity_modeling.models.placement import Placement
from service_capacity_modeling.models.placement import Requirement
from service_capacity_modeling.models.placement import ShapesDoNotMix


def vcpus(count: float, **bounds) -> CPURequirement:
    """CPU in vCPUs of default_reference_shape, where one vCPU is one NCU."""
    return CPURequirement(
        count=count, reference_shape=default_reference_shape, **bounds
    )


# A container slice: 8 reference vCPUs, 32 GiB, 500 GiB of local disk.
CONTAINER = NodeResources(
    shape=default_reference_shape,
    cpu=NodeCPU(count=8),
    memory=NodeMemory(gib=32),
    disk=NodeDisk(gib=500),
    network_mbps=10_000,
)


def test_each_resource_asks_for_its_own_node_count():
    placement = place(
        Requirement(
            cpu=vcpus(20),
            memory=MemoryRequirement(gib=100),
            disk=DiskRequirement(gib=1_600),
            network_mbps=25_000,
        ),
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

    requirement = Requirement(cpu=vcpus(40))  # 5 nodes of 8 vCPUs

    rounded = place(requirement, CONTAINER, cluster_size=pow2)
    floored = place(requirement, CONTAINER, cluster_size=pow2, min_count=12)

    assert isinstance(rounded, Placement) and rounded.node_count == 8
    assert isinstance(floored, Placement) and floored.node_count == 12


def test_cpu_bounds_hold_only_the_cpu_count():
    capped = place(
        Requirement(cpu=vcpus(80, max_total=24), memory=MemoryRequirement(gib=6 * 32)),
        CONTAINER,
    )

    assert isinstance(capped, Placement)
    assert capped.node_counts.required_nodes_by_type[NodeCountConstraint.cpu] == 3
    assert capped.node_count == 6  # memory still needs 6


def test_write_buffer_divides_over_each_nodes_pool():
    node = CONTAINER.model_copy(
        update={"memory": NodeMemory(gib=32, write_buffer_gib=2)}
    )

    placement = place(
        Requirement(cpu=vcpus(1), memory=MemoryRequirement(write_buffer_gib=9)), node
    )

    assert isinstance(placement, Placement) and placement.node_count == 5


def test_disk_on_a_node_without_disk_does_not_fit():
    diskless = CONTAINER.model_copy(update={"disk": NodeDisk()})

    placement = place(Requirement(cpu=vcpus(1), disk=DiskRequirement(gib=10)), diskless)

    assert isinstance(placement, NodeCannotServe)
    assert placement.shapes == [default_reference_shape.name]
    assert placement.bottlenecks == [Bottleneck.disk_capacity]


def test_a_node_missing_several_resources_names_each():
    bare = NodeResources(shape=default_reference_shape, cpu=NodeCPU(count=8))

    placement = place(
        Requirement(
            cpu=vcpus(1),
            memory=MemoryRequirement(gib=4),
            disk=DiskRequirement(gib=10),
            network_mbps=100,
        ),
        bare,
    )

    assert isinstance(placement, NodeCannotServe)
    assert placement.bottlenecks == [
        Bottleneck.memory,
        Bottleneck.network,
        Bottleneck.disk_capacity,
    ]
    assert placement.reason == (
        "The node has no memory or network or disk_capacity for the requirement"
    )


def test_ec2_node_resources_match_compute_stateful_zone():
    """Placing a requirement measured on i4i.2xlarge on that EC2 node gives
    the zone the existing core-based entry point sizes."""
    i4i = shapes.region("us-east-1").instances["i4i.2xlarge"]
    assert i4i.drive is not None

    placement = place(
        Requirement(
            cpu=CPURequirement(count=552, reference_shape=i4i),
            memory=MemoryRequirement(gib=300),
            disk=DiskRequirement(gib=20_000),
        ),
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


@pytest.mark.parametrize("read_fraction", [0.0, 0.3, 1.0])
def test_disk_io_weighs_the_drives_read_and_write_iops(read_fraction):
    """A local drive's IO limit is its read and write IOPS weighted by the
    requirement's share of reads, in bytes."""
    m5d = shapes.instance("m5d.4xlarge")
    drive = m5d.drive
    assert drive is not None and drive.read_io_per_s and drive.write_io_per_s
    per_node = (
        (
            read_fraction * drive.read_io_per_s
            + (1 - read_fraction) * drive.write_io_per_s
        )
        * drive.block_size_kib
        * 1024
    )

    placement = place(
        Requirement(
            cpu=vcpus(1),
            disk=DiskRequirement(
                gib=1, io_bytes_per_s=per_node * 7.5, read_fraction=read_fraction
            ),
        ),
        stateful_node_resources(m5d),
    )

    assert isinstance(placement, Placement)
    assert (
        placement.node_counts.required_nodes_by_type[NodeCountConstraint.disk_iops] == 8
    )


def test_a_node_with_no_disk_io_does_not_fit_disk_io():
    """A known 0 IOPS is no IO at all, not an unknown limit."""
    requirement = Requirement(
        cpu=vcpus(1), disk=DiskRequirement(gib=1, io_bytes_per_s=100)
    )

    def node(io_bytes_per_s: Optional[float]) -> NodeResources:
        return NodeResources(
            shape=default_reference_shape,
            cpu=NodeCPU(count=8),
            disk=NodeDisk(
                gib=10,
                read_bytes_per_s=io_bytes_per_s,
                write_bytes_per_s=io_bytes_per_s,
            ),
        )

    none = place(requirement, node(0))
    unknown = place(requirement, node(None))

    assert isinstance(none, DoesNotFit)
    assert none.bottlenecks == [Bottleneck.disk_iops]
    assert isinstance(unknown, Placement) and unknown.node_count == 1


def test_parallelism_needs_physical_cores_not_vcpus():
    """m5.large and m7a.large both have 2 vCPUs, but m5.large has 1 physical
    core, so it cannot run 2 threads truly in parallel."""
    requirement = Requirement(cpu=vcpus(4, parallelism=2))

    one_core = place(requirement, stateful_node_resources(shapes.instance("m5.large")))
    two_cores = place(
        requirement, stateful_node_resources(shapes.instance("m7a.large"))
    )

    assert isinstance(one_core, NodeCannotServe)
    assert one_core.bottlenecks == [Bottleneck.cpu]
    assert isinstance(two_cores, Placement)


def test_a_container_slice_has_its_share_of_physical_cores():
    i4i = shapes.instance("i4i.2xlarge")  # 8 vCPUs on 4 physical cores
    slice_of_i4i = NodeResources(shape=i4i, cpu=NodeCPU(count=4))

    assert isinstance(
        place(Requirement(cpu=vcpus(1, parallelism=2)), slice_of_i4i), Placement
    )
    assert isinstance(
        place(Requirement(cpu=vcpus(1, parallelism=3)), slice_of_i4i), DoesNotFit
    )


def _whole(name: str, memory_gib: float = 0, disk_gib: float = 0) -> NodeResources:
    shape = shapes.instance(name)
    return NodeResources(
        shape=shape,
        cpu=NodeCPU(count=shape.cpu),
        memory=NodeMemory(gib=memory_gib),
        disk=NodeDisk(gib=disk_gib),
    )


@pytest.mark.parametrize(
    "isolation, names, mixes",
    [
        (Isolation.performance, ["m7a.2xlarge", "m7a.2xlarge"], True),
        (Isolation.performance, ["c7a.2xlarge", "m7a.2xlarge"], False),
        (Isolation.balanced, ["c7a.2xlarge", "m7a.2xlarge", "r7a.2xlarge"], True),
        (Isolation.balanced, ["m7a.2xlarge", "m7a.4xlarge"], False),
        (Isolation.balanced, ["m7a.2xlarge", "m7i.2xlarge"], False),
        (Isolation.balanced, ["c5.large", "m5.large"], False),
        (Isolation.efficient, ["m7a.2xlarge", "m7i.2xlarge"], True),
    ],
)
def test_isolation_decides_which_shapes_one_cluster_runs(isolation, names, mixes):
    """Performance runs one shape. Balanced runs one CPU at one size: c7a,
    m7a and r7a of one size share it; another size, m7i and c5 do not.
    Efficient runs any."""
    first, *rest = [_whole(name) for name in names]

    placed = place(
        Requirement(cpu=vcpus(16), isolation=isolation), first, mixed_with=rest
    )

    if mixes:
        assert isinstance(placed, Placement)
    else:
        assert isinstance(placed, ShapesDoNotMix)
        assert placed.shapes == list(dict.fromkeys(names))
        assert placed.bottlenecks == []


def test_shapes_mix_only_when_the_requirement_asks():
    """Isolation defaults to performance, so mixing is opt-in."""
    placed = place(
        Requirement(cpu=vcpus(16)),
        _whole("m7a.2xlarge"),
        mixed_with=[_whole("m7i.2xlarge")],
    )

    assert isinstance(placed, ShapesDoNotMix)


def test_isolation_does_not_reject_one_shape():
    """Isolation only limits mixing, so a performance requirement places on
    any single shape."""
    placement = place(
        Requirement(cpu=vcpus(16), isolation=Isolation.performance), CONTAINER
    )

    assert isinstance(placement, Placement) and placement.node_count == 2


def test_a_mixed_cluster_needs_what_its_smallest_member_needs():
    """Every node holds an equal share, so 100 GiB needs 7 nodes of 16 GiB
    whether they are c7a or a mix with r7a, which alone would need 2."""
    requirement = Requirement(
        cpu=vcpus(16),
        memory=MemoryRequirement(gib=100),
        isolation=Isolation.balanced,
    )
    c7a = _whole("c7a.2xlarge", memory_gib=16)
    r7a = _whole("r7a.2xlarge", memory_gib=64)

    alone = [place(requirement, node) for node in (c7a, r7a)]
    mixed = place(requirement, r7a, mixed_with=[c7a])

    assert [p.node_count for p in alone if isinstance(p, Placement)] == [7, 2]
    assert isinstance(mixed, Placement) and mixed.node_count == 7


def test_a_mixed_partition_layout_fits_on_every_member():
    """Partitions of 100 GiB fit 3 to a 300 GiB node but 1 to a 100 GiB one,
    so a mixed cluster holds 1 per node."""
    requirement = Requirement(
        cpu=vcpus(1),
        partitions=Partitions(count=6, disk_gib_per_partition=100, min_replicas=1),
    )
    big = _whole("m7a.2xlarge", disk_gib=300)
    small = _whole("m7a.2xlarge", disk_gib=100)

    alone = place(requirement, big)
    mixed = place(requirement, big, mixed_with=[small])

    assert isinstance(alone, Placement) and alone.node_count == 2
    assert isinstance(mixed, Placement)
    assert mixed.node_count == 6
    assert mixed.layout == Layout(partitions_per_node=1, replica_count=1)


def test_a_mixed_cluster_names_the_member_that_cannot_serve():
    requirement = Requirement(
        cpu=vcpus(4, parallelism=2), isolation=Isolation.efficient
    )

    placed = place(
        requirement,
        stateful_node_resources(shapes.instance("m7a.large")),
        mixed_with=[stateful_node_resources(shapes.instance("m5.large"))],
    )

    assert isinstance(placed, NodeCannotServe)
    assert placed.shapes == ["m5.large"]


def test_a_count_limit_is_not_a_node_that_cannot_serve():
    """Twelve partitions on nodes that hold one each need 12 nodes, so a
    max_count of 6 is a limit the caller set, not a lack in the node."""
    requirement = Requirement(
        cpu=vcpus(1),
        partitions=Partitions(count=12, disk_gib_per_partition=100, min_replicas=1),
    )

    placed = place(requirement, _whole("m7a.2xlarge", disk_gib=100), max_count=6)

    assert isinstance(placed, LimitNotMet)
    assert placed.shapes == ["m7a.2xlarge"]


def test_an_ineligible_node_is_reported_not_asserted():
    """size_stateful_zone hands place()'s DoesNotFit back for the model to
    turn into an Excuse."""
    m5 = shapes.instance("m5.large")

    zone = size_stateful_zone(
        Requirement(cpu=vcpus(4, parallelism=2), memory=MemoryRequirement(gib=4)),
        stateful_node_resources(m5),
        shapes.region("us-east-1").drives["gp3"],
    )

    assert isinstance(zone, DoesNotFit)
    excuse = zone.excuse(m5.name, "gp3")
    assert excuse.bottleneck == Bottleneck.cpu
    assert "1 physical cores" in excuse.reason


def test_whole_partitions_need_more_nodes_than_dividing_evenly():
    """10 partitions of 300 GiB with 2 replicas on 500 GiB nodes: one
    partition fits per node, so 20 nodes, where dividing the same 6,000 GiB
    evenly asks for 12."""
    node = NodeResources(
        shape=default_reference_shape, cpu=NodeCPU(count=16), disk=NodeDisk(gib=500)
    )

    evenly = place(Requirement(cpu=vcpus(1), disk=DiskRequirement(gib=6_000)), node)
    whole = place(
        Requirement(
            cpu=vcpus(1),
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
        cpu=vcpus(1),
        partitions=Partitions(
            count=1,
            disk_gib_per_partition=40,
            memory_gib_per_partition=40,
            min_replicas=3,
        ),
    )

    def node(memory_gib: float) -> NodeResources:
        return NodeResources(
            shape=default_reference_shape,
            cpu=NodeCPU(count=16),
            memory=NodeMemory(gib=memory_gib),
            disk=NodeDisk(gib=100),
        )

    roomy = place(requirement, node(64))
    small = place(requirement, node(32))

    assert isinstance(roomy, Placement) and roomy.node_count == 3
    assert isinstance(small, DoesNotFit)
    assert small.bottlenecks == [Bottleneck.memory]


def test_min_count_nodes_hold_whole_copies():
    """One partition, one replica needed, at least 4 nodes: each of the 4
    nodes holds a full copy instead of 3 sitting empty."""
    node = NodeResources(
        shape=default_reference_shape, cpu=NodeCPU(count=8), disk=NodeDisk(gib=10)
    )
    requirement = Requirement(
        cpu=vcpus(0),
        partitions=Partitions(count=1, disk_gib_per_partition=1, min_replicas=1),
    )

    placed = place(requirement, node, min_count=4)

    assert isinstance(placed, Placement) and placed.node_count == 4
    assert placed.layout == Layout(partitions_per_node=1, replica_count=4)


def test_partitions_bring_their_own_disk_and_memory():
    with pytest.raises(ValueError, match="disk.gib cannot be set with partitions"):
        Requirement(
            cpu=vcpus(1),
            disk=DiskRequirement(gib=10),
            partitions=Partitions(count=1, disk_gib_per_partition=10, min_replicas=1),
        )


def test_a_layout_pin_needs_partitions():
    with pytest.raises(ValueError, match="needs partitions"):
        place(
            Requirement(cpu=vcpus(1)),
            CONTAINER,
            layout=Layout(partitions_per_node=1, replica_count=1),
        )


def test_a_pinned_count_holds_evenly_divided_data():
    """16 vCPUs need 2 nodes of 8. A cluster that scales by shape at 4
    nodes runs 4, and at 1 node, or capped at 1, the CPU does not fit."""
    requirement = Requirement(cpu=vcpus(16))

    pinned = place(requirement, CONTAINER, count=4)
    too_few = place(requirement, CONTAINER, count=1)
    capped = place(requirement, CONTAINER, max_count=1)

    assert isinstance(pinned, Placement) and pinned.node_count == 4
    for placed in (too_few, capped):
        assert isinstance(placed, LimitNotMet)
        assert placed.bottlenecks == [Bottleneck.cpu]
    assert place(requirement, CONTAINER, max_count=3) == place(requirement, CONTAINER)


def test_max_count_holds_after_attached_drives_add_nodes():
    """20,000 GiB on gp3 volumes of at most a third of their maximum size
    needs more nodes than the CPU does, so a cap of 2 nodes fails on disk."""
    m7a = shapes.instance("m7a.2xlarge")
    requirement = Requirement(cpu=vcpus(1), disk=DiskRequirement(gib=20_000))
    gp3 = shapes.region("us-east-1").drives["gp3"]

    uncapped = size_stateful_zone(requirement, stateful_node_resources(m7a), gp3)
    capped = size_stateful_zone(
        requirement, stateful_node_resources(m7a), gp3, max_count=2
    )

    assert not isinstance(uncapped, DoesNotFit) and uncapped.count > 2
    assert isinstance(capped, LimitNotMet)
    assert capped.bottlenecks == [Bottleneck.disk_capacity]


def test_a_node_without_room_for_one_partition_cannot_serve_any_pin():
    requirement = Requirement(
        cpu=vcpus(1),
        partitions=Partitions(count=6, disk_gib_per_partition=100, min_replicas=1),
    )
    node = _whole("m7a.2xlarge", disk_gib=50)

    for pin in (
        {"layout": Layout(partitions_per_node=1, replica_count=1)},
        {"count": 6},
        {},
    ):
        placed = place(requirement, node, **pin)
        assert isinstance(placed, NodeCannotServe)
        assert placed.bottlenecks == [Bottleneck.disk_capacity]


def test_measured_cpu_leaves_each_node_its_headroom():
    node = CONTAINER.model_copy(update={"cpu": NodeCPU(count=8, headroom=0.25)})

    staffed = place(Requirement(cpu=vcpus(48)), node)
    measured = place(Requirement(cpu=vcpus(48, add_headroom=True)), node)

    assert isinstance(staffed, Placement) and staffed.node_count == 6
    assert isinstance(measured, Placement) and measured.node_count == 8


@pytest.mark.parametrize(
    "candidate", ["i4i.xlarge", "i4i.8xlarge", "m5d.4xlarge", "r6id.12xlarge"]
)
@pytest.mark.parametrize(
    "cpu_percent, derived",
    [
        (40.0, None),
        # Hot CPU that may not grow past the current zone's CPU.
        (80.0, Buffer(intent=BufferIntent.scale_down, ratio=2, components=["cpu"])),
        # Cold CPU that keeps the current zone's CPU.
        (5.0, Buffer(intent=BufferIntent.preserve, components=["cpu"])),
    ],
)
def test_measured_cpu_places_like_the_per_candidate_cores(
    candidate, cpu_percent, derived
):
    """One shape-free CPU requirement asks each shape for the CPU node count
    the per-candidate core requirement asked for, derived bounds included."""
    current = shapes.instance("i4i.2xlarge")
    zone = RequirementFromCurrentCapacity(
        current_capacity=CurrentZoneClusterCapacity(
            cluster_instance_name=current.name,
            cluster_instance=current,
            cluster_instance_count=certain_int(12),
            cpu_utilization=certain_float(cpu_percent),
        ),
        buffers=Buffers(
            desired={"compute": Buffer(ratio=1.5, components=["compute"])},
            derived={} if derived is None else {"compute": derived},
        ),
    )
    instance = shapes.instance(candidate)
    cores = normalize_cores_float(zone.cpu(instance), instance, current)

    placement = place(
        Requirement(cpu=zone.cpu_requirement()), stateful_node_resources(instance)
    )

    assert isinstance(placement, Placement)
    by_type = placement.node_counts.required_nodes_by_type
    assert by_type[NodeCountConstraint.cpu] == math.ceil(cores / instance.cpu)
