"""
Place a requirement on nodes of any shape.

A model states what a workload needs once, without picking a shape (a
``Requirement``). A node states what it offers after reservations
(``NodeResources``): an EC2 instance, a container slice, or anything else with
CPU, memory, disk and network. ``place`` answers how many of those nodes give
every resource and all of the data a place to live, on one shape or a mix the
requirement's ``isolation`` allows. Otherwise it says why: ``NodeCannotServe``,
``LimitNotMet`` or ``ShapesDoNotMix``, each a ``DoesNotFit``.

CPU is stated as a vCPU count plus the shape it was measured on. ``place``
converts both sides to NCUs (normalized compute units, see ``cpu_to_ncu``)
internally, so one requirement can be placed on any shape.
"""

import math
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

from pydantic import BaseModel
from pydantic import Field
from pydantic import model_validator

from service_capacity_modeling.enum_utils import StrEnum
from service_capacity_modeling.interface import Bottleneck
from service_capacity_modeling.interface import default_reference_shape
from service_capacity_modeling.interface import Excuse
from service_capacity_modeling.interface import Instance
from service_capacity_modeling.interface import NodeCountConstraint
from service_capacity_modeling.interface import NodeCountContext


def ceil_ignoring_float_error(value: float) -> int:
    """Ceil a need without adding a unit for float error.

    Converting cores between shapes multiplies by a speed ratio, so an exact
    whole number can come out a few ulps over (966 * 4/7 is
    552.0000000000001) and a plain ceil adds a core, and sometimes a node.
    """
    return math.ceil(value * (1 - 1e-9))


def cpu_to_ncu(cpu: float, shape: Instance) -> float:
    """Convert vCPUs of shape to NCUs (normalized compute units).

    One NCU is one vCPU of default_reference_shape (2.3 GHz, cpu_ipc_scale
    1.0), so NCUs compare and add up across instance types. A vCPU is worth
    shape.cpu_ghz * shape.cpu_ipc_scale / 2.3 NCUs; cpu_ipc_scale already
    folds in the hyperthreading factor, so pass vCPUs, not physical cores.
    """
    shape_speed = shape.cpu_ghz * shape.cpu_ipc_scale
    ncu_speed = default_reference_shape.cpu_ghz * default_reference_shape.cpu_ipc_scale
    return cpu * (shape_speed / ncu_speed)


class Isolation(StrEnum):
    """Which node shapes one cluster may mix, as ``place``'s ``mixed_with``
    checks."""

    performance = "performance"  # one shape
    # One CPU at one size, any memory: c7a.4xlarge, m7a.4xlarge and r7a.4xlarge
    # mix, so any of them replaces another one for one.
    balanced = "balanced"
    efficient = "efficient"  # any shapes


class CPURequirement(BaseModel):
    """A workload's CPU, in vCPUs of the shape it was measured on."""

    count: float = Field(ge=0, description="vCPUs of reference_shape.")
    reference_shape: Instance = Field(description="The shape count was measured on.")
    add_headroom: bool = Field(
        default=False,
        description="True when count is measured busy CPU, such as current "
        "utilization, so each node keeps its headroom idle. False when count "
        "already allows for queueing, as sqrt staffing does.",
    )
    min_total: Optional[float] = Field(
        default=None,
        ge=0,
        description="Fewest vCPUs of reference_shape the CPU node count may "
        "add up to, counting all of each node's CPU, such as to keep a current "
        "cluster's CPU.",
    )
    max_total: Optional[float] = Field(
        default=None,
        gt=0,
        description="Most vCPUs of reference_shape the CPU node count may add "
        "up to, such as to not grow past a current cluster's CPU. It caps only "
        "the CPU count; memory, disk or network can still need more nodes. "
        "min_total wins when the two conflict.",
    )
    parallelism: int = Field(
        default=0,
        ge=0,
        description="Fewest physical cores one node must have, to run this "
        "many threads truly in parallel. Not scaled by reference_shape.",
    )


class MemoryRequirement(BaseModel):
    gib: float = Field(default=0, ge=0, description="Total, divided over the nodes.")
    write_buffer_gib: float = Field(
        default=0,
        ge=0,
        description="Total write-buffer memory, such as Cassandra memtables, "
        "divided over each node's write-buffer pool.",
    )


class DiskRequirement(BaseModel):
    gib: float = Field(default=0, ge=0, description="Total, on local disk.")
    io_bytes_per_s: float = Field(default=0, ge=0, description="Total, on local disk.")
    read_fraction: float = Field(
        default=0, ge=0, le=1, description="Share of IO requests that are reads."
    )


class Partitions(BaseModel):
    """Data stored in whole partitions, so every replica of every partition
    needs a node with room for it, and no node holds two replicas of the same
    partition.

    For example, 10 partitions of 300 GiB with 2 replicas on 500 GiB nodes
    need 20 nodes, one partition each. Divided evenly, the same 6,000 GiB
    would ask for 12. With ``count=1``, every node holds the whole data set.
    """

    count: int = Field(ge=1)
    disk_gib_per_partition: float = Field(
        gt=0, description="Disk one replica of a partition needs, with headroom."
    )
    memory_gib_per_partition: Optional[float] = Field(
        default=None,
        gt=0,
        description="Memory one replica of a partition needs, with headroom. "
        "None places partitions by disk only.",
    )
    min_replicas: int = Field(ge=1)
    max_partitions_per_node: Optional[int] = Field(default=None, ge=1)


class Requirement(BaseModel):
    """What a workload needs, independent of node shape.

    Without ``partitions``, every total divides evenly over the nodes, like
    water poured across them. That is the fewest nodes that could work, and
    close to exact when the data's partitions are much smaller than a node,
    as in Cassandra and Kafka. ``cpu.parallelism`` holds for every node, and
    ``isolation`` limits which shapes ``place``'s ``mixed_with`` may add.

    With ``partitions``, disk and memory come from the per-partition sizes,
    and CPU is met by adding whole replicas. Leave the other totals at 0;
    network, disk IO, write buffer and CPU bounds are not modeled for
    partitions yet.
    """

    cpu: CPURequirement
    memory: MemoryRequirement = MemoryRequirement()
    disk: DiskRequirement = DiskRequirement()
    network_mbps: float = Field(default=0, ge=0)
    isolation: Isolation = Field(
        default=Isolation.performance,
        description="Mixing shapes in one cluster is opt-in.",
    )
    partitions: Optional[Partitions] = None

    @model_validator(mode="after")
    def _partitions_carry_disk_and_memory(self) -> "Requirement":
        if self.partitions is None:
            return self
        totals = {
            "memory.gib": self.memory.gib,
            "memory.write_buffer_gib": self.memory.write_buffer_gib,
            "network_mbps": self.network_mbps,
            "disk.gib": self.disk.gib,
            "disk.io_bytes_per_s": self.disk.io_bytes_per_s,
            "cpu.min_total": self.cpu.min_total,
            "cpu.max_total": self.cpu.max_total,
        }
        set_totals = [name for name, value in totals.items() if value]
        if set_totals:
            raise ValueError(
                f"{', '.join(set_totals)} cannot be set with partitions: disk "
                "and memory come from the partitions, and the rest is not "
                "modeled for partitions yet"
            )
        return self


class NodeCPU(BaseModel):
    count: float = Field(
        gt=0,
        description="vCPUs given to the workload; a container can get fewer "
        "than shape.cpu.",
    )
    headroom: float = Field(
        default=0,
        ge=0,
        lt=1,
        description="Fraction of count kept idle so requests do not queue, "
        "which shrinks as a node has more vCPUs. Applied only to a CPU "
        "requirement with add_headroom.",
    )


class NodeMemory(BaseModel):
    gib: float = Field(
        default=0, ge=0, description="Used only when the requirement needs memory."
    )
    write_buffer_gib: float = Field(
        default=0,
        ge=0,
        description="This node's write-buffer pool. 0 places no write-buffer limit.",
    )


class NodeDisk(BaseModel):
    gib: float = Field(default=0, ge=0, description="Local disk. 0 when none.")
    read_bytes_per_s: Optional[float] = Field(
        default=None, ge=0, description="None when unknown."
    )
    write_bytes_per_s: Optional[float] = Field(
        default=None,
        ge=0,
        description="None when unknown. An unknown side the requirement uses "
        "places no IO limit; a known 0 offers none.",
    )


class NodeResources(BaseModel):
    """What one node offers after reservations."""

    shape: Instance = Field(
        description="Sets each vCPU's speed and physical cores, and is the "
        "instance a plan reports."
    )
    cpu: NodeCPU
    memory: NodeMemory = NodeMemory()
    disk: NodeDisk = NodeDisk()
    network_mbps: float = Field(
        default=0, ge=0, description="Used only when the requirement needs network."
    )


class DoesNotFit(BaseModel):
    """Why no count of these nodes fits. ``place`` returns one of three
    kinds: ``NodeCannotServe``, ``LimitNotMet`` or ``ShapesDoNotMix``.

    Each kind lists ``bottlenecks`` in a fixed order, and the list is empty
    when the limit is not a resource, such as a node count.
    """

    reason: str
    shapes: List[str] = Field(description="The shapes the misfit is about.")
    bottlenecks: List[Bottleneck] = Field(default_factory=list)

    def excuse(self, instance: str, drive: str) -> Excuse:
        """This misfit as a model's Excuse. Excuse holds one bottleneck, so
        it gets the first, and context lists them all when there are several."""
        return Excuse(
            instance=instance,
            drive=drive,
            reason=self.reason,
            bottleneck=self.bottlenecks[0] if self.bottlenecks else None,
            context=(
                {"bottlenecks": [b.value for b in self.bottlenecks]}
                if len(self.bottlenecks) > 1
                else {}
            ),
        )


class NodeCannotServe(DoesNotFit):
    """A node cannot serve the requirement at any count: it has fewer
    physical cores than ``cpu.parallelism``, no memory, disk, network or disk
    IO the requirement needs, or no room for one partition. ``shapes`` names
    the nodes that lack it."""


class LimitNotMet(DoesNotFit):
    """The nodes can serve the requirement, but not within ``max_count`` or
    at a pinned count or layout. ``bottlenecks`` lists what keeps the pin from
    fitting; empty when only the node count limits it."""


class ShapesDoNotMix(DoesNotFit):
    """The requirement's ``isolation`` does not let one cluster run these
    shapes together. No resource bottlenecks it."""


class Layout(BaseModel):
    """How partitions sit on nodes: partitions per node and complete copies."""

    partitions_per_node: int = Field(ge=1)
    replica_count: int = Field(ge=1)


class Placement(BaseModel):
    """How many nodes the requirement needs, and why.

    ``node_counts`` explains an evenly divided requirement: how many nodes
    each resource asked for. ``layout`` explains a partitioned one: how many
    partitions sit on each node, and how many complete copies there are.
    """

    node_count: int
    node_counts: Optional[NodeCountContext] = None
    layout: Optional[Layout] = None


def round_zone_count(  # pylint: disable=too-many-arguments
    *,
    count_cpu: int,
    count_memory: int,
    count_network: int,
    count_disk_capacity: int,
    count_disk_iops: int,
    cluster_size: Callable[[int], int],
    min_count: int,
) -> Placement:
    """Take the most nodes any resource needs, round that to a cluster size
    the technology allows, then apply the minimum count."""
    raw_count = max(
        count_cpu, count_memory, count_network, count_disk_capacity, count_disk_iops
    )
    cluster_size_count = cluster_size(raw_count)
    return Placement(
        node_count=max(cluster_size_count, min_count),
        node_counts=NodeCountContext.from_counts(
            count_cpu=count_cpu,
            count_memory=count_memory,
            count_network=count_network,
            count_disk_capacity=count_disk_capacity,
            count_disk_iops=count_disk_iops,
            cluster_size_count=cluster_size_count,
            min_count=min_count,
        ),
    )


def place(  # pylint: disable=too-many-arguments
    requirement: Requirement,
    node: NodeResources,
    *,
    mixed_with: Sequence[NodeResources] = (),
    cluster_size: Optional[Callable[[int], int]] = None,
    min_count: int = 0,
    max_count: Optional[int] = None,
    layout: Optional[Layout] = None,
    count: Optional[int] = None,
) -> Union[Placement, NodeCannotServe, LimitNotMet, ShapesDoNotMix]:
    """Count how many nodes like ``node`` the ``requirement`` needs.

    Without partitions, each resource total asks for ceil(total / per-node
    amount) nodes. The most any resource asks for is rounded by
    ``cluster_size`` (such as to a power of two), then raised to
    ``min_count``. ``count`` pins the node count, as when the cluster scales
    by shape at a fixed count, and ``max_count`` caps it.

    With partitions, as many whole partitions go on each node as fit on its
    disk and in its memory, and the node count is nodes per replica times
    replicas: at least ``min_replicas``, more if CPU needs them or if
    ``min_count`` leaves room for whole copies. The fewest
    nodes win, between ``min_count`` and ``max_count``; ties go to more
    replicas, then denser packing. ``layout`` pins the layout and ``count``
    the node count; with only ``count``, the densest layout on exactly that
    many nodes wins. ``cluster_size`` does not apply.

    ``mixed_with`` adds shapes the cluster may also run, when the
    requirement's ``isolation`` lets them mix. Every node takes an equal share,
    so the count is the one any mix of the shapes can run: the most any of
    them needs alone, and for partitions a layout that fits on each.

    Returns ``ShapesDoNotMix`` when the isolation forbids the mix,
    ``NodeCannotServe`` when a node lacks something at any count, and
    ``LimitNotMet`` when ``max_count`` or a pin cannot be met, each with
    every bottleneck.
    """
    if requirement.partitions is None:
        if layout is not None:
            raise ValueError("layout needs partitions")
    elif cluster_size is not None:
        raise ValueError("cluster_size does not apply to partitions")
    nodes = [node, *mixed_with]
    conflict = _shapes_do_not_mix(requirement, nodes)
    if conflict is not None:
        return conflict
    for member in nodes:
        unfit = _unfit_cpu(requirement, member)
        if unfit is not None:
            return unfit
    if requirement.partitions is None:
        return _place_evenly(
            requirement,
            nodes,
            cluster_size or (lambda nodes: nodes),
            min_count,
            count,
            max_count,
        )
    return _PartitionFit(
        requirement=requirement,
        partitions=requirement.partitions,
        nodes=nodes,
        min_count=min_count,
        max_count=max_count,
    ).place(layout, count)


def count_limit_not_met(
    placement: Placement,
    count: Optional[int],
    max_count: Optional[int],
    shapes: List[str],
) -> Optional[LimitNotMet]:
    """Why an evenly divided placement misses a pinned ``count`` or
    ``max_count``, or None. Names each resource that alone needs more nodes."""
    limits = [limit for limit in (count, max_count) if limit is not None]
    if not limits or placement.node_count <= min(limits):
        return None
    limit = min(limits)
    assert placement.node_counts is not None
    by_type = placement.node_counts.required_nodes_by_type
    return LimitNotMet(
        reason=f"Needs {placement.node_count} nodes, more than {limit}",
        shapes=shapes,
        bottlenecks=[
            bottleneck
            for bottleneck in (
                Bottleneck.cpu,
                Bottleneck.memory,
                Bottleneck.network,
                Bottleneck.disk_capacity,
                Bottleneck.disk_iops,
            )
            if by_type.get(NodeCountConstraint(bottleneck.value), 0) > limit
        ],
    )


def _names(nodes: Sequence[NodeResources]) -> List[str]:
    """Each shape name once, in the order given."""
    return list(dict.fromkeys(node.shape.name for node in nodes))


def _shapes_do_not_mix(
    requirement: Requirement, nodes: Sequence[NodeResources]
) -> Optional[ShapesDoNotMix]:
    """Why the requirement's isolation keeps these nodes out of one cluster,
    or None. A performance cluster runs one shape. A balanced cluster runs
    one CPU (clock, IPC and cores per vCPU) at one vCPU count, so only memory
    differs, the way c7a, m7a and r7a of one size do."""
    isolation = requirement.isolation
    cpus = {
        (
            node.shape.cpu_ghz,
            node.shape.cpu_ipc_scale,
            node.shape.cores / node.shape.cpu,
            node.cpu.count,
        )
        for node in nodes
    }
    names = _names(nodes)
    if (isolation == Isolation.performance and len(names) > 1) or (
        isolation == Isolation.balanced and len(cpus) > 1
    ):
        return ShapesDoNotMix(
            reason=f"A {isolation} cluster cannot run {', '.join(names)} together",
            shapes=names,
        )
    return None


def _unfit_cpu(
    requirement: Requirement, node: NodeResources
) -> Optional[NodeCannotServe]:
    """Why the node's CPU cannot serve the requirement at all, or None."""
    cores = node.cpu.count * node.shape.cores / node.shape.cpu
    if cores < requirement.cpu.parallelism:
        return NodeCannotServe(
            reason=f"{node.shape.name} has {cores:g} physical cores, fewer than "
            f"the {requirement.cpu.parallelism} the workload runs in parallel",
            shapes=[node.shape.name],
            bottlenecks=[Bottleneck.cpu],
        )
    return None


def _node_ncu(node: NodeResources) -> float:
    return cpu_to_ncu(node.cpu.count, node.shape)


def _usable_ncu(cpu: CPURequirement, node: NodeResources) -> float:
    """NCUs of one node the CPU may use: all of them, or with add_headroom,
    what is left after the node's queueing headroom."""
    if cpu.add_headroom:
        return _node_ncu(node) * (1 - node.cpu.headroom)
    return _node_ncu(node)


def _cpu_count(cpu: CPURequirement, node: NodeResources) -> int:
    """Nodes for CPU, then held between the total-CPU bounds."""
    count = ceil_ignoring_float_error(
        cpu_to_ncu(cpu.count, cpu.reference_shape) / _usable_ncu(cpu, node)
    )
    if cpu.max_total is not None:
        count = min(
            count,
            ceil_ignoring_float_error(
                cpu_to_ncu(cpu.max_total, cpu.reference_shape) / _node_ncu(node)
            ),
        )
    if cpu.min_total is not None:
        count = max(
            count,
            ceil_ignoring_float_error(
                cpu_to_ncu(cpu.min_total, cpu.reference_shape) / _node_ncu(node)
            ),
        )
    return count


def _memory_count(memory: MemoryRequirement, node: NodeMemory) -> Optional[int]:
    """Nodes for memory and the write buffer, or None if the node has no
    memory for a requirement that needs some."""
    count = 0
    if memory.gib > 0:
        if node.gib <= 0:
            return None
        count = math.ceil(memory.gib / node.gib)
        # Division can round an exact capacity boundary above an integer.
        if (count - 1) * node.gib >= memory.gib:
            count -= 1
    if memory.write_buffer_gib > 0 and node.write_buffer_gib > 0:
        count = max(count, math.ceil(memory.write_buffer_gib / node.write_buffer_gib))
    return count


def _place_evenly(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    requirement: Requirement,
    nodes: Sequence[NodeResources],
    cluster_size: Callable[[int], int],
    min_count: int,
    count: Optional[int],
    max_count: Optional[int],
) -> Union[Placement, NodeCannotServe, LimitNotMet]:
    """Each resource asks for the most nodes any of the shapes needs for it
    alone, since every node holds an equal share."""
    per_node = []
    for node in nodes:
        counts = _even_counts(requirement, node)
        if isinstance(counts, NodeCannotServe):
            return counts
        per_node.append(counts)
    most = {name: max(counts[name] for counts in per_node) for name in per_node[0]}
    placed = round_zone_count(
        count_cpu=most["count_cpu"],
        count_memory=most["count_memory"],
        count_network=most["count_network"],
        count_disk_capacity=most["count_disk_capacity"],
        count_disk_iops=most["count_disk_iops"],
        cluster_size=cluster_size,
        min_count=max(min_count, count or 0),
    )
    return count_limit_not_met(placed, count, max_count, _names(nodes)) or placed


def _even_counts(
    requirement: Requirement, node: NodeResources
) -> Union[Dict[str, int], NodeCannotServe]:
    """Nodes like ``node`` each resource asks for, by round_zone_count's
    argument names."""
    count_memory = _memory_count(requirement.memory, node.memory)
    needs_network = requirement.network_mbps > 0
    disk = requirement.disk
    needs_disk = disk.gib > 0
    read, write = node.disk.read_bytes_per_s, node.disk.write_bytes_per_s
    # An unknown (None) side the requirement uses places no IO limit; a known
    # 0 offers no IO at all.
    node_io_bytes_per_s: Optional[float] = None
    if (read is not None or disk.read_fraction == 0) and (
        write is not None or disk.read_fraction == 1
    ):
        node_io_bytes_per_s = disk.read_fraction * (read or 0) + (
            1 - disk.read_fraction
        ) * (write or 0)
    # Name every resource the node lacks, so adding one does not reveal the next.
    missing = []
    if count_memory is None:
        missing.append(Bottleneck.memory)
    if needs_network and node.network_mbps <= 0:
        missing.append(Bottleneck.network)
    if needs_disk and node.disk.gib <= 0:
        missing.append(Bottleneck.disk_capacity)
    if disk.io_bytes_per_s > 0 and node_io_bytes_per_s == 0:
        missing.append(Bottleneck.disk_iops)
    if missing or count_memory is None:
        names = " or ".join(bottleneck.value for bottleneck in missing)
        return NodeCannotServe(
            reason=f"The node has no {names} for the requirement",
            shapes=[node.shape.name],
            bottlenecks=missing,
        )

    count_network = 0
    if needs_network:
        count_network = math.ceil(requirement.network_mbps / node.network_mbps)

    count_disk_capacity = 0
    if needs_disk:
        count_disk_capacity = math.ceil(disk.gib / node.disk.gib)

    count_disk_iops = 0
    if disk.io_bytes_per_s > 0 and node_io_bytes_per_s:
        count_disk_iops = math.ceil(disk.io_bytes_per_s / node_io_bytes_per_s)

    return {
        "count_cpu": _cpu_count(requirement.cpu, node),
        "count_memory": count_memory,
        "count_network": count_network,
        "count_disk_capacity": count_disk_capacity,
        "count_disk_iops": count_disk_iops,
    }


def floor_ignoring_float_error(ratio: float) -> int:
    """Floor a capacity ratio without losing an exact fit to float error."""
    return math.floor(math.nextafter(ratio, math.inf))


class _PartitionFit(BaseModel):
    """One partitioned requirement on a mix of nodes: the per-node partition
    limits every one of them meets."""

    requirement: Requirement
    partitions: Partitions
    nodes: List[NodeResources]
    min_count: int
    max_count: Optional[int]

    @property
    def by_disk(self) -> int:
        per_partition = self.partitions.disk_gib_per_partition
        return min(
            floor_ignoring_float_error(node.disk.gib / per_partition)
            for node in self.nodes
        )

    @property
    def by_memory(self) -> Optional[int]:
        per_partition = self.partitions.memory_gib_per_partition
        if per_partition is None:
            return None
        return min(
            floor_ignoring_float_error(node.memory.gib / per_partition)
            for node in self.nodes
        )

    @property
    def max_partitions_per_node(self) -> int:
        limits = [
            self.by_disk,
            self.by_memory,
            self.partitions.max_partitions_per_node,
        ]
        return min(limit for limit in limits if limit is not None)

    def _within_max(self, node_count: int) -> bool:
        return self.max_count is None or node_count <= self.max_count

    def replicas_needed(self, nodes_for_one_copy: int) -> int:
        """The fewest copies that meet the minimum and the CPU need, on the
        node with the least usable CPU."""
        cpu = self.requirement.cpu
        usable = min(_usable_ncu(cpu, node) for node in self.nodes)
        cpu_per_copy = nodes_for_one_copy * usable
        return max(
            self.partitions.min_replicas,
            ceil_ignoring_float_error(
                cpu_to_ncu(cpu.count, cpu.reference_shape) / cpu_per_copy
            ),
        )

    def at_layout(self, layout: Layout) -> Optional[Placement]:
        """The placement at exactly this layout, or None if it does not fit."""
        ppn, rf = layout.partitions_per_node, layout.replica_count
        if ppn > self.max_partitions_per_node:
            return None
        nodes_for_one_copy = math.ceil(self.partitions.count / ppn)
        if rf < self.replicas_needed(nodes_for_one_copy):
            return None
        node_count = max(nodes_for_one_copy * rf, self.min_count)
        if not self._within_max(node_count):
            return None
        return Placement(node_count=node_count, layout=layout)

    def bottlenecks_for(self, layout: Layout) -> List[Bottleneck]:
        """Every resource that keeps a node from hosting this layout, in the
        order disk, memory, CPU. Empty when the limit is the node count or the
        partition cap instead."""
        found = []
        if layout.partitions_per_node > self.by_disk:
            found.append(Bottleneck.disk_capacity)
        by_memory = self.by_memory
        if by_memory is not None and layout.partitions_per_node > by_memory:
            found.append(Bottleneck.memory)
        nodes_for_one_copy = math.ceil(
            self.partitions.count / layout.partitions_per_node
        )
        if layout.replica_count < self.replicas_needed(nodes_for_one_copy):
            found.append(Bottleneck.cpu)
        return found

    def layouts_at_count(self, count: int) -> List[Layout]:
        """Each way to split ``count`` nodes into equal copies, densest first."""
        return [
            Layout(
                partitions_per_node=math.ceil(self.partitions.count / per_copy),
                replica_count=count // per_copy,
            )
            for per_copy in range(1, count + 1)
            if count % per_copy == 0
        ]

    def fewest_nodes(self) -> Optional[Placement]:
        """The fewest nodes within max_count; ties go to more replicas, then
        denser packing."""
        best: Optional[Tuple[int, int, int]] = None
        # A node holds at most every partition. Reporting more would make a
        # pinned copy of this layout ask for disk for partitions that don't exist.
        ppn = min(self.max_partitions_per_node, self.partitions.count)
        while ppn > 0:
            nodes_for_one_copy = math.ceil(self.partitions.count / ppn)
            # Nodes min_count adds beyond the need hold whole copies where they fit.
            rf = max(
                self.replicas_needed(nodes_for_one_copy),
                self.min_count // nodes_for_one_copy,
            )
            nodes = max(nodes_for_one_copy * rf, self.min_count)
            if self._within_max(nodes):
                candidate = (nodes, -rf, -ppn)
                if best is None or candidate < best:
                    best = candidate
            # Every skipped density has the same nodes per copy, so the same
            # replicas and node count; the current density wins their tie.
            ppn = (self.partitions.count - 1) // nodes_for_one_copy
        if best is None:
            return None
        return self.at_layout(
            Layout(partitions_per_node=-best[2], replica_count=-best[1])
        )

    def at_pinned_layout(
        self, layout: Layout, count: Optional[int]
    ) -> Union[Placement, LimitNotMet]:
        placed = self.at_layout(layout)
        if placed is not None and count in (None, placed.node_count):
            return placed
        on_count = "" if count is None else f" on {count} nodes"
        return LimitNotMet(
            reason=f"Cannot host {layout.partitions_per_node} partitions per node "
            f"at {layout.replica_count} replicas{on_count}",
            shapes=_names(self.nodes),
            bottlenecks=self.bottlenecks_for(layout),
        )

    def at_pinned_count(self, count: int) -> Union[Placement, LimitNotMet]:
        # Blame the densest layout whose partitions fit on disk and in memory,
        # else the sparsest layout, and list every resource that bottlenecks it.
        blamed: Optional[Layout] = None
        for candidate in self.layouts_at_count(count):
            placed = self.at_layout(candidate)
            if placed is not None and placed.node_count == count:
                return placed
            if blamed is None or set(self.bottlenecks_for(blamed)) - {Bottleneck.cpu}:
                blamed = candidate
        return LimitNotMet(
            reason=f"Cannot host {count} nodes",
            shapes=_names(self.nodes),
            bottlenecks=[] if blamed is None else self.bottlenecks_for(blamed),
        )

    def place(
        self, layout: Optional[Layout], count: Optional[int]
    ) -> Union[Placement, NodeCannotServe, LimitNotMet]:
        if self.max_partitions_per_node < 1:
            return self.no_room_for_one()
        if layout is not None:
            return self.at_pinned_layout(layout, count)
        if count is not None:
            return self.at_pinned_count(count)
        return self.unpinned()

    def unpinned(self) -> Union[Placement, LimitNotMet]:
        placed = self.fewest_nodes()
        if placed is not None:
            return placed
        return LimitNotMet(
            reason=f"Needs more than {self.max_count} nodes",
            shapes=_names(self.nodes),
        )

    def no_room_for_one(self) -> NodeCannotServe:
        """Each node without disk or memory for one partition, and what it
        lacks."""
        disk, memory = (
            self.partitions.disk_gib_per_partition,
            self.partitions.memory_gib_per_partition,
        )
        no_disk = [
            node
            for node in self.nodes
            if floor_ignoring_float_error(node.disk.gib / disk) < 1
        ]
        no_memory = [
            node
            for node in self.nodes
            if memory is not None
            and floor_ignoring_float_error(node.memory.gib / memory) < 1
        ]
        found = []
        if no_disk:
            found.append(Bottleneck.disk_capacity)
        if no_memory:
            found.append(Bottleneck.memory)
        return NodeCannotServe(
            reason="One partition does not fit on a node",
            shapes=_names([*no_disk, *no_memory]),
            bottlenecks=found,
        )
