"""
Place a requirement on nodes of any shape.

A model states what a workload needs once, without picking a shape (a
``Requirement``). A node states what it offers after reservations
(``NodeResources``): an EC2 instance, a container slice, or anything else with
CPU, memory, disk and network. ``place`` answers how many of those nodes give
every resource and all of the data a place to live, or returns ``DoesNotFit``
with every resource that bottlenecks it. ``mixable`` answers whether one
cluster may run several node shapes at once.

CPU is stated as a vCPU count plus the shape it was measured on. ``place``
converts both sides to NCUs (normalized compute units, see ``cpu_to_ncu``)
internally, so one requirement can be placed on any shape.
"""

import math
from typing import Callable
from typing import Dict
from typing import FrozenSet
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
    """How much interference and variance a workload tolerates."""

    performance = "performance"  # CPU pressure near zero, consistent cores
    balanced = "balanced"  # default for tier 0 and 1
    efficient = "efficient"  # a small steady-state CPU queue is fine


class Consistency(StrEnum):
    """Measured per-core performance consistency within one node."""

    high = "high"  # under 10% per-core variation
    medium = "medium"  # under 25%
    low = "low"  # more, or a shared host


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
    as in Cassandra and Kafka. ``cpu.parallelism`` and ``isolation`` hold for
    every node.

    With ``partitions``, disk and memory come from the per-partition sizes,
    and CPU is met by adding whole replicas. Leave the other totals at 0;
    network, disk IO, write buffer and CPU bounds are not modeled for
    partitions yet.
    """

    cpu: CPURequirement
    memory: MemoryRequirement = MemoryRequirement()
    disk: DiskRequirement = DiskRequirement()
    network_mbps: float = Field(default=0, ge=0)
    isolation: Optional[Isolation] = Field(
        default=None, description="None places on any node."
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
    consistency: Optional[Consistency] = Field(
        default=None, description="None when unmeasured."
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
        description="None when unknown. A node with neither places no IO limit.",
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
    """Why no count of these nodes fits: a reason, and every resource that
    bottlenecks the placement.

    Each caller lists ``bottlenecks`` in a fixed order. The list is empty
    when the limit is not a resource, such as a node count.
    """

    reason: str
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


# Core consistency each isolation level accepts; an unmeasured node passes
# every level but performance.
_ALLOWED_CONSISTENCY: Dict[Isolation, FrozenSet[Optional[Consistency]]] = {
    Isolation.performance: frozenset({Consistency.high}),
    Isolation.balanced: frozenset({Consistency.high, Consistency.medium, None}),
    Isolation.efficient: frozenset({*Consistency, None}),
}


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
    cluster_size: Optional[Callable[[int], int]] = None,
    min_count: int = 0,
    max_count: Optional[int] = None,
    layout: Optional[Layout] = None,
    count: Optional[int] = None,
) -> Union[Placement, DoesNotFit]:
    """Count how many nodes like ``node`` the ``requirement`` needs.

    Without partitions, each resource total asks for ceil(total / per-node
    amount) nodes. The most any resource asks for is rounded by
    ``cluster_size`` (such as to a power of two), then raised to
    ``min_count``.

    With partitions, as many whole partitions go on each node as fit on its
    disk and in its memory, and the node count is nodes per replica times
    replicas: at least ``min_replicas``, more if CPU needs them. The fewest
    nodes win, between ``min_count`` and ``max_count``; ties go to more
    replicas, then denser packing. ``layout`` pins the layout and ``count``
    the node count; with only ``count``, the densest layout on exactly that
    many nodes wins. ``cluster_size`` does not apply.

    Returns ``DoesNotFit`` with every bottleneck when the node has fewer
    physical cores than ``cpu.parallelism``, cores too inconsistent for
    ``isolation``, lacks a resource the requirement needs, when one partition
    does not fit on a node, or when a limit or pin cannot be met.
    """
    if requirement.partitions is None:
        if layout is not None or count is not None or max_count is not None:
            raise ValueError("layout, count and max_count need partitions")
    elif cluster_size is not None:
        raise ValueError("cluster_size does not apply to partitions")
    unfit = _unfit_cpu(requirement, node)
    if unfit is not None:
        return unfit
    if requirement.partitions is None:
        return _place_evenly(
            requirement, node, cluster_size or (lambda nodes: nodes), min_count
        )
    fit = _PartitionFit(
        requirement=requirement,
        partitions=requirement.partitions,
        node=node,
        min_count=min_count,
        max_count=max_count,
    )
    if layout is not None:
        return fit.at_pinned_layout(layout, count)
    if count is not None:
        return fit.at_pinned_count(count)
    return fit.unpinned()


def mixable(
    requirement: Requirement, nodes: Sequence[NodeResources]
) -> Optional[DoesNotFit]:
    """Whether one cluster may run all of these nodes at the requirement's
    isolation: None when it may, else why not.

    Each node's core consistency must suit the isolation level, and a
    performance cluster runs one shape only. A spread limit on per-node CPU
    for balanced clusters is not set yet.
    """
    for node in nodes:
        unfit = _unfit_cpu(requirement, node)
        if unfit is not None:
            return unfit
    shapes = sorted({node.shape.name for node in nodes})
    if requirement.isolation == Isolation.performance and len(shapes) > 1:
        return DoesNotFit(
            reason=f"A performance cluster runs one shape, not {', '.join(shapes)}",
            bottlenecks=[Bottleneck.cpu],
        )
    return None


def _unfit_cpu(requirement: Requirement, node: NodeResources) -> Optional[DoesNotFit]:
    """Why the node's CPU cannot serve the requirement at all, or None."""
    cores = node.cpu.count * node.shape.cores / node.shape.cpu
    if cores < requirement.cpu.parallelism:
        return DoesNotFit(
            reason=f"{node.shape.name} has {cores:g} physical cores, fewer than "
            f"the {requirement.cpu.parallelism} the workload runs in parallel",
            bottlenecks=[Bottleneck.cpu],
        )
    isolation = requirement.isolation
    if isolation is not None and (
        node.cpu.consistency not in _ALLOWED_CONSISTENCY[isolation]
    ):
        return DoesNotFit(
            reason=f"{node.shape.name} cores are {node.cpu.consistency or 'unmeasured'}"
            f", too inconsistent for {isolation} isolation",
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


def _place_evenly(
    requirement: Requirement,
    node: NodeResources,
    cluster_size: Callable[[int], int],
    min_count: int,
) -> Union[Placement, DoesNotFit]:
    count_memory = _memory_count(requirement.memory, node.memory)
    needs_network = requirement.network_mbps > 0
    disk = requirement.disk
    needs_disk = disk.gib > 0
    # Name every resource the node lacks, so adding one does not reveal the next.
    missing = []
    if count_memory is None:
        missing.append(Bottleneck.memory)
    if needs_network and node.network_mbps <= 0:
        missing.append(Bottleneck.network)
    if needs_disk and node.disk.gib <= 0:
        missing.append(Bottleneck.disk_capacity)
    if missing or count_memory is None:
        names = " or ".join(bottleneck.value for bottleneck in missing)
        return DoesNotFit(
            reason=f"The node has no {names} for the requirement",
            bottlenecks=missing,
        )

    count_network = 0
    if needs_network:
        count_network = math.ceil(requirement.network_mbps / node.network_mbps)

    count_disk_capacity = 0
    if needs_disk:
        count_disk_capacity = math.ceil(disk.gib / node.disk.gib)

    count_disk_iops = 0
    node_io_bytes_per_s = disk.read_fraction * (node.disk.read_bytes_per_s or 0) + (
        1 - disk.read_fraction
    ) * (node.disk.write_bytes_per_s or 0)
    if disk.io_bytes_per_s > 0 and node_io_bytes_per_s > 0:
        count_disk_iops = math.ceil(disk.io_bytes_per_s / node_io_bytes_per_s)

    return round_zone_count(
        count_cpu=_cpu_count(requirement.cpu, node),
        count_memory=count_memory,
        count_network=count_network,
        count_disk_capacity=count_disk_capacity,
        count_disk_iops=count_disk_iops,
        cluster_size=cluster_size,
        min_count=min_count,
    )


def floor_ignoring_float_error(ratio: float) -> int:
    """Floor a capacity ratio without losing an exact fit to float error."""
    return math.floor(math.nextafter(ratio, math.inf))


class _PartitionFit(BaseModel):
    """One partitioned requirement on one node: the per-node partition limits."""

    requirement: Requirement
    partitions: Partitions
    node: NodeResources
    min_count: int
    max_count: Optional[int]

    @property
    def by_disk(self) -> int:
        return floor_ignoring_float_error(
            self.node.disk.gib / self.partitions.disk_gib_per_partition
        )

    @property
    def by_memory(self) -> Optional[int]:
        per_partition = self.partitions.memory_gib_per_partition
        if per_partition is None:
            return None
        return floor_ignoring_float_error(self.node.memory.gib / per_partition)

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
        """The fewest copies that meet the minimum and the CPU need."""
        cpu = self.requirement.cpu
        cpu_per_copy = nodes_for_one_copy * _usable_ncu(cpu, self.node)
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
            rf = self.replicas_needed(nodes_for_one_copy)
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
    ) -> Union[Placement, DoesNotFit]:
        placed = self.at_layout(layout)
        if placed is not None and count in (None, placed.node_count):
            return placed
        on_count = "" if count is None else f" on {count} nodes"
        return DoesNotFit(
            reason=f"Cannot host {layout.partitions_per_node} partitions per node "
            f"at {layout.replica_count} replicas{on_count}",
            bottlenecks=self.bottlenecks_for(layout),
        )

    def at_pinned_count(self, count: int) -> Union[Placement, DoesNotFit]:
        # Blame the densest layout whose partitions fit on disk and in memory,
        # else the sparsest layout, and list every resource that bottlenecks it.
        blamed: Optional[Layout] = None
        for candidate in self.layouts_at_count(count):
            placed = self.at_layout(candidate)
            if placed is not None and placed.node_count == count:
                return placed
            if blamed is None or set(self.bottlenecks_for(blamed)) - {Bottleneck.cpu}:
                blamed = candidate
        return DoesNotFit(
            reason=f"Cannot host {count} nodes",
            bottlenecks=[] if blamed is None else self.bottlenecks_for(blamed),
        )

    def unpinned(self) -> Union[Placement, DoesNotFit]:
        placed = self.fewest_nodes()
        if placed is not None:
            return placed
        if self.max_partitions_per_node >= 1:
            return DoesNotFit(reason=f"Needs more than {self.max_count} nodes")
        by_memory = self.by_memory
        found = []
        if self.by_disk < 1:
            found.append(Bottleneck.disk_capacity)
        if by_memory is not None and by_memory < 1:
            found.append(Bottleneck.memory)
        return DoesNotFit(
            reason="One partition does not fit on a node",
            bottlenecks=found,
        )
