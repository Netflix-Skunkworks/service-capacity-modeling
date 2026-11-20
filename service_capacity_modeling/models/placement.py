"""
Place a requirement on nodes of any shape.

A model states what a workload needs once, without picking a shape (a
``Requirement``). A node states what it offers after reservations
(``NodeResources``): an EC2 instance, a container size, or anything else with
CPU, memory, disk and network. ``place`` answers how many of those nodes give
every resource and all of the data a place to live, or returns ``DoesNotFit``
with every resource that bottlenecks it.

CPU on both sides is counted in NCUs (normalized compute units, see
``cpu_to_ncu``), so one requirement can be placed on any shape.
"""

import math
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

from pydantic import BaseModel
from pydantic import Field
from pydantic import model_validator

from service_capacity_modeling.interface import Bottleneck
from service_capacity_modeling.interface import NodeCountContext


def ceil_ignoring_float_error(value: float) -> int:
    """Ceil a need without adding a unit for float error.

    Converting cores between shapes multiplies by a speed ratio, so an exact
    whole number can come out a few ulps over (966 * 4/7 is
    552.0000000000001) and a plain ceil adds a core, and sometimes a node.
    """
    return math.ceil(value * (1 - 1e-9))


class NodeResources(BaseModel):
    """What one node offers after reservations."""

    ncu: float = Field(gt=0, description="NCUs (normalized compute units).")
    cpu_headroom: float = Field(
        default=0,
        ge=0,
        lt=1,
        description="Fraction of ncu kept free so requests do not queue, "
        "which shrinks as a node has more vCPUs. Applied only to a "
        "requirement whose NCUs are busy NCUs.",
    )
    disk_gib: float = Field(
        default=0, ge=0, description="Local disk. 0 when the node has none."
    )
    memory_gib: float = Field(
        default=0, ge=0, description="Used only when the requirement needs memory."
    )
    network_mbps: float = Field(
        default=0, ge=0, description="Used only when the requirement needs network."
    )
    disk_io_bytes_per_s: Optional[float] = Field(
        default=None,
        ge=0,
        description="Local disk IO, weighted by the requirement's read/write "
        "mix. None when unknown, which places no IO limit.",
    )
    write_buffer_gib: float = Field(
        default=0,
        ge=0,
        description="Memory for buffering writes. 0 when the node models no "
        "write buffer, which places no write-buffer limit.",
    )


class DoesNotFit(BaseModel):
    """Why no count of these nodes fits: a reason, and every resource that
    bottlenecks the placement.

    Each caller lists ``bottlenecks`` in a fixed order. The list is empty
    when the limit is not a resource, such as a node count.
    """

    reason: str
    bottlenecks: List[Bottleneck] = Field(default_factory=list)


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
    as in Cassandra and Kafka.

    With ``partitions``, disk and memory come from the per-partition sizes,
    and CPU is met by adding whole replicas. Leave the other totals at 0;
    network, disk IO and write buffer are not modeled for partitions yet.
    """

    ncu: float = Field(ge=0, description="NCUs (normalized compute units).")
    ncu_is_busy: bool = Field(
        default=False,
        description="True when ncu is measured busy compute, such as current "
        "utilization, so each node keeps its cpu_headroom free. False when "
        "ncu already allows for queueing, as sqrt staffing does.",
    )
    min_total_ncu: Optional[float] = Field(
        default=None,
        ge=0,
        description="Fewest NCUs the nodes may have in total, counting all "
        "of each node's NCUs, such as to keep a current cluster's CPU.",
    )
    max_total_ncu: Optional[float] = Field(
        default=None,
        gt=0,
        description="Most NCUs the nodes may have in total, counting all of "
        "each node's NCUs, such as to not grow past a current cluster's CPU. "
        "min_total_ncu wins when the two conflict.",
    )
    memory_gib: float = Field(default=0, ge=0)
    network_mbps: float = Field(default=0, ge=0)
    disk_gib: float = Field(default=0, ge=0)
    disk_io_bytes_per_s: float = Field(
        default=0, ge=0, description="On local disk, weighted by read/write mix."
    )
    write_buffer_gib: float = Field(
        default=0,
        ge=0,
        description="Write-buffer memory, which no single node may run short of.",
    )
    partitions: Optional[Partitions] = None

    @model_validator(mode="after")
    def _partitions_carry_disk_and_memory(self) -> "Requirement":
        if self.partitions is None:
            return self
        totals = {
            "memory_gib": self.memory_gib,
            "network_mbps": self.network_mbps,
            "disk_gib": self.disk_gib,
            "disk_io_bytes_per_s": self.disk_io_bytes_per_s,
            "write_buffer_gib": self.write_buffer_gib,
            "min_total_ncu": self.min_total_ncu,
            "max_total_ncu": self.max_total_ncu,
        }
        set_totals = [name for name, value in totals.items() if value]
        if set_totals:
            raise ValueError(
                f"{', '.join(set_totals)} cannot be set with partitions: disk "
                "and memory come from the partitions, and the rest is not "
                "modeled for partitions yet"
            )
        return self


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

    Returns ``DoesNotFit`` with every bottleneck when the node lacks a
    resource the requirement needs, when one partition does not fit on a
    node, or when a limit or pin cannot be met.
    """
    if requirement.partitions is None:
        if layout is not None or count is not None or max_count is not None:
            raise ValueError("layout, count and max_count need partitions")
        return _place_evenly(
            requirement, node, cluster_size or (lambda nodes: nodes), min_count
        )
    if cluster_size is not None:
        raise ValueError("cluster_size does not apply to partitions")
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


def usable_ncu(requirement: Requirement, node: NodeResources) -> float:
    """NCUs of one node the requirement may use: all of them, or for busy
    NCUs, what is left after the node's queueing headroom."""
    if requirement.ncu_is_busy:
        return node.ncu * (1 - node.cpu_headroom)
    return node.ncu


def _cpu_count(requirement: Requirement, node: NodeResources) -> int:
    """Nodes for CPU, then held between the total-NCU bounds."""
    count = ceil_ignoring_float_error(requirement.ncu / usable_ncu(requirement, node))
    if requirement.max_total_ncu is not None:
        count = min(
            count, ceil_ignoring_float_error(requirement.max_total_ncu / node.ncu)
        )
    if requirement.min_total_ncu is not None:
        count = max(
            count, ceil_ignoring_float_error(requirement.min_total_ncu / node.ncu)
        )
    return count


def _memory_count(requirement: Requirement, node: NodeResources) -> Optional[int]:
    """Nodes for memory and the write buffer, or None if the node has no
    memory for a requirement that needs some."""
    count = 0
    if requirement.memory_gib > 0:
        if node.memory_gib <= 0:
            return None
        count = math.ceil(requirement.memory_gib / node.memory_gib)
        # Division can round an exact capacity boundary above an integer.
        if (count - 1) * node.memory_gib >= requirement.memory_gib:
            count -= 1
    if requirement.write_buffer_gib > 0 and node.write_buffer_gib > 0:
        count = max(
            count, math.ceil(requirement.write_buffer_gib / node.write_buffer_gib)
        )
    return count


def _place_evenly(
    requirement: Requirement,
    node: NodeResources,
    cluster_size: Callable[[int], int],
    min_count: int,
) -> Union[Placement, DoesNotFit]:
    count_memory = _memory_count(requirement, node)
    needs_network = requirement.network_mbps > 0
    needs_disk = requirement.disk_gib > 0
    # Name every resource the node lacks, so adding one does not reveal the next.
    missing = []
    if count_memory is None:
        missing.append(Bottleneck.memory)
    if needs_network and node.network_mbps <= 0:
        missing.append(Bottleneck.network)
    if needs_disk and node.disk_gib <= 0:
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
        count_disk_capacity = math.ceil(requirement.disk_gib / node.disk_gib)

    count_disk_iops = 0
    if requirement.disk_io_bytes_per_s > 0 and node.disk_io_bytes_per_s:
        count_disk_iops = math.ceil(
            requirement.disk_io_bytes_per_s / node.disk_io_bytes_per_s
        )

    return round_zone_count(
        count_cpu=_cpu_count(requirement, node),
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
            self.node.disk_gib / self.partitions.disk_gib_per_partition
        )

    @property
    def by_memory(self) -> Optional[int]:
        per_partition = self.partitions.memory_gib_per_partition
        if per_partition is None:
            return None
        return floor_ignoring_float_error(self.node.memory_gib / per_partition)

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
        cpu_per_copy = nodes_for_one_copy * usable_ncu(self.requirement, self.node)
        return max(
            self.partitions.min_replicas,
            ceil_ignoring_float_error(self.requirement.ncu / cpu_per_copy),
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
