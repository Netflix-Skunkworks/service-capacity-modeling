"""
Place a partitioned, replicated dataset on nodes of any shape.

A workload states what it needs once, without picking a shape
(``PartitionedRequirement``). A node states what it offers after reservations
(``NodeSupply``): an EC2 instance, a container size, or anything else with
cores, disk and memory. ``place`` answers how many of those nodes hold the
data, with how many partitions on each node and how many copies.

Without a pin, ``place`` returns the fewest nodes, then prefers more
replicas, then denser packing. A caller can instead pin the layout or the
node count; when the pin does not fit, ``place`` says which resource blocks it.

CPU on both sides is counted in NCUs (normalized compute units, see
``cpu_to_ncu``), so a requirement can be placed on any shape without
re-deriving it.
"""

import math
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

from pydantic import BaseModel
from pydantic import Field

from service_capacity_modeling.interface import Bottleneck
from service_capacity_modeling.models.common import ceil_ignoring_float_error


class PartitionedRequirement(BaseModel):
    """What a partitioned dataset needs, independent of any node shape."""

    partitions: int = Field(ge=1)
    disk_gib_per_partition: float = Field(
        gt=0, description="Disk one copy of a partition needs, with headroom."
    )
    memory_gib_per_partition: Optional[float] = Field(
        default=None,
        gt=0,
        description="Memory one copy of a partition needs, with headroom. "
        "None places partitions by disk only.",
    )
    cpu_ncu: float = Field(
        ge=0,
        description="NCUs (normalized compute units) to serve all traffic. Any "
        "copy serves reads, so this is spread across every node.",
    )
    min_replicas: int = Field(ge=1)
    max_nodes: int = Field(ge=1)
    max_partitions_per_node: Optional[int] = Field(default=None, ge=1)


class NodeSupply(BaseModel):
    """What one node offers the dataset after reservations."""

    cpu_ncu: float = Field(gt=0, description="NCUs (normalized compute units).")
    disk_gib: float = Field(ge=0)
    memory_gib: float = Field(
        default=0, ge=0, description="Used only when the requirement places memory."
    )


class Layout(BaseModel):
    """How partitions sit on nodes: partitions per node and complete copies."""

    partitions_per_node: int = Field(ge=1)
    replica_count: int = Field(ge=1)


class Placement(BaseModel):
    """The dataset placed on some number of one kind of node."""

    node_count: int
    replica_count: int
    partitions_per_node: int
    nodes_for_one_copy: int
    nodes_for_cpu: int
    max_partitions_per_node_by_disk: int
    max_partitions_per_node_by_memory: Optional[int] = None
    # What the placement uses on each node, comparable with NodeSupply.
    disk_gib_per_node: float
    memory_gib_per_node: float
    cpu_utilization: float


class Shortfall(BaseModel):
    """Why nothing fits: a reason, and the blocking resource when one is."""

    reason: str
    bottleneck: Optional[Bottleneck] = None


def _floor(ratio: float) -> int:
    """Floor a capacity ratio without losing an exact fit to float error."""
    return math.floor(math.nextafter(ratio, math.inf))


class _Fit(BaseModel):
    """One requirement on one node: the per-node partition limits."""

    requirement: PartitionedRequirement
    node: NodeSupply

    @property
    def by_disk(self) -> int:
        return _floor(self.node.disk_gib / self.requirement.disk_gib_per_partition)

    @property
    def by_memory(self) -> Optional[int]:
        per_partition = self.requirement.memory_gib_per_partition
        if per_partition is None:
            return None
        return _floor(self.node.memory_gib / per_partition)

    @property
    def max_partitions_per_node(self) -> int:
        limits = [
            self.by_disk,
            self.by_memory,
            self.requirement.max_partitions_per_node,
        ]
        return min(limit for limit in limits if limit is not None)

    def replicas_needed(self, nodes_for_one_copy: int) -> int:
        """The fewest copies that meet the minimum and the CPU need."""
        cpu_per_copy = nodes_for_one_copy * self.node.cpu_ncu
        return max(
            self.requirement.min_replicas,
            ceil_ignoring_float_error(self.requirement.cpu_ncu / cpu_per_copy),
        )

    def at_layout(self, layout: Layout) -> Optional[Placement]:
        """The placement at exactly this layout, or None if it does not fit."""
        ppn, rf = layout.partitions_per_node, layout.replica_count
        if ppn > self.max_partitions_per_node:
            return None
        nodes_for_one_copy = math.ceil(self.requirement.partitions / ppn)
        if rf < self.replicas_needed(nodes_for_one_copy):
            return None
        # At least 2 nodes for availability.
        node_count = max(nodes_for_one_copy * rf, 2)
        if node_count > self.requirement.max_nodes:
            return None
        memory_per_partition = self.requirement.memory_gib_per_partition or 0
        return Placement(
            node_count=node_count,
            replica_count=rf,
            partitions_per_node=ppn,
            nodes_for_one_copy=nodes_for_one_copy,
            nodes_for_cpu=ceil_ignoring_float_error(
                self.requirement.cpu_ncu / self.node.cpu_ncu
            ),
            max_partitions_per_node_by_disk=self.by_disk,
            max_partitions_per_node_by_memory=self.by_memory,
            disk_gib_per_node=ppn * self.requirement.disk_gib_per_partition,
            memory_gib_per_node=ppn * memory_per_partition,
            cpu_utilization=self.requirement.cpu_ncu
            / (nodes_for_one_copy * rf * self.node.cpu_ncu),
        )

    def layout_shortfall(self, layout: Layout) -> Optional[Bottleneck]:
        """The resource that keeps a node from hosting this layout, or None
        when the limit is the node count or the partition cap instead."""
        if layout.partitions_per_node > self.by_disk:
            return Bottleneck.disk_capacity
        by_memory = self.by_memory
        if by_memory is not None and layout.partitions_per_node > by_memory:
            return Bottleneck.memory
        nodes_for_one_copy = math.ceil(
            self.requirement.partitions / layout.partitions_per_node
        )
        if layout.replica_count < self.replicas_needed(nodes_for_one_copy):
            return Bottleneck.cpu
        return None

    def layouts_at_count(self, count: int) -> List[Layout]:
        """Each way to split ``count`` nodes into equal copies, densest first."""
        return [
            Layout(
                partitions_per_node=math.ceil(self.requirement.partitions / per_copy),
                replica_count=count // per_copy,
            )
            for per_copy in range(1, count + 1)
            if count % per_copy == 0
        ]

    def fewest_nodes(self) -> Optional[Placement]:
        """The fewest nodes within max_nodes; ties go to more replicas, then
        denser packing."""
        best: Optional[Tuple[int, int, int]] = None
        # A node holds at most every partition. Reporting more would make a
        # pinned copy of this layout ask for disk for partitions that don't exist.
        ppn = min(self.max_partitions_per_node, self.requirement.partitions)
        while ppn > 0:
            nodes_for_one_copy = math.ceil(self.requirement.partitions / ppn)
            rf = self.replicas_needed(nodes_for_one_copy)
            nodes = max(nodes_for_one_copy * rf, 2)
            if nodes <= self.requirement.max_nodes:
                candidate = (nodes, -rf, -ppn)
                if best is None or candidate < best:
                    best = candidate
            # Every skipped density has the same nodes per copy, so the same
            # replicas and node count; the current density wins their tie.
            ppn = (self.requirement.partitions - 1) // nodes_for_one_copy
        if best is None:
            return None
        return self.at_layout(
            Layout(partitions_per_node=-best[2], replica_count=-best[1])
        )

    def at_pinned_layout(
        self, layout: Layout, count: Optional[int]
    ) -> Union[Placement, Shortfall]:
        placed = self.at_layout(layout)
        if placed is not None and count in (None, placed.node_count):
            return placed
        on_count = "" if count is None else f" on {count} nodes"
        return Shortfall(
            reason=f"Cannot host {layout.partitions_per_node} partitions per node "
            f"at {layout.replica_count} replicas{on_count}",
            bottleneck=self.layout_shortfall(layout),
        )

    def at_pinned_count(self, count: int) -> Union[Placement, Shortfall]:
        # Blame the densest layout whose partitions fit on disk and in memory,
        # else the sparsest layout's resource.
        blame: Optional[Bottleneck] = None
        blamed = False
        for candidate in self.layouts_at_count(count):
            placed = self.at_layout(candidate)
            if placed is not None and placed.node_count == count:
                return placed
            if not blamed:
                blame = self.layout_shortfall(candidate)
                blamed = blame not in (Bottleneck.disk_capacity, Bottleneck.memory)
        return Shortfall(reason=f"Cannot host {count} nodes", bottleneck=blame)

    def unpinned(self) -> Union[Placement, Shortfall]:
        placed = self.fewest_nodes()
        if placed is not None:
            return placed
        if self.max_partitions_per_node >= 1:
            return Shortfall(
                reason=f"Needs more than {self.requirement.max_nodes} nodes"
            )
        by_memory = self.by_memory
        short_on_memory = by_memory is not None and by_memory < self.by_disk
        return Shortfall(
            reason="One partition does not fit on a node",
            bottleneck=Bottleneck.memory
            if short_on_memory
            else Bottleneck.disk_capacity,
        )


def place(
    requirement: PartitionedRequirement,
    node: NodeSupply,
    layout: Optional[Layout] = None,
    count: Optional[int] = None,
) -> Union[Placement, Shortfall]:
    """Place ``requirement`` on nodes like ``node``.

    With ``layout``, use exactly that layout (and, with ``count``, require that
    node count too). With only ``count``, use the densest layout on exactly
    that many nodes, which leaves the most replicas for CPU and availability.
    With neither, use the fewest nodes.
    """
    fit = _Fit(requirement=requirement, node=node)
    if layout is not None:
        return fit.at_pinned_layout(layout, count)
    if count is not None:
        return fit.at_pinned_count(count)
    return fit.unpinned()
