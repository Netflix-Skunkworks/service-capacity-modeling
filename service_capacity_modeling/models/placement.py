"""
Estimate how many nodes of any shape a requirement needs.

A model states what a workload needs once, without picking a shape (a
requirement, like summed Kubernetes resource requests). A node states what
it offers after reservations (``NodeAllocatable``, like a Kubernetes node's
allocatable resources): an EC2 instance, a container size, or anything else
with CPU, memory, disk and network. ``estimate_zone_nodes`` answers how many
of those nodes meet the requirement, or returns ``Unschedulable`` with every
resource the node is insufficient on.

CPU on both sides is counted in NCUs (normalized compute units, see
``cpu_to_ncu``), so one requirement can be estimated on any shape.
"""

import math
from typing import Callable
from typing import List
from typing import Optional
from typing import Union

from pydantic import BaseModel
from pydantic import Field

from service_capacity_modeling.interface import Bottleneck
from service_capacity_modeling.interface import NodeCountContext


def ceil_ignoring_float_error(value: float) -> int:
    """Ceil a need without adding a unit for float error.

    Converting cores between shapes multiplies by a speed ratio, so an exact
    whole number can come out a few ulps over (966 * 4/7 is
    552.0000000000001) and a plain ceil adds a core, and sometimes a node.
    """
    return math.ceil(value * (1 - 1e-9))


class NodeAllocatable(BaseModel):
    """What one node offers after reservations."""

    ncu: float = Field(gt=0, description="NCUs (normalized compute units).")
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


class Unschedulable(BaseModel):
    """Why no count of these nodes fits: a reason, and every resource the
    node is insufficient on.

    Each caller lists ``insufficient`` in a fixed order. The list is empty
    when the limit is not a resource, such as a node count.
    """

    reason: str
    insufficient: List[Bottleneck] = Field(default_factory=list)


class ZoneRequirement(BaseModel):
    """What one zone's copy of a workload needs, independent of node shape.

    Each resource is a total that spreads evenly over the zone's nodes.
    """

    ncu: float = Field(ge=0, description="NCUs (normalized compute units).")
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


class ZoneNodeEstimate(BaseModel):
    """How many nodes a zone needs, and how many each resource asked for."""

    count: int
    node_counts: NodeCountContext


def round_zone_count(  # pylint: disable=too-many-arguments
    *,
    count_cpu: int,
    count_memory: int,
    count_network: int,
    count_disk_capacity: int,
    count_disk_iops: int,
    cluster_size: Callable[[int], int],
    min_count: int,
) -> ZoneNodeEstimate:
    """Take the most nodes any resource needs, round that to a cluster size
    the technology allows, then apply the minimum count."""
    raw_count = max(
        count_cpu, count_memory, count_network, count_disk_capacity, count_disk_iops
    )
    cluster_size_count = cluster_size(raw_count)
    return ZoneNodeEstimate(
        count=max(cluster_size_count, min_count),
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


def _memory_count(requirement: ZoneRequirement, node: NodeAllocatable) -> Optional[int]:
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


def estimate_zone_nodes(
    requirement: ZoneRequirement,
    node: NodeAllocatable,
    cluster_size: Callable[[int], int] = lambda count: count,
    min_count: int = 0,
) -> Union[ZoneNodeEstimate, Unschedulable]:
    """Estimate how many nodes like ``node`` one zone's ``requirement`` needs.

    ``cluster_size`` rounds the node count to one the technology allows, such
    as a power of two; ``min_count`` is the floor after rounding.
    """
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
        return Unschedulable(
            reason=f"The node has no {names} for the requirement",
            insufficient=missing,
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
        count_cpu=ceil_ignoring_float_error(requirement.ncu / node.ncu),
        count_memory=count_memory,
        count_network=count_network,
        count_disk_capacity=count_disk_capacity,
        count_disk_iops=count_disk_iops,
        cluster_size=cluster_size,
        min_count=min_count,
    )
