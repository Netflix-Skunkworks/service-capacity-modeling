"""
Fit a requirement onto nodes of any shape.

A model states what a workload needs once, without picking a shape (a
requirement). A node states what it offers after reservations
(``NodeSupply``): an EC2 instance, a container size, or anything else with
CPU, memory, disk and network. Placement answers how many of those nodes
meet the requirement, or says which resource keeps them from it (a
``Shortfall``).

CPU on both sides is counted in NCUs (normalized compute units, see
``cpu_to_ncu``), so one requirement can be placed on any shape.
"""

import math
from typing import Callable
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


class NodeSupply(BaseModel):
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


class Shortfall(BaseModel):
    """Why nothing fits: a reason, and the blocking resource when one is."""

    reason: str
    bottleneck: Optional[Bottleneck] = None


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


class ZonePlacement(BaseModel):
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
) -> ZonePlacement:
    """Take the most nodes any resource needs, round that to a cluster size
    the technology allows, then apply the minimum count."""
    raw_count = max(
        count_cpu, count_memory, count_network, count_disk_capacity, count_disk_iops
    )
    cluster_size_count = cluster_size(raw_count)
    return ZonePlacement(
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


def _memory_count(requirement: ZoneRequirement, node: NodeSupply) -> Optional[int]:
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


def place_zone(
    requirement: ZoneRequirement,
    node: NodeSupply,
    cluster_size: Callable[[int], int] = lambda count: count,
    min_count: int = 0,
) -> Union[ZonePlacement, Shortfall]:
    """Place one zone's ``requirement`` on nodes like ``node``.

    ``cluster_size`` rounds the node count to one the technology allows, such
    as a power of two; ``min_count`` is the floor after rounding.
    """
    count_memory = _memory_count(requirement, node)
    if count_memory is None:
        return Shortfall(
            reason="The node has no memory for the requirement",
            bottleneck=Bottleneck.memory,
        )

    count_network = 0
    if requirement.network_mbps > 0:
        if node.network_mbps <= 0:
            return Shortfall(reason="The node has no network for the requirement")
        count_network = math.ceil(requirement.network_mbps / node.network_mbps)

    count_disk_capacity = 0
    if requirement.disk_gib > 0:
        if node.disk_gib <= 0:
            return Shortfall(
                reason="The node has no local disk for the requirement",
                bottleneck=Bottleneck.disk_capacity,
            )
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
