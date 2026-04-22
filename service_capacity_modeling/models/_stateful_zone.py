import logging
import math
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Tuple

from service_capacity_modeling.interface import Drive
from service_capacity_modeling.interface import Instance
from service_capacity_modeling.interface import NodeCountContext
from service_capacity_modeling.interface import ZoneClusterCapacity
from service_capacity_modeling.models import utils

logger = logging.getLogger(__name__)


# pylint: disable=too-many-locals
# pylint: disable=too-many-statements
def compute_stateful_zone(  # pylint: disable=too-many-positional-arguments
    instance: Instance,
    drive: Drive,
    needed_cores: int,
    needed_disk_gib: float,
    needed_memory_gib: float,
    needed_network_mbps: float,
    # Cloud drives may need to scale for IOs, and datastores might need more
    # or less IOs for a given data size as well as space
    # Contract for disk ios is
    # (per_node_size_gib, node_count) -> (read_ios, write_ios)
    required_disk_ios: Callable[[float, int], Tuple[float, float]] = lambda size_gib,
    count: (0, 0),
    # Some stateful clusters have sidecars that take memory
    reserve_memory: Callable[[float], float] = lambda x: 0,
    # How much write buffer we get per instance (usually a percentage of
    # the reserved memory, e.g. for buffering writes in heap)
    write_buffer: Callable[[float], float] = lambda x: 0,
    required_write_buffer_gib: float = 0,
    # Round the raw node count to a technology-specific cluster size
    # (for example even counts, powers of two, or an existing base size).
    cluster_size: Callable[[int], int] = lambda x: x,
    min_count: int = 0,
    adjusted_disk_io_needed: float = 0.0,
    read_write_ratio: float = 0.0,
    # Maximum EBS volume size per node. Default caps at 1/3 max to leave
    # growth headroom. Models can override for clusters with known disk needs.
    max_node_disk_gib: Callable[[Drive], int] = lambda d: math.ceil(d.max_size_gib / 3),
    include_node_count_breakdown: bool = False,
) -> ZoneClusterCapacity:
    count_cpu = math.ceil(needed_cores / instance.cpu)
    count_memory = math.ceil(
        needed_memory_gib / (instance.ram_gib - reserve_memory(instance.ram_gib))
    )
    if write_buffer(instance.ram_gib) > 0:
        count_memory = max(
            count_memory,
            math.ceil(required_write_buffer_gib / (write_buffer(instance.ram_gib))),
        )

    count_network = math.ceil(needed_network_mbps / instance.net_mbps)
    count_disk_capacity = 0
    count_disk_iops = 0

    if instance.drive is not None and instance.drive.size_gib > 0:
        count_disk_capacity, count_disk_iops = _local_disk_node_counts(
            instance=instance,
            needed_disk_gib=needed_disk_gib,
            adjusted_disk_io_needed=adjusted_disk_io_needed,
            read_write_ratio=read_write_ratio,
        )

    attached_drives: List[Drive] = []
    if instance.drive is None and needed_disk_gib > 0:
        count_disk_capacity, count_disk_iops, attached_drives = _attached_drive_plan(
            drive=drive,
            needed_disk_gib=needed_disk_gib,
            count_cpu=count_cpu,
            count_memory=count_memory,
            count_network=count_network,
            cluster_size=cluster_size,
            min_count=min_count,
            required_disk_ios=required_disk_ios,
            max_node_disk_gib=max_node_disk_gib,
        )

    raw_count = max(
        count_cpu, count_memory, count_network, count_disk_capacity, count_disk_iops
    )
    cluster_size_count = cluster_size(raw_count)
    count = max(cluster_size_count, min_count)

    logger.debug(
        "For (cpu, memory_gib, disk_gib) = (%s, %s, %s) need (%s, %s, %s, %s)",
        needed_cores,
        needed_memory_gib,
        needed_disk_gib,
        count,
        instance.name,
        attached_drives,
        count * instance.annual_cost
        + (attached_drives[0].annual_cost * count if attached_drives else 0),
    )

    cluster_params: Dict[str, Any] = {}
    if include_node_count_breakdown:
        cluster_params = NodeCountContext.from_counts(
            count_cpu=count_cpu,
            count_memory=count_memory,
            count_network=count_network,
            count_disk_capacity=count_disk_capacity,
            count_disk_iops=count_disk_iops,
            cluster_size_count=cluster_size_count,
            min_count=min_count,
        ).model_dump(mode="json")

    return ZoneClusterCapacity(
        cluster_type="stateful-cluster",
        count=count,
        instance=instance,
        attached_drives=attached_drives,
        cluster_params=cluster_params,
    )


def gp2_gib_for_io(read_ios: float) -> int:
    return int(max(1, read_ios // 3))


def cloud_gib_for_io(drive: Drive, total_ios: float, space_gib: float) -> int:
    if drive.name == "gp2":
        return gp2_gib_for_io(total_ios)
    return int(space_gib)


def _local_disk_node_counts(
    *,
    instance: Instance,
    needed_disk_gib: float,
    adjusted_disk_io_needed: float,
    read_write_ratio: float,
) -> Tuple[int, int]:
    assert instance.drive is not None
    count_disk_capacity = math.ceil(needed_disk_gib / instance.drive.size_gib)
    count_disk_iops = 0
    if adjusted_disk_io_needed != 0.0:
        instance_read_iops = (
            instance.drive.read_io_per_s
            if instance.drive.read_io_per_s is not None
            else 0
        )
        assert isinstance(instance_read_iops, int)
        instance_write_iops = (
            instance.drive.write_io_per_s
            if instance.drive.write_io_per_s is not None
            else 0
        )
        assert isinstance(instance_write_iops, int)
        instance_adjusted_io = (
            (
                read_write_ratio * float(instance_read_iops)
                + (1.0 - read_write_ratio) * float(instance_write_iops)
            )
            * instance.drive.block_size_kib
            * 1024.0
        )
        if instance_adjusted_io != 0.0:
            count_disk_iops = math.ceil(adjusted_disk_io_needed / instance_adjusted_io)
    return count_disk_capacity, count_disk_iops


def _attached_drive_plan(
    *,
    drive: Drive,
    needed_disk_gib: float,
    count_cpu: int,
    count_memory: int,
    count_network: int,
    cluster_size: Callable[[int], int],
    min_count: int,
    required_disk_ios: Callable[[float, int], Tuple[float, float]],
    max_node_disk_gib: Callable[[Drive], int],
) -> Tuple[int, int, List[Drive]]:
    preliminary_resource_count = max(count_cpu, count_memory, count_network)
    preliminary_count = max(cluster_size(preliminary_resource_count), min_count)

    space_gib = max(1, math.ceil(needed_disk_gib / preliminary_count))
    read_io, write_io = required_disk_ios(space_gib, preliminary_count)
    read_io, write_io = (
        utils.next_n(read_io, n=200),
        utils.next_n(write_io, n=200),
    )
    io_gib = cloud_gib_for_io(drive, read_io + write_io, space_gib)
    ebs_gib = utils.next_n(max(1, io_gib, space_gib), n=100)

    count_disk_capacity = 0
    max_size = max_node_disk_gib(drive)
    if max_size > 0:
        count_disk_capacity = math.ceil(needed_disk_gib / max_size)
    if ebs_gib > max_size > 0:
        count_disk_capacity = max(
            count_disk_capacity,
            math.ceil(preliminary_count * ebs_gib / max_size),
        )
        ebs_gib = int(max_size)

    effective_count = max(
        cluster_size(max(preliminary_count, count_disk_capacity)), min_count
    )
    read_io, write_io = required_disk_ios(space_gib, effective_count)
    read_io, write_io = (
        utils.next_n(read_io, n=200),
        utils.next_n(write_io, n=200),
    )

    count_disk_iops = 0
    if (read_io + write_io) > drive.max_io_per_s:
        ratio = (read_io + write_io) / drive.max_io_per_s
        count_disk_iops = math.ceil(effective_count * ratio)
        iops_count = max(cluster_size(count_disk_iops), min_count)
        read_io, write_io = required_disk_ios(space_gib, iops_count)
        read_io, write_io = (
            utils.next_n(read_io, n=200),
            utils.next_n(write_io, n=200),
        )

    attached_drive = drive.model_copy()
    attached_drive.size_gib = ebs_gib
    attached_drive.read_io_per_s = int(round(read_io, 2))
    attached_drive.write_io_per_s = int(round(write_io, 2))
    return count_disk_capacity, count_disk_iops, [attached_drive]
