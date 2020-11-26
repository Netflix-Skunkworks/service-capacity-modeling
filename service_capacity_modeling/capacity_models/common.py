import logging
import math

from service_capacity_modeling.capacity_models import utils
from service_capacity_modeling.models import CapacityDesires
from service_capacity_modeling.models import Drive
from service_capacity_modeling.models import Instance
from service_capacity_modeling.models import RegionClusterCapacity
from service_capacity_modeling.models import ZoneClusterCapacity


logger = logging.getLogger(__name__)


# In square root staffing we have to take into account the QOS parameter
# Which is related to the probability that a user queues. On low tier clusters
# (aka critical clusters) we want a lower probability of queueing
def _QOS(tier: int) -> float:
    # TODO: put the math in here for why we pick these
    if tier == 0:
        return 2
    elif tier == 1:
        return 1.5
    elif tier == 2:
        return 1.2
    else:
        return 1


def _sqrt_staffed_cores(rps: float, latency_s: float, qos: float) -> int:
    # Square root staffing
    # s = a + Q*sqrt(a)
    return int(math.ceil((rps * latency_s) + qos * math.sqrt(rps * latency_s)))


def sqrt_staffed_cores(desires: CapacityDesires) -> int:
    """Computes cores given a sqrt staffing model"""
    qos = _QOS(desires.service_tier)
    read_rps, read_lat = (
        desires.query_pattern.estimated_read_per_second.mid,
        desires.query_pattern.estimated_mean_read_latency_ms.mid / 1000.0,
    )
    write_rps, write_lat = (
        desires.query_pattern.estimated_write_per_second.mid,
        desires.query_pattern.estimated_mean_write_latency_ms.mid / 1000.0,
    )

    read_cores = _sqrt_staffed_cores(read_rps, read_lat, qos)
    write_cores = _sqrt_staffed_cores(write_rps, write_lat, qos)

    return read_cores + write_cores


def simple_network_mbps(desires: CapacityDesires) -> int:
    """Computes network mbps with a simple model"""
    read_bytes_per_second = (
        desires.query_pattern.estimated_read_per_second.mid
        * desires.query_pattern.estimated_mean_read_size_bytes.mid
    )
    write_bytes_per_second = (
        desires.query_pattern.estimated_write_per_second.mid
        * desires.query_pattern.estimated_mean_write_size_bytes.mid
    )

    net_bytes_per_sec = read_bytes_per_second + write_bytes_per_second

    return int(max(1, math.ceil(net_bytes_per_sec / 125000)))


def compute_stateless_region(
    instance: Instance,
    needed_cores: int,
    needed_memory_gib: float,
    needed_network_mbps: float,
    # Faster CPUs can execute operations faster
    core_reference_ghz: float,
    num_zones: int = 3,
) -> RegionClusterCapacity:
    """Computes a regional cluster of a stateless app

    Basically just takes into cpu, memory, and network

    returns: (count of instances, annual cost in dollars)
    """

    # Stateless apps basically just use CPU resources and network
    needed_cores = int(max(1, needed_cores // (instance.cpu_ghz / core_reference_ghz)))

    count = max(2, math.ceil(needed_cores / instance.cpu))

    # Now take into account the network bandwidth
    count = max(count, math.ceil(needed_network_mbps / instance.net_mbps))

    # Now take into account the needed memory
    count = max(count, math.ceil(needed_memory_gib / instance.ram_gib))

    # Try to keep zones balanced
    count = utils.next_n(count, num_zones)

    return RegionClusterCapacity(
        cluster_type="stateless-app",
        count=count,
        instance=instance,
        annual_cost=count * instance.annual_cost,
    )


# pylint: disable=too-many-locals
def compute_stateful_zone(
    instance: Instance,
    drive: Drive,
    needed_cores: int,
    needed_disk_gib: int,
    needed_memory_gib: int,
    needed_network_mbps: float,
    # EBS may need to scale for IOs, and datastores might need more
    # or less IOs for a given data size as well as space
    required_disk_ios,
    required_disk_space,
    # Some stateful clusters have sidecars that take memory
    reserve_memory,
    # Some stateful clusters have preferences on per zone sizing
    cluster_size,
    # Faster CPUs can execute operations faster
    core_reference_ghz: float,
) -> ZoneClusterCapacity:

    # Normalize the cores of this instance type to the latency reference
    needed_cores = int(max(1, needed_cores // (instance.cpu_ghz / core_reference_ghz)))

    # Datastores often require disk headroom for e.g. compaction and such
    if instance.drive is not None:
        needed_disk_gib = required_disk_space(needed_disk_gib)

    # How many instances do we need for the CPU
    count = math.ceil(needed_cores / instance.cpu)

    # How many instances do we need for the ram, taking into account
    # reserved memory for sidecars
    count = max(
        count,
        math.ceil(
            needed_memory_gib / (instance.ram_gib - reserve_memory(instance.ram_gib))
        ),
    )

    # How many instances do we need for the network
    count = max(count, math.ceil(needed_network_mbps / instance.net_mbps))

    # How many instances do we need for the disk
    if instance.drive is not None and instance.drive.size_gib > 0:
        count = max(count, needed_disk_gib // instance.drive.size_gib)

    count = cluster_size(count)
    cost = count * instance.annual_cost

    attached_drives = []
    if instance.drive is None:
        # If we don't have disks attach GP2 in at 50% space overprovision
        # because we can only (as of 2020-10-31) scale EBS once per 6 hours

        # Note that ebs is provisioned _per node_ and must be chosen for
        # the max of space and IOS
        space_gib = max(1, (needed_disk_gib * 2) // count)
        io_gib = _gp2_gib_for_io(required_disk_ios(needed_disk_gib // count))

        # Provision EBS in increments of 100 GiB
        ebs_gib = utils.next_n(max(1, max(io_gib, space_gib)), n=100)
        attached_drive = drive.copy()
        attached_drive.size_gib = ebs_gib

        # TODO: appropriately handle RAID setups for throughput requirements
        attached_drives.append(attached_drive)

        cost = cost + attached_drive.annual_cost

    logger.debug(
        "For (cpu, memory_gib, disk_gib) = (%s, %s, %s) need (%s, %s, %s, %s)",
        needed_cores,
        needed_memory_gib,
        needed_disk_gib,
        count,
        instance.name,
        attached_drives,
        cost,
    )

    return ZoneClusterCapacity(
        cluster_type="stateful-cluster",
        count=count,
        instance=instance,
        attached_drives=attached_drives,
        annual_cost=cost,
    )


# AWS GP2 gives 3 IOS / gb stored.
def _gp2_gib_for_io(read_ios) -> int:
    return int(max(1, read_ios // 3))
