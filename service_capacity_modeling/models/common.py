import logging
import math
import random
from typing import Callable
from typing import Optional

from service_capacity_modeling.interface import AVG_ITEM_SIZE_BYTES
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import CapacityPlan
from service_capacity_modeling.interface import CapacityRequirement
from service_capacity_modeling.interface import certain_float
from service_capacity_modeling.interface import certain_int
from service_capacity_modeling.interface import Clusters
from service_capacity_modeling.interface import Drive
from service_capacity_modeling.interface import Instance
from service_capacity_modeling.interface import Interval
from service_capacity_modeling.interface import RegionClusterCapacity
from service_capacity_modeling.interface import ZoneClusterCapacity
from service_capacity_modeling.models import utils


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


class WorkingSetEstimator:
    def __init__(self):
        self._cache = {}

    def working_set_percent(
        self,
        # latency distributions of the read SLOs versus the drives
        # expressed as scipy rv_continuous objects
        drive_read_latency_dist,
        read_slo_latency_dist,
        # what is our target percentile for hitting disk
        # Note that lower will decrease the amount we hit disk
        target_percentile: float = 0.10,
        min_working_set: float = 0.01,
    ) -> Interval:
        # random cache eviction
        if len(self._cache) >= 100:
            self._cache.pop(random.choice(tuple(self._cache.keys())))

        cache_key = (
            id(drive_read_latency_dist),
            id(read_slo_latency_dist),
            target_percentile,
        )
        # Cached because ppf in particular is _really_ slow
        if cache_key not in self._cache:
            # The inverse CDF, basically what percentile do we want to target
            # to be all on disk.
            target_latency = read_slo_latency_dist.ppf(target_percentile)

            # What percent of disk reads will fall below this latency SLO
            lat = max(drive_read_latency_dist.sf(target_latency), min_working_set)
            self._cache[cache_key] = certain_float(lat)
        return self._cache[cache_key]


_working_set_estimator = WorkingSetEstimator()


def working_set_from_drive_and_slo(
    # latency distributions of the read SLOs versus the drives
    # expressed as scipy rv_continuous objects
    drive_read_latency_dist,
    read_slo_latency_dist,
    estimated_working_set: Optional[Interval] = None,
    # what is our target percentile slo latency that we allow to hit disk
    # Note that lower will decrease the amount we hit disk
    target_percentile: float = 0.10,
    min_working_set: float = 0.01,
) -> Interval:
    if estimated_working_set is not None:
        return estimated_working_set

    return _working_set_estimator.working_set_percent(
        drive_read_latency_dist=drive_read_latency_dist,
        read_slo_latency_dist=read_slo_latency_dist,
        target_percentile=target_percentile,
        min_working_set=min_working_set,
    )


def item_count_from_state(
    estimated_state_size_gib: Interval,
    estimated_state_item_count: Optional[Interval] = None,
) -> Interval:
    if estimated_state_item_count is not None:
        return estimated_state_item_count

    return certain_int(
        int(estimated_state_size_gib.mid * 1024 * 1024 * 1024) // AVG_ITEM_SIZE_BYTES
    )


def _add_optional_float(
    left: Optional[float], right: Optional[float]
) -> Optional[float]:
    if left is None and right is None:
        return None
    if left is None:
        return right
    if right is None:
        return left
    return left + right


def _add_interval(left: Interval, right: Interval) -> Interval:
    return Interval(
        low=(left.low + right.low),
        mid=(left.mid + right.mid),
        high=(left.high + right.high),
        confidence=min(left.confidence, right.confidence),
        model_with=left.model_with,
        minimum_value=_add_optional_float(left.minimum_value, right.minimum_value),
        maximum_value=_add_optional_float(left.maximum_value, right.maximum_value),
    )


def _noop_zone(x: ZoneClusterCapacity) -> ZoneClusterCapacity:
    return x


def _noop_region(x: RegionClusterCapacity) -> RegionClusterCapacity:
    return x


def merge_plan(
    left: Optional[CapacityPlan],
    right: Optional[CapacityPlan],
    zonal_transform: Callable[[ZoneClusterCapacity], ZoneClusterCapacity] = _noop_zone,
    regional_transform: Callable[
        [RegionClusterCapacity], RegionClusterCapacity
    ] = _noop_region,
) -> Optional[CapacityPlan]:
    if left is None or right is None:
        return None

    left_req = left.requirement
    right_req = right.requirement

    merged_requirement = CapacityRequirement(
        core_reference_ghz=min(
            left_req.core_reference_ghz, right_req.core_reference_ghz
        ),
        cpu_cores=_add_interval(left_req.cpu_cores, right_req.cpu_cores),
        mem_gib=_add_interval(left_req.mem_gib, right_req.mem_gib),
        network_mbps=_add_interval(left_req.network_mbps, right_req.network_mbps),
        disk_gib=_add_interval(left_req.disk_gib, right_req.disk_gib),
    )
    left_cluster = left.candidate_clusters
    right_cluster = right.candidate_clusters

    merged_clusters = Clusters(
        total_annual_cost=_add_interval(
            left_cluster.total_annual_cost, right_cluster.total_annual_cost
        ),
        zonal=(
            [zonal_transform(z) for z in left_cluster.zonal]
            + [zonal_transform(z) for z in right_cluster.zonal]
        ),
        regional=(
            [regional_transform(z) for z in left_cluster.regional]
            + [regional_transform(z) for z in right_cluster.regional]
        ),
    )
    return CapacityPlan(
        requirement=merged_requirement, candidate_clusters=merged_clusters
    )
