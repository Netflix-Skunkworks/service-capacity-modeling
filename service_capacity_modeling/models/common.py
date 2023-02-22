import logging
import math
import random
from decimal import Decimal
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple

from service_capacity_modeling.interface import AVG_ITEM_SIZE_BYTES
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import CapacityPlan
from service_capacity_modeling.interface import certain_float
from service_capacity_modeling.interface import certain_int
from service_capacity_modeling.interface import Clusters
from service_capacity_modeling.interface import Drive
from service_capacity_modeling.interface import Instance
from service_capacity_modeling.interface import Interval
from service_capacity_modeling.interface import RegionClusterCapacity
from service_capacity_modeling.interface import RegionContext
from service_capacity_modeling.interface import Requirements
from service_capacity_modeling.interface import ServiceCapacity
from service_capacity_modeling.interface import ZoneClusterCapacity
from service_capacity_modeling.models import utils

logger = logging.getLogger(__name__)


SECONDS_IN_YEAR = 31556926


# In square root staffing we have to take into account the QOS parameter
# Which is related to the probability that a user queues. On low tier clusters
# (aka critical clusters) we want a lower probability of queueing
def _QOS(tier: int) -> float:
    # Halfin-Whitt delay function
    # P(queue) ~= [1 + B * normal_cdf(B) / normal_pdf(B)] ^ -1
    #
    # P(queue) ~= 0.01
    if tier == 0:
        return 2.375
    # P(queue) ~= 0.05
    elif tier == 1:
        return 1.761
    # P(queue) ~= 0.2
    elif tier == 2:
        return 1.16
    # P(queue) ~= 0.29 ~= 0.3
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


def network_services(
    service_type: str,
    context: RegionContext,
    desires: CapacityDesires,
    copies_per_region: int,
) -> List[ServiceCapacity]:
    result = []
    # Network transfer is for every other zone and then for every region
    # other than us as well.
    num_zones = copies_per_region - 1
    num_regions = context.num_regions - 1

    # have bytes and / second
    size = desires.query_pattern.estimated_mean_write_size_bytes.mid
    wps = desires.query_pattern.estimated_write_per_second.mid
    # need gib and / year

    txfer_gib = (wps * size / (1024 * 1024 * 1024)) * (SECONDS_IN_YEAR)

    inter_txfer = context.services.get("net.inter.region", None)
    if inter_txfer:
        price_per_gib = inter_txfer.annual_cost_per_gib
        result.append(
            ServiceCapacity(
                service_type=f"{service_type}.net.inter.region",
                annual_cost=(price_per_gib * txfer_gib * num_regions),
                service_params={"txfer_gib": txfer_gib, "num_regions": num_regions},
            )
        )

    intra_txfer = context.services.get("net.intra.region", None)
    if intra_txfer:
        price_per_gib = intra_txfer.annual_cost_per_gib
        result.append(
            ServiceCapacity(
                service_type=f"{service_type}.net.intra.region",
                annual_cost=(price_per_gib * txfer_gib * num_zones),
                service_params={"txfer_gib": txfer_gib, "num_zones": num_zones},
            )
        )
    return result


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
    needed_cores = math.ceil(
        max(1, needed_cores // (instance.cpu_ghz / core_reference_ghz))
    )

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
    needed_disk_gib: float,
    needed_memory_gib: float,
    needed_network_mbps: float,
    # Faster CPUs can execute operations faster
    core_reference_ghz: float,
    # Cloud drives may need to scale for IOs, and datastores might need more
    # or less IOs for a given data size as well as space
    # Contract for disk ios is
    # (per_node_size_gib, node_count) -> (read_ios, write_ios)
    required_disk_ios: Callable[
        [float, int], Tuple[float, float]
    ] = lambda size_gib, count: (0, 0),
    required_disk_space: Callable[[float], float] = lambda size_gib: size_gib,
    # The maximum amount of state we can hold per node in the database
    # typically you don't want stateful systems going much higher than a
    # few TiB so that recovery functions properly
    max_local_disk_gib: float = 2048,
    # Some stateful clusters have sidecars that take memory
    reserve_memory: Callable[[float], float] = lambda x: 0,
    # How much write buffer we get per instance (usually a percentage of
    # the reserved memory, e.g. for buffering writes in heap)
    write_buffer: Callable[[float], float] = lambda x: 0,
    required_write_buffer_gib: float = 0,
    # Some stateful clusters have preferences on per zone sizing
    cluster_size: Callable[[int], int] = lambda x: x,
    min_count: int = 0,
) -> ZoneClusterCapacity:
    # Normalize the cores of this instance type to the latency reference
    needed_cores = math.ceil(
        max(1, needed_cores // (instance.cpu_ghz / core_reference_ghz))
    )

    # Datastores often require disk headroom for e.g. compaction and such
    if instance.drive is not None:
        needed_disk_gib = math.ceil(required_disk_space(needed_disk_gib))

    # How many instances do we need for the CPU
    count = math.ceil(needed_cores / instance.cpu)

    # How many instances do we need for the ram, taking into account
    # reserved memory for the application and system
    count = max(
        count,
        math.ceil(
            needed_memory_gib / (instance.ram_gib - reserve_memory(instance.ram_gib))
        ),
    )
    # Account for if the stateful service needs a certain amount of reserved
    # memory for a given throughput.
    if write_buffer(instance.ram_gib) > 0:
        count = max(
            count,
            math.ceil(required_write_buffer_gib / (write_buffer(instance.ram_gib))),
        )

    # How many instances do we need for the network
    count = max(count, math.ceil(needed_network_mbps / instance.net_mbps))

    # How many instances do we need for the disk
    if (
        instance.drive is not None
        and instance.drive.size_gib > 0
        and max_local_disk_gib > 0
    ):
        disk_per_node = min(max_local_disk_gib, instance.drive.size_gib)
        count = max(count, math.ceil(needed_disk_gib / disk_per_node))

    count = max(cluster_size(count), min_count)
    cost = count * instance.annual_cost

    attached_drives = []
    if instance.drive is None and required_disk_space(needed_disk_gib) > 0:
        # If we don't have disks attach the cloud drive with enough
        # space and IO for the requirement

        # Note that cloud drivers are provisioned _per node_ and must be chosen for
        # the max of space and IOS.
        space_gib = max(1, math.ceil(required_disk_space(needed_disk_gib) / count))
        read_io, write_io = required_disk_ios(space_gib, count)
        read_io, write_io = (
            utils.next_n(read_io, n=200),
            utils.next_n(write_io, n=200),
        )
        total_ios = read_io + write_io
        io_gib = cloud_gib_for_io(drive, total_ios, space_gib)

        # Provision EBS in increments of 200 GiB
        ebs_gib = utils.next_n(max(1, io_gib, space_gib), n=200)

        # When initially provisioniong we don't want to attach more than
        # 1/3 the maximum volume size in one node (preferring more nodes
        # with smaller volumes)
        max_size = drive.max_size_gib / 3
        if ebs_gib > max_size > 0:
            ratio = ebs_gib / max_size
            count = max(cluster_size(math.ceil(count * ratio)), min_count)
            cost = count * instance.annual_cost
            ebs_gib = max_size

        read_io, write_io = required_disk_ios(space_gib, count)
        read_io, write_io = (
            utils.next_n(read_io, n=200),
            utils.next_n(write_io, n=200),
        )
        attached_drive = drive.copy()
        attached_drive.size_gib = ebs_gib
        attached_drive.read_io_per_s = int(round(read_io, 2))
        attached_drive.write_io_per_s = int(round(write_io, 2))

        # TODO: appropriately handle RAID setups for throughput requirements
        attached_drives.append(attached_drive)

        cost = cost + (attached_drive.annual_cost * count)

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
def gp2_gib_for_io(read_ios) -> int:
    return int(max(1, read_ios // 3))


def cloud_gib_for_io(drive, read_ios, space_gib) -> int:
    if drive.name == "gp2":
        return gp2_gib_for_io(read_ios)
    else:
        return space_gib


class WorkingSetEstimator:
    def __init__(self):
        self._cache = {}

    def working_set_percent(
        self,
        # latency distributions of the read SLOs versus the drives
        # expressed as scipy rv_continuous objects
        drive_read_latency_dist,
        read_slo_latency_dist,
        # what percentile of disk latency should we target for keeping in
        # memory. Not as this is _increased_ more memory will be reserved
        target_percentile: float = 0.90,
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
            # How fast is the drive at the target percentile
            minimum_drive_latency = drive_read_latency_dist.ppf(target_percentile)

            # How much of the read latency SLO lies below the minimum
            # drive latency. So for example if EBS's 99% is 1.7ms and we
            # 45% of our read SLO lies below that then we need at least 45%
            # of our data to be stored in memory.
            required_percent = float(read_slo_latency_dist.cdf(minimum_drive_latency))

            self._cache[cache_key] = certain_float(
                max(required_percent, min_working_set)
            )
        return self._cache[cache_key]


_working_set_estimator = WorkingSetEstimator()


def working_set_from_drive_and_slo(
    # latency distributions of the read SLOs versus the drives
    # expressed as scipy rv_continuous objects
    drive_read_latency_dist,
    read_slo_latency_dist,
    estimated_working_set: Optional[Interval] = None,
    # what percentile of disk latency should we target for keeping in
    # memory. Not as this is _increased_ more memory will be reserved
    target_percentile: float = 0.90,
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


def merge_requirements(
    left_req: Requirements,
    right_req: Requirements,
) -> Requirements:
    merged_zonal, merged_regional = [], []
    for req in list(left_req.zonal) + list(right_req.zonal):
        merged_zonal.append(req)
    for req in list(left_req.regional) + list(right_req.regional):
        merged_regional.append(req)

    merged_regrets = set(left_req.regrets) | set(right_req.regrets)

    return Requirements(
        zonal=merged_zonal, regional=merged_regional, regrets=tuple(merged_regrets)
    )


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

    merged_requirements = merge_requirements(left.requirements, right.requirements)

    left_cluster = left.candidate_clusters
    right_cluster = right.candidate_clusters

    merged_annual_costs = {}
    all_sources = set(
        left_cluster.annual_costs.keys() | right_cluster.annual_costs.keys()
    )

    merged_annual_costs = {
        k: (
            left_cluster.annual_costs.get(k, Decimal(0))
            + right_cluster.annual_costs.get(k, Decimal(0))
        )
        for k in all_sources
    }

    merged_clusters = Clusters(
        annual_costs=merged_annual_costs,
        zonal=(
            [zonal_transform(z) for z in left_cluster.zonal]
            + [zonal_transform(z) for z in right_cluster.zonal]
        ),
        regional=(
            [regional_transform(z) for z in left_cluster.regional]
            + [regional_transform(z) for z in right_cluster.regional]
        ),
        services=(list(left_cluster.services) + list(right_cluster.services)),
    )
    return CapacityPlan(
        requirements=merged_requirements, candidate_clusters=merged_clusters
    )
