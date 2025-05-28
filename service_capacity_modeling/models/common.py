import logging
import math
import random
from decimal import Decimal
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from service_capacity_modeling.hardware import shapes
from service_capacity_modeling.interface import AVG_ITEM_SIZE_BYTES
from service_capacity_modeling.interface import Buffer
from service_capacity_modeling.interface import BufferComponent
from service_capacity_modeling.interface import BufferIntent
from service_capacity_modeling.interface import Buffers
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import CapacityPlan
from service_capacity_modeling.interface import CapacityRequirement
from service_capacity_modeling.interface import certain_float
from service_capacity_modeling.interface import certain_int
from service_capacity_modeling.interface import Clusters
from service_capacity_modeling.interface import CurrentClusterCapacity
from service_capacity_modeling.interface import CurrentClusters
from service_capacity_modeling.interface import default_reference_shape
from service_capacity_modeling.interface import Drive
from service_capacity_modeling.interface import Instance
from service_capacity_modeling.interface import Interval
from service_capacity_modeling.interface import RegionClusterCapacity
from service_capacity_modeling.interface import RegionContext
from service_capacity_modeling.interface import Requirements
from service_capacity_modeling.interface import ServiceCapacity
from service_capacity_modeling.interface import ZoneClusterCapacity
from service_capacity_modeling.models import utils
from service_capacity_modeling.models.headroom_strategy import HeadroomStrategy
from service_capacity_modeling.models.headroom_strategy import (
    QueuingBasedHeadroomStrategy,
)

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
    return math.ceil((rps * latency_s) + qos * math.sqrt(rps * latency_s))


def sqrt_staffed_cores(desires: CapacityDesires) -> int:
    """Computes cores given a sqrt staffing model

    Little's Law: Concurrency = Average Rate * Average Latency
    For example: 0.1 average concurrency = 100 / second * 1 millisecond

    However, if you provision for average, when statistically unlikely traffic
    spikes happen, you will queue, creating _latency_.

    Square root staffing says to avoid that latency instead of provisioning
    average number of cores, you provision

    Cores = (Rate * Latency) + (QoS * sqrt(Rate * Latency))
    Cores = (Required cores) + (Safety margin)

    Pick higher QoS to minimize the probability of queueing. In our case we do it
    based on tier.
    """
    qos = _QOS(desires.service_tier)
    read_rps, read_lat = (
        desires.query_pattern.estimated_read_per_second.mid,
        desires.query_pattern.estimated_mean_read_latency_ms.mid / 1000.0,
    )
    write_rps, write_lat = (
        desires.query_pattern.estimated_write_per_second.mid,
        desires.query_pattern.estimated_mean_write_latency_ms.mid / 1000.0,
    )

    total_rate = read_rps + write_rps
    weighted_latency = (
        (read_rps / total_rate) * read_lat + (write_rps / total_rate) * write_lat
        if total_rate > 0
        else 0
    )
    # The alternative is to staff each workload separately, but that over
    # provisions cores as f(x) + f(y) > f(x+y) (sqrt staffing isn't linear)
    # read_cores = _sqrt_staffed_cores(read_rps, read_lat, qos)
    # write_cores = _sqrt_staffed_cores(write_rps, write_lat, qos)

    return _sqrt_staffed_cores(total_rate, weighted_latency, qos)


def normalize_cores(
    core_count: float,
    target_shape: Instance,
    reference_shape: Optional[Instance] = None,
) -> int:
    """Calculates equivalent cores on a target shape relative to a reference

    Takes into account relative core frequency and IPC factor from the hardware
    description to give a rough estimate of how many equivalent cores you need
    in a target_shape to have the core_count number of cores on the reference_shape
    """
    if reference_shape is None:
        reference_shape = default_reference_shape

    target_speed = target_shape.cpu_ghz * target_shape.cpu_ipc_scale
    reference_speed = reference_shape.cpu_ghz * reference_shape.cpu_ipc_scale
    return max(1, math.ceil(core_count / (target_speed / reference_speed)))


def _reserved_headroom(
    cpu: int, cpu_boost: float = 1.0, strategy: Optional[HeadroomStrategy] = None
) -> float:
    # Adjust effective cores if e.g. is enabled
    # This accounts for the reduced effectiveness of virtual cores
    effective_cpu = float(cpu) * cpu_boost

    if strategy is None:
        strategy = QueuingBasedHeadroomStrategy()  # default strategy
    return strategy.calculate_reserved_headroom(effective_cpu)


def cpu_headroom_target(instance: Instance, buffers: Optional[Buffers] = None) -> float:
    """Determine an approximate headroom target for an instance.

    The headroom target should be the percentage of CPU that should be
    reserved for headroom to ensure sensible performance profile. In other words,
    we want to avoid queueing and just have service time.

    If buffer is None we leave the ultimate utilization_target to the caller, since
    we do not know how much operational headroom they want to leave
    (ie: success buffer). If passed, we will return the headroom for that buffer.

    For example, a response here of "headroom = 15%", means caller could
    decide with a success_buffer=1 to use a utilization_target of 85%.
    For success_buffer>1, they should target below 85% utilization.

    This is only suitable for "single-thread-like" workloads, which
    fortunately many stateless services are.

    For implementation see /notebooks/headroom-estimator.ipynb
    """

    # Physical cores(no hyper-threading) provide a performance boost.
    # For headroom, physical cores are weighted = 1.66 vs 1.0 for virtual cores.
    cpu_boost = 1.0 if instance.cores < instance.cpu else 1.0 / 0.6
    reserved_headroom = _reserved_headroom(instance.cpu, cpu_boost)
    if buffers is not None:
        cpu_ratio = buffer_for_components(
            buffers=buffers, components=[BufferComponent.cpu]
        ).ratio
        buffer_adjusted_headroom = (1.0 - reserved_headroom) / cpu_ratio
        effective_headroom = 1.0 - buffer_adjusted_headroom
        return round(effective_headroom, 2)
    else:
        return round(reserved_headroom, 2)


# When someone asks for the key, return any buffers that
# influence the component in the value
_default_buffer_fallbacks: Dict[str, List[str]] = {
    BufferComponent.compute: [BufferComponent.cpu],
    BufferComponent.storage: [BufferComponent.disk],
    BufferComponent.cpu: [BufferComponent.compute],
    BufferComponent.network: [BufferComponent.compute],
    BufferComponent.memory: [BufferComponent.storage],
    BufferComponent.disk: [BufferComponent.storage],
}


def buffer_for_components(
    buffers: Buffers,
    components: List[str],
    current_capacity: Optional[CurrentClusterCapacity] = None,
    component_fallbacks: Optional[Dict[str, List[str]]] = None,
) -> Buffer:
    """Calculates buffer for a given set of components, handling fallbacks

    Typical usage would be buffer_for_components(buffers, ["cpu"]) to get the
    cpu buffer or buffer_for_components(buffers, ["disk"]) to get the disk buffer,
    but you can also do like buffer_for_components(buffers, ["compute"]) which will
    pull any compute buffers or cpu buffers.

    Returns a Buffer containing:
        ratio: the composite ratio (e.g. 2.0 for 2x combined buffer)
        components: the components that ultimately matched after applying
        source: All the component buffers that made up the composite ratio
    """
    if component_fallbacks is None:
        component_fallbacks = _default_buffer_fallbacks

    unique_components = set(components)
    for component in components:
        unique_components = unique_components | set(
            component_fallbacks.get(component, [])
        )

    desired = {k: v.model_copy() for k, v in buffers.desired.items()}
    if current_capacity:
        if current_capacity.cluster_instance is None:
            cluster_instance = shapes.instance(current_capacity.cluster_instance_name)
        else:
            cluster_instance = current_capacity.cluster_instance
        # TODO: use cluster instance to reverse compute the buffers
        _ = cluster_instance

    ratio = 1.0
    sources = {}
    for name, buffer in desired.items():
        if any(i in unique_components for i in buffer.components):
            sources[name] = buffer
            ratio *= buffer.ratio
    if not sources:
        ratio = buffers.default.ratio

    return Buffer(
        ratio=ratio, components=sorted(list(unique_components)), sources=sources
    )


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
    num_zones = max(copies_per_region - 1, 0)
    num_regions = max(context.num_regions - 1, 0)

    # have bytes and / second
    size = desires.query_pattern.estimated_mean_write_size_bytes.mid
    wps = desires.query_pattern.estimated_write_per_second.mid
    # need gib and / year

    txfer_gib = (wps * size / (1024 * 1024 * 1024)) * (SECONDS_IN_YEAR)

    # For each cross region replication we have to pay to move bytes
    # inter region. This is the number of regions minus 1
    inter_txfer = context.services.get("net.inter.region", None)
    if inter_txfer:
        result.append(
            ServiceCapacity(
                service_type=f"{service_type}.net.inter.region",
                annual_cost=(inter_txfer.annual_cost_gib(txfer_gib) * num_regions),
                service_params={"txfer_gib": txfer_gib, "num_regions": num_regions},
            )
        )

    # Same zone is free, but we pay for replication from our zone to others
    intra_txfer = context.services.get("net.intra.region", None)
    if intra_txfer:
        result.append(
            ServiceCapacity(
                service_type=f"{service_type}.net.intra.region",
                annual_cost=(
                    intra_txfer.annual_cost_gib(txfer_gib)
                    * num_zones
                    * context.num_regions
                ),
                service_params={
                    "txfer_gib": txfer_gib,
                    "num_zones": num_zones,
                    "num_regions": context.num_regions,
                },
            )
        )
    return result


def compute_stateless_region(  # pylint: disable=too-many-positional-arguments
    instance: Instance,
    needed_cores: int,
    needed_memory_gib: float,
    needed_network_mbps: float,
    num_zones: int = 3,
) -> RegionClusterCapacity:
    """Computes a regional cluster of a stateless app

    Basically just takes into cpu, memory, and network

    returns: (count of instances, annual cost in dollars)
    """

    # Stateless apps basically just use CPU resources and network
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
    adjusted_disk_io_needed: float = 0.0,
    read_write_ratio: float = 0.0,
    # Max attached EBS volume size per node. Higher value here could allow
    # for a lower instance count (allows more vertical scaling vs forcing horizontal)
    max_attached_disk_gib: Optional[float] = None,
) -> ZoneClusterCapacity:
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
                count = max(
                    count, math.ceil(adjusted_disk_io_needed / instance_adjusted_io)
                )

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

        # Provision EBS in increments of 100 GiB
        ebs_gib = utils.next_n(max(1, io_gib, space_gib), n=100)

        # When initially provisioniong we don't want to attach more than
        # 1/3 the maximum volume size in one node (preferring more nodes
        # with smaller volumes)
        max_size = drive.max_size_gib / 3
        if max_attached_disk_gib is not None:
            max_size = max_attached_disk_gib

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
        if (read_io + write_io) > drive.max_io_per_s:
            ratio = (read_io + write_io) / drive.max_io_per_s
            count = max(cluster_size(math.ceil(count * ratio)), min_count)
            cost = count * instance.annual_cost
            read_io = utils.next_n(read_io * ratio, n=200)
            write_io = utils.next_n(write_io * ratio, n=200)

        attached_drive = drive.model_copy()
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


def cloud_gib_for_io(drive, total_ios, space_gib) -> int:
    if drive.name == "gp2":
        return gp2_gib_for_io(total_ios)
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


def derived_buffer_for_component(buffer: Dict[str, Buffer], components: List[str]):
    scale = 0.0
    preserve = False

    if not buffer:
        return scale, preserve

    for bfr in buffer.values():
        if any(component in components for component in bfr.components):
            if bfr.intent == BufferIntent.scale:
                scale = max(scale, bfr.ratio)
            if bfr.intent == BufferIntent.preserve:
                preserve = True

    return scale, preserve


def get_cores_from_current_capacity(
    current_capacity: CurrentClusterCapacity, buffers: Buffers, instance: Instance
):
    # compute cores required per zone
    cpu_success_buffer = (1 - cpu_headroom_target(instance, buffers)) * 100
    current_cpu_utilization = current_capacity.cpu_utilization.mid

    if current_capacity.cluster_instance is None:
        cluster_instance = shapes.instance(current_capacity.cluster_instance_name)
    else:
        cluster_instance = current_capacity.cluster_instance

    current_cores = cluster_instance.cpu * current_capacity.cluster_instance_count.mid

    scale, preserve = derived_buffer_for_component(buffers.derived, ["compute", "cpu"])
    # Scale and preserve for the same component should not be passed together.
    # If user passes it, then scale will be preferred over preserve.
    if scale > 0:
        # if the new cpu core is less than the current,
        # then take no action and return the current cpu cores
        new_cpu_utilization = current_cpu_utilization * scale
        core_scale_up_factor = max(1.0, new_cpu_utilization / cpu_success_buffer)
        return math.ceil(current_cores * core_scale_up_factor)

    if preserve:
        return current_cores

    return int(current_cores * (current_cpu_utilization / cpu_success_buffer))


def get_memory_from_current_capacity(
    current_capacity: CurrentClusterCapacity, buffers: Buffers
):
    # compute memory required per zone
    current_memory_utilization = (
        current_capacity.memory_utilization_gib.mid
        * current_capacity.cluster_instance_count.mid
    )

    if current_capacity.cluster_instance is None:
        cluster_instance = shapes.instance(current_capacity.cluster_instance_name)
    else:
        cluster_instance = current_capacity.cluster_instance

    zonal_ram_allocated = (
        cluster_instance.ram_gib * current_capacity.cluster_instance_count.mid
    )

    # These are the desired buffers
    memory_buffer = buffer_for_components(
        buffers=buffers, components=[BufferComponent.memory]
    )

    scale, preserve = derived_buffer_for_component(
        buffers.derived, ["memory", "storage"]
    )
    # Scale and preserve for the same component should not be passed together.
    # If user passes it, then scale will be preferred over preserve.
    if scale > 0:
        # if the new required memory is less than the current,
        # then take no action and return the current ram
        return max(
            current_memory_utilization * scale * memory_buffer.ratio,
            zonal_ram_allocated,
        )

    if preserve:
        return zonal_ram_allocated

    return current_memory_utilization * memory_buffer.ratio


def get_network_from_current_capacity(
    current_capacity: CurrentClusterCapacity, buffers: Buffers
):
    # compute network required per zone
    current_network_utilization = (
        current_capacity.network_utilization_mbps.mid
        * current_capacity.cluster_instance_count.mid
    )

    if current_capacity.cluster_instance is None:
        cluster_instance = shapes.instance(current_capacity.cluster_instance_name)
    else:
        cluster_instance = current_capacity.cluster_instance

    zonal_network_allocated = (
        cluster_instance.net_mbps * current_capacity.cluster_instance_count.mid
    )

    # These are the desired buffers
    network_buffer = buffer_for_components(
        buffers=buffers, components=[BufferComponent.network]
    )

    scale, preserve = derived_buffer_for_component(
        buffers.derived, ["compute", "network"]
    )
    # Scale and preserve for the same component should not be passed together.
    # If user passes it, then scale will be preferred over preserve.
    if scale > 0:
        # if the new required network is less than the current,
        # then take no action and return the current bandwidth
        return max(
            current_network_utilization * scale * network_buffer.ratio,
            zonal_network_allocated,
        )

    if preserve:
        return zonal_network_allocated

    return current_network_utilization * network_buffer.ratio


def get_disk_from_current_capacity(
    current_capacity: CurrentClusterCapacity, buffers: Buffers
):
    # compute disk required per zone
    current_disk_utilization = (
        current_capacity.disk_utilization_gib.mid
        * current_capacity.cluster_instance_count.mid
    )

    if current_capacity.cluster_instance is None:
        cluster_instance = shapes.instance(current_capacity.cluster_instance_name)
    else:
        cluster_instance = current_capacity.cluster_instance

    if cluster_instance.drive is not None:
        instance_disk_allocated = cluster_instance.drive.max_size_gib
    else:
        assert current_capacity.cluster_drive is not None, "Drive should not be None"
        instance_disk_allocated = current_capacity.cluster_drive.size_gib

    zonal_disk_allocated = (
        instance_disk_allocated * current_capacity.cluster_instance_count.mid
    )

    # These are the desired buffers
    disk_buffer = buffer_for_components(
        buffers=buffers, components=[BufferComponent.disk]
    )

    scale, preserve = derived_buffer_for_component(buffers.derived, ["storage", "disk"])
    # Scale and preserve for the same component should not be passed together.
    # If user passes it, then scale will be preferred over preserve.
    if scale > 0:
        # if the new required disk is less than the current,
        # then take no action and return the current disk
        return max(
            current_disk_utilization * scale * disk_buffer.ratio, zonal_disk_allocated
        )
    if preserve:
        # preserve the current disk size for the zone
        return zonal_disk_allocated

    return current_disk_utilization * disk_buffer.ratio


def zonal_requirements_from_current(
    current_cluster: CurrentClusters,
    buffers: Buffers,
    instance: Instance,
    reference_shape: Instance,
) -> CapacityRequirement:
    if current_cluster is not None and current_cluster.zonal[0] is not None:
        current_capacity: CurrentClusterCapacity = current_cluster.zonal[0]
        needed_cores = normalize_cores(
            get_cores_from_current_capacity(current_capacity, buffers, instance),
            instance,
            reference_shape,
        )
        needed_network_mbps = get_network_from_current_capacity(
            current_capacity, buffers
        )
        needed_memory_gib = get_memory_from_current_capacity(current_capacity, buffers)
        needed_disk_gib = get_disk_from_current_capacity(current_capacity, buffers)

        return CapacityRequirement(
            requirement_type="zonal-capacity",
            cpu_cores=certain_int(needed_cores),
            mem_gib=certain_float(needed_memory_gib),
            disk_gib=certain_float(needed_disk_gib),
            network_mbps=certain_float(needed_network_mbps),
            reference_shape=current_capacity.cluster_instance,
        )
    else:
        raise ValueError("Please check if current_cluster is populated correctly.")
