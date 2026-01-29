# pylint: disable=too-many-lines
import logging
import math
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Set

from pydantic import BaseModel
from pydantic import Field

from service_capacity_modeling.interface import AccessConsistency
from service_capacity_modeling.interface import AccessPattern
from service_capacity_modeling.interface import Buffer
from service_capacity_modeling.interface import BufferComponent
from service_capacity_modeling.interface import Buffers
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import CapacityPlan
from service_capacity_modeling.interface import CapacityRequirement
from service_capacity_modeling.interface import certain_float
from service_capacity_modeling.interface import certain_int
from service_capacity_modeling.interface import ClusterCapacity
from service_capacity_modeling.interface import Clusters
from service_capacity_modeling.interface import Consistency
from service_capacity_modeling.interface import CurrentClusterCapacity
from service_capacity_modeling.interface import DataShape
from service_capacity_modeling.interface import Drive
from service_capacity_modeling.interface import FixedInterval
from service_capacity_modeling.interface import GlobalConsistency
from service_capacity_modeling.interface import Instance
from service_capacity_modeling.interface import Interval
from service_capacity_modeling.interface import QueryPattern
from service_capacity_modeling.interface import RegionContext
from service_capacity_modeling.interface import Requirements
from service_capacity_modeling.interface import ServiceCapacity
from service_capacity_modeling.models import CapacityModel
from service_capacity_modeling.models import CostAwareModel
from service_capacity_modeling.models.common import buffer_for_components
from service_capacity_modeling.models.common import cluster_infra_cost
from service_capacity_modeling.models.common import compute_stateful_zone
from service_capacity_modeling.models.common import DerivedBuffers
from service_capacity_modeling.models.common import get_effective_disk_per_node_gib
from service_capacity_modeling.models.common import network_services
from service_capacity_modeling.models.common import normalize_cores
from service_capacity_modeling.models.common import simple_network_mbps
from service_capacity_modeling.models.common import sqrt_staffed_cores
from service_capacity_modeling.models.common import working_set_from_drive_and_slo
from service_capacity_modeling.models.common import zonal_requirements_from_current
from service_capacity_modeling.models.utils import is_power_of_2
from service_capacity_modeling.models.utils import next_doubling
from service_capacity_modeling.models.utils import next_power_of_2
from service_capacity_modeling.stats import dist_for_interval

logger = logging.getLogger(__name__)

BACKGROUND_BUFFER = "background"
CRITICAL_TIERS: Set[int] = {0, 1}
# cluster size aka nodes per ASG
CRITICAL_TIER_MIN_CLUSTER_SIZE = 2


def _write_buffer_gib_zone(
    desires: CapacityDesires, zones_per_region: int, flushes_before_compaction: int = 4
) -> float:
    # Cassandra has to buffer writes before flushing to disk, and assuming
    # we will compact every 4 flushes and we want no more than 2 redundant
    # compactions in an hour, we want <= 4**2 = 16 flushes per hour
    # or a flush of data every 3600 / 16 = 225 seconds
    write_bytes_per_second = (
        desires.query_pattern.estimated_write_per_second.mid
        * desires.query_pattern.estimated_mean_write_size_bytes.mid
    )

    compactions_per_hour = 2
    hour_in_seconds = 60 * 60

    write_buffer_gib = (
        (write_bytes_per_second * hour_in_seconds)
        / (flushes_before_compaction**compactions_per_hour)
    ) / (1 << 30)

    return float(write_buffer_gib) / zones_per_region


def _get_cores_from_desires(desires: CapacityDesires, instance: Instance) -> int:
    cpu_buffer = buffer_for_components(
        buffers=desires.buffers, components=[BACKGROUND_BUFFER]
    )

    # We have no existing utilization to go from
    reference_shape = desires.reference_shape
    # Keep half of the cores free for background work (compaction, backup, repair).
    needed_cores = math.ceil(sqrt_staffed_cores(desires) * cpu_buffer.ratio)

    needed_cores = normalize_cores(
        core_count=needed_cores,
        target_shape=instance,
        reference_shape=reference_shape,
    )
    return needed_cores


def _get_disk_from_desires(desires: CapacityDesires, copies_per_region: int) -> int:
    disk_buffer = buffer_for_components(
        buffers=desires.buffers, components=[BufferComponent.disk]
    )
    # Do not add disk buffers now as memory calculation is done on the disk usage
    return math.ceil(
        (1.0 / desires.data_shape.estimated_compression_ratio.mid)
        * desires.data_shape.estimated_state_size_gib.mid
        * copies_per_region
        * disk_buffer.ratio
    )


def _get_min_count(
    tier: int,
    required_cluster_size: Optional[int],
    needed_disk_gib: float,
    disk_per_node_gib: float,
    cluster_size_lambda: Callable[[int], int],
) -> int:
    """
    Compute the minimum number of nodes required for a zone.

    This function is used to prevent the planner from allocating clusters that
    would exceed the max data per node or under the required cluster size for
    a tier or existing cluster
    """

    # Cassandra clusters should aim to be at least 2 nodes per zone to start
    # out with for tier 0 or tier 1. This gives us more room to "up-color"]
    # clusters.
    min_nodes_for_tier = 2 if tier in CRITICAL_TIERS else 0

    # Prevent allocating clusters that exceed the max data per node.
    min_nodes_for_disk = math.ceil(needed_disk_gib / disk_per_node_gib)

    # Take the max of the following in order to avoid:
    # (1) if `required_cluster_size` < `min_nodes_for_disk`, don't let the planner
    #     pick a shape that would exceed the max data per node
    #
    #     For example, if we need 4TiB of disk, and the max data per node is 1TiB,
    #     Regardless of the `required_cluster_size`, we cannot allocate less than 4
    #     nodes because that would exceed the max data per node.
    #
    # (2) if `required_cluster_size` > `min_nodes_for_disk`, don't let the
    #     node density requirement affect the min count because the required
    #     cluster size already meets the node density requirement.
    #
    #     For example, if we need 4TiB of disk, and the max data per node is 1TiB,
    #     and the upstream requires >= 8 nodes, we can allocate 8 nodes because
    #     each node would only have 500GB of data.
    min_count = max(
        min_nodes_for_tier,
        required_cluster_size or 0,
        min_nodes_for_disk,
    )
    # Ensure that the min count is an increment of the cluster size constraint (doubling)
    return cluster_size_lambda(min_count)


def _zonal_requirement_for_new_cluster(
    desires: CapacityDesires,
    instance: Instance,
    copies_per_region: int,
    zones_per_region: int,
) -> CapacityRequirement:
    needed_cores = _get_cores_from_desires(desires, instance)
    needed_disk = _get_disk_from_desires(desires, copies_per_region)

    # Keep some of the bandwidth available for backup and repair streaming
    network_buffer = buffer_for_components(
        buffers=desires.buffers, components=[BACKGROUND_BUFFER]
    )
    needed_network_mbps = simple_network_mbps(desires) * network_buffer.ratio

    # Now convert to per zone
    needed_cores = max(1, needed_cores // zones_per_region)
    needed_disk = max(1, needed_disk // zones_per_region)

    return CapacityRequirement(
        requirement_type="zonal-capacity",
        cpu_cores=certain_int(needed_cores),
        disk_gib=certain_float(needed_disk),
        network_mbps=certain_float(needed_network_mbps),
    )


# pylint: disable=too-many-locals
# pylint: disable=too-many-positional-arguments
def _estimate_cassandra_requirement(
    instance: Instance,
    desires: CapacityDesires,
    working_set: float,
    reads_per_second: float,
    max_rps_to_disk: int,
    zones_per_region: int = 3,
    copies_per_region: int = 3,
) -> CapacityRequirement:
    # Input: regional desires → Output: zonal requirement
    disk_buffer = buffer_for_components(
        buffers=desires.buffers, components=[BufferComponent.disk]
    )
    memory_preserve = False
    reference_shape = desires.reference_shape
    current_capacity = _get_current_capacity(desires)

    # If the cluster is already provisioned
    if current_capacity and desires.current_clusters is not None:
        capacity_requirement = zonal_requirements_from_current(
            desires.current_clusters,
            desires.buffers,
            instance,
            reference_shape,
        )
        disk_derived_buffer = DerivedBuffers.for_components(
            desires.buffers.derived, [BufferComponent.disk]
        )
        disk_used_gib = (
            current_capacity.disk_utilization_gib.mid
            * current_capacity.cluster_instance_count.mid
            * disk_derived_buffer.scale
        )
        memory_preserve = DerivedBuffers.for_components(
            desires.buffers.derived, [BufferComponent.memory]
        ).preserve
    else:
        # If the cluster is not yet provisioned
        capacity_requirement = _zonal_requirement_for_new_cluster(
            desires, instance, copies_per_region, zones_per_region
        )
        disk_used_gib = capacity_requirement.disk_gib.mid / disk_buffer.ratio

    needed_cores = math.ceil(capacity_requirement.cpu_cores.mid)
    needed_disk = capacity_requirement.disk_gib.mid
    needed_network_mbps = capacity_requirement.network_mbps.mid

    # it can be 0 for cases where disk utilization is not passed as a part of current cluster capacity
    if needed_disk == 0:
        needed_disk = max(
            1, _get_disk_from_desires(desires, copies_per_region) // zones_per_region
        )
        # We want to compute memory based on the data size.
        disk_used_gib = needed_disk / disk_buffer.ratio

    # Rough estimate of how many instances we would need just for the CPU
    # Note that this is a lower bound, we might end up with more.
    estimated_cores_per_region = math.ceil(
        (needed_cores * zones_per_region) / instance.cpu
    )
    # Generally speaking we want fewer than some number of reads per second
    # hitting disk per instance. If we don't have many reads we don't need to
    # hold much data in memory.
    instance_rps = max(1, reads_per_second // estimated_cores_per_region)
    disk_rps = instance_rps * _cass_io_per_read(
        max(1, (disk_used_gib * zones_per_region) // estimated_cores_per_region)
    )
    rps_working_set = min(1.0, disk_rps / max_rps_to_disk)

    # Cassandra can defer writes either by buffering in memory or by
    # waiting longer before recompacting (the min-threshold on the
    # L0 compactions or STCS compactions)
    min_threshold = 4
    write_buffer_gib = _write_buffer_gib_zone(
        desires=desires,
        zones_per_region=zones_per_region,
        flushes_before_compaction=min_threshold,
    )

    while write_buffer_gib > 12 and min_threshold < 16:
        min_threshold *= 2
        write_buffer_gib = _write_buffer_gib_zone(
            desires=desires,
            zones_per_region=zones_per_region,
            flushes_before_compaction=min_threshold,
        )

    if current_capacity and current_capacity.cluster_instance and memory_preserve:
        # remove base memory and heap from per node ram and then
        # multiply by number of nodes in a zone to compute the zonal requirement.
        reserve_memory = _get_base_memory(desires) + _cass_heap(
            current_capacity.cluster_instance.ram_gib
        )
        needed_memory = (
            current_capacity.cluster_instance.ram_gib - reserve_memory
        ) * current_capacity.cluster_instance_count.mid
        write_buffer_gib = 0
    else:
        # If disk RPS will be smaller than our target because there are no
        # reads, we don't need to hold as much data in memory.
        # For c*, we can skip memory buffer and can just keep using the heap and write buffer calc
        # Eventually we'll want to phrase those heap, read cache, and write cache as buffers
        needed_memory = (
            min(working_set, rps_working_set) * disk_used_gib * zones_per_region
        )
        # Now convert to per zone
        needed_memory = max(1, int(needed_memory // zones_per_region))

    logger.debug(
        "Need (cpu, mem, disk, working) = (%s, %s, %s, %f)",
        needed_cores,
        needed_memory,
        needed_disk,
        working_set,
    )

    return CapacityRequirement(
        requirement_type="cassandra-zonal",
        reference_shape=reference_shape,
        cpu_cores=certain_int(needed_cores),
        mem_gib=certain_float(needed_memory),
        disk_gib=certain_float(needed_disk),
        network_mbps=certain_float(needed_network_mbps),
        context={
            "working_set": min(working_set, rps_working_set),
            "rps_working_set": rps_working_set,
            "disk_slo_working_set": working_set,
            "replication_factor": copies_per_region,
            "compression_ratio": round(
                1.0 / desires.data_shape.estimated_compression_ratio.mid, 2
            ),
            "read_per_second": reads_per_second,
            "write_buffer_gib": write_buffer_gib,
            "min_threshold": min_threshold,
        },
    )


def _get_current_cluster_size(desires: CapacityDesires) -> int:
    current_capacity = _get_current_capacity(desires)
    if current_capacity is None:
        return 0
    return math.ceil(current_capacity.cluster_instance_count.mid)


def _get_current_capacity(desires: CapacityDesires) -> Optional[CurrentClusterCapacity]:
    current_capacity = (
        None
        if desires.current_clusters is None
        else (
            desires.current_clusters.zonal[0]
            if len(desires.current_clusters.zonal)
            else desires.current_clusters.regional[0]
        )
    )
    return current_capacity


def _upsert_params(cluster: Any, params: Dict[str, Any]) -> None:
    if cluster.cluster_params:
        cluster.cluster_params.update(params)
    else:
        cluster.cluster_params = params


def _get_cluster_size_lambda(
    current_cluster_size: int,
    required_cluster_size: Optional[int],
) -> Callable[[int], int]:
    if required_cluster_size:
        return lambda x: next_doubling(x, base=required_cluster_size)
    elif current_cluster_size and not is_power_of_2(current_cluster_size):
        return lambda x: next_doubling(x, base=current_cluster_size)
    else:  # New provisionings
        return next_power_of_2


# pylint: disable=too-many-locals
# pylint: disable=too-many-return-statements
# flake8: noqa: C901
def _estimate_cassandra_cluster_zonal(  # pylint: disable=too-many-positional-arguments
    instance: Instance,
    drive: Drive,
    context: RegionContext,
    desires: CapacityDesires,
    zones_per_region: int = 3,
    copies_per_region: int = 3,
    require_local_disks: bool = False,
    require_attached_disks: bool = False,
    required_cluster_size: Optional[int] = None,
    max_rps_to_disk: int = 500,
    max_local_data_per_node_gib: int = 1280,
    max_attached_data_per_node_gib: int = 2048,
    max_regional_size: int = 192,
    max_write_buffer_percent: float = 0.25,
    max_table_buffer_percent: float = 0.11,
) -> Optional[CapacityPlan]:
    # Netflix Cassandra doesn't like to deploy on really small instances
    if instance.cpu < 2 or instance.ram_gib <= 16:
        return None

    # if we're not allowed to use gp2, skip EBS only types
    if instance.drive is None and require_local_disks:
        return None

    # if we're not allowed to use local disks, skip ephems
    if instance.drive is not None and require_attached_disks:
        return None

    # Cassandra only deploys on gp2 and gp3 drives right now
    if drive.name not in ("gp2", "gp3"):
        return None

    rps = desires.query_pattern.estimated_read_per_second.mid // zones_per_region
    write_per_sec = (
        desires.query_pattern.estimated_write_per_second.mid // zones_per_region
    )
    write_bytes_per_sec = round(
        write_per_sec * desires.query_pattern.estimated_mean_write_size_bytes.mid
    )
    read_bytes_per_sec = rps * desires.query_pattern.estimated_mean_read_size_bytes.mid
    # Write IO will be 1 to commitlog + 2 writes (plus 2 reads) in the first
    # hour during compaction.
    # Writes are sequential
    # Reads are random
    write_io_per_sec = (1 + 4) * max(
        1, write_bytes_per_sec // (drive.seq_io_size_kib * 1024)
    )
    read_io_per_sec = max(rps, read_bytes_per_sec // (drive.rand_io_size_kib * 1024))

    # Based on the disk latency and the read latency SLOs we adjust our
    # working set to keep more or less data in RAM. Faster drives need
    # less fronting RAM.
    ws_drive = instance.drive or drive
    working_set = working_set_from_drive_and_slo(
        drive_read_latency_dist=dist_for_interval(ws_drive.read_io_latency_ms),
        read_slo_latency_dist=dist_for_interval(
            desires.query_pattern.read_latency_slo_ms
        ),
        estimated_working_set=desires.data_shape.estimated_working_set_percent,
        # This is about right for a database, a cache probably would want
        # to increase this even more.
        target_percentile=0.95,
    ).mid

    requirement = _estimate_cassandra_requirement(
        instance=instance,
        desires=desires,
        working_set=working_set,
        reads_per_second=rps,
        max_rps_to_disk=max_rps_to_disk,
        zones_per_region=zones_per_region,
        copies_per_region=copies_per_region,
    )

    # Adjust the min count to adjust to prevent too much data on a single
    needed_disk_gib = int(requirement.disk_gib.mid)
    disk_buffer_ratio = buffer_for_components(
        buffers=desires.buffers, components=[BufferComponent.disk]
    ).ratio
    disk_per_node_gib = get_effective_disk_per_node_gib(
        instance,
        drive,
        disk_buffer_ratio,
        max_local_data_per_node_gib=max_local_data_per_node_gib,
        max_attached_data_per_node_gib=max_attached_data_per_node_gib,
    )

    current_cluster_size = _get_current_cluster_size(desires)
    cluster_size_lambda = _get_cluster_size_lambda(
        current_cluster_size, required_cluster_size
    )
    min_count = _get_min_count(
        tier=desires.service_tier,
        required_cluster_size=required_cluster_size,
        needed_disk_gib=needed_disk_gib,
        disk_per_node_gib=disk_per_node_gib,
        cluster_size_lambda=cluster_size_lambda,
    )

    base_mem = _get_base_memory(desires)

    heap_fn = _cass_heap_for_write_buffer(
        instance=instance,
        max_zonal_size=max_regional_size // zones_per_region,
        write_buffer_gib=requirement.context["write_buffer_gib"],
        buffer_percent=(max_write_buffer_percent * max_table_buffer_percent),
    )

    cluster = compute_stateful_zone(
        instance=instance,
        drive=drive,
        needed_cores=int(requirement.cpu_cores.mid),
        needed_disk_gib=needed_disk_gib,
        needed_memory_gib=int(requirement.mem_gib.mid),
        needed_network_mbps=requirement.network_mbps.mid,
        # Take into account the reads per read
        # from the per node dataset using leveled compaction
        required_disk_ios=lambda size, count: (
            _cass_io_per_read(size) * math.ceil(read_io_per_sec / count),
            write_io_per_sec / count,
        ),
        # C* clusters provision in powers of 2 because doubling
        cluster_size=cluster_size_lambda,
        min_count=min_count,
        # TODO: Take reserve memory calculation into account during buffer calculation
        # C* heap usage takes away from OS page cache memory
        reserve_memory=lambda x: base_mem + heap_fn(x),
        # C* heap buffers the writes at roughly a rate of
        # memtable_cleanup_threshold * memtable_size. At Netflix this
        # is 0.11 * 25 * heap
        write_buffer=lambda x: heap_fn(x) * max_write_buffer_percent * 0.25,
        required_write_buffer_gib=float(requirement.context["write_buffer_gib"]),
    )

    # Communicate to the actual provision that if we want reduced RF
    params = {
        "cassandra.keyspace.rf": copies_per_region,
        # In order to handle high write loads we have to shift memory
        # to heap memory, communicate with C* about this
        "cassandra.heap.gib": heap_fn(instance.ram_gib),
        "cassandra.heap.write.percent": max_write_buffer_percent,
        "cassandra.heap.table.percent": max_table_buffer_percent,
        "cassandra.compaction.min_threshold": requirement.context["min_threshold"],
    }
    _upsert_params(cluster, params)

    # Sometimes we don't want modify cluster topology, so only allow
    # topologies that match the desired zone size
    if required_cluster_size is not None and cluster.count != required_cluster_size:
        return None

    # Cassandra clusters generally should try to stay under some total number
    # of nodes. Orgs do this for all kinds of reasons such as
    #   * Security group limits. Since you must have < 500 rules if you're
    #       ingressing public ips)
    #   * Maintenance. If your restart script does one node at a time you want
    #       smaller clusters so your restarts don't take months.
    #   * Schema propagation. Since C* must gossip out changes to schema the
    #       duration of this can increase a lot with > 500 node clusters.
    if cluster.count > (max_regional_size // zones_per_region):
        return None

    # Calculate service costs (network + backup)
    cap_services = NflxCassandraCapacityModel.service_costs(
        service_type=NflxCassandraCapacityModel.service_name,
        context=context,
        desires=desires,
        extra_model_arguments={"copies_per_region": copies_per_region},
    )

    cluster.cluster_type = NflxCassandraCapacityModel.cluster_type
    zonal_clusters = [cluster] * zones_per_region

    # Account for the clusters, backup, and network costs
    cassandra_costs = NflxCassandraCapacityModel.cluster_costs(
        service_type=NflxCassandraCapacityModel.service_name,
        zonal_clusters=zonal_clusters,
    )
    cassandra_costs.update({s.service_type: s.annual_cost for s in cap_services})

    clusters = Clusters(
        annual_costs=cassandra_costs,
        zonal=zonal_clusters,
        regional=[],
        services=cap_services,
    )

    return CapacityPlan(
        requirements=Requirements(zonal=[requirement] * zones_per_region),
        candidate_clusters=clusters,
    )


# C* LCS has 160 MiB sstables by default and 10 sstables per level
def _cass_io_per_read(node_size_gib: float, sstable_size_mb: int = 160) -> int:
    gb = node_size_gib * 1024
    sstables = max(1, gb // sstable_size_mb)
    # 10 sstables per level, plus 1 for L0 (avg)
    levels = 1 + int(math.ceil(math.log(sstables, 10)))
    # One disk IO per data read and one per index read (assume we miss
    # the key cache)
    return 2 * levels


def _get_base_memory(desires: CapacityDesires) -> float:
    return (
        desires.data_shape.reserved_instance_app_mem_gib
        + desires.data_shape.reserved_instance_system_mem_gib
    )


def _cass_heap_for_write_buffer(
    instance: Instance,
    write_buffer_gib: float,
    max_zonal_size: int,
    buffer_percent: float,
) -> Callable[[float], float]:
    # If there is no way we can get enough heap with the max zonal size, try
    # letting max heap grow to 31 GiB per node to get more write buffer
    if write_buffer_gib > (
        max_zonal_size * _cass_heap(instance.ram_gib) * buffer_percent
    ):
        return lambda x: _cass_heap(x, max_heap_gib=30)
    else:
        return _cass_heap


# C* follows the following formula for calculating heap
def _cass_heap(node_memory_gib: float, max_heap_gib: float = 30) -> float:
    # OSS Cassandra does this
    # max(min(node_memory_gib // 2, 4), min(node_memory_gib // 4, max_heap_gib))

    # Netflix Cassandra does this
    return min(max(4, node_memory_gib // 2), max_heap_gib)


def _target_rf(desires: CapacityDesires, user_copies: Optional[int]) -> int:
    if user_copies is not None:
        assert user_copies > 1
        return user_copies

    # Due to the relaxed durability and consistency requirements we can
    # run with RF=2
    consistency = desires.query_pattern.access_consistency.same_region
    if (
        desires.data_shape.durability_slo_order.mid < 1000
        and consistency is not None
        and consistency.target_consistency != AccessConsistency.read_your_writes
    ):
        return 2
    return 3


class NflxCassandraArguments(BaseModel):
    """Configuration arguments for the Netflix Cassandra capacity model.

    This model centralizes all tunable parameters with their defaults.
    Use `from_extra_model_arguments()` to parse a dict into a validated instance.
    """

    copies_per_region: Optional[int] = Field(
        default=None,
        description="How many copies of the data will exist e.g. RF=3. If unsupplied"
        " this will be deduced from durability and consistency desires",
    )
    require_local_disks: bool = Field(
        default=True,
        description="If local (ephemeral) drives are required",
    )
    require_attached_disks: bool = Field(
        default=False,
        description="If attached (ebs) drives are required",
    )
    required_cluster_size: Optional[int] = Field(
        default=None,
        description="Require zonal clusters to be this size (force vertical scaling)",
    )
    max_rps_to_disk: int = Field(
        default=500,
        description="How many disk IOs should be allowed to hit disk per instance",
    )
    max_regional_size: int = Field(
        default=192,
        description="What is the maximum size of a cluster in this region",
    )
    max_local_data_per_node_gib: int = Field(
        default=1280,
        description="Maximum data per node for local disk instances (GiB)",
    )
    max_attached_data_per_node_gib: int = Field(
        default=2048,
        description="Maximum data per node for attached disk instances (GiB)",
    )
    max_write_buffer_percent: float = Field(
        default=0.25,
        description="The amount of heap memory that can be used to buffer writes. "
        "Note that if there are more than 100k writes this will "
        "automatically adjust to 0.5",
    )
    max_table_buffer_percent: float = Field(
        default=0.11,
        description="How much of heap memory can be used for a single table. "
        "Note that if there are more than 100k writes this will "
        "automatically adjust to 0.2",
    )

    @classmethod
    def from_extra_model_arguments(
        cls, extra_model_arguments: Dict[str, Any]
    ) -> "NflxCassandraArguments":
        """Parse extra_model_arguments dict into a validated NflxCassandraArguments.

        This centralizes default values - any field not in extra_model_arguments
        will use the default defined in this model.

        Handles legacy field name mappings:
        - max_local_disk_gib -> max_local_data_per_node_gib (if not explicitly set)
        """
        # Handle legacy field name: max_local_disk_gib -> max_local_data_per_node_gib
        args = dict(extra_model_arguments)
        if "max_local_data_per_node_gib" not in args and "max_local_disk_gib" in args:
            args["max_local_data_per_node_gib"] = args["max_local_disk_gib"]

        # Pydantic will use defaults for any missing fields
        return cls.model_validate(args)


class NflxCassandraCapacityModel(CapacityModel, CostAwareModel):
    service_name = "cassandra"
    cluster_type = "cassandra"

    def __init__(self) -> None:
        pass

    @staticmethod
    def get_required_cluster_size(
        tier: int, extra_model_arguments: Dict[str, Any]
    ) -> Optional[int]:
        required_cluster_size: Optional[int] = (
            math.ceil(extra_model_arguments["required_cluster_size"])
            if "required_cluster_size" in extra_model_arguments
            else None
        )

        if tier not in CRITICAL_TIERS or required_cluster_size is None:
            return required_cluster_size

        # If the upstream explicitly set a cluster size, make sure it is
        # at least CRITICAL_TIER_MIN_CLUSTER_SIZE. We cannot do a max
        # of the two because the horizontal scaling is disabled
        if required_cluster_size < CRITICAL_TIER_MIN_CLUSTER_SIZE:
            raise ValueError(
                f"Required cluster size must be at least "
                f"{CRITICAL_TIER_MIN_CLUSTER_SIZE=} when "
                f"service tier({tier}) is a "
                f"critical tier({CRITICAL_TIERS}). "
                f"If it is an existing cluster, horizontally "
                f"scale the cluster to be >= "
                f"{CRITICAL_TIER_MIN_CLUSTER_SIZE}"
            )

        return required_cluster_size

    @staticmethod
    def service_costs(
        service_type: str,
        context: RegionContext,
        desires: CapacityDesires,
        extra_model_arguments: Dict[str, Any],
    ) -> List[ServiceCapacity]:
        # C* service costs: network + backup
        copies_per_region: int = _target_rf(
            desires, extra_model_arguments.get("copies_per_region")
        )

        services: List[ServiceCapacity] = []
        services.extend(
            network_services(service_type, context, desires, copies_per_region)
        )

        if desires.data_shape.durability_slo_order.mid >= 1000:
            blob = context.services.get("blob.standard", None)
            if blob:
                # Calculate backup disk from desires (same as capacity_plan)
                # This ensures consistent backup costs regardless of how requirement was built
                backup_disk_gib = max(
                    1,
                    _get_disk_from_desires(desires, copies_per_region)
                    // context.zones_in_region,
                )
                services.append(
                    ServiceCapacity(
                        service_type=f"{service_type}.backup.{blob.name}",
                        annual_cost=blob.annual_cost_gib(backup_disk_gib),
                        service_params={
                            "nines_required": (
                                1 - 1.0 / desires.data_shape.durability_slo_order.mid
                            )
                        },
                    )
                )

        return services

    @staticmethod
    def cluster_costs(
        service_type: str,
        zonal_clusters: Sequence[ClusterCapacity] = (),
        regional_clusters: Sequence[ClusterCapacity] = (),
    ) -> Dict[str, float]:
        return cluster_infra_cost(
            service_type,
            zonal_clusters,
            regional_clusters,
            cluster_type=NflxCassandraCapacityModel.cluster_type,
        )

    @staticmethod
    def capacity_plan(
        instance: Instance,
        drive: Drive,
        context: RegionContext,
        desires: CapacityDesires,
        extra_model_arguments: Dict[str, Any],
    ) -> Optional[CapacityPlan]:
        # Parse extra_model_arguments into a validated model with centralized defaults
        args = NflxCassandraArguments.from_extra_model_arguments(extra_model_arguments)

        # Use durability and consistency to compute RF if not explicitly set
        copies_per_region = _target_rf(desires, args.copies_per_region)

        # Validate required_cluster_size for critical tiers
        required_cluster_size: Optional[int] = (
            NflxCassandraCapacityModel.get_required_cluster_size(
                desires.service_tier, extra_model_arguments
            )
        )

        # Apply caps to buffer percentages
        max_write_buffer_percent = min(0.5, args.max_write_buffer_percent)
        max_table_buffer_percent = min(0.5, args.max_table_buffer_percent)

        # Adjust heap defaults for high write clusters
        if (
            desires.query_pattern.estimated_write_per_second.mid >= 100_000
            and desires.data_shape.estimated_state_size_gib.mid >= 100
        ):
            max_write_buffer_percent = max(0.5, max_write_buffer_percent)
            max_table_buffer_percent = max(0.2, max_table_buffer_percent)

        return _estimate_cassandra_cluster_zonal(
            instance=instance,
            drive=drive,
            context=context,
            desires=desires,
            zones_per_region=context.zones_in_region,
            copies_per_region=copies_per_region,
            require_local_disks=args.require_local_disks,
            require_attached_disks=args.require_attached_disks,
            required_cluster_size=required_cluster_size,
            max_rps_to_disk=args.max_rps_to_disk,
            max_regional_size=args.max_regional_size,
            max_local_data_per_node_gib=args.max_local_data_per_node_gib,
            max_write_buffer_percent=max_write_buffer_percent,
            max_table_buffer_percent=max_table_buffer_percent,
        )

    @staticmethod
    def description() -> str:
        return "Netflix Streaming Cassandra Model"

    @staticmethod
    def extra_model_arguments_schema() -> Dict[str, Any]:
        return NflxCassandraArguments.model_json_schema()

    @staticmethod
    def default_buffers() -> Buffers:
        return Buffers(
            default=Buffer(ratio=1.5),
            desired={
                "compute": Buffer(ratio=1.5, components=[BufferComponent.compute]),
                "storage": Buffer(ratio=4.0, components=[BufferComponent.storage]),
                # Cassandra reserves headroom in both cpu and network for background
                # work and tasks
                "background": Buffer(
                    ratio=2.0,
                    components=[
                        BufferComponent.cpu,
                        BufferComponent.network,
                        BACKGROUND_BUFFER,
                    ],
                ),
            },
        )

    @staticmethod
    def default_desires(
        user_desires: CapacityDesires, extra_model_arguments: Dict[str, Any]
    ) -> CapacityDesires:
        acceptable_consistency = {
            None,
            AccessConsistency.best_effort,
            AccessConsistency.eventual,
            AccessConsistency.read_your_writes,
            AccessConsistency.never,
        }
        for key, value in user_desires.query_pattern.access_consistency:
            if value.target_consistency not in acceptable_consistency:
                raise ValueError(
                    f"Cassandra can only provide {acceptable_consistency} access."
                    f"User asked for {key}={value}"
                )

        # Lower RF = less write compute
        rf = _target_rf(
            user_desires, extra_model_arguments.get("copies_per_region", None)
        )
        if rf < 3:
            rf_write_latency = Interval(low=0.2, mid=0.6, high=2, confidence=0.98)
        else:
            rf_write_latency = Interval(low=0.4, mid=1, high=2, confidence=0.98)

        # By supplying these buffers we can deconstruct observed utilization into
        # load versus buffer.
        buffers = NflxCassandraCapacityModel.default_buffers()
        if user_desires.query_pattern.access_pattern == AccessPattern.latency:
            return CapacityDesires(
                query_pattern=QueryPattern(
                    access_pattern=AccessPattern.latency,
                    access_consistency=GlobalConsistency(
                        same_region=Consistency(
                            target_consistency=AccessConsistency.read_your_writes,
                        ),
                        cross_region=Consistency(
                            target_consistency=AccessConsistency.eventual,
                        ),
                    ),
                    estimated_mean_read_size_bytes=Interval(
                        low=128, mid=1024, high=65536, confidence=0.95
                    ),
                    estimated_mean_write_size_bytes=Interval(
                        low=64, mid=256, high=1024, confidence=0.95
                    ),
                    # Cassandra point queries usualy take just around 2ms
                    # of on CPU time for reads and 1ms for writes
                    estimated_mean_read_latency_ms=Interval(
                        low=0.4, mid=2, high=5, confidence=0.98
                    ),
                    estimated_mean_write_latency_ms=rf_write_latency,
                    # Assume point queries, "Single digit milliseconds SLO"
                    read_latency_slo_ms=FixedInterval(
                        minimum_value=0.2,
                        maximum_value=10,
                        low=0.4,
                        mid=2,
                        high=5,
                        confidence=0.98,
                    ),
                    write_latency_slo_ms=FixedInterval(
                        minimum_value=0.2,
                        maximum_value=10,
                        low=0.4,
                        mid=1,
                        high=4,
                        confidence=0.98,
                    ),
                ),
                # Most latency sensitive cassandra clusters are in the
                # < 1TiB range
                data_shape=DataShape(
                    estimated_state_size_gib=Interval(
                        low=10, mid=100, high=1000, confidence=0.98
                    ),
                    # Cassandra compresses with LZ4 by default
                    estimated_compression_ratio=Interval(
                        minimum_value=1.1,
                        maximum_value=8,
                        low=2,
                        mid=3,
                        high=5,
                        confidence=0.98,
                    ),
                    # We dynamically allocate the C* JVM memory in the plan
                    # but account for the Priam sidecar here
                    reserved_instance_app_mem_gib=4,
                ),
                buffers=buffers,
            )
        else:
            return CapacityDesires(
                query_pattern=QueryPattern(
                    access_pattern=AccessPattern.throughput,
                    access_consistency=GlobalConsistency(
                        same_region=Consistency(
                            target_consistency=AccessConsistency.read_your_writes,
                        ),
                        cross_region=Consistency(
                            target_consistency=AccessConsistency.eventual,
                        ),
                    ),
                    estimated_mean_read_size_bytes=Interval(
                        low=128, mid=1024, high=65536, confidence=0.95
                    ),
                    estimated_mean_write_size_bytes=Interval(
                        low=128, mid=1024, high=65536, confidence=0.95
                    ),
                    # Cassandra scan queries usually take longer
                    estimated_mean_read_latency_ms=Interval(
                        low=0.2, mid=5, high=20, confidence=0.98
                    ),
                    # Usually throughput clusters are running RF=2
                    # Maybe revise this?
                    estimated_mean_write_latency_ms=Interval(
                        low=0.2, mid=0.6, high=2, confidence=0.98
                    ),
                    # Assume they're scanning -> slow reads
                    read_latency_slo_ms=FixedInterval(
                        minimum_value=1,
                        maximum_value=100,
                        low=2,
                        mid=8,
                        high=90,
                        confidence=0.98,
                    ),
                    # Assume they're doing BATCH writes
                    write_latency_slo_ms=FixedInterval(
                        minimum_value=0.5,
                        maximum_value=20,
                        low=1,
                        mid=2,
                        high=8,
                        confidence=0.98,
                    ),
                ),
                data_shape=DataShape(
                    estimated_state_size_gib=Interval(
                        low=100, mid=1000, high=4000, confidence=0.98
                    ),
                    # Cassandra compresses with LZ4 by default
                    estimated_compression_ratio=Interval(
                        low=2, mid=3, high=5, confidence=0.98
                    ),
                    # We dynamically allocate the C* JVM memory in the plan
                    # but account for the Priam sidecar here
                    reserved_instance_app_mem_gib=4,
                ),
                buffers=buffers,
            )


nflx_cassandra_capacity_model = NflxCassandraCapacityModel()
