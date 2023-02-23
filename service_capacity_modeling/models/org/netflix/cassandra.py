import logging
import math
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional

from pydantic import BaseModel
from pydantic import Field

from service_capacity_modeling.interface import AccessConsistency
from service_capacity_modeling.interface import AccessPattern
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import CapacityPlan
from service_capacity_modeling.interface import CapacityRequirement
from service_capacity_modeling.interface import certain_float
from service_capacity_modeling.interface import certain_int
from service_capacity_modeling.interface import Clusters
from service_capacity_modeling.interface import Consistency
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
from service_capacity_modeling.models.common import compute_stateful_zone
from service_capacity_modeling.models.common import network_services
from service_capacity_modeling.models.common import simple_network_mbps
from service_capacity_modeling.models.common import sqrt_staffed_cores
from service_capacity_modeling.models.common import working_set_from_drive_and_slo
from service_capacity_modeling.models.utils import next_power_of_2
from service_capacity_modeling.stats import dist_for_interval

logger = logging.getLogger(__name__)


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


def _estimate_cassandra_requirement(
    instance: Instance,
    desires: CapacityDesires,
    working_set: float,
    reads_per_second: float,
    max_rps_to_disk: int,
    zones_per_region: int = 3,
    copies_per_region: int = 3,
) -> CapacityRequirement:
    """Estimate the capacity required for one zone given a regional desire

    The input desires should be the **regional** desire, and this function will
    return the zonal capacity requirement
    """
    # Keep half of the cores free for background work (compaction, backup, repair)
    needed_cores = sqrt_staffed_cores(desires) * 2
    # Keep half of the bandwidth available for backup
    needed_network_mbps = simple_network_mbps(desires) * 2

    needed_disk = math.ceil(
        (1.0 / desires.data_shape.estimated_compression_ratio.mid)
        * desires.data_shape.estimated_state_size_gib.mid
        * copies_per_region,
    )

    # Rough estimate of how many instances we would need just for the the CPU
    # Note that this is a lower bound, we might end up with more.
    needed_cores = math.ceil(
        max(1, needed_cores // (instance.cpu_ghz / desires.core_reference_ghz))
    )
    rough_count = math.ceil(needed_cores / instance.cpu)

    # Generally speaking we want fewer than some number of reads per second
    # hitting disk per instance. If we don't have many reads we don't need to
    # hold much data in memory.
    instance_rps = max(1, reads_per_second // rough_count)
    disk_rps = instance_rps * _cass_io_per_read(max(1, needed_disk // rough_count))
    rps_working_set = min(1.0, disk_rps / max_rps_to_disk)

    # If disk RPS will be smaller than our target because there are no
    # reads, we don't need to hold as much data in memory
    needed_memory = min(working_set, rps_working_set) * needed_disk

    # Now convert to per zone
    needed_cores = max(1, needed_cores // zones_per_region)
    needed_disk = max(1, needed_disk // zones_per_region)
    needed_memory = max(1, int(needed_memory // zones_per_region))
    logger.debug(
        "Need (cpu, mem, disk, working) = (%s, %s, %s, %f)",
        needed_cores,
        needed_memory,
        needed_disk,
        working_set,
    )

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

    return CapacityRequirement(
        requirement_type="cassandra-zonal",
        core_reference_ghz=desires.core_reference_ghz,
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


def _upsert_params(cluster, params):
    if cluster.cluster_params:
        cluster.cluster_params.update(params)
    else:
        cluster.cluster_params = params


# pylint: disable=too-many-locals
# flake8: noqa: C901
def _estimate_cassandra_cluster_zonal(
    instance: Instance,
    drive: Drive,
    context: RegionContext,
    desires: CapacityDesires,
    zones_per_region: int = 3,
    copies_per_region: int = 3,
    require_local_disks: bool = False,
    required_cluster_size: Optional[int] = None,
    max_rps_to_disk: int = 500,
    max_local_disk_gib: int = 2048,
    max_regional_size: int = 96,
    max_write_buffer_percent: float = 0.25,
    max_table_buffer_percent: float = 0.11,
) -> Optional[CapacityPlan]:

    # Netflix Cassandra doesn't like to deploy on really small instances
    if instance.cpu < 2 or instance.ram_gib < 14:
        return None

    # if we're not allowed to use gp2, skip EBS only types
    if instance.drive is None and require_local_disks:
        return None

    # Cassandra only deploys on gp2 drives right now
    if drive.name != "gp2":
        return None

    rps = desires.query_pattern.estimated_read_per_second.mid // zones_per_region
    write_per_sec = (
        desires.query_pattern.estimated_write_per_second.mid // zones_per_region
    )
    write_bytes_per_sec = (
        write_per_sec * desires.query_pattern.estimated_mean_write_size_bytes.mid
    )
    # Write IO will be 1 to commitlog + 2 writes (plus 2 reads) in the first
    # hour during compaction.
    # https://aws.amazon.com/ebs/volume-types/ says IOPS are 16k for io2/gp2
    # so for now we're just hardcoding.
    write_io_per_sec = (1 + 4) * max(1, write_bytes_per_sec // 16384)

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

    # Cassandra clusters should aim to be at least 2 nodes per zone to start
    # out with for tier 0 or tier 1. This gives us more room to "up-color"]
    # clusters.
    min_count = 0
    if desires.service_tier <= 1:
        min_count = 2

    base_mem = (
        desires.data_shape.reserved_instance_app_mem_gib
        + desires.data_shape.reserved_instance_system_mem_gib
    )

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
        needed_disk_gib=int(requirement.disk_gib.mid),
        needed_memory_gib=int(requirement.mem_gib.mid),
        needed_network_mbps=requirement.network_mbps.mid,
        core_reference_ghz=requirement.core_reference_ghz,
        # Take into account the reads per read
        # from the per node dataset using leveled compaction
        required_disk_ios=lambda size, count: (
            _cass_io_per_read(size) * math.ceil(rps / count),
            write_io_per_sec / count,
        ),
        # C* requires ephemeral disks to be 25% full because compaction
        # and replacement time if we're underscaled.
        required_disk_space=lambda x: x * 4,
        # C* clusters cannot recover data from neighbors quickly so we
        # want to avoid clusters with more than 1 TiB of local state
        max_local_disk_gib=max_local_disk_gib,
        # C* clusters provision in powers of 2 because doubling
        cluster_size=next_power_of_2,
        min_count=max(min_count, required_cluster_size or 0),
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

    # Durable Cassandra clusters backup to S3
    # TODO use the write rate and estimated write size to estimate churn
    # over the retention period.
    cap_services = []
    if desires.data_shape.durability_slo_order.mid >= 1000:
        blob = context.services.get("blob.standard", None)
        if blob:
            cap_services = [
                ServiceCapacity(
                    service_type=f"cassandra.backup.{blob.name}",
                    annual_cost=blob.annual_cost_per_gib * requirement.disk_gib.mid,
                    service_params={
                        "nines_required": (
                            1 - 1.0 / desires.data_shape.durability_slo_order.mid
                        )
                    },
                )
            ]

    network_costs = network_services("cassandra", context, desires, copies_per_region)
    if network_costs:
        cap_services.extend(network_costs)

    # Account for the clusters, backup, and network costs
    cassandra_costs = {
        "cassandra.zonal-clusters": zones_per_region * cluster.annual_cost,
    }
    for s in cap_services:
        cassandra_costs[f"{s.service_type}"] = s.annual_cost

    cluster.cluster_type = "cassandra"
    clusters = Clusters(
        annual_costs=cassandra_costs,
        zonal=[cluster] * zones_per_region,
        regional=[],
        services=cap_services,
    )

    return CapacityPlan(
        requirements=Requirements(zonal=[requirement] * zones_per_region),
        candidate_clusters=clusters,
    )


# C* LCS has 160 MiB sstables by default and 10 sstables per level
def _cass_io_per_read(node_size_gib, sstable_size_mb=160):
    gb = node_size_gib * 1024
    sstables = max(1, gb // sstable_size_mb)
    # 10 sstables per level, plus 1 for L0 (avg)
    levels = 1 + int(math.ceil(math.log(sstables, 10)))
    # One disk IO per data read and one per index read (assume we miss
    # the key cache)
    return 2 * levels


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
        return lambda x: _cass_heap(x, max_heap_gib=31)
    else:
        return _cass_heap


# C* follows the following formula for calculating heap
def _cass_heap(node_memory_gib: float, max_heap_gib: float = 12) -> float:
    return max(min(node_memory_gib // 2, 4), min(node_memory_gib // 4, max_heap_gib))


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
    copies_per_region: int = Field(
        default=3,
        description="How many copies of the data will exist e.g. RF=3. If unsupplied"
        " this will be deduced from durability and consistency desires",
    )
    require_local_disks: bool = Field(
        default=False,
        description="If local (ephemeral) drives are required",
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
        default=96,
        description="What is the maximum size of a cluster in this region",
    )
    max_local_disk_gib: int = Field(
        default=2048,
        description="The maximum amount of data we store per machine",
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


class NflxCassandraCapacityModel(CapacityModel):
    @staticmethod
    def capacity_plan(
        instance: Instance,
        drive: Drive,
        context: RegionContext,
        desires: CapacityDesires,
        extra_model_arguments: Dict[str, Any],
    ) -> Optional[CapacityPlan]:
        # Use durabiliy and consistency to compute RF.
        copies_per_region = _target_rf(
            desires, extra_model_arguments.get("copies_per_region", None)
        )
        require_local_disks: bool = extra_model_arguments.get(
            "require_local_disks", False
        )
        required_cluster_size: Optional[int] = extra_model_arguments.get(
            "required_cluster_size", None
        )
        max_rps_to_disk: int = extra_model_arguments.get("max_rps_to_disk", 500)
        max_regional_size: int = extra_model_arguments.get("max_regional_size", 96)
        max_local_disk_gib: int = extra_model_arguments.get("max_local_disk_gib", 2048)
        max_write_buffer_percent: float = min(
            0.5, extra_model_arguments.get("max_write_buffer_percent", 0.25)
        )
        max_table_buffer_percent: float = min(
            0.5, extra_model_arguments.get("max_table_buffer_percent", 0.11)
        )

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
            require_local_disks=require_local_disks,
            required_cluster_size=required_cluster_size,
            max_rps_to_disk=max_rps_to_disk,
            max_regional_size=max_regional_size,
            max_local_disk_gib=max_local_disk_gib,
            max_write_buffer_percent=max_write_buffer_percent,
            max_table_buffer_percent=max_table_buffer_percent,
        )

    @staticmethod
    def description():
        return "Netflix Streaming Cassandra Model"

    @staticmethod
    def extra_model_arguments_schema() -> Dict[str, Any]:
        return NflxCassandraArguments.schema()

    @staticmethod
    def default_desires(user_desires, extra_model_arguments: Dict[str, Any]):
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
            )


nflx_cassandra_capacity_model = NflxCassandraCapacityModel()
