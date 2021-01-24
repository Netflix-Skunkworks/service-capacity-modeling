import logging
import math
from typing import Dict
from typing import Optional
from typing import Sequence
from typing import Tuple

from service_capacity_modeling.interface import AccessPattern
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import CapacityPlan
from service_capacity_modeling.interface import CapacityRequirement
from service_capacity_modeling.interface import certain_float
from service_capacity_modeling.interface import certain_int
from service_capacity_modeling.interface import Clusters
from service_capacity_modeling.interface import DataShape
from service_capacity_modeling.interface import Drive
from service_capacity_modeling.interface import FixedInterval
from service_capacity_modeling.interface import Instance
from service_capacity_modeling.interface import Interval
from service_capacity_modeling.interface import QueryPattern
from service_capacity_modeling.interface import Service
from service_capacity_modeling.models import CapacityModel
from service_capacity_modeling.models.common import compute_stateful_zone
from service_capacity_modeling.models.common import simple_network_mbps
from service_capacity_modeling.models.common import sqrt_staffed_cores
from service_capacity_modeling.models.common import working_set_from_drive_and_slo
from service_capacity_modeling.models.utils import next_power_of_2
from service_capacity_modeling.stats import gamma_for_interval


logger = logging.getLogger(__name__)


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

    needed_disk = desires.data_shape.estimated_state_size_gib.mid * copies_per_region

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
    rps_working_set = disk_rps / max_rps_to_disk

    # If disk RPS will be smaller than our target because there are no
    # reads, we don't need to hold as much data in memory
    needed_memory = min(working_set, rps_working_set) * needed_disk

    # Now convert to per zone
    needed_cores = needed_cores // zones_per_region
    needed_disk = needed_disk // zones_per_region
    needed_memory = int(needed_memory // zones_per_region)
    logger.info(
        "Need (cpu, mem, disk, working) = (%s, %s, %s, %f)",
        needed_cores,
        needed_memory,
        needed_disk,
        working_set,
    )

    return CapacityRequirement(
        core_reference_ghz=desires.core_reference_ghz,
        cpu_cores=certain_int(needed_cores),
        mem_gib=certain_float(needed_memory),
        disk_gib=certain_float(needed_disk),
        network_mbps=certain_float(needed_network_mbps),
        context={
            "working_set": min(working_set, rps_working_set),
            "rps_working_set": rps_working_set,
            "disk_slo_working_set": working_set,
        },
    )


# pylint: disable=too-many-locals
def _estimate_cassandra_cluster_zonal(
    instance: Instance,
    drive: Drive,
    desires: CapacityDesires,
    zones_per_region: int = 3,
    copies_per_region: int = 3,
    allow_gp2: bool = True,
    required_cluster_size: Optional[int] = None,
    max_rps_to_disk: int = 500,
    max_regional_size: int = 96,
) -> Optional[CapacityPlan]:

    # Cassandra only deploys on gp2 drives right now
    if drive.name != "gp2":
        return None

    # Cassandra doesn't like to deploy on really small instances
    if instance.cpu < 8:
        return None

    # if we're not allowed to use gp2, skip EBS only types
    if instance.drive is None and not allow_gp2:
        if not allow_gp2:
            return None

    rps = desires.query_pattern.estimated_read_per_second.mid // zones_per_region

    # Based on the disk latency and the read latency SLOs we adjust our
    # working set to keep more or less data in RAM. Faster drives need
    # less fronting RAM.
    ws_drive = instance.drive or drive
    working_set = working_set_from_drive_and_slo(
        drive_read_latency_dist=gamma_for_interval(ws_drive.read_io_latency_ms),
        read_slo_latency_dist=gamma_for_interval(
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

    cluster = compute_stateful_zone(
        instance=instance,
        # Only run C* on gp2
        drive=drive,
        needed_cores=int(requirement.cpu_cores.mid),
        needed_disk_gib=int(requirement.disk_gib.mid),
        needed_memory_gib=int(requirement.mem_gib.mid),
        needed_network_mbps=requirement.network_mbps.mid,
        # Assume that by provisioning enough memory we'll get
        # a 90% hit rate, but take into account the reads per read
        # from the per node dataset using leveled compaction
        # FIXME: I feel like this can be improved
        required_disk_ios=lambda x: _cass_io_per_read(x) * math.ceil(0.1 * rps),
        # C* requires ephemeral disks to be 25% full because compaction
        # and replacement time if we're underscaled.
        required_disk_space=lambda x: x * 4,
        # C* clusters provision in powers of 2 because doubling
        cluster_size=next_power_of_2,
        min_count=max(min_count, required_cluster_size or 0),
        # C* heap usage takes away from OS page cache memory
        reserve_memory=lambda x: max(min(x // 2, 4), min(x // 4, 12)),
        core_reference_ghz=requirement.core_reference_ghz,
    )

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

    cluster.cluster_type = "cassandra"
    clusters = Clusters(
        total_annual_cost=certain_float(zones_per_region * cluster.annual_cost),
        zonal=[cluster] * zones_per_region,
        regional=list(),
    )

    return CapacityPlan(requirement=requirement, candidate_clusters=clusters)


# C* LCS has 160 MiB sstables by default and 10 sstables per level
def _cass_io_per_read(node_size_gib, sstable_size_mb=160):
    gb = node_size_gib * 1024
    sstables = max(1, gb // sstable_size_mb)
    # 10 sstables per level, plus 1 for L0 (avg)
    levels = 1 + int(math.ceil(math.log(sstables, 10)))
    return levels


class NflxCassandraCapacityModel(CapacityModel):
    @staticmethod
    def capacity_plan(
        instance: Instance,
        drive: Drive,
        services: Dict[str, Service],
        desires: CapacityDesires,
        **kwargs,
    ) -> Optional[CapacityPlan]:
        zones_per_region: int = kwargs.pop("zones_per_region", 3)
        copies_per_region: int = kwargs.pop("copies_per_region", 3)
        allow_gp2: bool = kwargs.pop("allow_gp2", True)
        required_cluster_size: Optional[int] = kwargs.pop("required_cluster_size", None)
        max_rps_to_disk: int = kwargs.pop("max_rps_to_disk", 500)
        max_regional_size: int = kwargs.pop("max_regional_size", 96)

        return _estimate_cassandra_cluster_zonal(
            instance=instance,
            drive=drive,
            desires=desires,
            zones_per_region=zones_per_region,
            copies_per_region=copies_per_region,
            allow_gp2=allow_gp2,
            required_cluster_size=required_cluster_size,
            max_rps_to_disk=max_rps_to_disk,
            max_regional_size=max_regional_size,
        )

    @staticmethod
    def description():
        return "Netflix Streaming Cassandra Model"

    @staticmethod
    def extra_model_arguments() -> Sequence[Tuple[str, str, str]]:
        return (
            ("zones_per_region", "int = 3", "How many zones exist in the region"),
            (
                "copies_per_region",
                "int = 3",
                "How many copies of the data will exist e.g. RF=3",
            ),
            ("allow_gp2", "bool = 0", "If gp2 drives should be permitted"),
            (
                "max_rps_to_disk",
                "int = 500",
                "How many disk IOs should be allowed to hit disk per instance",
            ),
            (
                "max_regional_size",
                "int = 96",
                "What is the maximum size of a cluster in this region",
            ),
        )

    @staticmethod
    def default_desires(user_desires, **kwargs):
        if user_desires.query_pattern.access_pattern == AccessPattern.latency:
            return CapacityDesires(
                query_pattern=QueryPattern(
                    access_pattern=AccessPattern.latency,
                    estimated_mean_read_size_bytes=Interval(
                        low=128, mid=1024, high=65536, confidence=0.95
                    ),
                    estimated_mean_write_size_bytes=Interval(
                        low=64, mid=128, high=1024, confidence=0.95
                    ),
                    # Cassandra point queries usualy take just around 2ms
                    # of on CPU time for reads and 1ms for writes
                    estimated_mean_read_latency_ms=Interval(
                        low=0.4, mid=2, high=10, confidence=0.98
                    ),
                    estimated_mean_write_latency_ms=Interval(
                        low=0.4, mid=1, high=2, confidence=0.98
                    ),
                    # "Single digit milliseconds SLO"
                    read_latency_slo_ms=FixedInterval(
                        low=0.4, mid=2.5, high=10, confidence=0.98
                    ),
                    write_latency_slo_ms=FixedInterval(
                        low=0.4, mid=1, high=10, confidence=0.98
                    ),
                ),
                # Most latency sensitive cassandra clusters are in the
                # < 1TiB range
                data_shape=DataShape(
                    estimated_state_size_gib=Interval(
                        low=10, mid=100, high=1000, confidence=0.98
                    )
                ),
            )
        else:
            return CapacityDesires(
                query_pattern=QueryPattern(
                    access_pattern=AccessPattern.latency,
                    estimated_mean_read_size_bytes=Interval(
                        low=128, mid=1024, high=65536, confidence=0.95
                    ),
                    estimated_mean_write_size_bytes=Interval(
                        low=64, mid=128, high=1024, confidence=0.95
                    ),
                    # Cassandra scan queries usually take longer
                    estimated_mean_read_latency_ms=Interval(
                        low=0.2, mid=5, high=20, confidence=0.98
                    ),
                    estimated_mean_write_latency_ms=Interval(
                        low=0.2, mid=0.6, high=2, confidence=0.98
                    ),
                    # "Single digit milliseconds SLO"
                    read_latency_slo_ms=FixedInterval(
                        low=0.4, mid=8, high=100, confidence=0.98
                    ),
                    write_latency_slo_ms=FixedInterval(
                        low=0.4, mid=2, high=10, confidence=0.98
                    ),
                ),
                data_shape=DataShape(
                    estimated_state_size_gib=Interval(
                        low=100, mid=1000, high=4000, confidence=0.98
                    )
                ),
            )
