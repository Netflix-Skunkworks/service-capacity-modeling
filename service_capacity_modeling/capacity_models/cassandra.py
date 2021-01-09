import logging
import math
from typing import Optional

from service_capacity_modeling.capacity_models import CapacityModel
from service_capacity_modeling.capacity_models.common import compute_stateful_zone
from service_capacity_modeling.capacity_models.common import simple_network_mbps
from service_capacity_modeling.capacity_models.common import sqrt_staffed_cores
from service_capacity_modeling.capacity_models.utils import next_power_of_2
from service_capacity_modeling.models import CapacityDesires
from service_capacity_modeling.models import CapacityPlan
from service_capacity_modeling.models import CapacityRequirement
from service_capacity_modeling.models import certain_float
from service_capacity_modeling.models import certain_int
from service_capacity_modeling.models import Clusters
from service_capacity_modeling.models import Drive
from service_capacity_modeling.models import Instance
from service_capacity_modeling.stats import gamma_for_interval


logger = logging.getLogger(__name__)


def _estimate_cassandra_requirement(
    desires: CapacityDesires,
    working_set: float,
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
    needed_memory = working_set * needed_disk

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
) -> Optional[CapacityPlan]:

    if instance.drive is None:
        # if we're not allowed to use gp2, skip EBS only types
        if not allow_gp2:
            return None

    # Cassandra only deploys on gp2 drives right now
    if drive.name != "gp2":
        return None

    # Cassandra doesn't like to deploy on really small instances
    if instance.cpu < 8:
        return None

    rps = desires.query_pattern.estimated_read_per_second.mid // zones_per_region

    # Based on the disk latency and the read latency SLOs we adjust our
    # working set to keep more or less data in RAM. Faster drives need
    # less fronting RAM.
    working_set = desires.data_shape.working_set_percent(
        drive_read_latency_dist=gamma_for_interval(drive.read_io_latency_ms),
        read_slo_latency_dist=gamma_for_interval(
            desires.query_pattern.read_latency_slo_ms
        ),
        target_percentile=0.10,
    ).mid

    requirement = _estimate_cassandra_requirement(
        desires=desires,
        working_set=working_set,
        zones_per_region=zones_per_region,
        copies_per_region=copies_per_region,
    )

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
        # C* heap usage takes away from OS page cache memory
        reserve_memory=lambda x: max(min(x // 2, 4), min(x // 4, 12)),
        core_reference_ghz=requirement.core_reference_ghz,
    )

    # Sometimes we don't want modify cluster topology, so only allow
    # topologies that match the desired zone size
    if required_cluster_size is not None and cluster.count != required_cluster_size:
        return None

    # Cassandra clusters shouldn't be more than 32 nodes per zone
    if cluster.count <= 32:
        cluster.cluster_type = "cassandra"
    else:
        return None

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
        desires: CapacityDesires,
        **kwargs,
    ) -> Optional[CapacityPlan]:
        zones_per_region: int = kwargs.pop("zones_per_region", 3)
        copies_per_region: int = kwargs.pop("copies_per_region", 3)
        allow_gp2: bool = kwargs.pop("allow_gp2", True)
        required_cluster_size: Optional[int] = kwargs.pop("required_cluster_size", None)

        return _estimate_cassandra_cluster_zonal(
            instance=instance,
            drive=drive,
            desires=desires,
            zones_per_region=zones_per_region,
            copies_per_region=copies_per_region,
            allow_gp2=allow_gp2,
            required_cluster_size=required_cluster_size,
        )
