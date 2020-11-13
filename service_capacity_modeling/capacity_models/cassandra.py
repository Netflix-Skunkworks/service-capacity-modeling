import logging
import math
from typing import List
from typing import Optional

from service_capacity_modeling.capacity_models.common import compute_stateful_zone
from service_capacity_modeling.capacity_models.common import simple_network_mbps
from service_capacity_modeling.capacity_models.common import sqrt_staffed_cores
from service_capacity_modeling.capacity_models.utils import next_power_of_2
from service_capacity_modeling.capacity_models.utils import reduce_by_family
from service_capacity_modeling.models import CapacityDesires
from service_capacity_modeling.models import CapacityPlan
from service_capacity_modeling.models import CapacityRequirement
from service_capacity_modeling.models import certain_float
from service_capacity_modeling.models import certain_int
from service_capacity_modeling.models import Hardware
from service_capacity_modeling.models import ZoneClusterCapacity


logger = logging.getLogger(__name__)


# pylint: disable=too-many-locals
def estimate_cassandra_cluster_zone(
    hardware: Hardware,
    desires: CapacityDesires,
    zones_per_region: int = 3,
    copies_per_region: int = 3,
    allow_gp2: bool = True,
    required_cluster_size: Optional[int] = None,
) -> CapacityPlan:
    """Estimate the capacity required for one zone given a regional desire

    The input desires should be the **regional** desire, and this function will
    return a list of potential zone configuraitons that would meet that need

    Note that
    """
    # Keep half of the cores free for background work (compaction, backup, repair)
    needed_cores = sqrt_staffed_cores(desires) * 2
    # Keep half of the bandwidth available for backup
    needed_network_mbps = simple_network_mbps(desires) * 2

    needed_disk = desires.data_shape.estimated_state_size_gb.mid * copies_per_region
    needed_memory = desires.data_shape.estimated_working_set_percent.mid * needed_disk

    # Now convert to per zone
    needed_cores = needed_cores // zones_per_region
    needed_disk = needed_disk // zones_per_region
    needed_memory = int(needed_memory // zones_per_region)
    rps = desires.query_pattern.estimated_read_per_second.mid // zones_per_region

    logger.info(
        "Need (cpu, mem, disk) = (%s, %s, %s)", needed_cores, needed_memory, needed_disk
    )

    topologies: List[ZoneClusterCapacity] = []
    for instance in hardware.instances.values():
        if instance.drive is None:
            # if we're not allowed to use gp2, skip EBS only types
            if not allow_gp2:
                continue

        # Cassandra doesn't like to deploy on really small instances
        if instance.cpu < 8:
            continue

        cluster = compute_stateful_zone(
            instance=instance,
            # Only run C* on gp2
            drive=hardware.drives["gp2"],
            needed_cores=needed_cores,
            needed_disk_gib=needed_disk,
            needed_memory_gib=needed_memory,
            needed_network_mbps=needed_network_mbps,
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
            core_reference_ghz=hardware.core_reference_ghz,
        )

        # Sometimes we don't want modify cluster topology, so only allow
        # topologies that match the desired zone size
        if required_cluster_size is not None and cluster.count != required_cluster_size:
            continue

        # Cassandra clusters shouldn't be more than 32 nodes per zone
        if cluster.count <= 32:
            zc = cluster.copy()
            zc.cluster_type = "cassandra"
            topologies.append(zc)

    # We only want one kind from each family
    return CapacityPlan(
        capacity_requirement=CapacityRequirement(
            cpu_reference_ghz=hardware.core_reference_ghz,
            cpu_cores=certain_int(needed_cores),
            mem_gib=certain_float(needed_memory),
            disk_gib=certain_float(needed_disk),
            network_mbps=certain_float(needed_network_mbps),
        ),
        clusters=reduce_by_family(topologies)[:4],
    )


# C* LCS has 160 MiB sstables by default and 10 sstables per level
def _cass_io_per_read(node_size_gib, sstable_size_mb=160):
    gb = node_size_gib * 1024
    sstables = max(1, gb // sstable_size_mb)
    # 10 sstables per level, plus 1 for L0 (avg)
    levels = 1 + int(math.ceil(math.log(sstables, 10)))
    return levels
