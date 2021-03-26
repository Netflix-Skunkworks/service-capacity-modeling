import logging
from decimal import Decimal
from typing import Any
from typing import Dict
from typing import Optional
from typing import Sequence
from typing import Tuple

from service_capacity_modeling.interface import AccessConsistency
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
from service_capacity_modeling.interface import RegionContext
from service_capacity_modeling.interface import Requirements
from service_capacity_modeling.models import CapacityModel
from service_capacity_modeling.models.common import compute_stateful_zone
from service_capacity_modeling.models.common import simple_network_mbps
from service_capacity_modeling.models.common import sqrt_staffed_cores
from service_capacity_modeling.models.common import working_set_from_drive_and_slo
from service_capacity_modeling.models.utils import next_n
from service_capacity_modeling.stats import dist_for_interval


logger = logging.getLogger(__name__)


def _estimate_evcache_requirement(
    instance: Instance,
    desires: CapacityDesires,
    working_set: float,
    zones_per_region: int = 3,
    copies_per_region: int = 2,
) -> CapacityRequirement:
    """Estimate the capacity required for one zone given a regional desire

    The input desires should be the **regional** desire, and this function will
    return the zonal capacity requirement
    """
    # EVCache can run at full CPU utilization
    needed_cores = sqrt_staffed_cores(desires)

    # (FIXME): Need to figure out the right network approach here ...
    # Keep half of the bandwidth available for cache warmer
    needed_network_mbps = simple_network_mbps(desires) * 2

    needed_disk = round(
        desires.data_shape.estimated_state_size_gib.mid * copies_per_region,
        2,
    )

    # (FIXME): Is this the right statement?
    if instance.drive is None:
        # We can't currently store data on cloud drives, but we can put the
        # dataset into memory!
        needed_memory = needed_disk
        needed_disk = 0
    else:
        # We can store data on fast ephems (reducing the working set that must
        # be kept in RAM)
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
        requirement_type="evcache-zonal",
        core_reference_ghz=desires.core_reference_ghz,
        cpu_cores=certain_int(needed_cores),
        mem_gib=certain_float(needed_memory),
        disk_gib=certain_float(needed_disk),
        network_mbps=certain_float(needed_network_mbps),
        context={
            "working_set": working_set,
            "replication_factor": copies_per_region,
        },
    )


def _upsert_params(cluster, params):
    if cluster.cluster_params:
        cluster.cluster_params.update(params)
    else:
        cluster.cluster_params = params


# pylint: disable=too-many-locals
def _estimate_evcache_cluster_zonal(
    instance: Instance,
    drive: Drive,
    desires: CapacityDesires,
    zones_per_region: int = 3,
    copies_per_region: int = 3,
    max_local_disk_gib: int = 2048,
    max_regional_size: int = 288,
) -> Optional[CapacityPlan]:

    # evcache doesn't like to deploy on single cpu instances
    if instance.cpu < 2:
        return None

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
        # Caches have very tight latency SLOs, so we target a high
        # percentile of the drive latency distribution for WS calculation
        target_percentile=0.99,
    ).mid

    requirement = _estimate_evcache_requirement(
        instance=instance,
        desires=desires,
        working_set=working_set,
        zones_per_region=zones_per_region,
        copies_per_region=copies_per_region,
    )

    # evcache clusters should aim to be at least 2 nodes per zone to start
    # out with for tier 0 or tier 1. This gives us more room to "up-color"]
    # clusters.
    min_count = 0
    if desires.service_tier <= 1:
        min_count = 2

    base_mem = (
        desires.data_shape.reserved_instance_app_mem_gib
        + desires.data_shape.reserved_instance_system_mem_gib
    )

    cluster = compute_stateful_zone(
        instance=instance,
        drive=drive,
        needed_cores=int(requirement.cpu_cores.mid),
        needed_disk_gib=int(requirement.disk_gib.mid),
        needed_memory_gib=int(requirement.mem_gib.mid),
        needed_network_mbps=requirement.network_mbps.mid,
        # EVCache doesn't use cloud drives to store data, we will have
        # accounted for the data going on drives or memory via working set
        required_disk_ios=lambda x: 0,
        required_disk_space=lambda x: 0,
        max_local_disk_gib=max_local_disk_gib,
        # EVCache clusters should be balanced per zone
        cluster_size=lambda x: next_n(x, zones_per_region),
        min_count=max(min_count, 0),
        # EVCache takes away memory from evcache
        reserve_memory=lambda x: base_mem,
        core_reference_ghz=requirement.core_reference_ghz,
    )

    # Communicate to the actual provision that if we want reduced RF
    params = {"evcache.copies": copies_per_region}
    _upsert_params(cluster, params)

    # evcache clusters generally should try to stay under some total number
    # of nodes. Orgs do this for all kinds of reasons such as
    #   * Security group limits. Since you must have < 500 rules if you're
    #       ingressing public ips)
    #   * Maintenance. If your restart script does one node at a time you want
    #       smaller clusters so your restarts don't take months.
    #   * NxN network issues. Sometimes smaller clusters of bigger nodes
    #       are better for network propagation
    if cluster.count > (max_regional_size // zones_per_region):
        return None

    ec2_cost = zones_per_region * cluster.annual_cost

    cluster.cluster_type = "evcache"
    clusters = Clusters(
        total_annual_cost=round(Decimal(ec2_cost), 2),
        zonal=[cluster] * zones_per_region,
        regional=list(),
    )

    return CapacityPlan(
        requirements=Requirements(zonal=[requirement] * zones_per_region),
        candidate_clusters=clusters,
    )


class NflxEVCacheCapacityModel(CapacityModel):
    @staticmethod
    def capacity_plan(
        instance: Instance,
        drive: Drive,
        context: RegionContext,
        desires: CapacityDesires,
        extra_model_arguments: Dict[str, Any],
    ) -> Optional[CapacityPlan]:
        # TODO: Use read requirements to compute RF.
        copies_per_region: int = extra_model_arguments.get("copies_per_region", 2)
        max_regional_size: int = extra_model_arguments.get("max_regional_size", 288)
        # Very large nodes are hard to cache warm
        max_local_disk_gib: int = extra_model_arguments.get("max_local_disk_gib", 2048)

        return _estimate_evcache_cluster_zonal(
            instance=instance,
            drive=drive,
            desires=desires,
            zones_per_region=context.zones_in_region,
            copies_per_region=copies_per_region,
            max_regional_size=max_regional_size,
            max_local_disk_gib=max_local_disk_gib,
        )

    @staticmethod
    def description():
        return "Netflix Streaming EVCache (memcached) Model"

    @staticmethod
    def extra_model_arguments() -> Sequence[Tuple[str, str, str]]:
        return (
            (
                "copies_per_region",
                "int = 2",
                "How many copies of the data will exist e.g. RF=2. If unsupplied"
                " this will be deduced from durability and consistency desires",
            ),
            (
                "max_regional_size",
                "int = 288",
                "What is the maximum size of a cluster in this region",
            ),
            (
                "max_local_disk_gib",
                "int = 2048",
                "The maximum amount of data we store per machine",
            ),
        )

    @staticmethod
    def default_desires(user_desires, extra_model_arguments: Dict[str, Any]):
        acceptable_consistency = set((AccessConsistency.best_effort,))
        for key, value in user_desires.query_pattern.access_consistency:
            if value.target_consistency not in acceptable_consistency:
                raise ValueError(
                    f"evcache can only provide {acceptable_consistency} access."
                    f"User asked for {key}={value}"
                )

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
                    # memcache point queries usualy take just around 100us
                    # of on CPU time for reads and writes. Memcache is very
                    # fast
                    estimated_mean_read_latency_ms=Interval(
                        low=0.01, mid=0.1, high=0.2, confidence=0.98
                    ),
                    estimated_mean_write_latency_ms=Interval(
                        low=0.01, mid=0.1, high=0.2, confidence=0.98
                    ),
                    # Assume point queries, "1 millisecond SLO"
                    read_latency_slo_ms=FixedInterval(
                        minimum_value=0.1,
                        maximum_value=4,
                        low=0.15,
                        mid=0.5,
                        high=1,
                        confidence=0.98,
                    ),
                    write_latency_slo_ms=FixedInterval(
                        minimum_value=0.2,
                        maximum_value=10,
                        low=0.4,
                        mid=1,
                        high=5,
                        confidence=0.98,
                    ),
                ),
                # Most latency sensitive evcache clusters are in the
                # < 100GiB range
                data_shape=DataShape(
                    estimated_state_size_gib=Interval(
                        low=10, mid=20, high=100, confidence=0.98
                    ),
                    # account for the evcar sidecar here
                    reserved_instance_app_mem_gib=0.5,
                ),
            )
        else:
            return CapacityDesires(
                # (FIXME): Need to pair with memcache folks on the exact values
                query_pattern=QueryPattern(
                    access_pattern=AccessPattern.throughput,
                    estimated_mean_read_size_bytes=Interval(
                        low=128, mid=1024, high=65536, confidence=0.95
                    ),
                    estimated_mean_write_size_bytes=Interval(
                        low=128, mid=1024, high=65536, confidence=0.95
                    ),
                    # evcache bulk reads usually take slightly longer
                    estimated_mean_read_latency_ms=Interval(
                        low=0.01, mid=0.15, high=0.3, confidence=0.98
                    ),
                    # evcache bulk puts usually take slightly longer
                    estimated_mean_write_latency_ms=Interval(
                        low=0.01, mid=0.15, high=0.3, confidence=0.98
                    ),
                    # Assume they're multi-getting -> slow reads
                    read_latency_slo_ms=FixedInterval(
                        minimum_value=0.1,
                        maximum_value=10,
                        low=0.2,
                        mid=0.5,
                        high=1,
                        confidence=0.98,
                    ),
                    # Assume they're multi-setting writes
                    write_latency_slo_ms=FixedInterval(
                        minimum_value=0.1,
                        maximum_value=10,
                        low=0.2,
                        mid=0.5,
                        high=1,
                        confidence=0.98,
                    ),
                ),
                data_shape=DataShape(
                    # Typical ML models
                    estimated_state_size_gib=Interval(
                        low=10, mid=100, high=1000, confidence=0.98
                    ),
                    # We dynamically allocate the memcache JVM memory in the
                    # plan but account for the Priam sidecar here
                    reserved_instance_app_mem_gib=0.5,
                ),
            )
