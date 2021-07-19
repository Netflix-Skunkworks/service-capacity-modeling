import logging
import math
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
    copies_per_region: int,
    zones_per_region: int = 3,
) -> Tuple[CapacityRequirement, Tuple[str, ...]]:
    """Estimate the capacity required for one zone given a regional desire

    The input desires should be the **regional** desire, and this function will
    return the zonal capacity requirement
    """
    # EVCache can run at full CPU utilization
    needed_cores = sqrt_staffed_cores(desires)

    # (Arun): Keep 20% of available bandwidth for cache warmer
    needed_network_mbps = simple_network_mbps(desires) * 1.25

    needed_disk = math.ceil(
        desires.data_shape.estimated_state_size_gib.mid * copies_per_region,
    )

    regrets: Tuple[str, ...] = ("spend", "mem")
    # (Arun): As of 2021 we are using ephemerals exclusively and do not
    # use cloud drives
    if instance.drive is None:
        # We can't currently store data on cloud drives, but we can put the
        # dataset into memory!
        needed_memory = float(needed_disk)
        needed_disk = 0
    else:
        # We can store data on fast ephems (reducing the working set that must
        # be kept in RAM)
        needed_memory = float(working_set) * float(needed_disk)
        regrets = ("spend", "disk", "mem")

    # Now convert to per zone
    needed_cores = max(1, needed_cores // zones_per_region)
    if needed_disk > 0:
        needed_disk = max(1, needed_disk // zones_per_region)
    else:
        needed_disk = needed_disk // zones_per_region
    needed_memory = max(1, int(needed_memory // zones_per_region))
    logger.debug(
        "Need (cpu, mem, disk, working) = (%s, %s, %s, %f)",
        needed_cores,
        needed_memory,
        needed_disk,
        working_set,
    )

    return (
        CapacityRequirement(
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
        ),
        regrets,
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
    max_regional_size: int = 999,
    min_instance_memory_gib: int = 12,
) -> Optional[CapacityPlan]:

    # EVCache doesn't like to deploy on single CPU instances
    if instance.cpu < 2:
        return None

    # EVCache doesn't like to deploy to instances with < 7 GiB of ram
    if instance.ram_gib < min_instance_memory_gib:
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

    requirement, regrets = _estimate_evcache_requirement(
        instance=instance,
        desires=desires,
        working_set=working_set,
        zones_per_region=zones_per_region,
        copies_per_region=copies_per_region,
    )

    # Account for sidecars and base system memory
    base_mem = (
        desires.data_shape.reserved_instance_app_mem_gib
        + desires.data_shape.reserved_instance_system_mem_gib
    )

    # (Arun) We currently reserve extra memory for the OS as instances get
    # larger to account for additional overhead. Note that the
    # reserved_instance_system_mem_gib has a base of 1 GiB OSMEM so this
    # just represents the variable component
    def reserve_memory(instance_mem_gib):
        # (Joey) From the chart it appears to be about a 3% overhead for
        # OS memory.
        variable_os = int(instance_mem_gib * 0.03)
        return base_mem + variable_os

    requirement.context["osmem"] = reserve_memory(instance.ram_gib)

    # EVCache clusters aim to be at least 2 nodes per zone to start
    # out with for tier 0
    min_count = 0
    if desires.service_tier < 1:
        min_count = 2

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
        # Sidecars and Variable OS Memory
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
        requirements=Requirements(
            zonal=[requirement] * zones_per_region, regrets=regrets
        ),
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
        # (Arun) EVCache defaults to RF=3 for tier 0 and tier 1
        default_copies = 3
        if desires.service_tier > 1:
            default_copies = 2
        copies_per_region: int = extra_model_arguments.get(
            "copies_per_region", default_copies
        )
        max_regional_size: int = extra_model_arguments.get("max_regional_size", 999)
        # Very large nodes are hard to cache warm
        max_local_disk_gib: int = extra_model_arguments.get(
            "max_local_disk_gib", 1024 * 6
        )
        # Very small nodes are hard to run memcache on
        # (Arun) We do not deploy to less than 12 GiB
        min_instance_memory_gib: int = extra_model_arguments.get(
            "min_instance_memory_gib", 12
        )

        return _estimate_evcache_cluster_zonal(
            instance=instance,
            drive=drive,
            desires=desires,
            zones_per_region=context.zones_in_region,
            copies_per_region=copies_per_region,
            max_regional_size=max_regional_size,
            max_local_disk_gib=max_local_disk_gib,
            min_instance_memory_gib=min_instance_memory_gib,
        )

    @staticmethod
    def description():
        return "Netflix Streaming EVCache (memcached) Model"

    @staticmethod
    def extra_model_arguments() -> Sequence[Tuple[str, str, str]]:
        return (
            (
                "copies_per_region",
                "int = 3",
                "How many copies of the data will exist e.g. RF=3. If unsupplied"
                " this will be deduced from tier",
            ),
            (
                "max_regional_size",
                "int = 999",
                "What is the maximum size of a cluster in this region",
            ),
            (
                "max_local_disk_gib",
                "int = 6144",
                "The maximum amount of data we store per machine",
            ),
            (
                "min_instance_memory_gib",
                "int = 12",
                "The minimum amount of instance memory to allow",
            ),
        )

    @staticmethod
    def default_desires(user_desires, extra_model_arguments: Dict[str, Any]):
        acceptable_consistency = set(
            (AccessConsistency.best_effort, AccessConsistency.never, None)
        )

        for key, value in user_desires.query_pattern.access_consistency:
            if value.target_consistency not in acceptable_consistency:
                raise ValueError(
                    f"EVCache can only provide {acceptable_consistency} access."
                    f"User asked for {key}={value}"
                )

        if user_desires.query_pattern.access_pattern == AccessPattern.latency:
            return CapacityDesires(
                query_pattern=QueryPattern(
                    access_pattern=AccessPattern.latency,
                    access_consistency=GlobalConsistency(
                        same_region=Consistency(
                            target_consistency=AccessConsistency.best_effort
                        ),
                        # By default EVCache does not globally replicate
                        cross_region=Consistency(
                            target_consistency=AccessConsistency.never
                        ),
                    ),
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
                data_shape=DataShape(
                    # (Arun): Most latency sensitive < 600GiB
                    estimated_state_size_gib=Interval(
                        low=10, mid=100, high=600, confidence=0.98
                    ),
                    # (Arun): The management sidecar takes 512 MiB
                    reserved_instance_app_mem_gib=0.5,
                    # account for the memcached connection memory
                    # and system requirements.
                    # (Arun) We currently use 1 GiB for connection memory
                    reserved_instance_system_mem_gib=(1 + 2),
                ),
            )
        else:
            return CapacityDesires(
                # (FIXME): Need to pair with memcache folks on the exact values
                query_pattern=QueryPattern(
                    access_pattern=AccessPattern.throughput,
                    access_consistency=GlobalConsistency(
                        same_region=Consistency(
                            target_consistency=AccessConsistency.best_effort
                        ),
                        # By default EVCache does not globally replicate
                        cross_region=Consistency(
                            target_consistency=AccessConsistency.never
                        ),
                    ),
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
                    # Typical ML models go up to TiB
                    estimated_state_size_gib=Interval(
                        low=10, mid=100, high=1000, confidence=0.98
                    ),
                    # (Arun): The management sidecar takes 512 MiB
                    reserved_instance_app_mem_gib=0.5,
                    # account for the memcached connection memory
                    # and system requirements.
                    # (Arun) We currently use 1 GiB base for connection memory
                    reserved_instance_system_mem_gib=(1 + 2),
                ),
            )


nflx_evcache_capacity_model = NflxEVCacheCapacityModel()
