import logging
import math
from typing import Any
from typing import Dict
from typing import List
from typing import Sequence
from typing import Optional
from typing import Tuple

from pydantic import BaseModel
from pydantic import Field

from service_capacity_modeling.enum_utils import StrEnum
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
from service_capacity_modeling.interface import Clusters
from service_capacity_modeling.interface import Consistency
from service_capacity_modeling.interface import DataShape
from service_capacity_modeling.interface import Drive
from service_capacity_modeling.interface import FixedInterval
from service_capacity_modeling.interface import GlobalConsistency
from service_capacity_modeling.interface import Instance
from service_capacity_modeling.interface import Interval
from service_capacity_modeling.interface import QueryPattern
from service_capacity_modeling.interface import ClusterCapacity
from service_capacity_modeling.interface import RegionContext
from service_capacity_modeling.interface import Requirements
from service_capacity_modeling.interface import ServiceCapacity
from service_capacity_modeling.models import CapacityModel
from service_capacity_modeling.models import CostAwareModel
from service_capacity_modeling.models.common import buffer_for_components
from service_capacity_modeling.models.common import cluster_infra_cost
from service_capacity_modeling.models.common import compute_stateful_zone
from service_capacity_modeling.models.common import get_effective_disk_per_node_gib
from service_capacity_modeling.models.common import network_services
from service_capacity_modeling.models.common import normalize_cores
from service_capacity_modeling.models.common import RequirementFromCurrentCapacity
from service_capacity_modeling.models.common import simple_network_mbps
from service_capacity_modeling.models.common import sqrt_staffed_cores
from service_capacity_modeling.models.common import working_set_from_drive_and_slo
from service_capacity_modeling.stats import dist_for_interval

logger = logging.getLogger(__name__)


class Replication(StrEnum):
    none = "none"
    sets = "sets"
    evicts = "evicts"


def calculate_read_cpu_time_evcache_ms(read_size_bytes: float) -> float:
    # Fitted a curve based on some data that we crunched from couple of
    # read heavy clusters
    # In memory
    #  250 bit - 10 micros
    # 1520 bit - 41 micros
    # 8250 bit - 66 micros
    # On disk
    # 24   KiB - 133 micros
    # 40   KiB - 158 top of our curve
    # Fit a logistic curve, requiring it to go through first
    # point
    read_latency_ms = 979.4009 + (-0.06853492 - 979.4009) / math.pow(
        (1 + math.pow(read_size_bytes / 13061.23, 0.180864)), 0.0002819491
    )
    return max(read_latency_ms, 0.005)


def calculate_spread_cost(
    cluster_size: int, max_cost: float = 100000, min_cost: float = 0.0
) -> float:
    if cluster_size > 10:
        return min_cost
    if cluster_size < 2:
        return max_cost
    return min_cost + (max_cost - cluster_size * (max_cost - min_cost) / 30.0)


def calculate_vitals_for_capacity_planner(
    desires: CapacityDesires,
    instance: Instance,
    current_memory_gib: float,
    current_disk_gib: float,
) -> Tuple[float, float, float, float]:
    # First calculate assuming new deployment
    needed_cores = normalize_cores(
        core_count=sqrt_staffed_cores(desires),
        target_shape=instance,
        reference_shape=desires.reference_shape,
    )
    needed_network_mbps = float(simple_network_mbps(desires))
    needed_memory_gib = current_memory_gib
    needed_disk_gib = current_disk_gib

    # Check if we can apply optimizations based on current cluster capacity
    current_capacity = (
        desires.current_clusters.zonal[0]
        if desires.current_clusters and desires.current_clusters.zonal
        else None
    )
    if not current_capacity:
        return needed_cores, needed_network_mbps, needed_memory_gib, needed_disk_gib
    requirements = RequirementFromCurrentCapacity(
        current_capacity=current_capacity,
        buffers=desires.buffers,
    )
    needed_cores = normalize_cores(
        core_count=requirements.cpu(instance_candidate=instance),
        target_shape=instance,
        reference_shape=current_capacity.cluster_instance,
    )
    needed_network_mbps = float(requirements.network_mbps)
    needed_disk_gib = float(
        requirements.disk_gib if current_capacity.cluster_drive is not None else 0.0
    )
    needed_memory_gib = requirements.mem_gib
    return needed_cores, needed_network_mbps, needed_memory_gib, needed_disk_gib


def _estimate_evcache_requirement(
    instance: Instance,
    desires: CapacityDesires,
    working_set: Optional[float],
    copies_per_region: int,
) -> Tuple[CapacityRequirement, Tuple[str, ...]]:
    """Estimate the capacity required for one zone given a regional desire

    The input desires should be the **regional** desire, and this function will
    return the zonal capacity requirement
    """
    regrets: Tuple[str, ...] = ("spend", "mem")
    state_size = desires.data_shape.estimated_state_size_gib
    needed_memory = state_size.mid
    item_count = desires.data_shape.estimated_state_item_count
    payload_greater_than_classic = False
    if state_size is not None and item_count is not None and item_count.mid != 0:
        payload_size = (state_size.mid * 1024.0 * 1024.0 * 1024.0) / (item_count.mid)
        if payload_size > 200.0:
            payload_greater_than_classic = True
    else:
        if desires.query_pattern.estimated_mean_read_size_bytes.mid > 200.0:
            payload_greater_than_classic = True

    # (Arun): As of 2021 we are using ephemerals exclusively and do not
    # use cloud drives
    if working_set is None or (
        desires.data_shape.estimated_state_size_gib.mid < 110.0
        and payload_greater_than_classic
    ):
        # We can't currently store data on cloud drives, but we can put the
        # dataset into memory!
        needed_memory = float(needed_memory)
        needed_disk = 0.0
    else:
        # We can store data on fast ephems (reducing the working set that must
        # be kept in RAM)
        needed_disk = needed_memory
        needed_memory = float(working_set) * float(needed_memory)
        regrets = ("spend", "disk", "mem")

    (
        needed_cores,
        needed_network_mbps,
        needed_memory,
        needed_disk,
    ) = calculate_vitals_for_capacity_planner(
        desires, instance, needed_memory, needed_disk
    )

    # For EVCache, writes go to all zones
    # Regional reads can also go to any one zone due to app's zone affinity
    needed_cores = max(1, needed_cores)
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
            reference_shape=desires.reference_shape,
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


def _upsert_params(cluster: Any, params: Dict[str, Any]) -> None:
    if cluster.cluster_params:
        cluster.cluster_params.update(params)
    else:
        cluster.cluster_params = params


# pylint: disable=too-many-locals
def _estimate_evcache_cluster_zonal(  # noqa: C901,E501 pylint: disable=too-many-positional-arguments
    instance: Instance,
    drive: Drive,
    desires: CapacityDesires,
    context: RegionContext,
    copies_per_region: int = 3,
    max_local_data_per_node_gib: int = 2048,
    max_regional_size: int = 10000,
    min_instance_memory_gib: int = 12,
    cross_region_replication: Replication = Replication.none,
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

    if ws_drive:
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
    else:
        working_set = None

    requirement, regrets = _estimate_evcache_requirement(
        instance=instance,
        desires=desires,
        working_set=working_set,
        copies_per_region=copies_per_region,
    )

    # reject instances without ephemeral drives is the requirements need disk
    if requirement.disk_gib.mid > 0.0 and instance.drive is None:
        return None

    # Account for sidecars and base system memory
    base_mem = (
        desires.data_shape.reserved_instance_app_mem_gib
        + desires.data_shape.reserved_instance_system_mem_gib
    )

    # (Arun) We currently reserve extra memory for the OS as instances get
    # larger to account for additional overhead. Note that the
    # reserved_instance_system_mem_gib has a base of 1 GiB OSMEM so this
    # just represents the variable component
    def reserve_memory(instance_mem_gib: float) -> float:
        # (Joey) From the chart it appears to be about a 3% overhead for
        # OS memory.
        variable_os = int(instance_mem_gib * 0.03)
        return base_mem + variable_os

    requirement.context["osmem"] = reserve_memory(instance.ram_gib)
    # EVCache clusters aim to be at least 2 nodes per zone to start
    # out with for tier 0
    min_count_for_tier = 0
    if desires.service_tier < 1:
        min_count_for_tier = 2

    is_disk_io_constraint: bool = requirement.disk_gib.mid > 0.0
    adjusted_disk_io_needed = 0.0
    read_write_ratio = 0.0
    if is_disk_io_constraint:
        reads_per_sec = desires.query_pattern.estimated_read_per_second.mid
        writes_per_sec = desires.query_pattern.estimated_write_per_second.mid
        read_size = desires.query_pattern.estimated_mean_read_size_bytes.mid
        write_size = desires.query_pattern.estimated_mean_write_size_bytes.mid
        read_disk_io_needed = reads_per_sec * read_size
        write_disk_io_needed = writes_per_sec * write_size
        adjusted_disk_io_needed = read_disk_io_needed + write_disk_io_needed
        # Giving headroom for cachewarming and region squeeze
        adjusted_disk_io_needed = 1.4 * adjusted_disk_io_needed
        read_write_ratio = reads_per_sec / (reads_per_sec + writes_per_sec)

    needed_disk_gib = int(requirement.disk_gib.mid)
    disk_buffer_ratio = buffer_for_components(
        desires.buffers, [BufferComponent.disk]
    ).ratio
    max_data_per_node_gib = get_effective_disk_per_node_gib(
        instance,
        drive,
        disk_buffer_ratio,
        max_local_data_per_node_gib=max_local_data_per_node_gib,
    )
    min_count_for_data = math.ceil(needed_disk_gib / max_data_per_node_gib)
    cluster = compute_stateful_zone(
        instance=instance,
        drive=drive,
        needed_cores=int(requirement.cpu_cores.mid),
        needed_disk_gib=needed_disk_gib,
        needed_memory_gib=int(requirement.mem_gib.mid),
        needed_network_mbps=requirement.network_mbps.mid,
        min_count=max(min_count_for_data, min_count_for_tier),
        adjusted_disk_io_needed=adjusted_disk_io_needed,
        read_write_ratio=read_write_ratio,
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
    if cluster.count > (max_regional_size // copies_per_region):
        return None

    # Calculate service costs (network transfer) using the model's service_costs method
    services = NflxEVCacheCapacityModel.service_costs(
        service_type=NflxEVCacheCapacityModel.service_name,
        context=context,
        desires=desires,
        extra_model_arguments={
            "copies_per_region": copies_per_region,
            "cross_region_replication": cross_region_replication.value,
        },
    )

    cluster.cluster_type = NflxEVCacheCapacityModel.cluster_type
    zonal_clusters = [cluster] * copies_per_region

    evcache_costs = NflxEVCacheCapacityModel.cluster_costs(
        service_type=NflxEVCacheCapacityModel.service_name,
        zonal_clusters=zonal_clusters,
    )
    evcache_costs.update({s.service_type: s.annual_cost for s in services})

    clusters = Clusters(
        annual_costs=evcache_costs,
        zonal=zonal_clusters,
        regional=[],
        services=services,
    )

    return CapacityPlan(
        requirements=Requirements(
            zonal=[requirement] * copies_per_region, regrets=regrets
        ),
        candidate_clusters=clusters,
    )


class NflxEVCacheArguments(BaseModel):
    copies_per_region: int = Field(
        default=3,
        description="How many copies of the data will exist e.g. RF=3. If not supplied"
        " this will be deduced from tier",
    )
    max_regional_size: int = Field(
        default=10000,
        description="What is the maximum size of a cluster in this region",
    )
    max_local_disk_gib: int = Field(
        default=6144,
        description="The maximum amount of data we store per machine",
    )
    min_instance_memory_gib: int = Field(
        default=12,
        description="The minimum amount of instance memory to allow",
    )
    cross_region_replication: Replication = Field(
        default=Replication.none,
        description=(
            "Whether this evcache service does cross region replication. "
            "By default we do no replication"
        ),
    )


class NflxEVCacheCapacityModel(CapacityModel, CostAwareModel):
    service_name = "evcache"
    cluster_type = "evcache"

    @staticmethod
    def cluster_costs(
        service_type: str,
        zonal_clusters: Sequence[ClusterCapacity] = (),
        regional_clusters: Sequence[ClusterCapacity] = (),
    ) -> Dict[str, float]:
        # Adds "{service_type}.spread.cost" penalty for small clusters
        filtered_zonal = [
            c
            for c in zonal_clusters
            if c.cluster_type == NflxEVCacheCapacityModel.cluster_type
        ]

        costs = cluster_infra_cost(
            service_type,
            filtered_zonal,
            regional_clusters,
            cluster_type=NflxEVCacheCapacityModel.cluster_type,
        )

        # Add spread cost penalty for small clusters
        if filtered_zonal:
            cluster_count = filtered_zonal[0].count
            costs[f"{service_type}.spread.cost"] = calculate_spread_cost(cluster_count)

        return costs

    @staticmethod
    def service_costs(
        service_type: str,
        context: RegionContext,
        desires: CapacityDesires,
        extra_model_arguments: Dict[str, Any],
    ) -> List[ServiceCapacity]:
        # Network costs depend on cross_region_replication mode:
        # - 'none': No network costs (default)
        # - 'sets': Full write size replicated cross-region
        # - 'evicts': Only 128-byte keys replicated (DELETE operations)
        # Default to 'none' for composite models (like Key-Value) that compose
        # EVCache without specifying cross_region_replication
        cross_region_replication = Replication(
            extra_model_arguments.get("cross_region_replication", "none")
        )

        match cross_region_replication:
            case Replication.sets:
                copies: int = extra_model_arguments["copies_per_region"]
                return network_services(service_type, context, desires, copies)
            case Replication.evicts:
                copies = extra_model_arguments["copies_per_region"]
                # For evicts mode, only replicate 128-byte keys (DELETE operations)
                modified = desires.model_copy(deep=True)
                modified.query_pattern.estimated_mean_write_size_bytes = certain_int(
                    128
                )
                return network_services(service_type, context, modified, copies)
            case Replication.none:
                return []

    @staticmethod
    def capacity_plan(
        instance: Instance,
        drive: Drive,
        context: RegionContext,
        desires: CapacityDesires,
        extra_model_arguments: Dict[str, Any],
    ) -> Optional[CapacityPlan]:
        # (Arun) EVCache defaults to RF=3 for tier 0 and tier 1
        default_copies = context.zones_in_region
        copies_per_region: int = int(
            extra_model_arguments.get("copies_per_region", default_copies)
        )
        max_regional_size: int = int(
            extra_model_arguments.get("max_regional_size", 10000)
        )
        # Very large nodes are hard to cache warm
        max_local_data_per_node_gib: int = int(
            extra_model_arguments.get(
                "max_local_data_per_node_gib",
                extra_model_arguments.get("max_local_disk_gib", 1024 * 5),
            )
        )
        # Very small nodes are hard to run memcache on
        # (Arun) We do not deploy to less than 12 GiB
        min_instance_memory_gib: int = int(
            extra_model_arguments.get("min_instance_memory_gib", 12)
        )
        cross_region_replication = Replication(
            extra_model_arguments.get("cross_region_replication", "none")
        )

        return _estimate_evcache_cluster_zonal(
            instance=instance,
            drive=drive,
            desires=desires,
            copies_per_region=copies_per_region,
            max_regional_size=max_regional_size,
            max_local_data_per_node_gib=max_local_data_per_node_gib,
            min_instance_memory_gib=min_instance_memory_gib,
            cross_region_replication=cross_region_replication,
            context=context,
        )

    @staticmethod
    def description() -> str:
        return "Netflix Streaming EVCache (memcached) Model"

    @staticmethod
    def extra_model_arguments_schema() -> Dict[str, Any]:
        return NflxEVCacheArguments.model_json_schema()

    @staticmethod
    def default_desires(
        user_desires: CapacityDesires, extra_model_arguments: Dict[str, Any]
    ) -> CapacityDesires:
        acceptable_consistency = {
            AccessConsistency.best_effort,
            AccessConsistency.never,
            None,
        }

        for key, value in user_desires.query_pattern.access_consistency:
            if value.target_consistency not in acceptable_consistency:
                raise ValueError(
                    f"EVCache can only provide {acceptable_consistency} access."
                    f"User asked for {key}={value}"
                )

        estimated_read_size: Interval = Interval(
            **user_desires.query_pattern.model_dump().get(
                "estimated_mean_read_size_bytes",
                user_desires.query_pattern.model_dump().get(
                    "estimated_mean_write_size_bytes",
                    {"low": 16, "mid": 1024, "high": 65536, "confidence": 0.95},
                ),
            )
        )
        estimated_read_latency_ms: Interval = Interval(
            low=calculate_read_cpu_time_evcache_ms(estimated_read_size.low),
            mid=calculate_read_cpu_time_evcache_ms(estimated_read_size.mid),
            high=calculate_read_cpu_time_evcache_ms(estimated_read_size.high),
            confidence=estimated_read_size.confidence,
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
                    estimated_mean_read_size_bytes=estimated_read_size,
                    estimated_mean_write_size_bytes=Interval(
                        low=64, mid=512, high=1024, confidence=0.95
                    ),
                    # evcache read latency is sensitive to payload size
                    # so this is computed above
                    estimated_mean_read_latency_ms=estimated_read_latency_ms,
                    # evcache bulk puts usually take slightly longer
                    estimated_mean_write_latency_ms=Interval(
                        low=0.01, mid=0.01, high=0.01, confidence=0.98
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
                    reserved_instance_app_mem_gib=1,
                    # account for the memcached connection memory
                    # and system requirements.
                    # (Arun) We currently use 1 GiB for connection memory
                    reserved_instance_system_mem_gib=(1 + 2),
                ),
                buffers=Buffers(
                    default=Buffer(ratio=1.5),
                    desired={
                        "cpu": Buffer(
                            ratio=1.5, components=[BufferComponent.cpu]
                        ),  # ~70%
                        "storage": Buffer(
                            ratio=1.25, components=[BufferComponent.storage]
                        ),  # ~80%
                        "memory": Buffer(
                            ratio=1.25, components=[BufferComponent.memory]
                        ),  # 80%
                        "network": Buffer(
                            ratio=1.5, components=[BufferComponent.network]
                        ),  # ~70%
                    },
                ),
            )
        else:
            return CapacityDesires(
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
                    estimated_mean_read_size_bytes=estimated_read_size,
                    estimated_mean_write_size_bytes=Interval(
                        low=128, mid=1024, high=65536, confidence=0.95
                    ),
                    # evcache read latency is sensitive to payload size
                    # so this is computed above
                    estimated_mean_read_latency_ms=estimated_read_latency_ms,
                    # evcache bulk puts usually take slightly longer
                    estimated_mean_write_latency_ms=Interval(
                        low=0.01, mid=0.01, high=0.01, confidence=0.98
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
                    reserved_instance_app_mem_gib=1,
                    # account for the memcached connection memory
                    # and system requirements.
                    # (Arun) We currently use 1 GiB base for connection memory
                    reserved_instance_system_mem_gib=(1 + 2),
                ),
                buffers=Buffers(
                    default=Buffer(ratio=1.5),
                    desired={
                        "cpu": Buffer(
                            ratio=1.5, components=[BufferComponent.cpu]
                        ),  # ~70%
                        "storage": Buffer(
                            ratio=1.25, components=[BufferComponent.storage]
                        ),  # ~80%
                        "memory": Buffer(
                            ratio=1.2, components=[BufferComponent.memory]
                        ),  # 80%
                        "network": Buffer(
                            ratio=1.5, components=[BufferComponent.network]
                        ),  # ~70%
                    },
                ),
            )


nflx_evcache_capacity_model = NflxEVCacheCapacityModel()
