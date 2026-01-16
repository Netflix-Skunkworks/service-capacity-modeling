import logging
import math
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple

from pydantic import BaseModel
from pydantic import Field

from service_capacity_modeling.interface import AccessConsistency
from service_capacity_modeling.interface import AccessPattern
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import CapacityPlan
from service_capacity_modeling.interface import CapacityRequirement
from service_capacity_modeling.interface import certain_float
from service_capacity_modeling.interface import certain_int
from service_capacity_modeling.interface import ClusterCapacity
from service_capacity_modeling.interface import Clusters
from service_capacity_modeling.interface import Consistency
from service_capacity_modeling.interface import DataShape
from service_capacity_modeling.interface import Drive
from service_capacity_modeling.interface import FixedInterval
from service_capacity_modeling.interface import GlobalConsistency
from service_capacity_modeling.interface import Instance
from service_capacity_modeling.interface import Interval
from service_capacity_modeling.interface import Platform
from service_capacity_modeling.interface import QueryPattern
from service_capacity_modeling.interface import RegionClusterCapacity
from service_capacity_modeling.interface import RegionContext
from service_capacity_modeling.interface import Requirements
from service_capacity_modeling.interface import ServiceCapacity
from service_capacity_modeling.models import CapacityModel
from service_capacity_modeling.models.common import normalize_cores
from service_capacity_modeling.models.common import simple_network_mbps
from service_capacity_modeling.models.common import sqrt_staffed_cores

logger = logging.getLogger(__name__)


def _estimate_aurora_requirement(
    instance: Instance, desires: CapacityDesires, db_type: str
) -> CapacityRequirement:
    """Estimate the capacity required for one region given a regional desire. Unlike
    Cassandra aurora instances are deployed per region, not zone. The input desires
    should be the **regional** desire, and this function will return the regional
    capacity requirement
    Todo: We'll start with what we have for RDS, given there shouldn't be too
    much difference in engine itself
    """

    if db_type == "postgres":
        needed_cores = sqrt_staffed_cores(desires) * 1.2  # 20% head room for VACUUM
    else:
        needed_cores = sqrt_staffed_cores(desires) * 1.1  # Just a guess!

    needed_cores = normalize_cores(
        core_count=needed_cores,
        target_shape=instance,
        reference_shape=desires.reference_shape,
    )

    # 20% head room For replication, backups etc.
    needed_network_mbps = simple_network_mbps(desires) * 1.2
    needed_disk = round(desires.data_shape.estimated_state_size_gib.mid, 2)

    if (
        desires.data_shape.estimated_working_set_percent
        and desires.data_shape.estimated_working_set_percent.mid
    ):
        working_set_percent = desires.data_shape.estimated_working_set_percent.mid
    else:
        working_set_percent = 0.10

    needed_memory = (
        desires.data_shape.estimated_state_size_gib.mid * working_set_percent
    )

    return CapacityRequirement(
        requirement_type="aurora-regional",
        reference_shape=desires.reference_shape,
        cpu_cores=certain_int(needed_cores),
        mem_gib=certain_float(needed_memory),
        disk_gib=certain_float(needed_disk),
        network_mbps=certain_float(needed_network_mbps),
    )


# MySQL default block size is 16KiB, PostGreSQL is 8KiB Number of reads for B-Tree
# are given by log of total pages to the base of B-Tree fan out factor
def _rds_required_disk_ios(
    disk_size_gib: int, db_type: str, btree_fan_out: int = 100
) -> float:
    disk_size_kb = disk_size_gib * 1024 * 1024
    if db_type == "postgres":
        default_block_size = 8  # KiB
    else:
        default_block_size = 16  # MySQL default block size in KiB

    pages = max(1, disk_size_kb // default_block_size)
    return math.log(pages, btree_fan_out)


# This is a start, we should iterate based on the actual work load
def _estimate_io_cost(
    db_type: str,
    desires: CapacityDesires,
    read_io_price: float,
    write_io_price: float,
    cache_hit_rate: float = 0.8,
) -> float:
    if db_type == "postgres":
        read_byte_per_io = 8192
    else:
        read_byte_per_io = 16384

    write_byte_per_io = 4096

    r_io = desires.query_pattern.estimated_read_per_second.mid * math.ceil(
        desires.query_pattern.estimated_mean_read_size_bytes.mid / read_byte_per_io
    )
    # Assuming write can be batched
    w_io = (
        desires.query_pattern.estimated_write_per_second.mid
        * desires.query_pattern.estimated_mean_write_size_bytes.mid
        / write_byte_per_io
    )

    r_cost = r_io * (1 - cache_hit_rate) * read_io_price
    w_cost = w_io * write_io_price
    return r_cost + w_cost


def _compute_aurora_region(  # pylint: disable=too-many-positional-arguments
    instance: Instance,
    drive: Drive,  # always to be Aurora Storage
    needed_cores: int,
    needed_disk_gib: int,
    needed_memory_gib: int,
    needed_network_mbps: float,
    required_disk_ios: Callable[[int], float],
    required_disk_space: Callable[[int], float],
    desires: CapacityDesires,
) -> Optional[RegionClusterCapacity]:
    """Computes a regional cluster of a Aurora service

    Basically just verifies that a single instance of a passed in instance type can
    support required cpu, memory and network since we can't scale Aurora horizontally by
    adding more instances like Cassandra. Count of instance is always 1
    """

    # TODO: This probably needs to be used ...
    _ = required_disk_ios

    # We can't scale Aurora horizontally by adding more nodes like we can for
    # Cassandra
    # so single instance must meet the whole cpu, memory and network bandwidth
    # requirement
    # Disk Scaling will be handled differently
    if (
        instance.cpu < needed_cores
        or instance.ram_gib < needed_memory_gib
        or instance.net_mbps < needed_network_mbps
    ):
        return None

    # calculate storage cost
    attached_drives = []
    attached_drive = drive.model_copy()
    attached_drive.size_gib = math.ceil(
        max(1, required_disk_space(needed_disk_gib))
    )  # todo: Figure out the IO vs disk
    attached_drives.append(attached_drive)

    # IO cost is now calculated separately via service_costs() method
    # to ensure consistent cost calculation between recommendations
    # and baseline extraction
    cluster_annual_cost = instance.annual_cost + attached_drive.annual_cost

    logger.debug(
        "For (cpu, memory_gib, disk_gib) = (%s, %s, %s) need ( %s, %s, %s)",
        needed_cores,
        needed_memory_gib,
        needed_disk_gib,
        instance.name,
        attached_drives,
        cluster_annual_cost,
    )

    # TODO (ramsrivatsak): In future we need to leverage read traffic and model the
    # number of reader instances based on the number of read IOPS.
    # We add a reader instance if we are deploying a tier 0 and tier 1 service.
    # Writer instance + Reader instance = 2. For other service tiers the writer instance
    # is enough.
    instance_count = 2 if desires.service_tier <= 1 else 1

    return RegionClusterCapacity(
        cluster_type="aurora-cluster",
        count=instance_count,
        instance=instance,
        attached_drives=attached_drives,
        annual_cost=cluster_annual_cost,
        cluster_params={"instance_cost": instance.annual_cost},
    )


def _estimate_aurora_regional(
    instance: Instance,
    drive: Drive,
    context: RegionContext,
    desires: CapacityDesires,
    extra_model_arguments: Dict[str, Any],
) -> Optional[CapacityPlan]:
    db_type = extra_model_arguments.get("aurora.engine", "postgres")
    if db_type == "postgres":
        if Platform.aurora_postgres not in instance.platforms:
            return None
    else:
        if Platform.aurora_mysql not in instance.platforms:
            return None

    if drive.name != "aurora":
        return None

    # Aurora cannot make tier 0 service guarantees in terms partitions or availability
    if desires.service_tier == 0:
        return None

    requirement = _estimate_aurora_requirement(instance, desires, db_type)
    rps = desires.query_pattern.estimated_read_per_second.mid

    cluster: Optional[RegionClusterCapacity] = _compute_aurora_region(
        instance=instance,
        drive=drive,
        needed_cores=int(requirement.cpu_cores.mid),
        needed_disk_gib=int(requirement.disk_gib.mid),
        needed_memory_gib=int(requirement.mem_gib.mid),
        needed_network_mbps=requirement.network_mbps.mid,
        required_disk_ios=lambda size_gib: _rds_required_disk_ios(size_gib, db_type)
        * math.ceil(0.1 * rps),
        required_disk_space=lambda x: x * 1.2,  # Unscientific random guess!
        desires=desires,
    )

    if not cluster:
        return None

    # Replicas: explicit parameter overrides service_tier-based default
    # This allows users to specify exact topology for baseline extraction
    explicit_replicas = extra_model_arguments.get("aurora.replicas")
    if explicit_replicas is not None:
        replicas = int(explicit_replicas)
    elif desires.service_tier < 3:
        replicas = 2  # Writer + 1 read replica for tier 0-2
    else:
        replicas = 1  # Writer only for tier 3+

    # Store replicas in cluster_params for cluster_costs() to use
    cluster.cluster_params["aurora.replicas"] = replicas

    regional_clusters = [cluster]

    # Calculate infrastructure costs (instance + storage)
    aurora_costs = NflxAuroraCapacityModel.cluster_costs(
        service_type="aurora-cluster", regional_clusters=regional_clusters
    )

    # Store drive IO prices in extra_model_arguments for service_costs() to use
    extra_args_with_io = extra_model_arguments.copy()
    extra_args_with_io["aurora.read_io_price"] = drive.annual_cost_per_read_io[0][1]
    extra_args_with_io["aurora.write_io_price"] = drive.annual_cost_per_write_io[0][1]

    # Calculate service costs (IO operations)
    services = NflxAuroraCapacityModel.service_costs(
        service_type="aurora-cluster",
        context=context,
        desires=desires,
        requirement=requirement,
        extra_model_arguments=extra_args_with_io,
    )

    # Merge service costs into annual_costs
    for service in services:
        aurora_costs[service.service_type] = service.annual_cost

    clusters = Clusters(
        annual_costs=aurora_costs,
        zonal=[],
        regional=regional_clusters,
        services=services,
    )

    return CapacityPlan(
        requirements=Requirements(
            zonal=[],
            regional=[requirement] * replicas,
            regrets=("spend", "disk", "mem"),
        ),
        candidate_clusters=clusters,
    )


class NflxAuroraArguments(BaseModel):
    aurora_engine: str = Field(
        alias="aurora.engine",
        default="mysql",
        description="Aurora Database type",
    )
    aurora_replicas: Optional[int] = Field(
        alias="aurora.replicas",
        default=None,
        description=(
            "Number of Aurora replicas (writer + read replicas). "
            "If not specified, defaults to 2 for tier 0-2 (writer + 1 read replica) "
            "or 1 for tier 3+ (writer only). For baseline extraction, pass the "
            "actual replica count from your deployment."
        ),
    )


class NflxAuroraCapacityModel(CapacityModel):
    @staticmethod
    def capacity_plan(
        instance: Instance,
        drive: Drive,
        context: RegionContext,
        desires: CapacityDesires,
        extra_model_arguments: Dict[str, Any],
    ) -> Optional[CapacityPlan]:
        return _estimate_aurora_regional(
            instance=instance,
            drive=drive,
            context=context,
            desires=desires,
            extra_model_arguments=extra_model_arguments,
        )

    @staticmethod
    def service_costs(
        service_type: str,
        context: RegionContext,
        desires: CapacityDesires,
        requirement: CapacityRequirement,
        extra_model_arguments: Dict[str, Any],
    ) -> List[ServiceCapacity]:
        """Calculate Aurora-specific service costs (IO operations).

        Aurora IO costs are calculated based on read/write query patterns
        and the aurora drive's IO pricing. This method is called both
        during recommendation (from capacity_plan) and baseline extraction
        (from extract_baseline_plan) to ensure cost parity.

        IO prices must be passed via extra_model_arguments:
        - aurora.read_io_price: Cost per read IO
        - aurora.write_io_price: Cost per write IO
        """
        db_type = extra_model_arguments.get("aurora.engine", "postgres")

        read_io_price = extra_model_arguments.get("aurora.read_io_price")
        write_io_price = extra_model_arguments.get("aurora.write_io_price")

        # If we don't have IO prices, return empty (no IO costs)
        if read_io_price is None or write_io_price is None:
            return []

        io_cost = _estimate_io_cost(
            db_type,
            desires,
            read_io_price,
            write_io_price,
        )

        if io_cost > 0:
            return [
                ServiceCapacity(
                    service_type=f"{service_type}.io",
                    annual_cost=io_cost,
                )
            ]
        return []

    @staticmethod
    def cluster_costs(
        service_type: str,
        zonal_clusters: Sequence[ClusterCapacity] = (),
        regional_clusters: Sequence[ClusterCapacity] = (),
    ) -> Dict[str, float]:
        """Calculate Aurora cluster infrastructure costs.

        Aurora has a writer node (annual_cost) plus read replicas
        (instance_cost per replica). Both values are stored in cluster_params.
        """
        if not regional_clusters:
            return {}

        total_cost = 0.0
        for cluster in regional_clusters:
            replicas: int = cluster.cluster_params["aurora.replicas"]
            instance_cost: float = cluster.cluster_params["instance_cost"]
            # Writer cost + read replica costs
            total_cost += cluster.annual_cost + (replicas - 1) * instance_cost

        return {f"{service_type}.regional-clusters": total_cost}

    @staticmethod
    def description() -> str:
        return "Netflix Aurora Cluster Model"

    @staticmethod
    def extra_model_arguments_schema() -> Dict[str, Any]:
        return NflxAuroraArguments.model_json_schema()

    @staticmethod
    def allowed_platforms() -> Tuple[Platform, ...]:
        return Platform.aurora_mysql, Platform.aurora_mysql

    @staticmethod
    def default_desires(
        user_desires: CapacityDesires, extra_model_arguments: Dict[str, Any]
    ) -> CapacityDesires:
        return CapacityDesires(
            query_pattern=QueryPattern(
                access_pattern=AccessPattern.latency,
                access_consistency=GlobalConsistency(
                    same_region=Consistency(
                        target_consistency=AccessConsistency.serializable_stale,
                    ),
                    cross_region=Consistency(
                        target_consistency=AccessConsistency.never,
                    ),
                ),
                # can't really make latency/throughput trade-offs with RDS
                estimated_mean_read_size_bytes=Interval(
                    low=128, mid=1024, high=65536, confidence=0.90
                ),
                estimated_mean_write_size_bytes=Interval(
                    low=64, mid=512, high=2048, confidence=0.90
                ),
                # probably closer to CRDB than Cassandra. Query by PK in MySQL
                # theoretically takes total of ~300 us end to end, but
                # PostgreSQL is usually slower ...
                estimated_mean_read_latency_ms=Interval(
                    low=1, mid=3.5, high=100, confidence=0.90
                ),
                estimated_mean_write_latency_ms=Interval(
                    low=1, mid=5, high=200, confidence=0.90
                ),
                # Single row fetch by PK in MySQL takes total of ~300 ms end to end
                read_latency_slo_ms=FixedInterval(
                    minimum_value=10,
                    maximum_value=2000,
                    low=300,
                    mid=800,
                    high=2000,
                    confidence=0.90,
                ),
                write_latency_slo_ms=FixedInterval(
                    minimum_value=10,
                    maximum_value=2000,
                    low=300,
                    mid=800,
                    high=2000,
                    confidence=0.90,
                ),
            ),
            # Assume that the working set is between 20% by default
            data_shape=DataShape(
                estimated_working_set_percent=Interval(
                    low=0.05, mid=0.10, high=0.20, confidence=0.8
                )
            ),
        )


nflx_aurora_capacity_model = NflxAuroraCapacityModel()
