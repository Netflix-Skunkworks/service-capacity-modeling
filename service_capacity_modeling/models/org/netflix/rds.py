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
from service_capacity_modeling.interface import RegionClusterCapacity
from service_capacity_modeling.interface import RegionContext
from service_capacity_modeling.interface import Requirements
from service_capacity_modeling.models import CapacityModel
from service_capacity_modeling.models.common import gp2_gib_for_io
from service_capacity_modeling.models.common import normalize_cores
from service_capacity_modeling.models.common import simple_network_mbps
from service_capacity_modeling.models.common import sqrt_staffed_cores

logger = logging.getLogger(__name__)


def _estimate_rds_requirement(
    instance: Instance, desires: CapacityDesires, db_type: str
) -> CapacityRequirement:
    """Estimate the capacity required for one region given a regional desire. Unlike
    Cassandra RDS instances are deployed per region, not zone. The input desires
    should be the **regional** desire, and this function will return the regional
    capacity requirement
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
        requirement_type="rds-regional",
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


def _compute_rds_region(  # pylint: disable=too-many-positional-arguments
    instance: Instance,
    drive: Drive,
    needed_cores: int,
    needed_disk_gib: int,
    needed_memory_gib: int,
    needed_network_mbps: float,
    required_disk_ios: Callable[[int], float],
    required_disk_space: Callable[[int], int],
    reference_shape: Instance,
) -> Optional[RegionClusterCapacity]:
    """Computes a regional cluster of a RDS service

    Basically just verifies that a single instance of a passed in instance type can
    support required cpu, memory and network since we can't scale RDS horizontally by
    adding more instances like Cassandra. Count of instance is always 1
    """

    needed_cores = normalize_cores(
        core_count=needed_cores, target_shape=instance, reference_shape=reference_shape
    )

    # We can't scale RDS horizontally by adding more nodes like we can for Cassandra
    # so single instance must meet the whole cpu, disk, memory and network bandwidth
    # requirement
    if (
        instance.cpu < needed_cores
        or instance.ram_gib < needed_memory_gib
        or instance.net_mbps < needed_network_mbps
    ):
        return None

    # calculate storage cost
    attached_drives = []
    space_gib = max(1, required_disk_space(needed_disk_gib))
    io_gib = gp2_gib_for_io(required_disk_ios(needed_disk_gib))
    rds_gib = max(io_gib, space_gib)
    attached_drive = drive.model_copy()
    attached_drive.size_gib = rds_gib
    attached_drives.append(attached_drive)
    total_annual_cost = instance.annual_cost + attached_drive.annual_cost

    logger.debug(
        "For (cpu, memory_gib, disk_gib) = (%s, %s, %s) need ( %s, %s, %s)",
        needed_cores,
        needed_memory_gib,
        needed_disk_gib,
        instance.name,
        attached_drives,
        total_annual_cost,
    )

    return RegionClusterCapacity(
        cluster_type="rds-cluster",
        count=1,
        instance=instance,
        attached_drives=attached_drives,
        annual_cost=total_annual_cost,
    )


def _estimate_rds_regional(
    instance: Instance,
    drive: Drive,
    desires: CapacityDesires,
    extra_model_arguments: Dict[str, Any],
) -> Optional[CapacityPlan]:
    instance_family = instance.family
    if instance_family not in ("m5", "r5"):
        return None

    if drive.name != "gp2":
        return None

    # RDS cannot make tier 0 service guarantees in terms partitions or availability
    if desires.service_tier == 0:
        return None

    db_type = extra_model_arguments.get("rds.engine", "mysql")
    requirement = _estimate_rds_requirement(instance, desires, db_type)
    rps = desires.query_pattern.estimated_read_per_second.mid

    cluster: Optional[RegionClusterCapacity] = _compute_rds_region(
        instance=instance,
        drive=drive,
        needed_cores=int(requirement.cpu_cores.mid),
        needed_disk_gib=int(requirement.disk_gib.mid),
        needed_memory_gib=int(requirement.mem_gib.mid),
        needed_network_mbps=requirement.network_mbps.mid,
        required_disk_ios=lambda size_gib: _rds_required_disk_ios(size_gib, db_type)
        * math.ceil(0.1 * rps),
        required_disk_space=lambda x: math.ceil(x * 1.2),  # Unscientific random guess!
        reference_shape=desires.reference_shape,
    )

    if not cluster:
        return None

    if desires.service_tier < 3:
        replicas = 2
    else:
        replicas = 1

    costs = {"rds-cluster.regional-clusters": cluster.annual_cost * replicas}
    clusters = Clusters(
        annual_costs=costs,
        zonal=[],
        regional=[cluster],
    )

    return CapacityPlan(
        requirements=Requirements(
            zonal=[],
            regional=[requirement] * replicas,
            regrets=("spend", "disk", "mem"),
        ),
        candidate_clusters=clusters,
    )


class NflxRDSArguments(BaseModel):
    rds_engine: str = Field(
        alias="rds.engine",
        default="mysql",
        description="RDS Database type",
    )


class NflxRDSCapacityModel(CapacityModel):
    @staticmethod
    def capacity_plan(
        instance: Instance,
        drive: Drive,
        context: RegionContext,
        desires: CapacityDesires,
        extra_model_arguments: Dict[str, Any],
    ) -> Optional[CapacityPlan]:
        return _estimate_rds_regional(
            instance=instance,
            drive=drive,
            desires=desires,
            extra_model_arguments=extra_model_arguments,
        )

    @staticmethod
    def description() -> str:
        return "Netflix RDS Cluster Model"

    @staticmethod
    def extra_model_arguments_schema() -> Dict[str, Any]:
        return NflxRDSArguments.model_json_schema()

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
                # probably closer to CRDB than Cassandra. Query by PK in MySQL takes
                # total of ~300 ms end to end
                estimated_mean_read_latency_ms=Interval(
                    low=5, mid=30, high=150, confidence=0.90
                ),
                estimated_mean_write_latency_ms=Interval(
                    low=5, mid=30, high=150, confidence=0.90
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
            data_shape=DataShape(
                estimated_working_set_percent=Interval(
                    low=0.05, mid=0.50, high=0.70, confidence=0.8
                )
            ),
        )


nflx_rds_capacity_model = NflxRDSCapacityModel()
