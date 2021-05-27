import math
from typing import Any, FrozenSet
from typing import Dict
from typing import Optional
from typing import Sequence
from typing import Tuple

from service_capacity_modeling.interface import AccessPattern
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import CapacityPlan
from service_capacity_modeling.interface import CapacityRequirement
from service_capacity_modeling.interface import Clusters
from service_capacity_modeling.interface import DataShape
from service_capacity_modeling.interface import Drive
from service_capacity_modeling.interface import FixedInterval
from service_capacity_modeling.interface import Instance
from service_capacity_modeling.interface import Interval
from service_capacity_modeling.interface import QueryPattern
from service_capacity_modeling.interface import RegionClusterCapacity
from service_capacity_modeling.interface import RegionContext
from service_capacity_modeling.interface import Requirements
from service_capacity_modeling.interface import certain_float
from service_capacity_modeling.interface import certain_int
from service_capacity_modeling.models import CapacityModel
from service_capacity_modeling.models.common import compute_rds_region
from service_capacity_modeling.models.common import simple_network_mbps
from service_capacity_modeling.models.common import sqrt_staffed_cores

valid_rds_instance_types: FrozenSet[str] = frozenset([
    "m5.large", "m5.xlarge", "m5.2xlarge", "m5.4xlarge", "m5.8xlarge", "m5.12xlrage", "m5.16xlarge", "m5.24xlarge",
    "m6g.large", "m6g.xlarge", "m6g.2xlarge", "m6g.4xlarge", "m6g.8xlarge", "m6g.12xlrage", "m6g.16xlarge",
    "r5.large", "r5.xlarge", "r5.2xlarge", "r5.4xlarge", "r5.8xlarge", "r5.12xlrage", "r5.16xlarge", "r5.24xlarge",
    "r6g.large", "r6g.xlarge", "r6g.2xlarge", "r6g.4xlarge", "r6g.8xlarge", "r6g.12xlrage", "r6g.16xlarge"
])


def _estimate_rds_requirement(
        instance: Instance,
        desires: CapacityDesires,
        db_type: str
) -> CapacityRequirement:
    """Estimate the capacity required for one region given a regional desire. Unlike Cassandra RDS instances are
    deployed per region, not zone.
    The input desires should be the **regional** desire, and this function will return the regional capacity requirement
    """

    if db_type == "postgres":
        needed_cores = sqrt_staffed_cores(desires) * 1.2  # 20% head room for VACUUM
    else:
        needed_cores = sqrt_staffed_cores(desires) * 1.1  # Unscientific guess!

    needed_cores = math.ceil(
        max(1, needed_cores // (instance.cpu_ghz / desires.core_reference_ghz))
    )

    needed_network_mbps = simple_network_mbps(desires) * 1.2  # 20% head room For replication, backups etc.

    needed_disk = round(desires.data_shape.estimated_state_size_gib.mid, 2)

    if desires.data_shape.estimated_working_set_percent and desires.data_shape.estimated_working_set_percent.mid:
        working_set_percent = desires.data_shape.estimated_working_set_percent.mid
    else:
        working_set_percent = 0.10

    needed_memory = (desires.data_shape.estimated_state_size_gib.mid * working_set_percent)

    return CapacityRequirement(
        requirement_type="rds-regional",
        core_reference_ghz=desires.core_reference_ghz,
        cpu_cores=certain_int(needed_cores),
        mem_gib=certain_float(needed_memory),
        disk_gib=certain_float(needed_disk),
        network_mbps=certain_float(needed_network_mbps),
    )


# MySQL default block size is 16KiB, PostGreSQL is 8KiB
# Number of reads for B-Tree are given by log of total pages to the base of B-Tree fan out factor
def _rds_required_disk_ios(disk_size_gib: int, db_type: str, btree_fan_out: int = 100):
    disk_size_kb = disk_size_gib * 1024 * 1024
    if db_type == "postgres":
        default_block_size = 8  # KiB
    else:
        default_block_size = 16  # MySQL default block size in KiB

    pages = max(1, disk_size_kb // default_block_size)
    return math.log(pages, btree_fan_out)


def _estimate_rds_regional(
        instance: Instance,
        drive: Drive,
        desires: CapacityDesires,
        extra_model_arguments: Dict[str, Any],
) -> Optional[CapacityPlan]:
    if instance.name not in valid_rds_instance_types:
        return None

    if drive.name != "gp2":
        return None

    # RDS cannot make tier 0 service guarantees in terms partitions or availability
    if desires.service_tier == 0:
        return None

    db_type = extra_model_arguments.get("rds.engine", "mysql")
    requirement = _estimate_rds_requirement(instance, desires, db_type)
    rps = desires.query_pattern.estimated_read_per_second.mid

    cluster: Optional[RegionClusterCapacity] = compute_rds_region(
        instance=instance,
        drive=drive,
        needed_cores=int(requirement.cpu_cores.mid),
        needed_disk_gib=int(requirement.disk_gib.mid),
        needed_memory_gib=int(requirement.mem_gib.mid),
        needed_network_mbps=requirement.network_mbps.mid,
        required_disk_ios=lambda x: _rds_required_disk_ios(x, db_type) * math.ceil(0.1 * rps),
        required_disk_space=lambda x: x * 1.2,  # Unscientific random guess!
        core_reference_ghz=requirement.core_reference_ghz,
    )

    if not cluster:
        return None

    if desires.service_tier < 3:
        replicas = 2
    else:
        replicas = 1

    clusters = Clusters(
        total_annual_cost=round(cluster.annual_cost * replicas, 2),
        zonal=list(),
        regional=[cluster] * replicas,
    )

    return CapacityPlan(
        requirements=Requirements(
            zonal=list(),
            regional=[requirement] * replicas,
            regrets=("spend", "disk", "mem"),
        ),
        candidate_clusters=clusters,
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
            extra_model_arguments=extra_model_arguments
        )

    @staticmethod
    def description():
        return "Netflix RDS Cluster Model"

    @staticmethod
    def extra_model_arguments() -> Sequence[Tuple[str, str, str]]:
        return (
            (
                "rds.engine",
                "str = mysql",
                "RDS Database type",
            ),
        )

    @staticmethod
    def default_desires(user_desires, extra_model_arguments):
        return CapacityDesires(
            query_pattern=QueryPattern(
                access_pattern=AccessPattern.latency,  # can't really make latency/throughput trade-offs with RDS
                estimated_mean_read_size_bytes=Interval(
                    low=128, mid=1024, high=65536, confidence=0.90
                ),
                estimated_mean_write_size_bytes=Interval(
                    low=64, mid=512, high=2048, confidence=0.90
                ),
                # probably closer to CRDB than Cassandra. Query by PK in MySQL takes total of ~300 ms end to end
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
                    low=0.05,
                    mid=0.50,
                    high=0.70,
                    confidence=0.8
                )
            ),
        )


nflx_rds_capacity_model = NflxRDSCapacityModel()
