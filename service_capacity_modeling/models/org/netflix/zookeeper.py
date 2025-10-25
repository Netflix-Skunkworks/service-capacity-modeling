from typing import Any
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
from service_capacity_modeling.interface import ZoneClusterCapacity
from service_capacity_modeling.models import CapacityModel
from service_capacity_modeling.models.common import simple_network_mbps
from service_capacity_modeling.models.common import sqrt_staffed_cores


def _zk_requirement(
    instance: Instance,
    desires: CapacityDesires,
    heap_overhead: float,
    disk_overhead: float,
) -> Optional[CapacityRequirement]:
    # We only deploy Zookeeper to fast ephemeral storage
    # Due to fsync latency to the disk.
    if instance.drive is None:
        return None

    # Zookeeper can only really scale vertically, so let's determine if
    # this instance type meets our memory and CPU requirements and if
    # it does we make either a 3 node or 5 node cluster based on tier

    needed_cores = sqrt_staffed_cores(desires)
    needed_network_mbps = simple_network_mbps(desires) * 2
    # ZK stores all data on heap, say with ~25% overhead
    needed_memory = (
        desires.data_shape.estimated_state_size_gib.mid * heap_overhead
        + desires.data_shape.reserved_instance_app_mem_gib
        + desires.data_shape.reserved_instance_system_mem_gib
    )
    # To take into account snapshots, we might want to make this larger
    needed_disk = needed_memory * disk_overhead

    if (
        instance.cpu < needed_cores
        or instance.ram_gib < needed_memory
        or instance.drive.size_gib < needed_disk
        or instance.net_mbps < needed_network_mbps
    ):
        return None

    return CapacityRequirement(
        requirement_type="zk-zonal",
        reference_shape=desires.reference_shape,
        cpu_cores=certain_int(needed_cores),
        mem_gib=certain_float(needed_memory),
        disk_gib=certain_float(needed_disk),
        network_mbps=certain_float(needed_network_mbps),
    )


class NflxZookeeperArguments(BaseModel):
    heap_overhead: float = Field(
        default=1.25,
        description="Amount of heap overhead per byte stored",
    )
    snapshot_overhead: float = Field(
        default=4,
        description="Amount of disk overhead to keep for snapshots",
    )


class NflxZookeeperCapacityModel(CapacityModel):
    @staticmethod
    def capacity_plan(
        instance: Instance,
        drive: Drive,
        context: RegionContext,
        desires: CapacityDesires,
        extra_model_arguments: Dict[str, Any],
    ) -> Optional[CapacityPlan]:
        # We only deploy Zookeeper to 3 zone regions at this time
        if context.zones_in_region != 3:
            return None

        heap_overhead = extra_model_arguments.get("heap_overhead", 1.25)
        disk_overhead = extra_model_arguments.get("snapshot_overhead", 4)
        req = _zk_requirement(instance, desires, heap_overhead, disk_overhead)

        # This instance doesn't meet the requirement
        if req is None:
            return None

        # We have a viable instance, now either make 3 or 5 depending on tier
        def soln(n: int) -> ZoneClusterCapacity:
            return ZoneClusterCapacity(
                cluster_type="zk-zonal",
                count=n,
                instance=instance,
                annual_cost=(n * instance.annual_cost),
            )

        if desires.service_tier == 0:
            requirements = [req] * context.zones_in_region
            zonal = [soln(2), soln(2), soln(1)]
        else:
            requirements = [req] * context.zones_in_region
            zonal = [soln(1), soln(1), soln(1)]

        clusters = Clusters(
            annual_costs={
                "zk-zonal.zonal-clusters": (round(sum(z.annual_cost for z in zonal), 2))
            },
            zonal=zonal,
            regional=[],
            services=[],
        )

        return CapacityPlan(
            requirements=Requirements(
                zonal=requirements,
                # Zookeeper does not want to run out of disk or memory
                regrets=("spend", "disk", "mem"),
            ),
            candidate_clusters=clusters,
        )

    @staticmethod
    def description() -> str:
        return "Netflix Zookeeper Coordination Cluster Model"

    @staticmethod
    def extra_model_arguments_schema() -> Dict[str, Any]:
        return NflxZookeeperArguments.model_json_schema()

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
                estimated_mean_read_size_bytes=Interval(
                    low=128, mid=128, high=1024, confidence=0.95
                ),
                estimated_mean_write_size_bytes=Interval(
                    low=64, mid=128, high=1024, confidence=0.95
                ),
                estimated_mean_read_latency_ms=Interval(
                    low=0.2, mid=1, high=2, confidence=0.98
                ),
                estimated_mean_write_latency_ms=Interval(
                    low=0.2, mid=1, high=2, confidence=0.98
                ),
                # "Single digit milliseconds SLO"
                read_latency_slo_ms=FixedInterval(
                    minimum_value=0.5,
                    maximum_value=10,
                    low=1,
                    mid=2,
                    high=5,
                    confidence=0.98,
                ),
                write_latency_slo_ms=FixedInterval(
                    low=1, mid=2, high=5, confidence=0.98
                ),
            ),
            data_shape=DataShape(
                # We autosize our heap to the dataset, so don't account for
                # that here.
                reserved_instance_app_mem_gib=0,
                # Technically the number of connections to ZK matters for
                # this, but let's just say it's 1 GiB and be ok with that
                # (kernel, pipes, bolt, etc ...)
                reserved_instance_system_mem_gib=1,
            ),
        )


nflx_zookeeper_capacity_model = NflxZookeeperCapacityModel()
