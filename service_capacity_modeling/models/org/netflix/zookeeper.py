from typing import Any
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
from service_capacity_modeling.interface import RegionContext
from service_capacity_modeling.interface import Requirements
from service_capacity_modeling.interface import ZoneClusterCapacity
from service_capacity_modeling.models import CapacityModel
from service_capacity_modeling.models.common import simple_network_mbps
from service_capacity_modeling.models.common import sqrt_staffed_cores


class NflxZookeeperCapacityModel(CapacityModel):
    @staticmethod
    def capacity_plan(
        instance: Instance,
        drive: Drive,
        context: RegionContext,
        desires: CapacityDesires,
        extra_model_arguments: Dict[str, Any],
    ) -> Optional[CapacityPlan]:

        # We only deploy Zookeeper to fast ephemeral storage
        # Due to fsync latency to the disk.
        if instance.drive is None:
            return None

        # We only deploy Zookeeper to 3 zone regions at this time
        if context.zones_in_region != 3:
            return None

        # Only ZK team members can create tier zero ZK clusters since
        # we generally don't want them
        approving_zk_member = extra_model_arguments.get("zk_approver", None)
        if desires.service_tier == 0 and approving_zk_member is None:
            return None

        # Zookeeper can only really scale vertically, so let's determine if
        # this instance type meets our memory and CPU requirements and if
        # it does we make either a 3 node or 5 node cluster based on tier

        needed_cores = sqrt_staffed_cores(desires)
        needed_network_mbps = simple_network_mbps(desires) * 2
        # ZK stores all data on heap, say with ~25% overhead
        needed_memory = desires.data_shape.estimated_state_size_gib.mid * 1.25
        # To take into account snapshots, we might want to make this larger
        needed_disk = needed_memory * 2

        if (
            instance.cpu < needed_cores
            or instance.ram_gib < needed_memory
            or instance.drive.size_gib < needed_disk
        ):
            return None

        # We have a viable instance, now either make 3 or 5 depending on tier

        def soln(n) -> ZoneClusterCapacity:
            return ZoneClusterCapacity(
                cluster_type="stateful-cluster",
                count=n,
                instance=instance,
                annual_cost=(n * instance.annual_cost),
            )

        req = CapacityRequirement(
            requirement_type="zk-zonal",
            core_reference_ghz=desires.core_reference_ghz,
            cpu_cores=certain_int(needed_cores),
            mem_gib=certain_float(needed_memory),
            disk_gib=certain_float(needed_disk),
            network_mbps=certain_float(needed_network_mbps),
        )

        if desires.service_tier == 0:
            requirements = [req] * context.zones_in_region
            zonal = [soln(2), soln(2), soln(1)]
        else:
            requirements = [req] * context.zones_in_region
            zonal = [soln(1), soln(1), soln(1)]

        clusters = Clusters(
            total_annual_cost=round(sum(z.annual_cost for z in zonal), 2),
            zonal=zonal,
            regional=list(),
            services=list(),
        )

        return CapacityPlan(
            requirements=Requirements(zonal=requirements), candidate_clusters=clusters
        )

    @staticmethod
    def description():
        return "Netflix Zookeeper Coordination App Model"

    @staticmethod
    def extra_model_arguments() -> Sequence[Tuple[str, str, str]]:
        return (
            (
                "zk_approver",
                "str = None",
                "Zookeeper teammate who approved this cluster",
            ),
        )

    @staticmethod
    def default_desires(user_desires, extra_model_arguments):
        return CapacityDesires(
            query_pattern=QueryPattern(
                access_pattern=AccessPattern.latency,
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
                # Assume 4 GiB heaps
                reserved_instance_app_mem_gib=4
            ),
        )


nflx_zookeeper_capacity_model = NflxZookeeperCapacityModel()
