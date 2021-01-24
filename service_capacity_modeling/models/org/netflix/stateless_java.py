import math
from typing import Dict
from typing import Optional
from typing import Sequence
from typing import Tuple

from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import CapacityPlan
from service_capacity_modeling.interface import CapacityRequirement
from service_capacity_modeling.interface import certain_float
from service_capacity_modeling.interface import certain_int
from service_capacity_modeling.interface import Clusters
from service_capacity_modeling.interface import Drive
from service_capacity_modeling.interface import Instance
from service_capacity_modeling.interface import RegionClusterCapacity
from service_capacity_modeling.interface import Service
from service_capacity_modeling.models import CapacityModel
from service_capacity_modeling.models.common import compute_stateless_region
from service_capacity_modeling.models.common import simple_network_mbps
from service_capacity_modeling.models.common import sqrt_staffed_cores


def _estimate_java_app_requirement(
    desires: CapacityDesires,
    failover: bool = True,
    jvm_memory_overhead: float = 1.2,
) -> CapacityRequirement:
    needed_cores = sqrt_staffed_cores(desires)
    needed_network_mbps = simple_network_mbps(desires)

    if failover:
        # For failover provision at 40% utilization
        needed_cores = int(math.ceil(needed_cores * (1 / 0.4)))
        needed_network_mbps = int(math.ceil(needed_network_mbps * (1 / 0.4)))

    # Assume a Java application that can allocate about 1 GiB/s to heap
    # per 2 GiB of memory and assume that we have a ~2x memory overhead
    # due to the JVM overhead.
    # TODO: we should probably have the memory bandwidth attached to
    # the instance type, e.g. Intel CPUs and AMD CPUs have different
    # per core memory bandwidth.
    mem_allocation_mbps = needed_network_mbps * jvm_memory_overhead
    heap_allocation_gibps = mem_allocation_mbps / 8
    needed_memory_gib = heap_allocation_gibps * 2

    return CapacityRequirement(
        core_reference_ghz=desires.core_reference_ghz,
        cpu_cores=certain_int(needed_cores),
        mem_gib=certain_float(needed_memory_gib),
        network_mbps=certain_float(needed_network_mbps),
    )


def _estimate_java_app_region(
    instance: Instance,
    drive: Drive,
    desires: CapacityDesires,
    failover: bool = True,
    jvm_memory_overhead: float = 2,
    zones_per_region: int = 3,
) -> Optional[CapacityPlan]:

    if drive.name != "gp2":
        return None

    requirement = _estimate_java_app_requirement(desires, failover, jvm_memory_overhead)

    drive = drive.copy()
    drive.size_gib = 20
    attached_drives = (drive,)

    cluster: RegionClusterCapacity = compute_stateless_region(
        instance=instance,
        needed_cores=int(requirement.cpu_cores.mid),
        needed_memory_gib=requirement.mem_gib.mid,
        needed_network_mbps=requirement.network_mbps.mid,
        core_reference_ghz=requirement.core_reference_ghz,
        num_zones=zones_per_region,
    )
    cluster.cluster_type = "nflx-java-app"
    cluster.attached_drives = attached_drives

    # Generally don't want giant clusters
    # Especially not above 1000 because some load balancers struggle
    # with such large clusters

    if cluster.count <= 256:
        return CapacityPlan(
            requirement=requirement,
            candidate_clusters=Clusters(
                total_annual_cost=certain_float(cluster.annual_cost),
                regional=[cluster],
                zonal=list(),
            ),
        )
    return None


class NflxJavaAppCapacityModel(CapacityModel):
    @staticmethod
    def capacity_plan(
        instance: Instance,
        drive: Drive,
        services: Dict[str, Service],
        desires: CapacityDesires,
        **kwargs,
    ) -> Optional[CapacityPlan]:
        failover: bool = kwargs.pop("failover", True)
        jvm_memory_overhead: float = kwargs.pop("jvm_memory_overhead", 1.2)
        zones_per_region: int = kwargs.pop("zones_per_region", 3)

        return _estimate_java_app_region(
            instance=instance,
            drive=drive,
            desires=desires,
            failover=failover,
            jvm_memory_overhead=jvm_memory_overhead,
            zones_per_region=zones_per_region,
        )

    @staticmethod
    def description():
        return "Netflix Streaming Java App Model"

    @staticmethod
    def extra_model_arguments() -> Sequence[Tuple[str, str, str]]:
        return (
            ("zones_per_region", "int = 3", "How many zones exist in the region"),
            ("failover", "bool = 1", "If this app participates in failover"),
            (
                "jvm_memory_overhead",
                "float = 1.2",
                "How much overhead does the heap have per read byte",
            ),
        )
