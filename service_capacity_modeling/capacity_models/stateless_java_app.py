from typing import List

from service_capacity_modeling.capacity_models.common import compute_stateless_region
from service_capacity_modeling.capacity_models.common import simple_network_mbps
from service_capacity_modeling.capacity_models.common import sqrt_staffed_cores
from service_capacity_modeling.capacity_models.utils import reduce_by_family
from service_capacity_modeling.models import CapacityDesires
from service_capacity_modeling.models import CapacityPlan
from service_capacity_modeling.models import CapacityRequirement
from service_capacity_modeling.models import certain_float
from service_capacity_modeling.models import certain_int
from service_capacity_modeling.models import Hardware
from service_capacity_modeling.models import RegionClusterCapacity


def estimate_java_app_region(
    hardware: Hardware,
    desires: CapacityDesires,
    zones_per_region: int = 3,
    failover: bool = True,
    jvm_memory_overhead: float = 2,
) -> CapacityPlan:
    needed_cores = sqrt_staffed_cores(desires)
    needed_network_mbps = simple_network_mbps(desires)

    if failover:
        # For failover provision at 40% utilization
        needed_cores = needed_cores * (1 / 0.4)
        needed_network_mbps = needed_network_mbps * (1 / 0.4)

    # Assume a Java application that can allocate about 1 GiB/s to heap
    # per 2 GiB of memory and assume that we have a ~2x memory overhead
    # due to the JVM overhead.
    # TODO: we should probably have the memory bandwidth attached to
    # the instance type, e.g. Intel CPUs and AMD CPUs have different
    # per core memory bandwidth.
    mem_allocation_mbps = needed_network_mbps * jvm_memory_overhead
    heap_allocation_gibps = mem_allocation_mbps / 8
    needed_memory_gib = heap_allocation_gibps * 2

    topologies: List[RegionClusterCapacity] = []
    for instance in hardware.instances.values():
        cluster: RegionClusterCapacity = compute_stateless_region(
            instance=instance,
            needed_cores=needed_cores,
            needed_memory_gib=needed_memory_gib,
            needed_network_mbps=needed_network_mbps,
            core_reference_ghz=hardware.core_reference_ghz,
            num_zones=zones_per_region,
        )
        # Generally don't want giant clusters
        # Especially not above 1000 because some load balancers struggle
        # with such large clusters
        if cluster.count <= 256:
            topologies.append(
                RegionClusterCapacity(
                    cluster_type="java-app",
                    count=cluster.count,
                    instance=instance,
                    annual_cost=cluster.annual_cost,
                )
            )

    return CapacityPlan(
        capacity_requirement=CapacityRequirement(
            cpu_reference_ghz=hardware.core_reference_ghz,
            cpu_cores=certain_int(needed_cores),
            mem_gib=certain_float(needed_memory_gib),
            network_mbps=certain_float(needed_network_mbps),
        ),
        clusters=reduce_by_family(topologies)[:2],
    )
