import math
from typing import Any
from typing import Dict
from typing import Optional

from pydantic import BaseModel
from pydantic import Field

from service_capacity_modeling.interface import AccessConsistency
from service_capacity_modeling.interface import AccessPattern
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import CapacityPlan
from service_capacity_modeling.interface import CapacityRegretParameters
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
from service_capacity_modeling.models.common import compute_stateless_region
from service_capacity_modeling.models.common import network_services
from service_capacity_modeling.models.common import normalize_cores
from service_capacity_modeling.models.common import simple_network_mbps
from service_capacity_modeling.models.common import sqrt_staffed_cores


def _estimate_java_app_requirement(
    instance: Instance,
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

    # Adjust cores to clock frequency differences against measurement
    needed_cores = normalize_cores(
        core_count=needed_cores,
        target_shape=instance,
        reference_shape=desires.reference_shape,
    )

    # Assume a Java application that can allocate about 1 GiB/s to heap
    # per 2 GiB of heap with some amount of overhead on the network traffic.
    # So for example if we have 512 MiB of network traffic there is some
    # overhead associated with that...

    # TODO: we should probably have the memory bandwidth attached to
    # the instance type, e.g. Intel CPUs and AMD CPUs have different
    # per core memory bandwidth.
    mem_allocation_mbps = needed_network_mbps * jvm_memory_overhead
    heap_allocation_gibps = (mem_allocation_mbps / 8) / 1024
    network_heap = heap_allocation_gibps * 2

    needed_memory_gib = network_heap

    return CapacityRequirement(
        requirement_type="java-app",
        reference_shape=desires.reference_shape,
        cpu_cores=certain_int(needed_cores),
        mem_gib=certain_float(needed_memory_gib),
        network_mbps=certain_float(needed_network_mbps),
        context={
            "network_heap_gib": network_heap,
            "reserved_mem": desires.data_shape.reserved_instance_app_mem_gib,
        },
    )


def _estimate_java_app_region(  # pylint: disable=too-many-positional-arguments
    instance: Instance,
    drive: Drive,
    desires: CapacityDesires,
    context: RegionContext,
    root_disk_gib: int = 10,
    failover: bool = True,
    jvm_memory_overhead: float = 2,
) -> Optional[CapacityPlan]:

    if drive.name != "gp2":
        return None

    zones_per_region = context.zones_in_region
    requirement = _estimate_java_app_requirement(
        instance, desires, failover, jvm_memory_overhead
    )

    drive = drive.model_copy()
    drive.size_gib = root_disk_gib
    attached_drives = (drive,)

    cluster: RegionClusterCapacity = compute_stateless_region(
        instance=instance,
        needed_cores=int(requirement.cpu_cores.mid),
        needed_memory_gib=requirement.mem_gib.mid,
        needed_network_mbps=requirement.network_mbps.mid,
        num_zones=zones_per_region,
    )
    cluster.cluster_type = "nflx-java-app"
    cluster.attached_drives = attached_drives

    # Generally don't want giant clusters
    # Especially not above 1000 because some load balancers struggle
    # with such large clusters

    if cluster.count <= 256:
        costs = {"nflx-java-app.regional-clusters": cluster.annual_cost}
        # Assume stateless java stays in the same region but crosses a zone
        network = network_services(
            "nflx-java-app", RegionContext(num_regions=1), desires, copies_per_region=2
        )
        for s in network:
            costs[s.service_type] = s.annual_cost

        return CapacityPlan(
            requirements=Requirements(regional=[requirement]),
            candidate_clusters=Clusters(
                annual_costs=costs,
                regional=[cluster],
                zonal=[],
            ),
        )
    return None


class NflxJavaAppArguments(BaseModel):
    failover: bool = Field(
        default=True, description="If this app participates in failover"
    )
    jvm_memory_overhead: float = Field(
        default=1.2,
        description="How much overhead does the heap have per read byte",
    )
    root_disk_gib: int = Field(
        default=10, description="How many GiB of root volume to attach"
    )


class NflxJavaAppCapacityModel(CapacityModel):
    @staticmethod
    def capacity_plan(
        instance: Instance,
        drive: Drive,
        context: RegionContext,
        desires: CapacityDesires,
        extra_model_arguments: Dict[str, Any],
    ) -> Optional[CapacityPlan]:
        failover: bool = extra_model_arguments.get("failover", True)
        jvm_memory_overhead: float = extra_model_arguments.get(
            "jvm_memory_overhead", 1.2
        )
        root_disk_gib: int = extra_model_arguments.get("root_disk_gib", 10)

        return _estimate_java_app_region(
            instance=instance,
            drive=drive,
            desires=desires,
            failover=failover,
            root_disk_gib=root_disk_gib,
            jvm_memory_overhead=jvm_memory_overhead,
            context=context,
        )

    @staticmethod
    def description():
        return "Netflix Streaming Java App Model"

    @staticmethod
    def extra_model_arguments_schema() -> Dict[str, Any]:
        return NflxJavaAppArguments.model_json_schema()

    @staticmethod
    def regret(
        regret_params: CapacityRegretParameters,
        optimal_plan: CapacityPlan,
        proposed_plan: CapacityPlan,
    ) -> Dict[str, float]:
        regret = super(NflxJavaAppCapacityModel, NflxJavaAppCapacityModel).regret(
            regret_params, optimal_plan, proposed_plan
        )
        regret["disk_space"] = 0
        return regret

    @staticmethod
    def default_desires(user_desires, extra_model_arguments):
        if user_desires.query_pattern.access_pattern == AccessPattern.latency:
            return CapacityDesires(
                query_pattern=QueryPattern(
                    access_pattern=AccessPattern.latency,
                    access_consistency=GlobalConsistency(
                        same_region=Consistency(
                            target_consistency=AccessConsistency.read_your_writes,
                        ),
                        cross_region=Consistency(
                            target_consistency=AccessConsistency.never,
                        ),
                    ),
                    estimated_mean_read_size_bytes=Interval(
                        low=128, mid=1024, high=65536, confidence=0.95
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
        else:
            return CapacityDesires(
                query_pattern=QueryPattern(
                    access_pattern=AccessPattern.latency,
                    access_consistency=GlobalConsistency(
                        same_region=Consistency(
                            target_consistency=AccessConsistency.read_your_writes,
                        ),
                        cross_region=Consistency(
                            target_consistency=AccessConsistency.never,
                        ),
                    ),
                    estimated_mean_read_size_bytes=Interval(
                        low=128, mid=1024, high=65536, confidence=0.95
                    ),
                    estimated_mean_write_size_bytes=Interval(
                        low=64, mid=128, high=1024, confidence=0.95
                    ),
                    # Throughput ops can be slower
                    estimated_mean_read_latency_ms=Interval(
                        low=0.2, mid=4, high=8, confidence=0.98
                    ),
                    estimated_mean_write_latency_ms=Interval(
                        low=0.2, mid=1, high=5, confidence=0.98
                    ),
                    # "Tens of millisecond SLO"
                    read_latency_slo_ms=FixedInterval(
                        minimum_value=0.5,
                        maximum_value=100,
                        low=1,
                        mid=5,
                        high=40,
                        confidence=0.98,
                    ),
                    write_latency_slo_ms=FixedInterval(
                        minimum_value=0.5,
                        maximum_value=100,
                        low=1,
                        mid=5,
                        high=40,
                        confidence=0.98,
                    ),
                ),
                data_shape=DataShape(
                    # Assume 4 GiB heaps
                    reserved_instance_app_mem_gib=4
                ),
            )


nflx_java_app_capacity_model = NflxJavaAppCapacityModel()
