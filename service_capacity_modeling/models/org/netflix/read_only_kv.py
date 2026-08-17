"""
Netflix Read-Only Key-Value Capacity Model

A read-only data serving layer that loads data from offline sources
(e.g., S3) and serves read traffic online.

Key characteristics:
- Regional deployment (not zonal)
- Read-only after data population is complete
- Uses RocksDB as the storage backend
- Local disks only, attached disk not supported
- Define min_replica_count but determine the optimal number of replicas as output
"""

import logging
import math
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from pydantic import BaseModel
from pydantic import Field

from service_capacity_modeling.interface import AccessPattern
from service_capacity_modeling.interface import Buffer
from service_capacity_modeling.interface import BufferComponent
from service_capacity_modeling.interface import Buffers
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import CapacityPlan
from service_capacity_modeling.interface import CapacityRegretParameters
from service_capacity_modeling.interface import CapacityRequirement
from service_capacity_modeling.interface import certain_float
from service_capacity_modeling.interface import certain_int
from service_capacity_modeling.interface import Clusters
from service_capacity_modeling.interface import DataShape
from service_capacity_modeling.interface import Drive
from service_capacity_modeling.interface import FixedInterval
from service_capacity_modeling.interface import Instance
from service_capacity_modeling.interface import Interval
from service_capacity_modeling.interface import normalized_aws_size
from service_capacity_modeling.interface import Platform
from service_capacity_modeling.interface import QueryPattern
from service_capacity_modeling.interface import RegionClusterCapacity
from service_capacity_modeling.interface import RegionContext
from service_capacity_modeling.interface import Requirements
from service_capacity_modeling.interface import ServiceCapacity
from service_capacity_modeling.models import CapacityModel
from service_capacity_modeling.models import CostAwareModel
from service_capacity_modeling.models import RANK_PENALTIES
from service_capacity_modeling.models.common import buffer_for_components
from service_capacity_modeling.models.common import EFFECTIVE_DISK_PER_NODE_GIB
from service_capacity_modeling.models.common import get_effective_disk_per_node_gib
from service_capacity_modeling.models.common import upsert_params
from service_capacity_modeling.models.common import normalize_cores
from service_capacity_modeling.models.common import simple_network_mbps
from service_capacity_modeling.models.common import sqrt_staffed_cores
from service_capacity_modeling.models.common import usable_memory_gib
from service_capacity_modeling.models.org.netflix.partition_aware_algorithm import (
    CapacityProblem,
    search_for_min_nodes,
)

logger = logging.getLogger(__name__)

# A namespace opts in by supplying this named desired memory buffer.
SST_MEMORY_BUFFER_NAME = "read_only_kv_sst_memory"


def _compute_penalties(
    instance: Instance,
    large_instance_regret: float,
) -> Dict[str, float]:
    penalties: Dict[str, float] = {}

    instance_size = float(normalized_aws_size(instance.name))
    if large_instance_regret > 0 and instance_size > 4:
        penalties["large_instance"] = large_instance_regret * (instance_size - 4) / 4

    return penalties


class NflxReadOnlyKVArguments(BaseModel):
    """Configuration arguments for the Netflix Read-Only KV capacity model.

    Note: This model only supports local (ephemeral) disks. EBS/attached disks
    are not supported because the partition-aware algorithm relies on fixed disk
    capacity per instance to calculate partition placement and leverage spare
    disk space for additional replicas in compute-heavy workloads.
    """

    min_replica_count: int = Field(
        default=2,
        description="Minimum number of data replicas. "
        "Actual count may be higher for compute-heavy workloads.",
        ge=1,
        le=10,
    )
    total_num_partitions: int = Field(
        description="Total number of partitions for the dataset (required).",
        ge=1,
    )
    max_regional_size: int = Field(
        default=256,
        description="Maximum number of instances in the regional cluster.",
    )
    max_data_per_node_gib: int = Field(
        default=2048,
        description="Maximum data per node (GiB). "
        "Prevents overloading nodes with too much data.",
    )
    reserved_memory_gib: float = Field(
        default=8.0,
        description="Minimum memory reserved for the OS, JVM heap, block cache, "
        "and other processes (GiB). Data-shape reservations take precedence when "
        "their sum is larger.",
        ge=0,
    )
    sst_memory_overhead_ratio: float = Field(
        default=0.05,
        description="Planning assumption for resident RocksDB table-reader memory "
        "per complete data copy, as a ratio of SST data. Applied only when "
        f"the {SST_MEMORY_BUFFER_NAME!r} desired memory buffer is present.",
        ge=0,
        le=1,
    )
    large_instance_regret: float = Field(
        default=0.2,
        description="Graduated cost penalty for instance sizes above 4xlarge. "
        "Adds penalty * max(0, (normalized_size - 4) / 4) * cost to the "
        "effective sort cost. Set to 0 to disable.",
    )


# TODO: Data-block working set estimation remains disabled because
# working_set_from_drive_and_slo does not work well for large OODM datasets.
# Resident SST table-reader memory is modeled separately as a percentage of data.


def _estimate_read_only_kv_requirement(
    instance: Instance,
    desires: CapacityDesires,
    sst_memory_overhead_ratio: float,
    memory_buffer_ratio: float,
) -> CapacityRequirement:
    """Estimate the capacity requirement for the read-only KV regional cluster.

    Note: For the partition-aware algorithm, we calculate requirements for one
    complete data copy. The actual replica count is determined by the cluster
    computation based on partition placement and compute needs. Cluster parameters
    report the resulting total SST memory across all replicas.

    Args:
        instance: The compute instance being considered
        desires: User's capacity desires
        sst_memory_overhead_ratio: Resident SST memory as a ratio of SST data
        memory_buffer_ratio: Memory headroom applied to the SST estimate

    Returns:
        CapacityRequirement for the regional cluster
    """
    # Get compute buffer for CPU calculation
    compute_buffer = buffer_for_components(
        buffers=desires.buffers, components=[BufferComponent.cpu]
    )

    # CPU calculation using sqrt staffing model (independent of replicas)
    # Apply buffer first, then normalize to target instance shape
    needed_cores = math.ceil(sqrt_staffed_cores(desires) * compute_buffer.ratio)
    needed_cores = normalize_cores(
        core_count=needed_cores,
        target_shape=instance,
        reference_shape=desires.reference_shape,
    )

    needed_sst_memory_gib = (
        desires.data_shape.estimated_state_size_gib.mid
        * sst_memory_overhead_ratio
        * memory_buffer_ratio
    )

    # Network calculation (read-only, so only outbound read traffic)
    needed_network_mbps = simple_network_mbps(desires)

    return CapacityRequirement(
        requirement_type=NflxReadOnlyKVCapacityModel.cluster_type,
        reference_shape=desires.reference_shape,
        cpu_cores=certain_int(needed_cores),
        mem_gib=certain_float(needed_sst_memory_gib),
        disk_gib=certain_float(0),  # Disk computed via partition-aware algorithm
        network_mbps=certain_float(needed_network_mbps),
        context={},
    )


# pylint: disable-next=too-many-positional-arguments
def _compute_read_only_kv_regional_cluster(
    instance: Instance,
    requirement: CapacityRequirement,
    args: NflxReadOnlyKVArguments,
    partition_size_with_buffer_gib: float,
    disk_buffer_ratio: float,
    sst_memory_overhead_ratio: float,
    memory_buffer_ratio: float,
    reserved_memory_gib: float,
) -> Optional[RegionClusterCapacity]:
    """Compute the regional cluster configuration using the partition-aware algorithm.

    Partition-aware algorithm (local disks only):
    1. Calculate independent disk and resident SST memory partition ceilings.
    2. Evaluate each distinct nodes-per-copy placement below those ceilings.
    3. Calculate the replica count required by minimum RF and CPU for each placement.
    4. Select the fewest nodes, then prefer higher RF and denser packing on ties.

    Note: Only supports local disks. EBS/attached disk is not supported because:
    - EBS disk is provisioned exactly for data needs (no spare space)
    - The partition-aware algorithm relies on leveraging spare disk capacity

    Args:
        instance: The compute instance being considered (must have local disk)
        requirement: Calculated capacity requirement
        args: Read-only KV specific arguments
        partition_size_with_buffer_gib: Size of one partition with disk buffer applied
        disk_buffer_ratio: Disk buffer ratio for headroom
        sst_memory_overhead_ratio: Active resident SST memory estimate
        memory_buffer_ratio: Memory buffer ratio for headroom
        reserved_memory_gib: Total memory reserved outside SST table readers

    Returns:
        RegionClusterCapacity or None if configuration is not viable
    """
    # Only support instances with local disks
    if instance.drive is None:
        return None

    total_needed_cores = math.ceil(requirement.cpu_cores.mid)

    effective_disk_per_node = get_effective_disk_per_node_gib(
        instance=instance,
        drive=instance.drive,
        disk_buffer_ratio=disk_buffer_ratio,
        max_local_data_per_node_gib=args.max_data_per_node_gib,
    )

    sst_memory_per_partition_gib = requirement.mem_gib.mid / args.total_num_partitions
    available_sst_memory_per_node_gib = None
    if sst_memory_overhead_ratio > 0:
        usable_memory_per_node_gib = usable_memory_gib(
            instance, lambda _: reserved_memory_gib
        )
        if usable_memory_per_node_gib <= 0:
            return None
        available_sst_memory_per_node_gib = usable_memory_per_node_gib

    problem = CapacityProblem(
        n_partitions=args.total_num_partitions,
        partition_size_gib=partition_size_with_buffer_gib,
        disk_per_node_gib=effective_disk_per_node,
        memory_per_partition_gib=(
            sst_memory_per_partition_gib if sst_memory_overhead_ratio > 0 else None
        ),
        memory_per_node_gib=available_sst_memory_per_node_gib,
        cpu_per_node=instance.cpu,
        cpu_needed=total_needed_cores,
        min_rf=args.min_replica_count,
        max_nodes=args.max_regional_size,
    )

    result = search_for_min_nodes(problem)
    if result is None:
        return None

    cluster = RegionClusterCapacity(
        cluster_type="read-only-kv",
        count=result.node_count,
        instance=instance,
        attached_drives=tuple(),  # No attached drives
    )

    # Add cluster parameters for provisioning
    params: Dict[str, Any] = {
        "read-only-kv.total_num_partitions": args.total_num_partitions,
        "read-only-kv.min_replica_count": args.min_replica_count,
        "read-only-kv.replica_count": result.replica_count,
        "read-only-kv.partitions_per_node": result.partitions_per_node,
        "read-only-kv.nodes_for_one_copy": result.nodes_for_one_copy,
        "read-only-kv.nodes_for_cpu": math.ceil(total_needed_cores / instance.cpu),
        EFFECTIVE_DISK_PER_NODE_GIB: effective_disk_per_node,
    }
    if sst_memory_overhead_ratio > 0:
        params.update(
            {
                "read-only-kv.sst_memory_buffer": SST_MEMORY_BUFFER_NAME,
                "read-only-kv.sst_memory_overhead_ratio": sst_memory_overhead_ratio,
                "read-only-kv.memory_buffer_ratio": memory_buffer_ratio,
                "read-only-kv.reserved_memory_per_node_gib": reserved_memory_gib,
                "read-only-kv.buffered_sst_memory_per_partition_gib": (
                    sst_memory_per_partition_gib
                ),
                "read-only-kv.estimated_sst_memory_per_copy_gib": (
                    requirement.mem_gib.mid / memory_buffer_ratio
                ),
                "read-only-kv.buffered_sst_memory_per_copy_gib": (
                    requirement.mem_gib.mid
                ),
                "read-only-kv.total_buffered_sst_memory_gib": (
                    requirement.mem_gib.mid * result.replica_count
                ),
                "read-only-kv.available_sst_memory_per_node_gib": (
                    available_sst_memory_per_node_gib
                ),
                "read-only-kv.max_partitions_per_node_by_disk": (
                    result.max_partitions_per_node_by_disk
                ),
                "read-only-kv.max_partitions_per_node_by_memory": (
                    result.max_partitions_per_node_by_memory
                ),
            }
        )
    upsert_params(cluster, params)

    penalties = _compute_penalties(
        instance=instance,
        large_instance_regret=args.large_instance_regret,
    )
    if penalties:
        upsert_params(cluster, {RANK_PENALTIES: penalties})

    return cluster


def _estimate_read_only_kv_cluster(
    instance: Instance,
    desires: CapacityDesires,
    args: NflxReadOnlyKVArguments,
) -> Optional[CapacityPlan]:
    """Main function to estimate read-only KV cluster configuration.

    Note: Only supports instances with local disks. EBS is not supported
    because the partition-aware algorithm relies on fixed disk capacity to
    calculate partition placement.

    Args:
        instance: The compute instance being considered (must have local disk)
        desires: User's capacity desires
        args: Read-only KV specific arguments

    Returns:
        CapacityPlan or None if configuration is not viable
    """

    # Validate instance constraints
    # Minimum 30 GiB RAM: allows 32 GiB-class shapes in the hardware catalog
    # while keeping enough memory for RocksDB block cache and working set.
    if instance.cpu < 2 or instance.ram_gib < 30:
        return None

    # Only support instances with local disks
    # EBS not supported: partition-aware algorithm relies on fixed disk capacity
    if instance.drive is None:
        return None

    unreplicated_data_gib = desires.data_shape.estimated_state_size_gib.mid
    if unreplicated_data_gib <= 0:
        return None

    sst_memory_buffer = desires.buffers.desired.get(SST_MEMORY_BUFFER_NAME)
    sst_memory_overhead_ratio = 0.0
    memory_buffer_ratio = 1.0
    if args.sst_memory_overhead_ratio > 0 and sst_memory_buffer is not None:
        if set(sst_memory_buffer.components).intersection(
            {BufferComponent.memory, BufferComponent.storage}
        ):
            memory_buffer = buffer_for_components(
                buffers=desires.buffers, components=[BufferComponent.memory]
            )
            memory_buffer_ratio = memory_buffer.ratio
        else:
            memory_buffer_ratio = 0
        if memory_buffer_ratio <= 0:
            return None
        sst_memory_overhead_ratio = args.sst_memory_overhead_ratio

    # Calculate requirements
    per_copy_requirement = _estimate_read_only_kv_requirement(
        instance=instance,
        desires=desires,
        sst_memory_overhead_ratio=sst_memory_overhead_ratio,
        memory_buffer_ratio=memory_buffer_ratio,
    )

    # Compute disk buffer values inline (not via context)
    disk_buffer = buffer_for_components(
        buffers=desires.buffers, components=[BufferComponent.disk]
    )
    partition_size_gib = unreplicated_data_gib / args.total_num_partitions
    partition_size_with_buffer_gib = partition_size_gib * disk_buffer.ratio
    reserved_memory_gib = max(
        args.reserved_memory_gib,
        desires.data_shape.reserved_instance_app_mem_gib
        + desires.data_shape.reserved_instance_system_mem_gib,
    )

    # Compute cluster
    cluster = _compute_read_only_kv_regional_cluster(
        instance=instance,
        requirement=per_copy_requirement,
        args=args,
        partition_size_with_buffer_gib=partition_size_with_buffer_gib,
        disk_buffer_ratio=disk_buffer.ratio,
        sst_memory_overhead_ratio=sst_memory_overhead_ratio,
        memory_buffer_ratio=memory_buffer_ratio,
        reserved_memory_gib=reserved_memory_gib,
    )

    if cluster is None:
        return None

    replica_count = int(cluster.cluster_params["read-only-kv.replica_count"])
    regional_requirement = per_copy_requirement.model_copy(
        update={"mem_gib": per_copy_requirement.mem_gib.scale(replica_count)}
    )

    # Build cost breakdown using cluster_costs (consistent with CostAwareModel)
    rokv_costs = NflxReadOnlyKVCapacityModel.cluster_costs(
        service_type=NflxReadOnlyKVCapacityModel.service_name,
        regional_clusters=[cluster],
    )

    clusters = Clusters(
        annual_costs=rokv_costs,
        zonal=[],
        regional=[cluster],
        services=[],
    )
    penalty_ratio = sum(
        cluster.cluster_params.get(RANK_PENALTIES, {}).values(),
    )

    return CapacityPlan(
        requirements=Requirements(regional=[regional_requirement]),
        candidate_clusters=clusters,
        rank=clusters.total_annual_cost * (1 + penalty_ratio),
    )


class NflxReadOnlyKVCapacityModel(CapacityModel, CostAwareModel):
    """Netflix Read-Only Key-Value Capacity Model.

    A read-only data serving layer that:
    - Loads data from offline sources (data warehouse, batch processing)
    - Serves read traffic with low latency using RocksDB
    - Deploys regionally with configurable replication factor
    - Local (ephemeral) disks only - EBS not supported
    """

    service_name = "read-only-kv"
    cluster_type = "read-only-kv"

    @staticmethod
    def regret(
        regret_params: CapacityRegretParameters,
        optimal_plan: CapacityPlan,
        proposed_plan: CapacityPlan,
    ) -> Dict[str, float]:
        regrets = CapacityModel.regret(regret_params, optimal_plan, proposed_plan)

        if proposed_plan.candidate_clusters.regional:
            params = proposed_plan.candidate_clusters.regional[0].cluster_params
            penalties = params.get(RANK_PENALTIES, {})
            if "large_instance" in penalties:
                compute_cost = float(
                    sum(
                        c.annual_cost for c in proposed_plan.candidate_clusters.regional
                    )
                )
                regrets["large_instance"] = penalties["large_instance"] * compute_cost

        return regrets

    @staticmethod
    def service_costs(
        service_type: str,
        context: RegionContext,
        desires: CapacityDesires,
        extra_model_arguments: Dict[str, Any],
    ) -> List[ServiceCapacity]:
        # No additional service costs (read-only: no network replication, no backup)
        return []

    @staticmethod
    def capacity_plan(
        instance: Instance,
        drive: Drive,
        context: RegionContext,
        desires: CapacityDesires,
        extra_model_arguments: Dict[str, Any],
    ) -> Optional[CapacityPlan]:
        args = NflxReadOnlyKVArguments.model_validate(extra_model_arguments)

        # drive and context are unused: read-only KV uses instance.drive
        # (local disks only) and is not zone-balanced.
        _ = drive
        _ = context

        return _estimate_read_only_kv_cluster(
            instance=instance,
            desires=desires,
            args=args,
        )

    @staticmethod
    def description() -> str:
        return "Netflix Read-Only Key-Value Capacity Model"

    @staticmethod
    def extra_model_arguments_schema() -> Dict[str, Any]:
        return NflxReadOnlyKVArguments.model_json_schema()

    @staticmethod
    def allowed_platforms() -> Tuple[Platform, ...]:
        return (Platform.amd64, Platform.arm64)

    @staticmethod
    def default_buffers() -> Buffers:
        return Buffers(
            desired={
                # No background work, just traffic spikes
                "compute": Buffer(
                    ratio=1.5, components=[BufferComponent.compute]
                ),  # 67% utilization target
                # Read-only: can run at high disk utilization
                "disk": Buffer(ratio=1.15, components=[BufferComponent.disk]),  # 87%
            },
        )

    @staticmethod
    def default_desires(
        user_desires: CapacityDesires, extra_model_arguments: Dict[str, Any]
    ) -> CapacityDesires:
        buffers = NflxReadOnlyKVCapacityModel.default_buffers()

        # Both latency and throughput access patterns use the same defaults.
        # The actual capacity calculation is driven by user-provided values
        # (RPS, latency SLO, data size), so differentiated defaults are not needed.
        return CapacityDesires(
            query_pattern=QueryPattern(
                access_pattern=AccessPattern.latency,
                # Read-only: no writes
                estimated_mean_write_size_bytes=certain_float(0),
                estimated_mean_write_latency_ms=certain_float(0),
                # RocksDB point reads are fast when data is in cache
                estimated_mean_read_latency_ms=Interval(
                    low=0.5, mid=2, high=10, confidence=0.98
                ),
                # Single digit millisecond SLO for latency pattern
                read_latency_slo_ms=FixedInterval(
                    minimum_value=0.1,
                    maximum_value=20,
                    low=0.5,
                    mid=2,
                    high=10,
                    confidence=0.98,
                ),
            ),
            data_shape=DataShape(
                # Typical dataset size
                estimated_state_size_gib=Interval(
                    low=10, mid=100, high=1000, confidence=0.98
                ),
                estimated_compression_ratio=certain_float(1),
                # Reserved memory for OS and RocksDB overhead
                reserved_instance_app_mem_gib=4,
            ),
            buffers=buffers,
        )


nflx_read_only_kv_capacity_model = NflxReadOnlyKVCapacityModel()
