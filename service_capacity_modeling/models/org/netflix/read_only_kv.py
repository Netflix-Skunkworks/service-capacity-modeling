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
from typing import Sequence
from typing import Tuple

from pydantic import BaseModel
from pydantic import Field

from service_capacity_modeling.interface import AccessPattern
from service_capacity_modeling.interface import Buffer
from service_capacity_modeling.interface import ClusterCapacity
from service_capacity_modeling.interface import BufferComponent
from service_capacity_modeling.interface import Buffers
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
from service_capacity_modeling.interface import Platform
from service_capacity_modeling.interface import QueryPattern
from service_capacity_modeling.interface import RegionClusterCapacity
from service_capacity_modeling.interface import RegionContext
from service_capacity_modeling.interface import Requirements
from service_capacity_modeling.interface import ServiceCapacity
from service_capacity_modeling.models import CapacityModel
from service_capacity_modeling.models import CostAwareModel
from service_capacity_modeling.models.common import buffer_for_components
from service_capacity_modeling.models.common import cluster_infra_cost
from service_capacity_modeling.models.common import get_effective_disk_per_node_gib
from service_capacity_modeling.models.common import normalize_cores
from service_capacity_modeling.models.common import simple_network_mbps
from service_capacity_modeling.models.common import sqrt_staffed_cores
from service_capacity_modeling.models.org.netflix.partition_aware_algorithm import (
    CapacityProblem,
    search_for_max_rf,
)

logger = logging.getLogger(__name__)


def _upsert_params(cluster: Any, params: Dict[str, Any]) -> None:
    """Update or set cluster parameters."""
    if cluster.cluster_params:
        cluster.cluster_params.update(params)
    else:
        cluster.cluster_params = params


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
        description="Reserved memory for OS, bloom filters, index blocks, "
        "and other processes (GiB).",
        ge=0,
    )


# TODO: Memory estimation is currently disabled because working_set_from_drive_and_slo
# doesn't work well for large datasets (which is all of OODM use cases). The working
# set calculation assumes a relationship between drive latency and SLO that doesn't
# hold for large datasets where the working set is a small fraction of total data.
# For now, we rely on the 64 GiB minimum RAM filter on instances and don't use
# memory as a sizing constraint. Future work: implement a better memory estimation
# that considers actual access patterns and cache hit rates for large datasets.


def _estimate_read_only_kv_requirement(
    instance: Instance,
    desires: CapacityDesires,
    args: NflxReadOnlyKVArguments,
) -> CapacityRequirement:
    """Estimate the capacity requirement for the read-only KV regional cluster.

    Note: For the partition-aware algorithm, we calculate requirements that are
    independent of replica count. The actual replica count is determined by the
    cluster computation based on partition placement and compute needs.

    Args:
        instance: The compute instance being considered
        desires: User's capacity desires

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

    # Memory: not used as sizing constraint (see TODO at top of file)
    # Instances are filtered to require 64+ GiB RAM

    # Network calculation (read-only, so only outbound read traffic)
    needed_network_mbps = simple_network_mbps(desires)

    return CapacityRequirement(
        requirement_type="read-only-kv-regional",
        reference_shape=desires.reference_shape,
        cpu_cores=certain_int(needed_cores),
        mem_gib=certain_float(0),  # Not used (see TODO at top of file)
        disk_gib=certain_float(0),  # Disk computed via partition-aware algorithm
        network_mbps=certain_float(needed_network_mbps),
        context={},
    )


def _compute_read_only_kv_regional_cluster(
    instance: Instance,
    requirement: CapacityRequirement,
    args: NflxReadOnlyKVArguments,
    partition_size_with_buffer_gib: float,
    disk_buffer_ratio: float,
) -> Optional[RegionClusterCapacity]:
    """Compute the regional cluster configuration using the partition-aware algorithm.

    Partition-aware algorithm (local disks only):
    1. DISK FIRST: Calculate partitions_per_node based on local disk capacity
    2. Calculate nodes_for_one_copy = total_partitions / partitions_per_node
    3. Start with min_replica_count, calculate initial node count
    4. CHECK CPU: If not satisfied, increase replica_count (uses spare disk)

    Note: Only supports local disks. EBS/attached disk is not supported because:
    - EBS disk is provisioned exactly for data needs (no spare space)
    - The partition-aware algorithm relies on leveraging spare disk capacity

    Args:
        instance: The compute instance being considered (must have local disk)
        requirement: Calculated capacity requirement
        args: Read-only KV specific arguments
        partition_size_with_buffer_gib: Size of one partition with disk buffer applied
        disk_buffer_ratio: Disk buffer ratio for headroom

    Returns:
        RegionClusterCapacity or None if configuration is not viable
    """
    # Only support instances with local disks
    if instance.drive is None:
        return None

    total_needed_cores = math.ceil(requirement.cpu_cores.mid)

    # Step 1 (DISK): Calculate effective disk capacity per node using helper
    effective_disk_per_node = get_effective_disk_per_node_gib(
        instance=instance,
        drive=instance.drive,
        disk_buffer_ratio=disk_buffer_ratio,
        max_local_data_per_node_gib=args.max_data_per_node_gib,
    )

    # Step 2: Use partition-aware algorithm to find optimal configuration
    problem = CapacityProblem(
        n_partitions=args.total_num_partitions,
        partition_size_gib=partition_size_with_buffer_gib,
        disk_per_node_gib=effective_disk_per_node,
        cpu_per_node=instance.cpu,
        cpu_needed=total_needed_cores,
        min_rf=args.min_replica_count,
        max_nodes=args.max_regional_size,
    )

    result = search_for_max_rf(problem)
    if result is None:
        return None

    cluster = RegionClusterCapacity(
        cluster_type="read-only-kv",
        count=result.node_count,
        instance=instance,
        attached_drives=tuple(),  # No attached drives
    )

    # Add cluster parameters for provisioning
    params = {
        "read-only-kv.total_num_partitions": args.total_num_partitions,
        "read-only-kv.min_replica_count": args.min_replica_count,
        "read-only-kv.replica_count": result.replica_count,
        "read-only-kv.partitions_per_node": result.partitions_per_node,
        "read-only-kv.nodes_for_one_copy": result.nodes_for_one_copy,
        "read-only-kv.nodes_for_cpu": math.ceil(total_needed_cores / instance.cpu),
        "read-only-kv.effective_disk_per_node_gib": effective_disk_per_node,
    }
    _upsert_params(cluster, params)

    return cluster


def _estimate_read_only_kv_cluster(
    instance: Instance,
    drive: Drive,
    context: RegionContext,
    desires: CapacityDesires,
    args: NflxReadOnlyKVArguments,
) -> Optional[CapacityPlan]:
    """Main function to estimate read-only KV cluster configuration.

    Note: Only supports instances with local disks. EBS is not supported
    because the partition-aware algorithm relies on fixed disk capacity to
    calculate partition placement.

    Args:
        instance: The compute instance being considered (must have local disk)
        drive: The drive configuration (unused - local disks only)
        context: Regional context (unused - read-only KV is not zone-balanced)
        desires: User's capacity desires
        args: Read-only KV specific arguments

    Returns:
        CapacityPlan or None if configuration is not viable
    """
    # Mark unused parameters (required by CapacityModel interface)
    _ = drive  # Local disks only - use instance.drive
    _ = context  # Read-only KV is not zone-balanced

    # Validate instance constraints
    # Minimum 64 GiB RAM: ensures sufficient memory for RocksDB block cache
    # and working set without needing memory as a sizing constraint
    if instance.cpu < 2 or instance.ram_gib < 64:
        return None

    # Only support instances with local disks
    # EBS not supported: partition-aware algorithm relies on fixed disk capacity
    if instance.drive is None:
        return None

    # Calculate requirements
    requirement = _estimate_read_only_kv_requirement(
        instance=instance, desires=desires, args=args
    )

    # Compute disk buffer values inline (not via context)
    disk_buffer = buffer_for_components(
        buffers=desires.buffers, components=[BufferComponent.disk]
    )
    unreplicated_data_gib = desires.data_shape.estimated_state_size_gib.mid
    partition_size_gib = unreplicated_data_gib / args.total_num_partitions
    partition_size_with_buffer_gib = partition_size_gib * disk_buffer.ratio

    # Compute cluster
    cluster = _compute_read_only_kv_regional_cluster(
        instance=instance,
        requirement=requirement,
        args=args,
        partition_size_with_buffer_gib=partition_size_with_buffer_gib,
        disk_buffer_ratio=disk_buffer.ratio,
    )

    if cluster is None:
        return None

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

    return CapacityPlan(
        requirements=Requirements(regional=[requirement]),
        candidate_clusters=clusters,
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
    def service_costs(
        service_type: str,
        context: RegionContext,
        desires: CapacityDesires,
        extra_model_arguments: Dict[str, Any],
    ) -> List[ServiceCapacity]:
        # No additional service costs (read-only: no network replication, no backup)
        return []

    @staticmethod
    def cluster_costs(
        service_type: str,
        zonal_clusters: Sequence[ClusterCapacity] = (),
        regional_clusters: Sequence[ClusterCapacity] = (),
    ) -> Dict[str, float]:
        return cluster_infra_cost(
            service_type,
            zonal_clusters,
            regional_clusters,
            cluster_type=NflxReadOnlyKVCapacityModel.cluster_type,
        )

    @staticmethod
    def capacity_plan(
        instance: Instance,
        drive: Drive,
        context: RegionContext,
        desires: CapacityDesires,
        extra_model_arguments: Dict[str, Any],
    ) -> Optional[CapacityPlan]:
        args = NflxReadOnlyKVArguments.model_validate(extra_model_arguments)

        return _estimate_read_only_kv_cluster(
            instance=instance,
            drive=drive,
            context=context,
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
