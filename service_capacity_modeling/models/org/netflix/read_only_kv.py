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
from typing import Optional
from typing import Tuple

from pydantic import BaseModel
from pydantic import Field

from service_capacity_modeling.interface import AccessConsistency
from service_capacity_modeling.interface import AccessPattern
from service_capacity_modeling.interface import Buffer
from service_capacity_modeling.interface import BufferComponent
from service_capacity_modeling.interface import Buffers
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
from service_capacity_modeling.interface import Platform
from service_capacity_modeling.interface import QueryPattern
from service_capacity_modeling.interface import RegionClusterCapacity
from service_capacity_modeling.interface import RegionContext
from service_capacity_modeling.interface import Requirements
from service_capacity_modeling.models import CapacityModel
from service_capacity_modeling.models.common import buffer_for_components
from service_capacity_modeling.models.common import normalize_cores
from service_capacity_modeling.models.common import simple_network_mbps
from service_capacity_modeling.models.common import sqrt_staffed_cores
from service_capacity_modeling.models.common import working_set_from_drive_and_slo
from service_capacity_modeling.stats import dist_for_interval

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
    rocksdb_block_cache_percent: float = Field(
        default=0.3,
        description="Percentage of data to keep in RocksDB block cache. "
        "Higher values improve read latency but require more memory.",
        ge=0.0,
        le=1.0,
    )
    reserved_memory_gib: float = Field(
        default=8.0,
        description="Reserved memory for OS, bloom filters, index blocks, "
        "and other processes (GiB).",
        ge=0,
    )


def _get_data_size_gib(
    desires: CapacityDesires,
    replica_count: int,
) -> float:
    """Calculate total data size in GiB including replication.

    Args:
        desires: User's capacity desires containing data shape
        replica_count: Number of replicas (RF)

    Returns:
        Total data size in GiB
    """
    # Use estimated_state_size_gib directly (user-provided total data size)
    data_size_gib = desires.data_shape.estimated_state_size_gib.mid
    return data_size_gib * replica_count


def _get_memory_gib(
    desires: CapacityDesires,
    drive: Drive,
    replica_count: int,
    block_cache_percent: float,
) -> float:
    """Calculate total memory required for the cluster.

    Takes the max of (per replica):
    - Block cache: data * block_cache_percent
    - Working set: data * working_set_percent (based on drive latency vs SLO)

    Then multiplies by replica_count for total.
    Bloom filters and index blocks are covered by reserved_memory_gib.

    Args:
        desires: Capacity desires with data shape and SLO requirements
        drive: Drive to use for latency estimation
        replica_count: Number of replicas
        block_cache_percent: Percentage of data to keep in block cache

    Returns:
        Total memory required in GiB
    """
    unreplicated_data_gib = desires.data_shape.estimated_state_size_gib.mid

    block_cache_gib = unreplicated_data_gib * block_cache_percent

    working_set_percent = working_set_from_drive_and_slo(
        drive_read_latency_dist=dist_for_interval(drive.read_io_latency_ms),
        read_slo_latency_dist=dist_for_interval(
            desires.query_pattern.read_latency_slo_ms
        ),
        estimated_working_set=desires.data_shape.estimated_working_set_percent,
        target_percentile=0.95,
    ).mid
    working_set_gib = unreplicated_data_gib * working_set_percent

    memory_per_replica = max(block_cache_gib, working_set_gib)
    return memory_per_replica * replica_count


def _estimate_read_only_kv_requirement(
    instance: Instance,
    drive: Drive,
    desires: CapacityDesires,
    args: NflxReadOnlyKVArguments,
) -> CapacityRequirement:
    """Estimate the capacity requirement for the read-only KV regional cluster.

    Note: For the partition-aware algorithm, we calculate requirements that are
    independent of replica count. The actual replica count is determined by the
    cluster computation based on partition placement and compute needs.

    Args:
        instance: The compute instance being considered
        drive: The drive configuration
        desires: User's capacity desires
        args: Read-only KV specific arguments

    Returns:
        CapacityRequirement for the regional cluster
    """
    # Get buffers for each component
    compute_buffer = buffer_for_components(
        buffers=desires.buffers, components=[BufferComponent.cpu]
    )
    disk_buffer = buffer_for_components(
        buffers=desires.buffers, components=[BufferComponent.disk]
    )
    memory_buffer = buffer_for_components(
        buffers=desires.buffers, components=[BufferComponent.memory]
    )

    # Unreplicated data size
    unreplicated_data_gib = desires.data_shape.estimated_state_size_gib.mid

    # Partition size (unreplicated)
    partition_size_gib = unreplicated_data_gib / args.total_num_partitions

    # CPU calculation using sqrt staffing model (independent of replicas)
    raw_cores = sqrt_staffed_cores(desires)
    raw_cores = normalize_cores(
        core_count=raw_cores,
        target_shape=instance,
        reference_shape=desires.reference_shape,
    )
    # Apply compute buffer
    needed_cores = raw_cores * compute_buffer.ratio

    # Memory calculation PER REPLICA (not total)
    # This will be multiplied by actual replica count in cluster computation
    memory_per_replica_gib = _get_memory_gib(
        desires=desires,
        drive=instance.drive or drive,
        replica_count=1,  # Per-replica memory
        block_cache_percent=args.rocksdb_block_cache_percent,
    )
    memory_per_replica_gib = memory_per_replica_gib * memory_buffer.ratio

    # Network calculation (read-only, so only outbound read traffic)
    # Independent of replicas
    needed_network_mbps = simple_network_mbps(desires)

    return CapacityRequirement(
        requirement_type="read-only-kv-regional",
        reference_shape=desires.reference_shape,
        cpu_cores=certain_int(int(needed_cores)),
        mem_gib=certain_float(memory_per_replica_gib),  # Per-replica memory
        disk_gib=certain_float(partition_size_gib * disk_buffer.ratio),
        network_mbps=certain_float(needed_network_mbps),
        context={
            "min_replica_count": args.min_replica_count,
            "total_num_partitions": args.total_num_partitions,
            "unreplicated_data_gib": round(unreplicated_data_gib, 2),
            "partition_size_gib": round(partition_size_gib, 2),
            "raw_cores": round(raw_cores, 2),
            "memory_per_replica_gib": round(memory_per_replica_gib, 2),
            "block_cache_percent": args.rocksdb_block_cache_percent,
            "compute_buffer_ratio": compute_buffer.ratio,
            "disk_buffer_ratio": disk_buffer.ratio,
            "memory_buffer_ratio": memory_buffer.ratio,
        },
    )


def _compute_read_only_kv_regional_cluster(
    instance: Instance,
    requirement: CapacityRequirement,
    args: NflxReadOnlyKVArguments,
) -> Optional[RegionClusterCapacity]:
    """Compute the regional cluster configuration using the partition-aware algorithm.

    Partition-aware algorithm (local disks only):
    1. DISK FIRST: Calculate partitions_per_node based on local disk capacity
    2. Calculate nodes_for_one_copy = total_partitions / partitions_per_node
    3. Start with min_replica_count, calculate initial node count
    4. CHECK CPU & MEMORY: If not satisfied, increase replica_count (uses spare disk)

    Note: Only supports local disks. EBS/attached disk is not supported because:
    - EBS disk is provisioned exactly for data needs (no spare space)
    - The partition-aware algorithm relies on leveraging spare disk capacity

    Args:
        instance: The compute instance being considered (must have local disk)
        requirement: Calculated capacity requirement (with per-replica values)
        args: Read-only KV specific arguments

    Returns:
        RegionClusterCapacity or None if configuration is not viable
    """
    # Only support instances with local disks
    if instance.drive is None:
        return None

    total_needed_cores = int(requirement.cpu_cores.mid)
    total_memory_per_replica_gib = requirement.mem_gib.mid
    partition_size_with_buffer_gib = requirement.disk_gib.mid

    # Step 1 (DISK): Calculate effective disk capacity per node
    effective_disk_per_node = min(instance.drive.size_gib, args.max_data_per_node_gib)

    # Step 2 (DISK): Calculate partitions_per_node
    if partition_size_with_buffer_gib <= 0:
        return None
    partitions_per_node = int(effective_disk_per_node / partition_size_with_buffer_gib)
    if partitions_per_node < 1:
        # This instance type cannot fit even one partition
        return None

    # Step 3 (DISK): Calculate nodes needed for one copy of the dataset
    nodes_for_one_copy = math.ceil(args.total_num_partitions / partitions_per_node)

    # Calculate available memory per node
    available_memory_per_node = instance.ram_gib - args.reserved_memory_gib

    # Step 4: Start with min_replica_count, iterate until CPU & memory satisfied
    replica_count = args.min_replica_count

    while True:
        count = nodes_for_one_copy * replica_count

        # Ensure minimum of 2 nodes for redundancy
        count = max(2, count)

        # Check if count exceeds max cluster size
        if count > args.max_regional_size:
            return None

        # CHECK CPU: Primary constraint after disk
        cpu_satisfied = (count * instance.cpu) >= total_needed_cores

        # CHECK MEMORY: Total memory >= memory_per_replica * replica_count
        total_available_memory = count * available_memory_per_node
        total_needed_memory = total_memory_per_replica_gib * replica_count
        memory_satisfied = total_available_memory >= total_needed_memory

        if cpu_satisfied and memory_satisfied:
            break

        # Not satisfied, increase replicas to add more nodes
        replica_count += 1

    # Calculate cost (local disks only, no EBS cost)
    cost = count * instance.annual_cost

    cluster = RegionClusterCapacity(
        cluster_type="read-only-kv",
        count=count,
        instance=instance,
        attached_drives=tuple(),  # No attached drives
        annual_cost=cost,
    )

    # Add cluster parameters for provisioning
    params = {
        "read-only-kv.replica_count": replica_count,
        "read-only-kv.partitions_per_node": partitions_per_node,
    }
    _upsert_params(cluster, params)

    return cluster


def _estimate_read_only_kv_cluster(  # pylint: disable=unused-argument
    instance: Instance,
    drive: Drive,
    _context: RegionContext,
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
        _context: Regional context (unused - read-only KV is not zone-balanced)
        desires: User's capacity desires
        args: Read-only KV specific arguments

    Returns:
        CapacityPlan or None if configuration is not viable
    """
    # Validate instance constraints (minimum 64GB RAM per instance)
    if instance.cpu < 2 or instance.ram_gib < 64:
        return None

    # Only support instances with local disks
    # EBS not supported: partition-aware algorithm relies on fixed disk capacity
    if instance.drive is None:
        return None

    # Calculate requirements
    requirement = _estimate_read_only_kv_requirement(
        instance=instance,
        drive=instance.drive,  # Use instance's local drive
        desires=desires,
        args=args,
    )

    # Compute cluster
    cluster = _compute_read_only_kv_regional_cluster(
        instance=instance,
        requirement=requirement,
        args=args,
    )

    if cluster is None:
        return None

    # Build cost breakdown
    rokv_costs = {"read-only-kv.regional-cluster": cluster.annual_cost}

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


class NflxReadOnlyKVCapacityModel(CapacityModel):
    """Netflix Read-Only Key-Value Capacity Model.

    A read-only data serving layer that:
    - Loads data from offline sources (data warehouse, batch processing)
    - Serves read traffic with low latency using RocksDB
    - Deploys regionally with configurable replication factor
    - Supports both local (ephemeral) and attached (EBS) drives
    """

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
            _context=context,
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
            default=Buffer(ratio=1.25),
            desired={
                # No background work, just traffic spikes
                "compute": Buffer(
                    ratio=1.5, components=[BufferComponent.compute]
                ),  # 67%
                # Read-only: can run at high disk utilization
                "disk": Buffer(ratio=1.15, components=[BufferComponent.disk]),  # 87%
                # Memory headroom for RocksDB block cache
                "memory": Buffer(ratio=1.2, components=[BufferComponent.memory]),  # 83%
            },
        )

    @staticmethod
    def default_desires(
        user_desires: CapacityDesires, extra_model_arguments: Dict[str, Any]
    ) -> CapacityDesires:
        # Read-only KV only accepts read consistency models
        acceptable_consistency = {
            None,
            AccessConsistency.best_effort,
            AccessConsistency.eventual,
            AccessConsistency.never,
        }
        for key, value in user_desires.query_pattern.access_consistency:
            if value.target_consistency not in acceptable_consistency:
                raise ValueError(
                    f"Read-only KV can only provide "
                    f"{acceptable_consistency} access. User asked for {key}={value}"
                )

        buffers = NflxReadOnlyKVCapacityModel.default_buffers()

        if user_desires.query_pattern.access_pattern == AccessPattern.latency:
            return CapacityDesires(
                query_pattern=QueryPattern(
                    access_pattern=AccessPattern.latency,
                    access_consistency=GlobalConsistency(
                        same_region=Consistency(
                            target_consistency=AccessConsistency.eventual,
                        ),
                        cross_region=Consistency(
                            target_consistency=AccessConsistency.never,
                        ),
                    ),
                    # Read size (items are typically small to medium)
                    estimated_mean_read_size_bytes=Interval(
                        low=128, mid=1024, high=8192, confidence=0.95
                    ),
                    # No writes (read-only)
                    estimated_mean_write_size_bytes=Interval(
                        low=0, mid=0, high=0, confidence=1.0
                    ),
                    # RocksDB point reads are fast when data is in cache
                    estimated_mean_read_latency_ms=Interval(
                        low=0.1, mid=1, high=5, confidence=0.98
                    ),
                    estimated_mean_write_latency_ms=Interval(
                        low=0, mid=0, high=0, confidence=1.0
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
                    write_latency_slo_ms=FixedInterval(
                        low=0, mid=0, high=0, confidence=1.0
                    ),
                ),
                data_shape=DataShape(
                    # Typical dataset size
                    estimated_state_size_gib=Interval(
                        low=10, mid=100, high=1000, confidence=0.98
                    ),
                    # Typical item count
                    estimated_state_item_count=Interval(
                        low=1_000_000,
                        mid=100_000_000,
                        high=10_000_000_000,
                        confidence=0.98,
                    ),
                    # RocksDB uses LZ4/Snappy compression
                    estimated_compression_ratio=Interval(
                        minimum_value=1.1,
                        maximum_value=5,
                        low=1.5,
                        mid=2,
                        high=3,
                        confidence=0.98,
                    ),
                    # Reserved memory for OS and RocksDB overhead
                    reserved_instance_app_mem_gib=4,
                ),
                buffers=buffers,
            )
        else:
            # Throughput pattern - batch/scan operations
            return CapacityDesires(
                query_pattern=QueryPattern(
                    access_pattern=AccessPattern.throughput,
                    access_consistency=GlobalConsistency(
                        same_region=Consistency(
                            target_consistency=AccessConsistency.eventual,
                        ),
                        cross_region=Consistency(
                            target_consistency=AccessConsistency.never,
                        ),
                    ),
                    # Larger reads for throughput pattern (scans/batches)
                    estimated_mean_read_size_bytes=Interval(
                        low=1024, mid=8192, high=65536, confidence=0.95
                    ),
                    estimated_mean_write_size_bytes=Interval(
                        low=0, mid=0, high=0, confidence=1.0
                    ),
                    # Scan operations are slower
                    estimated_mean_read_latency_ms=Interval(
                        low=1, mid=10, high=50, confidence=0.98
                    ),
                    estimated_mean_write_latency_ms=Interval(
                        low=0, mid=0, high=0, confidence=1.0
                    ),
                    # Relaxed SLO for throughput pattern
                    read_latency_slo_ms=FixedInterval(
                        minimum_value=5,
                        maximum_value=500,
                        low=10,
                        mid=50,
                        high=200,
                        confidence=0.98,
                    ),
                    write_latency_slo_ms=FixedInterval(
                        low=0, mid=0, high=0, confidence=1.0
                    ),
                ),
                data_shape=DataShape(
                    # Larger datasets for throughput pattern
                    estimated_state_size_gib=Interval(
                        low=100, mid=1000, high=5000, confidence=0.98
                    ),
                    estimated_state_item_count=Interval(
                        low=10_000_000,
                        mid=1_000_000_000,
                        high=100_000_000_000,
                        confidence=0.98,
                    ),
                    estimated_compression_ratio=Interval(
                        minimum_value=1.1,
                        maximum_value=5,
                        low=1.5,
                        mid=2,
                        high=3,
                        confidence=0.98,
                    ),
                    reserved_instance_app_mem_gib=4,
                ),
                buffers=buffers,
            )


nflx_read_only_kv_capacity_model = NflxReadOnlyKVCapacityModel()
