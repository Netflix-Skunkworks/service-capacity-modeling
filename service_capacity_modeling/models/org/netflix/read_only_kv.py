"""
Netflix Read-Only Key-Value Capacity Model

A read-only data serving layer that loads data from offline sources
(e.g., S3) and serves read traffic online.

Key characteristics:
- Regional deployment (not zonal)
- Read-only after data population is complete
- Uses RocksDB as the storage backend
- Local disks preferred (EBS optional)
- RF=2 by default for redundancy
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
from service_capacity_modeling.models.utils import next_n
from service_capacity_modeling.stats import dist_for_interval

logger = logging.getLogger(__name__)


class NflxReadOnlyKVArguments(BaseModel):
    """Configuration arguments for the Netflix Read-Only KV capacity model."""

    replica_count: int = Field(
        default=2,
        description="Number of data replicas (RF). Default is 2 for redundancy.",
        ge=1,
        le=5,
    )
    require_local_disks: bool = Field(
        default=True,
        description="If local (ephemeral) drives are required. "
        "Set to False to allow EBS/attached drives.",
    )
    require_attached_disks: bool = Field(
        default=False,
        description="If attached (EBS) drives are required. "
        "Set to True to force EBS usage.",
    )
    max_regional_size: int = Field(
        default=256,
        description="Maximum number of instances in the regional cluster.",
    )
    max_local_data_per_node_gib: int = Field(
        default=1500,
        description="Maximum data per node for local disk instances (GiB). "
        "Prevents overloading nodes with too much data.",
    )
    max_attached_data_per_node_gib: int = Field(
        default=2048,
        description="Maximum data per node for attached disk instances (GiB).",
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

    # Calculate raw data size (before buffer)
    raw_data_size_gib = _get_data_size_gib(desires, args.replica_count)
    # Apply disk buffer
    data_size_gib = raw_data_size_gib * disk_buffer.ratio

    # CPU calculation using sqrt staffing model
    raw_cores = sqrt_staffed_cores(desires)
    raw_cores = normalize_cores(
        core_count=raw_cores,
        target_shape=instance,
        reference_shape=desires.reference_shape,
    )
    # Apply compute buffer
    needed_cores = raw_cores * compute_buffer.ratio

    # Memory calculation
    raw_memory_gib = _get_memory_gib(
        desires=desires,
        drive=instance.drive or drive,
        replica_count=args.replica_count,
        block_cache_percent=args.rocksdb_block_cache_percent,
    )
    needed_memory_gib = raw_memory_gib * memory_buffer.ratio

    # Network calculation (read-only, so only outbound read traffic)
    needed_network_mbps = simple_network_mbps(desires)

    return CapacityRequirement(
        requirement_type="read-only-kv-regional",
        reference_shape=desires.reference_shape,
        cpu_cores=certain_int(int(needed_cores)),
        mem_gib=certain_float(needed_memory_gib),
        disk_gib=certain_float(data_size_gib),
        network_mbps=certain_float(needed_network_mbps),
        context={
            "replica_count": args.replica_count,
            "raw_cores": round(raw_cores, 2),
            "raw_data_size_gib": round(raw_data_size_gib, 2),
            "data_size_gib": round(data_size_gib, 2),
            "raw_memory_gib": round(raw_memory_gib, 2),
            "block_cache_percent": args.rocksdb_block_cache_percent,
            "compute_buffer_ratio": compute_buffer.ratio,
            "disk_buffer_ratio": disk_buffer.ratio,
            "memory_buffer_ratio": memory_buffer.ratio,
        },
    )


def _compute_read_only_kv_regional_cluster(
    instance: Instance,
    drive: Drive,
    requirement: CapacityRequirement,
    args: NflxReadOnlyKVArguments,
) -> Optional[RegionClusterCapacity]:
    """Compute the regional cluster configuration for read-only KV.

    Args:
        instance: The compute instance being considered
        drive: The drive configuration
        requirement: Calculated capacity requirement
        args: Read-only KV specific arguments
    Returns:
        RegionClusterCapacity or None if configuration is not viable
    """
    needed_cores = int(requirement.cpu_cores.mid)
    needed_memory_gib = requirement.mem_gib.mid
    needed_disk_gib = requirement.disk_gib.mid
    needed_network_mbps = requirement.network_mbps.mid

    # Calculate available memory per instance (minus reserved)
    available_memory_per_instance = max(0, instance.ram_gib - args.reserved_memory_gib)
    if available_memory_per_instance <= 0:
        return None

    # Calculate count based on CPU
    count = max(2, math.ceil(needed_cores / instance.cpu))

    # Adjust count based on memory
    count = max(count, math.ceil(needed_memory_gib / available_memory_per_instance))

    # Adjust count based on network
    count = max(count, math.ceil(needed_network_mbps / instance.net_mbps))

    # Adjust count based on disk
    if instance.drive is not None:
        # Local disk
        max_data_per_node: float = min(
            instance.drive.size_gib, args.max_local_data_per_node_gib
        )
        count = max(count, math.ceil(needed_disk_gib / max_data_per_node))
    else:
        # Attached disk (EBS)
        max_data_per_node = min(drive.max_size_gib, args.max_attached_data_per_node_gib)
        count = max(count, math.ceil(needed_disk_gib / max_data_per_node))

    # Check max cluster size
    if count > args.max_regional_size:
        return None

    # Calculate cost
    cost = count * instance.annual_cost

    # Handle attached drives for EBS instances
    attached_drives: Tuple[Drive, ...] = tuple()
    if instance.drive is None and needed_disk_gib > 0:
        disk_per_node: float = math.ceil(needed_disk_gib / count)
        # Round up to 100 GiB increments for EBS
        disk_per_node = next_n(disk_per_node, 100)
        disk_per_node = min(disk_per_node, drive.max_size_gib)

        attached_drive = drive.model_copy()
        attached_drive.size_gib = int(disk_per_node)
        attached_drives = (attached_drive,)
        cost += attached_drive.annual_cost * count

    return RegionClusterCapacity(
        cluster_type="read-only-kv",
        count=count,
        instance=instance,
        attached_drives=attached_drives,
        annual_cost=cost,
    )


def _estimate_read_only_kv_cluster(  # pylint: disable=unused-argument
    instance: Instance,
    drive: Drive,
    _context: RegionContext,
    desires: CapacityDesires,
    args: NflxReadOnlyKVArguments,
) -> Optional[CapacityPlan]:
    """Main function to estimate read-only KV cluster configuration.

    Args:
        instance: The compute instance being considered
        drive: The drive configuration
        _context: Regional context (unused - read-only KV is not zone-balanced)
        desires: User's capacity desires
        args: Read-only KV specific arguments

    Returns:
        CapacityPlan or None if configuration is not viable
    """
    # Validate instance constraints
    if instance.cpu < 2 or instance.ram_gib < 8:
        return None

    # Filter based on disk requirements
    if instance.drive is None and args.require_local_disks:
        return None
    if instance.drive is not None and args.require_attached_disks:
        return None

    # Read-only KV supports gp2 and gp3 for attached drives
    if instance.drive is None and drive.name not in ("gp2", "gp3"):
        return None

    # Calculate requirements
    requirement = _estimate_read_only_kv_requirement(
        instance=instance,
        drive=drive,
        desires=desires,
        args=args,
    )

    # Compute cluster
    cluster = _compute_read_only_kv_regional_cluster(
        instance=instance,
        drive=drive,
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
                    ratio=1.25, components=[BufferComponent.compute]
                ),  # 80%
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
