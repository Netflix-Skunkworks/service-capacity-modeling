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
from typing import Union

from pydantic import BaseModel
from pydantic import Field
from pydantic import model_validator

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
from service_capacity_modeling.interface import Excuse
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
from service_capacity_modeling.models.common import cpu_to_ncu
from service_capacity_modeling.models.common import EFFECTIVE_DISK_PER_NODE_GIB
from service_capacity_modeling.models.common import get_effective_disk_per_node_gib
from service_capacity_modeling.models.common import upsert_params
from service_capacity_modeling.models.common import normalize_cores
from service_capacity_modeling.models.common import simple_network_mbps
from service_capacity_modeling.models.common import sqrt_staffed_cores
from service_capacity_modeling.models.common import usable_memory_gib
from service_capacity_modeling.models.org.netflix.partition_aware_algorithm import (
    Layout,
    NodeSupply,
    PartitionedRequirement,
    place,
    Placement,
    Shortfall,
)

logger = logging.getLogger(__name__)


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
        "per complete data copy, as a ratio of SST data. Applied only when a "
        "desired buffer explicitly targets memory.",
        ge=0,
        le=1,
    )
    large_instance_regret: float = Field(
        default=0.2,
        description="Graduated cost penalty for instance sizes above 4xlarge. "
        "Adds penalty * max(0, (normalized_size - 4) / 4) * cost to the "
        "effective sort cost. Set to 0 to disable.",
    )
    required_cluster_size: Optional[int] = Field(
        default=None,
        ge=1,
        description="Plan exactly this many nodes instead of the fewest. With "
        "no pinned layout, uses the densest partition layout that fits on "
        "that count. Shapes that cannot host it are excused.",
    )
    partitions_per_node: Optional[int] = Field(
        default=None,
        ge=1,
        description="Pin the partition layout: partitions on each node. Set "
        "with replica_count, for example to check that another instance type "
        "can host an existing cluster's layout.",
    )
    replica_count: Optional[int] = Field(
        default=None,
        ge=1,
        description="Pin the partition layout: complete data copies. Set "
        "with partitions_per_node.",
    )
    max_partitions_per_node: Optional[int] = Field(
        default=None,
        ge=1,
        description="Most partitions any node may hold, on top of the disk "
        "and memory limits.",
    )
    needed_cores: Optional[float] = Field(
        default=None,
        gt=0,
        description="CPU need in cores of desires.reference_shape, replacing "
        "the estimate from the query pattern, for example an autoscaler's own "
        "minimum. Compute buffers are not applied to it.",
    )

    @model_validator(mode="after")
    def layout_pin_is_complete(self) -> "NflxReadOnlyKVArguments":
        """A pinned layout needs both its density and its replica count."""
        if (self.partitions_per_node is None) != (self.replica_count is None):
            raise ValueError(
                "partitions_per_node and replica_count must be set together"
            )
        return self


# TODO: Data-block working set estimation remains disabled because
# working_set_from_drive_and_slo does not work well for large OODM datasets.
# Resident SST table-reader memory is modeled separately as a percentage of data.


class _MemoryPlacement(BaseModel):
    """SST memory placement, on only when a desired buffer targets memory."""

    sst_memory_overhead_ratio: float = 0.0
    memory_buffer_ratio: float = 1.0

    @property
    def enabled(self) -> bool:
        return self.sst_memory_overhead_ratio > 0


def _memory_placement(
    desires: CapacityDesires, args: NflxReadOnlyKVArguments
) -> Optional[_MemoryPlacement]:
    """The SST memory terms, or None when the memory buffer is not positive."""
    has_explicit_memory_buffer = any(
        BufferComponent.memory in buffer.components
        for buffer in desires.buffers.desired.values()
    )
    if args.sst_memory_overhead_ratio <= 0 or not has_explicit_memory_buffer:
        return _MemoryPlacement()
    # WFS buffer names are decorative and may change during conversion. An
    # explicit memory component is the durable opt-in signal. Do not inherit
    # generic storage headroom: disk-only placement remains the cost-optimal
    # default until a workload asks the model to reserve SST metadata memory.
    memory_buffer = buffer_for_components(
        buffers=desires.buffers,
        components=[BufferComponent.memory],
        component_fallbacks={},
    )
    if memory_buffer.ratio <= 0:
        return None
    return _MemoryPlacement(
        sst_memory_overhead_ratio=args.sst_memory_overhead_ratio,
        memory_buffer_ratio=memory_buffer.ratio,
    )


def read_only_kv_requirement(
    desires: CapacityDesires, args: NflxReadOnlyKVArguments
) -> Optional[PartitionedRequirement]:
    """What the dataset needs, on no particular shape.

    Pass it to ``place`` with ``read_only_kv_node(instance, ...)`` for an EC2
    shape, or with a ``NodeSupply`` for any other node, such as a container
    size. CPU is in NCUs (normalized compute units). ``desires`` must
    already include the model's defaults, as ``capacity_plan`` receives them.
    Returns None when there is no data or the memory buffer is not positive.
    """
    data_gib = desires.data_shape.estimated_state_size_gib.mid
    memory = _memory_placement(desires, args)
    if data_gib <= 0 or memory is None:
        return None

    cpu_cores = args.needed_cores
    if cpu_cores is None:
        compute_buffer = buffer_for_components(
            buffers=desires.buffers, components=[BufferComponent.cpu]
        )
        # Sqrt staffing, independent of replicas: any copy serves a read.
        cpu_cores = math.ceil(sqrt_staffed_cores(desires) * compute_buffer.ratio)

    disk_buffer = buffer_for_components(
        buffers=desires.buffers, components=[BufferComponent.disk]
    )
    partitions = args.total_num_partitions
    memory_per_partition = None
    if memory.enabled:
        memory_per_partition = (
            data_gib
            * memory.sst_memory_overhead_ratio
            * memory.memory_buffer_ratio
            / partitions
        )
    return PartitionedRequirement(
        partitions=partitions,
        disk_gib_per_partition=data_gib / partitions * disk_buffer.ratio,
        memory_gib_per_partition=memory_per_partition,
        cpu_ncu=cpu_to_ncu(cpu_cores, desires.reference_shape),
        min_replicas=args.min_replica_count,
        max_nodes=args.max_regional_size,
        max_partitions_per_node=args.max_partitions_per_node,
    )


def _reserved_memory_gib(
    desires: CapacityDesires, args: NflxReadOnlyKVArguments
) -> float:
    return max(
        args.reserved_memory_gib,
        desires.data_shape.reserved_instance_app_mem_gib
        + desires.data_shape.reserved_instance_system_mem_gib,
    )


def read_only_kv_node(
    instance: Instance, desires: CapacityDesires, args: NflxReadOnlyKVArguments
) -> Optional[NodeSupply]:
    """What one node of ``instance`` offers the dataset, or None for shapes
    the model does not run on.

    Needs at least 2 cores and 30 GiB of RAM (which admits 32 GiB-class
    shapes while keeping room for the RocksDB block cache), and local disks:
    placement relies on a fixed disk per node, so attached disks are out.
    """
    if instance.cpu < 2 or instance.ram_gib < 30 or instance.drive is None:
        return None
    disk_buffer = buffer_for_components(
        buffers=desires.buffers, components=[BufferComponent.disk]
    )
    reserved_gib = _reserved_memory_gib(desires, args)
    return NodeSupply(
        cpu_ncu=cpu_to_ncu(instance.cpu, instance),
        disk_gib=get_effective_disk_per_node_gib(
            instance=instance,
            drive=instance.drive,
            disk_buffer_ratio=disk_buffer.ratio,
            max_local_data_per_node_gib=args.max_data_per_node_gib,
        ),
        memory_gib=max(0.0, usable_memory_gib(instance, lambda _: reserved_gib)),
    )


def _cluster_params(
    requirement: PartitionedRequirement,
    node: NodeSupply,
    placed: Placement,
    memory: _MemoryPlacement,
    reserved_memory_gib: float,
) -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "read-only-kv.total_num_partitions": requirement.partitions,
        "read-only-kv.min_replica_count": requirement.min_replicas,
        "read-only-kv.replica_count": placed.replica_count,
        "read-only-kv.partitions_per_node": placed.partitions_per_node,
        "read-only-kv.nodes_for_one_copy": placed.nodes_for_one_copy,
        "read-only-kv.nodes_for_cpu": placed.nodes_for_cpu,
        # What the placement uses on each node.
        "read-only-kv.disk_gib_per_node": placed.disk_gib_per_node,
        "read-only-kv.cpu_utilization": placed.cpu_utilization,
        EFFECTIVE_DISK_PER_NODE_GIB: node.disk_gib,
    }
    if memory.enabled:
        assert requirement.memory_gib_per_partition is not None
        per_copy_gib = requirement.memory_gib_per_partition * requirement.partitions
        params.update(
            {
                "read-only-kv.memory_gib_per_node": placed.memory_gib_per_node,
                "read-only-kv.sst_memory_placement_enabled": True,
                "read-only-kv.sst_memory_overhead_ratio": (
                    memory.sst_memory_overhead_ratio
                ),
                "read-only-kv.memory_buffer_ratio": memory.memory_buffer_ratio,
                "read-only-kv.reserved_memory_per_node_gib": reserved_memory_gib,
                "read-only-kv.buffered_sst_memory_per_partition_gib": (
                    requirement.memory_gib_per_partition
                ),
                "read-only-kv.estimated_sst_memory_per_copy_gib": (
                    per_copy_gib / memory.memory_buffer_ratio
                ),
                "read-only-kv.buffered_sst_memory_per_copy_gib": per_copy_gib,
                "read-only-kv.total_buffered_sst_memory_gib": (
                    per_copy_gib * placed.replica_count
                ),
                "read-only-kv.available_sst_memory_per_node_gib": node.memory_gib,
                "read-only-kv.max_partitions_per_node_by_disk": (
                    placed.max_partitions_per_node_by_disk
                ),
                "read-only-kv.max_partitions_per_node_by_memory": (
                    placed.max_partitions_per_node_by_memory
                ),
            }
        )
    return params


def _pins(args: NflxReadOnlyKVArguments) -> Optional[Layout]:
    if args.partitions_per_node is None or args.replica_count is None:
        return None
    return Layout(
        partitions_per_node=args.partitions_per_node,
        replica_count=args.replica_count,
    )


def _estimate_read_only_kv_cluster(
    instance: Instance,
    desires: CapacityDesires,
    args: NflxReadOnlyKVArguments,
) -> Union[CapacityPlan, Excuse, None]:
    """Place the dataset's requirement on ``instance``.

    Returns the plan, an Excuse when a pinned layout or node count does not
    fit, or None when the shape is not viable or nothing fits unpinned.
    """
    requirement = read_only_kv_requirement(desires, args)
    node = read_only_kv_node(instance, desires, args)
    memory = _memory_placement(desires, args)
    if requirement is None or node is None or memory is None:
        return None
    assert instance.drive is not None  # read_only_kv_node requires local disks

    layout = _pins(args)
    placed = place(requirement, node, layout=layout, count=args.required_cluster_size)
    if isinstance(placed, Shortfall):
        if layout is None and args.required_cluster_size is None:
            return None
        return Excuse(
            instance=instance.name,
            drive=instance.drive.name,
            reason=placed.reason,
            bottleneck=placed.bottleneck,
        )

    cluster = RegionClusterCapacity(
        cluster_type="read-only-kv",
        count=placed.node_count,
        instance=instance,
        attached_drives=tuple(),  # No attached drives
    )
    reserved_gib = _reserved_memory_gib(desires, args)
    upsert_params(
        cluster, _cluster_params(requirement, node, placed, memory, reserved_gib)
    )
    penalties = _compute_penalties(
        instance=instance,
        large_instance_regret=args.large_instance_regret,
    )
    if penalties:
        upsert_params(cluster, {RANK_PENALTIES: penalties})

    per_copy_memory_gib = (
        requirement.memory_gib_per_partition or 0
    ) * requirement.partitions
    regional_requirement = CapacityRequirement(
        requirement_type=NflxReadOnlyKVCapacityModel.cluster_type,
        reference_shape=desires.reference_shape,
        cpu_cores=certain_int(
            normalize_cores(
                core_count=requirement.cpu_ncu,
                target_shape=instance,
            )
        ),
        mem_gib=certain_float(per_copy_memory_gib * placed.replica_count),
        disk_gib=certain_float(0),  # Disk is placed by partition, not summed
        network_mbps=certain_float(simple_network_mbps(desires)),
        context={},
    )

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
    ) -> Union[CapacityPlan, Excuse, None]:
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
