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
from dataclasses import dataclass
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

from pydantic import BaseModel
from pydantic import Field

from service_capacity_modeling.demand import Amounts
from service_capacity_modeling.demand import cores_for_ncu
from service_capacity_modeling.demand import Demand
from service_capacity_modeling.demand import fit
from service_capacity_modeling.demand import Layout
from service_capacity_modeling.demand import NodeRules
from service_capacity_modeling.demand import Resource
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
from service_capacity_modeling.interface import default_reference_shape
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
from service_capacity_modeling.models.common import EFFECTIVE_DISK_PER_NODE_GIB
from service_capacity_modeling.models.common import upsert_params
from service_capacity_modeling.models.common import normalize_cores
from service_capacity_modeling.models.common import normalize_cores_float
from service_capacity_modeling.models.common import simple_network_mbps
from service_capacity_modeling.models.common import sqrt_staffed_cores
from service_capacity_modeling.models.org.netflix.partition_aware_algorithm import (
    CapacityProblem,
    CapacityResult,
    max_partitions_per_node,
    plan_at_layout,
    replicas_needed,
    search_for_min_nodes,
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


# TODO: Data-block working set estimation remains disabled because
# working_set_from_drive_and_slo does not work well for large OODM datasets.
# Resident SST table-reader memory is modeled separately as a percentage of data.

# Layout keys a fallback must reproduce exactly to host the same placement.
_LAYOUT_KEYS = (
    "read-only-kv.replica_count",
    "read-only-kv.partitions_per_node",
    "read-only-kv.nodes_for_one_copy",
)


def _sst_memory_settings(
    desires: CapacityDesires, args: NflxReadOnlyKVArguments
) -> Optional[Tuple[float, float]]:
    """(sst_memory_overhead_ratio, memory_buffer_ratio), or None if invalid."""
    has_explicit_memory_buffer = any(
        BufferComponent.memory in buffer.components
        for buffer in desires.buffers.desired.values()
    )
    if args.sst_memory_overhead_ratio <= 0 or not has_explicit_memory_buffer:
        return 0.0, 1.0
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
    return args.sst_memory_overhead_ratio, memory_buffer.ratio


def _reject_shape(instance: Instance, drive: Drive) -> Optional[str]:
    del drive
    # Minimum 30 GiB RAM: allows 32 GiB-class shapes in the hardware catalog
    # while keeping enough memory for RocksDB block cache and working set.
    if instance.cpu < 2 or instance.ram_gib < 30:
        return "Too small for read-only KV"
    # EBS not supported: partition-aware algorithm relies on fixed disk capacity
    if instance.drive is None:
        return "Read-only KV requires local disks"
    return None


@dataclass(frozen=True)
class _PartitionLayout:
    """Places partitions on nodes for one shape; the read-only KV layout rule."""

    args: NflxReadOnlyKVArguments
    memory_placement: bool
    # The model's own CPU demand, kept in whole cores of the shape it was
    # measured on so an unchanged demand converts to cores exactly as the
    # model always has. A caller-supplied NCU converts with cores_for_ncu.
    ncu: float
    reference_cores: int
    reference_shape: Instance

    def __call__(
        self,
        instance: Instance,
        need: Amounts,
        have: Amounts,
        pinned: Optional[Layout],
    ) -> Union[Layout, str]:
        partition_gib = need[Resource.disk_gib] / self.args.total_num_partitions
        memory_per_partition = need[Resource.mem_gib] / self.args.total_num_partitions
        # CPU is placed in whole cores of this shape.
        if need[Resource.ncu] == self.ncu:
            cpu_needed = normalize_cores(
                self.reference_cores, instance, self.reference_shape
            )
        else:
            cpu_needed = cores_for_ncu(need[Resource.ncu], instance)
        problem = CapacityProblem(
            n_partitions=self.args.total_num_partitions,
            partition_size_gib=partition_gib,
            disk_per_node_gib=have[Resource.disk_gib],
            memory_per_partition_gib=(
                memory_per_partition if self.memory_placement else None
            ),
            memory_per_node_gib=(
                have[Resource.mem_gib] if self.memory_placement else None
            ),
            cpu_per_node=instance.cpu,
            cpu_needed=cpu_needed,
            min_rf=self.args.min_replica_count,
            max_nodes=self.args.max_regional_size,
        )
        if pinned is not None:
            return _host_pinned(problem, pinned)
        result = search_for_min_nodes(problem)
        if result is None:
            return "No partition placement fits"
        params: Dict[str, Any] = {
            "read-only-kv.replica_count": result.replica_count,
            "read-only-kv.partitions_per_node": result.partitions_per_node,
            "read-only-kv.nodes_for_one_copy": result.nodes_for_one_copy,
            "read-only-kv.nodes_for_cpu": math.ceil(cpu_needed / instance.cpu),
            "read-only-kv.cpu_needed": cpu_needed,
            "read-only-kv.max_partitions_per_node_by_disk": (
                result.max_partitions_per_node_by_disk
            ),
            "read-only-kv.max_partitions_per_node_by_memory": (
                result.max_partitions_per_node_by_memory
            ),
        }
        return Layout(
            count=result.node_count,
            params=params,
            need_per_node=_carried_per_node(problem, result.partitions_per_node),
        )


def _carried_per_node(problem: CapacityProblem, ppn: int) -> Dict[Resource, float]:
    """Disk and memory one node holds. ``ppn`` can exceed the partition count
    when one node fits every partition, so a node holds at most all of them."""
    placed = min(ppn, problem.n_partitions)
    return {
        Resource.disk_gib: placed * problem.partition_size_gib,
        Resource.mem_gib: placed * (problem.memory_per_partition_gib or 0.0),
    }


def _host_pinned(problem: CapacityProblem, pinned: Layout) -> Union[Layout, str]:
    """Whether this shape can host the pinned layout, under the same rules as
    the search. A pin with only a count gets the placement that fits it."""
    ppn = pinned.params.get("read-only-kv.partitions_per_node")
    rf = pinned.params.get("read-only-kv.replica_count")
    if ppn is None or rf is None:
        result = _place_at_count(problem, pinned.count)
        if result is None:
            return f"No partition placement fits {pinned.count} nodes"
    else:
        result = plan_at_layout(problem, int(ppn), int(rf))
        if result is None or result.node_count != pinned.count:
            return (
                f"Cannot host the layout: {_short_resource(problem, int(ppn), int(rf))}"
            )
    return Layout(
        count=result.node_count,
        params={
            "read-only-kv.replica_count": result.replica_count,
            "read-only-kv.partitions_per_node": result.partitions_per_node,
            "read-only-kv.nodes_for_one_copy": result.nodes_for_one_copy,
        },
        need_per_node=_carried_per_node(problem, result.partitions_per_node),
    )


def _place_at_count(problem: CapacityProblem, count: int) -> Optional[CapacityResult]:
    """The placement on exactly ``count`` nodes with the fewest nodes per copy,
    which leaves the most replicas for CPU and availability."""
    for nodes_for_one_copy in range(1, count + 1):
        if count % nodes_for_one_copy:
            continue
        ppn = math.ceil(problem.n_partitions / nodes_for_one_copy)
        result = plan_at_layout(problem, ppn, count // nodes_for_one_copy)
        if result is not None and result.node_count == count:
            return result
    return None


def _short_resource(problem: CapacityProblem, ppn: int, rf: int) -> str:
    """Why a shape cannot host ``ppn`` partitions per node at ``rf`` replicas."""
    disk_ppn, memory_ppn = max_partitions_per_node(problem)
    if ppn > disk_ppn:
        return str(Resource.disk_gib)
    if memory_ppn is not None and ppn > memory_ppn:
        return str(Resource.mem_gib)
    if rf < replicas_needed(problem, math.ceil(problem.n_partitions / ppn)):
        return str(Resource.ncu)
    return "node count"


def read_only_kv_demand(
    desires: CapacityDesires, args: NflxReadOnlyKVArguments
) -> Optional[Tuple[Demand, NodeRules]]:
    """Demand for one complete data copy, plus the partition layout rule.

    The replica count is an output: the layout rule places partitions and adds
    replicas for CPU. Disk and memory demand are per copy.
    """
    unreplicated_data_gib = desires.data_shape.estimated_state_size_gib.mid
    if unreplicated_data_gib <= 0:
        return None
    memory = _sst_memory_settings(desires, args)
    if memory is None:
        return None
    sst_memory_overhead_ratio, memory_buffer_ratio = memory

    compute_buffer = buffer_for_components(
        buffers=desires.buffers, components=[BufferComponent.cpu]
    )
    disk_buffer = buffer_for_components(
        buffers=desires.buffers, components=[BufferComponent.disk]
    )
    # CPU uses the sqrt staffing model (independent of replicas). Apply the
    # buffer in whole reference cores, then express them as NCU.
    reference_cores = math.ceil(sqrt_staffed_cores(desires) * compute_buffer.ratio)
    ncu = normalize_cores_float(
        core_count=reference_cores,
        target_shape=default_reference_shape,
        reference_shape=desires.reference_shape,
    )
    reserved_memory_gib = max(
        args.reserved_memory_gib,
        desires.data_shape.reserved_instance_app_mem_gib
        + desires.data_shape.reserved_instance_system_mem_gib,
    )
    demand = Demand(
        ncu=certain_float(ncu),
        mem_gib=certain_float(
            unreplicated_data_gib * sst_memory_overhead_ratio * memory_buffer_ratio
        ),
        disk_gib=certain_float(unreplicated_data_gib * disk_buffer.ratio),
        network_mbps=certain_float(simple_network_mbps(desires)),
    )
    rules = NodeRules(
        reserve_mem_gib=lambda _: reserved_memory_gib,
        max_data_per_node_gib=args.max_data_per_node_gib * disk_buffer.ratio,
        rejects=_reject_shape,
        layout=_PartitionLayout(
            args=args,
            memory_placement=sst_memory_overhead_ratio > 0,
            ncu=ncu,
            reference_cores=reference_cores,
            reference_shape=desires.reference_shape,
        ),
    )
    return demand, rules


def _estimate_read_only_kv_cluster(
    instance: Instance,
    desires: CapacityDesires,
    args: NflxReadOnlyKVArguments,
) -> Union[CapacityPlan, Excuse, None]:
    """Size one shape from the model's demand and build its plan."""
    stated = read_only_kv_demand(desires, args)
    if stated is None:
        return None
    demand, rules = stated
    # Local disks only: size against the instance's own drive, not the
    # attached drive the planner pairs it with.
    if instance.drive is None:
        return Excuse(
            instance=instance.name,
            drive="none",
            reason="Read-only KV requires local disks",
        )
    result = fit(demand, rules, instance, instance.drive)
    if isinstance(result, Excuse):
        return result

    params = result.layout.params
    replica_count = int(params["read-only-kv.replica_count"])
    cpu_needed = int(params["read-only-kv.cpu_needed"])
    memory_placement = demand.mem_gib.mid > 0
    sst_memory_settings = _sst_memory_settings(desires, args)
    assert sst_memory_settings is not None
    sst_memory_overhead_ratio, memory_buffer_ratio = sst_memory_settings

    cluster = RegionClusterCapacity(
        cluster_type="read-only-kv",
        count=result.count,
        instance=instance,
        attached_drives=tuple(),  # No attached drives
    )
    cluster_params: Dict[str, Any] = {
        "read-only-kv.total_num_partitions": args.total_num_partitions,
        "read-only-kv.min_replica_count": args.min_replica_count,
        "read-only-kv.replica_count": replica_count,
        "read-only-kv.partitions_per_node": params["read-only-kv.partitions_per_node"],
        "read-only-kv.nodes_for_one_copy": params["read-only-kv.nodes_for_one_copy"],
        "read-only-kv.nodes_for_cpu": params["read-only-kv.nodes_for_cpu"],
        EFFECTIVE_DISK_PER_NODE_GIB: result.supply_per_node[Resource.disk_gib],
    }
    if memory_placement:
        per_copy = demand.mem_gib.mid
        cluster_params.update(
            {
                "read-only-kv.sst_memory_placement_enabled": True,
                "read-only-kv.sst_memory_overhead_ratio": sst_memory_overhead_ratio,
                "read-only-kv.memory_buffer_ratio": memory_buffer_ratio,
                "read-only-kv.reserved_memory_per_node_gib": rules.reserve_mem_gib(
                    instance
                ),
                "read-only-kv.buffered_sst_memory_per_partition_gib": (
                    per_copy / args.total_num_partitions
                ),
                "read-only-kv.estimated_sst_memory_per_copy_gib": (
                    per_copy / memory_buffer_ratio
                ),
                "read-only-kv.buffered_sst_memory_per_copy_gib": per_copy,
                "read-only-kv.total_buffered_sst_memory_gib": per_copy * replica_count,
                "read-only-kv.available_sst_memory_per_node_gib": (
                    result.supply_per_node[Resource.mem_gib]
                ),
                "read-only-kv.max_partitions_per_node_by_disk": params[
                    "read-only-kv.max_partitions_per_node_by_disk"
                ],
                "read-only-kv.max_partitions_per_node_by_memory": params[
                    "read-only-kv.max_partitions_per_node_by_memory"
                ],
            }
        )
    upsert_params(cluster, cluster_params)

    penalties = _compute_penalties(
        instance=instance,
        large_instance_regret=args.large_instance_regret,
    )
    if penalties:
        upsert_params(cluster, {RANK_PENALTIES: penalties})

    regional_requirement = CapacityRequirement(
        requirement_type=NflxReadOnlyKVCapacityModel.cluster_type,
        reference_shape=desires.reference_shape,
        cpu_cores=certain_int(cpu_needed),
        mem_gib=demand.mem_gib.scale(replica_count),
        disk_gib=certain_float(0),  # Disk computed via partition-aware algorithm
        network_mbps=demand.network_mbps,
        context={},
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
    ) -> Union[CapacityPlan, Excuse, None]:
        args = NflxReadOnlyKVArguments.model_validate(extra_model_arguments)

        # drive and context are unused: read-only KV uses instance.drive
        # (local disks only) and is not zone-balanced.
        _ = (drive, context)

        return _estimate_read_only_kv_cluster(
            instance=instance,
            desires=desires,
            args=args,
        )

    @staticmethod
    def demand(
        drive: Drive,
        context: RegionContext,
        desires: CapacityDesires,
        extra_model_arguments: Dict[str, Any],
    ) -> Optional[Tuple[Demand, NodeRules]]:
        _ = (drive, context)
        args = NflxReadOnlyKVArguments.model_validate(extra_model_arguments)
        return read_only_kv_demand(desires, args)

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
