import math
from decimal import Decimal
from typing import Any
from typing import Dict
from typing import FrozenSet
from typing import NamedTuple
from typing import Optional
from typing import Tuple

from pydantic import BaseModel
from pydantic import Field
from pydantic import model_validator

from service_capacity_modeling.interface import AccessConsistency
from service_capacity_modeling.interface import AccessPattern
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
from service_capacity_modeling.interface import GIB_IN_BYTES
from service_capacity_modeling.interface import GlobalConsistency
from service_capacity_modeling.interface import Instance
from service_capacity_modeling.interface import Platform
from service_capacity_modeling.interface import QueryPattern
from service_capacity_modeling.interface import RegionClusterCapacity
from service_capacity_modeling.interface import RegionContext
from service_capacity_modeling.interface import Requirements
from service_capacity_modeling.models import CapacityModel
from service_capacity_modeling.models.common import simple_network_mbps
from service_capacity_modeling.models.org.netflix.iso_date_math import iso_to_seconds


VALKEY_MIN_CPU_SPEED_GHZ = 2.6
VALKEY_MAX_CPU_SPEED_GHZ = 3.5
VALKEY_MIN_OPS_PER_SECOND = 700_000
VALKEY_MAX_OPS_PER_SECOND = 1_000_000
VALKEY_MIN_LUA_OPS_PER_SECOND = 50_000
VALKEY_MAX_LUA_OPS_PER_SECOND = 150_000
VALKEY_DURABILITY_THRESHOLD = 1_000
VALKEY_AWS_MEMORY_RESERVATION = 0.25
VALKEY_MAX_READ_REPLICAS_PER_SHARD = 5
VALKEY_ENGINE_VERSION = "8.1"
VALKEY_ITEM_METADATA_BYTES = 31
VALKEY_DEFAULT_KEY_SIZE_BYTES = 16
VALKEY_DEFAULT_VALUE_SIZE_BYTES = 50
VALKEY_MAX_MEASURED_OVERHEAD_VALUE_BYTES = 128
VALKEY_DEFAULT_MAX_NODES_PER_CLUSTER = 90


def _valkey_ops_per_second(instance: Instance, use_lua: bool) -> int:
    """Estimate one shard's throughput from its single-thread CPU speed."""
    speed_ratio = min(
        1.0,
        max(
            0.0,
            (instance.cpu_ghz - VALKEY_MIN_CPU_SPEED_GHZ)
            / (VALKEY_MAX_CPU_SPEED_GHZ - VALKEY_MIN_CPU_SPEED_GHZ),
        ),
    )
    if use_lua:
        low, high = VALKEY_MIN_LUA_OPS_PER_SECOND, VALKEY_MAX_LUA_OPS_PER_SECOND
    else:
        low, high = VALKEY_MIN_OPS_PER_SECOND, VALKEY_MAX_OPS_PER_SECOND
    return round(low + speed_ratio * (high - low))


def _jemalloc_size_class(requested_bytes: int) -> int:
    """Return the allocator size class, extrapolating beyond the measured curve."""
    quantum = 8
    upper_bound = 64
    while requested_bytes > upper_bound:
        quantum *= 2
        upper_bound *= 2
    return max(quantum, math.ceil(requested_bytes / quantum) * quantum)


def _valkey_item_overhead_bytes(
    key_size_bytes: int,
    value_size_bytes: int,
) -> int:
    """Estimate Valkey 8.1 metadata and allocator slack for one item."""
    key_allocation = _jemalloc_size_class(key_size_bytes + 4)
    key_allocation_slack = key_allocation - key_size_bytes
    if value_size_bytes < 32:
        # The 8.1 curve has a separate compact-value branch below 32 bytes.
        baseline_16_byte_key_slack = 8
        return (
            VALKEY_ITEM_METADATA_BYTES
            + _jemalloc_size_class(value_size_bytes + 7)
            - value_size_bytes
            + key_allocation_slack
            - baseline_16_byte_key_slack
        )
    value_allocation = _jemalloc_size_class(value_size_bytes + 4)
    return (
        VALKEY_ITEM_METADATA_BYTES
        + key_allocation_slack
        + value_allocation
        - value_size_bytes
    )


class NflxValkeyArguments(BaseModel):
    key_size_bytes: int = Field(
        alias="valkey.key_size_bytes",
        default=VALKEY_DEFAULT_KEY_SIZE_BYTES,
        ge=0,
        description=(
            "Average key size within QueryPattern's total read/write item size."
        ),
    )
    ttl: str = Field(
        alias="valkey.ttl",
        default="PT24H",
        description=(
            "ISO-8601 key TTL. Used with key size, value size, and WPS when "
            "total state size is not supplied."
        ),
    )
    lua_write_percent: float = Field(
        alias="valkey.lua_write_percent",
        default=0,
        ge=0,
        le=1,
        description=(
            "Fraction of writes that execute Lua, from 0 to 1. Lua and ordinary "
            "operations consume weighted shares of the same node CPU capacity."
        ),
    )
    max_nodes_per_cluster: int = Field(
        alias="valkey.max_nodes_per_cluster",
        default=VALKEY_DEFAULT_MAX_NODES_PER_CLUSTER,
        ge=1,
        description=(
            "Maximum nodes allowed in one ElastiCache cluster. Defaults to AWS's "
            "standard 90-node cluster quota; increase only with an approved quota."
        ),
    )

    @model_validator(mode="after")
    def _check_ttl(self) -> "NflxValkeyArguments":
        if iso_to_seconds(self.ttl) <= 0:
            raise ValueError("valkey.ttl must be a positive, finite duration")
        return self


class _ValkeyMemoryRequirement(NamedTuple):
    item_count: float
    value_size_bytes: int
    item_payload_size_bytes: int
    item_memory_overhead_bytes: int
    item_overhead_extrapolated: bool
    estimated_state_size_gib: float
    memory_overhead_gib: float
    total_data_memory_gib: float


class _ValkeyCPURequirement(NamedTuple):
    simple_ops_per_second: int
    lua_ops_per_second: int
    write_cpu_units: float
    total_cpu_units: float
    write_network_mbps: float
    total_network_mbps: float


class _ValkeyTopology(NamedTuple):
    node_count: int
    shards: int
    node_copies: int


def _estimate_valkey_memory(
    desires: CapacityDesires,
    args: NflxValkeyArguments,
) -> _ValkeyMemoryRequirement:
    item_size_bytes = desires.query_pattern.estimated_mean_write_size_bytes.mid
    if item_size_bytes < args.key_size_bytes:
        raise ValueError(
            "Valkey total item size must be greater than or equal to key size"
        )
    compression_ratio = max(1.0, desires.data_shape.estimated_compression_ratio.mid)
    value_size_bytes = math.ceil(
        (item_size_bytes - args.key_size_bytes) / compression_ratio
    )
    item_payload_size_bytes = args.key_size_bytes + value_size_bytes
    item_memory_overhead_bytes = _valkey_item_overhead_bytes(
        key_size_bytes=args.key_size_bytes,
        value_size_bytes=value_size_bytes,
    )

    requested_item_count = desires.data_shape.estimated_state_item_count
    estimated_state_size_gib = desires.data_shape.estimated_state_size_gib.mid
    if requested_item_count is not None and requested_item_count.mid > 0:
        item_count = requested_item_count.mid
    elif estimated_state_size_gib > 0:
        if item_size_bytes <= 0:
            raise ValueError(
                "Valkey total item size must be positive when estimating per-item "
                "overhead for explicit state"
            )
        item_count = estimated_state_size_gib * GIB_IN_BYTES / item_size_bytes
    else:
        item_count = (
            desires.query_pattern.estimated_write_per_second.mid
            * iso_to_seconds(args.ttl)
        )

    estimated_state_size_gib = item_count * item_payload_size_bytes / GIB_IN_BYTES
    memory_overhead_gib = item_count * item_memory_overhead_bytes / GIB_IN_BYTES
    total_data_memory_gib = estimated_state_size_gib + memory_overhead_gib
    return _ValkeyMemoryRequirement(
        item_count=item_count,
        value_size_bytes=value_size_bytes,
        item_payload_size_bytes=item_payload_size_bytes,
        item_memory_overhead_bytes=item_memory_overhead_bytes,
        item_overhead_extrapolated=(
            value_size_bytes > VALKEY_MAX_MEASURED_OVERHEAD_VALUE_BYTES
        ),
        estimated_state_size_gib=estimated_state_size_gib,
        memory_overhead_gib=memory_overhead_gib,
        total_data_memory_gib=total_data_memory_gib,
    )


def _estimate_valkey_cpu(
    instance: Instance,
    desires: CapacityDesires,
    args: NflxValkeyArguments,
) -> _ValkeyCPURequirement:
    read_ops_per_second = desires.query_pattern.estimated_read_per_second.mid
    write_ops_per_second = desires.query_pattern.estimated_write_per_second.mid
    simple_ops_per_second = _valkey_ops_per_second(instance, use_lua=False)
    lua_ops_per_second = _valkey_ops_per_second(instance, use_lua=True)
    lua_write_ops_per_second = write_ops_per_second * args.lua_write_percent
    simple_write_ops_per_second = write_ops_per_second - lua_write_ops_per_second
    write_cpu_units = (
        simple_write_ops_per_second / simple_ops_per_second
        + lua_write_ops_per_second / lua_ops_per_second
    )
    return _ValkeyCPURequirement(
        simple_ops_per_second=simple_ops_per_second,
        lua_ops_per_second=lua_ops_per_second,
        write_cpu_units=write_cpu_units,
        total_cpu_units=(write_cpu_units + read_ops_per_second / simple_ops_per_second),
        write_network_mbps=(
            write_ops_per_second
            * desires.query_pattern.estimated_mean_write_size_bytes.mid
            * 8
            / 1_000_000
        ),
        total_network_mbps=simple_network_mbps(desires),
    )


def _select_valkey_topology(
    instance: Instance,
    desires: CapacityDesires,
    memory: _ValkeyMemoryRequirement,
    cpu: _ValkeyCPURequirement,
    args: NflxValkeyArguments,
) -> Optional[_ValkeyTopology]:
    usable_memory_per_node_gib = instance.ram_gib * (1 - VALKEY_AWS_MEMORY_RESERVATION)
    min_shards = max(
        1,
        math.ceil(cpu.write_cpu_units),
        math.ceil(cpu.write_network_mbps / instance.net_mbps),
        math.ceil(memory.total_data_memory_gib / usable_memory_per_node_gib),
    )
    min_node_copies = (
        2
        if desires.data_shape.durability_slo_order.mid >= VALKEY_DURABILITY_THRESHOLD
        else 1
    )
    required_nodes = max(
        math.ceil(cpu.total_cpu_units),
        math.ceil(cpu.total_network_mbps / instance.net_mbps),
    )
    if (
        required_nodes > args.max_nodes_per_cluster
        or min_shards * min_node_copies > args.max_nodes_per_cluster
    ):
        return None

    # Reads and keys are assumed uniform across shards, so every shard needs the
    # same replica count. Evaluate complete uniform topologies by total node count.
    topology: Optional[_ValkeyTopology] = None
    max_shards = args.max_nodes_per_cluster // min_node_copies
    for shards in range(
        min_shards,
        min(max_shards, max(min_shards, required_nodes)) + 1,
    ):
        node_copies = max(
            min_node_copies,
            math.ceil(required_nodes / shards),
        )
        if node_copies - 1 > VALKEY_MAX_READ_REPLICAS_PER_SHARD:
            continue
        candidate = _ValkeyTopology(
            node_count=shards * node_copies,
            shards=shards,
            node_copies=node_copies,
        )
        if candidate.node_count > args.max_nodes_per_cluster:
            continue
        if topology is None or candidate < topology:
            topology = candidate

    return topology


class NflxValkeyCapacityModel(CapacityModel):
    cluster_type = "valkey"

    @staticmethod
    def capacity_plan(
        instance: Instance,
        drive: Drive,
        context: RegionContext,
        desires: CapacityDesires,
        extra_model_arguments: Dict[str, Any],
    ) -> Optional[CapacityPlan]:
        _ = drive
        _ = context

        if Platform.valkey not in instance.platforms or instance.ram_gib <= 0:
            return None

        args = NflxValkeyArguments.model_validate(extra_model_arguments)
        memory = _estimate_valkey_memory(desires=desires, args=args)
        cpu = _estimate_valkey_cpu(instance=instance, desires=desires, args=args)
        topology = _select_valkey_topology(
            instance=instance,
            desires=desires,
            memory=memory,
            cpu=cpu,
            args=args,
        )
        if topology is None:
            return None
        read_replicas_per_shard = topology.node_copies - 1
        cluster_params = {
            "valkey.shards": topology.shards,
            "valkey.read_replicas_per_shard": read_replicas_per_shard,
            "valkey.max_read_replicas_per_shard": (VALKEY_MAX_READ_REPLICAS_PER_SHARD),
            "valkey.lua_write_percent": args.lua_write_percent,
            "valkey.simple_ops_per_second_per_node": cpu.simple_ops_per_second,
            "valkey.lua_ops_per_second_per_node": cpu.lua_ops_per_second,
            "valkey.cpu_capacity_units_required": cpu.total_cpu_units,
            "valkey.write_network_mbps_required": cpu.write_network_mbps,
            "valkey.network_mbps_required": cpu.total_network_mbps,
            "valkey.network_mbps_capacity": (topology.node_count * instance.net_mbps),
            "valkey.max_nodes_per_cluster": args.max_nodes_per_cluster,
            "valkey.assumes_uniform_key_distribution": True,
            "valkey.engine_version": VALKEY_ENGINE_VERSION,
            "valkey.item_count": memory.item_count,
            "valkey.value_size_bytes": memory.value_size_bytes,
            "valkey.item_payload_size_bytes": memory.item_payload_size_bytes,
            "valkey.item_memory_overhead_bytes": memory.item_memory_overhead_bytes,
            "valkey.item_overhead_extrapolated": memory.item_overhead_extrapolated,
            "valkey.estimated_state_size_gib": memory.estimated_state_size_gib,
            "valkey.memory_overhead_gib": memory.memory_overhead_gib,
            "valkey.total_data_memory_gib": memory.total_data_memory_gib,
            "valkey.aws_memory_reservation_percent": (VALKEY_AWS_MEMORY_RESERVATION),
            "valkey.usable_memory_per_node_gib": (
                instance.ram_gib * (1 - VALKEY_AWS_MEMORY_RESERVATION)
            ),
            "valkey.read_capacity_ops_per_second": round(
                max(0, topology.node_count - cpu.write_cpu_units)
                * cpu.simple_ops_per_second
            ),
            "valkey.write_capacity_ops_per_second": (
                round(
                    topology.shards
                    / (
                        (1 - args.lua_write_percent) / cpu.simple_ops_per_second
                        + args.lua_write_percent / cpu.lua_ops_per_second
                    )
                )
            ),
        }
        cluster = RegionClusterCapacity(
            cluster_type=NflxValkeyCapacityModel.cluster_type,
            count=topology.node_count,
            instance=instance,
            cluster_params=cluster_params,
        )
        requirement = CapacityRequirement(
            requirement_type="valkey-regional",
            reference_shape=instance,
            cpu_cores=certain_int(topology.node_count),
            mem_gib=certain_float(memory.total_data_memory_gib * topology.node_copies),
            network_mbps=certain_float(cpu.total_network_mbps),
            context=cluster_params,
        )
        return CapacityPlan(
            requirements=Requirements(
                regional=[requirement],
                regrets=("spend", "mem"),
            ),
            candidate_clusters=Clusters(
                annual_costs={
                    "valkey.regional-clusters": Decimal(str(cluster.annual_cost))
                },
                regional=[cluster],
            ),
        )

    @staticmethod
    def description() -> str:
        return "Netflix Valkey on ElastiCache Capacity Model"

    @staticmethod
    def extra_model_arguments_schema() -> Dict[str, Any]:
        return NflxValkeyArguments.model_json_schema()

    @staticmethod
    def preferred_families() -> Optional[FrozenSet[str]]:
        return frozenset(("cache.r7g", "cache.r8g"))

    @staticmethod
    def allowed_platforms() -> Tuple[Platform, ...]:
        return (Platform.valkey,)

    @staticmethod
    def default_desires(
        user_desires: CapacityDesires, extra_model_arguments: Dict[str, Any]
    ) -> CapacityDesires:
        args = NflxValkeyArguments.model_validate(extra_model_arguments)
        default_item_size_bytes = args.key_size_bytes + VALKEY_DEFAULT_VALUE_SIZE_BYTES
        return CapacityDesires(
            query_pattern=QueryPattern(
                access_pattern=AccessPattern.latency,
                access_consistency=GlobalConsistency(
                    same_region=Consistency(
                        target_consistency=AccessConsistency.linearizable_stale,
                    ),
                    cross_region=Consistency(
                        target_consistency=AccessConsistency.never,
                    ),
                ),
                estimated_mean_read_size_bytes=certain_int(default_item_size_bytes),
                estimated_mean_write_size_bytes=certain_int(default_item_size_bytes),
                estimated_mean_read_latency_ms=certain_float(1),
                estimated_mean_write_latency_ms=certain_float(1),
                read_latency_slo_ms=FixedInterval(
                    low=0.5, mid=2, high=5, confidence=0.98
                ),
                write_latency_slo_ms=FixedInterval(
                    low=0.5, mid=2, high=5, confidence=0.98
                ),
            ),
            data_shape=DataShape(
                reserved_instance_app_mem_gib=0,
                reserved_instance_system_mem_gib=0,
            ),
        )


nflx_valkey_capacity_model = NflxValkeyCapacityModel()
