import math
from decimal import Decimal
from typing import Any
from typing import Dict
from typing import FrozenSet
from typing import Optional
from typing import Tuple

from pydantic import BaseModel
from pydantic import Field

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


class NflxValkeyArguments(BaseModel):
    key_size_bytes: int = Field(
        alias="valkey.key_size_bytes",
        default=64,
        ge=0,
        description=(
            "Average key size. Used with value size, WPS, and TTL when total "
            "state size is not supplied."
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


class NflxValkeyCapacityModel(CapacityModel):
    cluster_type = "valkey"

    @staticmethod
    def capacity_plan(  # pylint: disable=too-many-locals
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
        read_ops_per_second = desires.query_pattern.estimated_read_per_second.mid
        write_ops_per_second = desires.query_pattern.estimated_write_per_second.mid
        compression_ratio = max(1.0, desires.data_shape.estimated_compression_ratio.mid)
        estimated_state_size_gib = desires.data_shape.estimated_state_size_gib.mid
        if estimated_state_size_gib <= 0:
            estimated_state_size_gib = (
                (
                    args.key_size_bytes
                    + desires.query_pattern.estimated_mean_write_size_bytes.mid
                )
                * write_ops_per_second
                * iso_to_seconds(args.ttl)
                / GIB_IN_BYTES
            )
        estimated_state_size_gib /= compression_ratio
        required_memory_gib = estimated_state_size_gib / (
            1 - VALKEY_AWS_MEMORY_RESERVATION
        )

        simple_ops_per_second = _valkey_ops_per_second(instance, use_lua=False)
        lua_ops_per_second = _valkey_ops_per_second(instance, use_lua=True)
        lua_write_ops_per_second = write_ops_per_second * args.lua_write_percent
        simple_write_ops_per_second = write_ops_per_second - lua_write_ops_per_second
        write_cpu_units = (
            simple_write_ops_per_second / simple_ops_per_second
            + lua_write_ops_per_second / lua_ops_per_second
        )
        total_cpu_units = write_cpu_units + (
            read_ops_per_second / simple_ops_per_second
        )
        min_shards = max(
            1,
            math.ceil(write_cpu_units),
            math.ceil(required_memory_gib / instance.ram_gib),
        )
        min_node_copies = (
            2
            if desires.data_shape.durability_slo_order.mid
            >= VALKEY_DURABILITY_THRESHOLD
            else 1
        )
        required_nodes = math.ceil(total_cpu_units)

        # Extra shards and read replicas use the same node type and price. Evaluate
        # the possible shard counts and retain the topology with the fewest nodes.
        topology: Optional[Tuple[int, int, int]] = None
        for shards in range(min_shards, max(min_shards, required_nodes) + 1):
            node_copies = max(
                min_node_copies,
                math.ceil(required_nodes / shards),
            )
            if node_copies - 1 > VALKEY_MAX_READ_REPLICAS_PER_SHARD:
                continue
            candidate = (shards * node_copies, shards, node_copies)
            if topology is None or candidate < topology:
                topology = candidate

        assert topology is not None
        node_count, shards, node_copies = topology
        read_replicas_per_shard = node_copies - 1
        cluster_params = {
            "valkey.shards": shards,
            "valkey.read_replicas_per_shard": read_replicas_per_shard,
            "valkey.max_read_replicas_per_shard": (VALKEY_MAX_READ_REPLICAS_PER_SHARD),
            "valkey.lua_write_percent": args.lua_write_percent,
            "valkey.simple_ops_per_second_per_node": simple_ops_per_second,
            "valkey.lua_ops_per_second_per_node": lua_ops_per_second,
            "valkey.cpu_capacity_units_required": total_cpu_units,
            "valkey.estimated_state_size_gib": estimated_state_size_gib,
            "valkey.required_memory_gib": required_memory_gib,
            "valkey.aws_memory_reservation_percent": (VALKEY_AWS_MEMORY_RESERVATION),
            "valkey.usable_memory_per_node_gib": (
                instance.ram_gib * (1 - VALKEY_AWS_MEMORY_RESERVATION)
            ),
            "valkey.read_capacity_ops_per_second": round(
                max(0, node_count - write_cpu_units) * simple_ops_per_second
            ),
            "valkey.write_capacity_ops_per_second": (
                round(
                    shards
                    / (
                        (1 - args.lua_write_percent) / simple_ops_per_second
                        + args.lua_write_percent / lua_ops_per_second
                    )
                )
            ),
        }
        cluster = RegionClusterCapacity(
            cluster_type=NflxValkeyCapacityModel.cluster_type,
            count=node_count,
            instance=instance,
            cluster_params=cluster_params,
        )
        requirement = CapacityRequirement(
            requirement_type="valkey-regional",
            reference_shape=instance,
            cpu_cores=certain_int(node_count),
            mem_gib=certain_float(required_memory_gib * node_copies),
            network_mbps=certain_float(simple_network_mbps(desires)),
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
                estimated_mean_read_size_bytes=certain_int(1024),
                estimated_mean_write_size_bytes=certain_int(1024),
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
