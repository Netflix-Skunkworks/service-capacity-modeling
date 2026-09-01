from decimal import Decimal
from typing import Optional

import pytest

from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.hardware import shapes
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import certain_int
from service_capacity_modeling.interface import DataShape
from service_capacity_modeling.interface import Drive
from service_capacity_modeling.interface import FixedInterval
from service_capacity_modeling.interface import GIB_IN_BYTES
from service_capacity_modeling.interface import Platform
from service_capacity_modeling.interface import QueryPattern
from service_capacity_modeling.interface import RegionContext
from service_capacity_modeling.models.org.netflix.valkey import _valkey_ops_per_second
from service_capacity_modeling.models.org.netflix.valkey import (
    _valkey_item_overhead_bytes,
)
from service_capacity_modeling.models.org.netflix.valkey import (
    nflx_valkey_capacity_model,
)
from tests.util import shape


PROPERTY_TEST_CONFIG = {
    "org.netflix.valkey": {
        "read_qps_range": (100_000, 2_000_000),
        "write_qps_range": (50_000, 1_000_000),
        "data_range_gib": (1, 10),
    },
}


def _plan_for_shape(
    instance_name: str,
    *,
    read_ops_per_second: int = 0,
    write_ops_per_second: int = 0,
    state_size_gib: Optional[int] = 1,
    item_size_bytes: int = 66,
    durable: bool = True,
    key_size_bytes: int = 16,
    ttl: str = "PT24H",
    lua_write_percent: float = 0,
    sync_durability_surcharge: Optional[dict[str, float]] = None,
):
    durability = 10_000 if durable else 100
    desires = CapacityDesires(
        query_pattern=QueryPattern(
            estimated_read_per_second=certain_int(read_ops_per_second),
            estimated_write_per_second=certain_int(write_ops_per_second),
            estimated_mean_read_size_bytes=certain_int(item_size_bytes),
            estimated_mean_write_size_bytes=certain_int(item_size_bytes),
        ),
        data_shape=DataShape(
            estimated_state_size_gib=certain_int(state_size_gib or 0),
            durability_slo_order=FixedInterval(
                low=durability,
                mid=durability,
                high=durability,
            ),
        ),
    )
    return nflx_valkey_capacity_model.capacity_plan(
        instance=shape(instance_name),
        drive=Drive.get_managed_drive(),
        context=RegionContext(),
        desires=desires,
        extra_model_arguments={
            "valkey.key_size_bytes": key_size_bytes,
            "valkey.ttl": ttl,
            "valkey.lua_write_percent": lua_write_percent,
            "valkey.sync_durability_surcharge": sync_durability_surcharge or {},
        },
    )


def _cluster(plan):
    assert plan is not None
    return plan.candidate_clusters.regional[0]


def test_only_seventh_and_eighth_generation_nodes_are_available():
    valkey_instances = [
        instance
        for instance in shapes.region("us-east-1").instances.values()
        if Platform.valkey in instance.platforms
    ]

    assert valkey_instances
    assert {instance.family for instance in valkey_instances} == {
        "cache.r7g",
        "cache.r8g",
    }


def test_default_value_size_is_50_bytes_plus_key():
    defaults = nflx_valkey_capacity_model.default_desires(CapacityDesires(), {})

    assert defaults.query_pattern.estimated_mean_read_size_bytes.mid == 66
    assert defaults.query_pattern.estimated_mean_write_size_bytes.mid == 66


def test_throughput_tracks_core_speed_within_benchmark_bounds():
    slow = shape("cache.r7g.large").model_copy(update={"cpu_ghz": 2.5})
    fast = slow.model_copy(update={"cpu_ghz": 3.5})

    assert _valkey_ops_per_second(slow, use_lua=False) == 700_000
    assert _valkey_ops_per_second(fast, use_lua=False) == 1_000_000
    assert _valkey_ops_per_second(slow, use_lua=True) == 50_000
    assert _valkey_ops_per_second(fast, use_lua=True) == 150_000


def test_taller_instances_do_not_increase_throughput():
    small = _cluster(
        _plan_for_shape("cache.r7g.large", write_ops_per_second=750_000, durable=False)
    )
    tall = _cluster(
        _plan_for_shape(
            "cache.r7g.16xlarge", write_ops_per_second=750_000, durable=False
        )
    )

    assert small.cluster_params["valkey.shards"] == 2
    assert tall.cluster_params["valkey.shards"] == 2
    assert (
        small.cluster_params["valkey.simple_ops_per_second_per_node"]
        == tall.cluster_params["valkey.simple_ops_per_second_per_node"]
    )


def test_item_memory_overhead_tracks_valkey_8_1_curve():
    assert _valkey_item_overhead_bytes(16, 64) == 55


def test_item_memory_overhead_jumps_at_benchmark_boundary():
    assert _valkey_item_overhead_bytes(16, 8) == 39
    assert _valkey_item_overhead_bytes(16, 9) == 38
    assert _valkey_item_overhead_bytes(16, 10) == 45
    assert _valkey_item_overhead_bytes(16, 31) == 40
    assert _valkey_item_overhead_bytes(16, 32) == 47
    assert _valkey_item_overhead_bytes(16, 124) == 43
    assert _valkey_item_overhead_bytes(16, 125) == 74
    assert _valkey_item_overhead_bytes(16, 1020) == 43
    assert _valkey_item_overhead_bytes(16, 1021) == 298


def test_memory_can_be_derived_from_key_value_wps_and_ttl():
    cluster = _cluster(
        _plan_for_shape(
            "cache.r7g.large",
            write_ops_per_second=100,
            state_size_gib=None,
            item_size_bytes=1024,
            key_size_bytes=64,
            ttl="PT10S",
            durable=False,
        )
    )

    item_count = 100 * 10
    state_size_gib = 1024 * item_count / GIB_IN_BYTES
    overhead_gib = 111 * item_count / GIB_IN_BYTES
    assert cluster.cluster_params["valkey.estimated_state_size_gib"] == pytest.approx(
        state_size_gib
    )
    assert cluster.cluster_params["valkey.item_memory_overhead_bytes"] == 111
    assert cluster.cluster_params["valkey.memory_overhead_gib"] == pytest.approx(
        overhead_gib
    )
    assert cluster.cluster_params["valkey.total_data_memory_gib"] == pytest.approx(
        state_size_gib + overhead_gib
    )


def test_explicit_item_count_is_preserved():
    item_count = 17_000_000
    plans = planner.plan_certain(
        model_name="org.netflix.valkey",
        region="us-east-1",
        desires=CapacityDesires(
            query_pattern=QueryPattern(
                estimated_mean_read_size_bytes=certain_int(53),
                estimated_mean_write_size_bytes=certain_int(53),
            ),
            data_shape=DataShape(
                estimated_state_item_count=certain_int(item_count),
            ),
        ),
        extra_model_arguments={"valkey.key_size_bytes": 16},
    )

    params = plans[0].candidate_clusters.regional[0].cluster_params
    assert params["valkey.item_count"] == item_count
    assert params["valkey.estimated_state_size_gib"] == pytest.approx(
        item_count * 53 / GIB_IN_BYTES
    )
    assert params["valkey.total_data_memory_gib"] == pytest.approx(
        item_count * (53 + 50) / GIB_IN_BYTES
    )


@pytest.mark.parametrize("ttl", ["PT0S", "-PT1H", "unlimited"])
def test_ttl_must_be_positive_and_finite(ttl):
    with pytest.raises(ValueError, match="positive, finite"):
        planner.plan_certain(
            model_name="org.netflix.valkey",
            region="us-east-1",
            desires=CapacityDesires(),
            extra_model_arguments={"valkey.ttl": ttl},
        )


def test_aws_memory_reservation_is_applied_to_upfront_state_size():
    cluster = _cluster(
        _plan_for_shape("cache.r7g.large", state_size_gib=10, durable=False)
    )

    item_count = 10 * GIB_IN_BYTES / (16 + 50)
    overhead_gib = item_count * 45 / GIB_IN_BYTES
    assert cluster.cluster_params["valkey.item_memory_overhead_bytes"] == 45
    assert cluster.cluster_params["valkey.total_data_memory_gib"] == pytest.approx(
        10 + overhead_gib
    )
    assert cluster.cluster_params["valkey.usable_memory_per_node_gib"] == (
        pytest.approx(13.07 * 0.75)
    )
    assert cluster.cluster_params["valkey.shards"] == 2


def test_item_overhead_extrapolation_is_reported():
    cluster = _cluster(
        _plan_for_shape(
            "cache.r7g.large",
            item_size_bytes=200,
            state_size_gib=1,
            durable=False,
        )
    )

    assert cluster.cluster_params["valkey.value_size_bytes"] == 184
    assert cluster.cluster_params["valkey.item_memory_overhead_bytes"] == 47
    assert cluster.cluster_params["valkey.item_overhead_extrapolated"] is True


@pytest.mark.parametrize(
    "durable, expected_nodes, expected_shards",
    [(False, 2, 1), (True, 10, 5)],
)
def test_cluster_shape_for_500k_writes_and_500k_reads(
    durable, expected_nodes, expected_shards
):
    cluster = _cluster(
        _plan_for_shape(
            "cache.r7g.large",
            write_ops_per_second=500_000,
            read_ops_per_second=500_000,
            durable=durable,
        )
    )

    assert cluster.count == expected_nodes
    assert cluster.cluster_params["valkey.shards"] == expected_shards
    assert cluster.cluster_params["valkey.read_replicas_per_shard"] == 1


@pytest.mark.parametrize(
    "lua_write_percent, write_ops_per_second",
    [(0, 100_000), (1, 50_000)],
)
def test_durable_write_throughput_limits(lua_write_percent, write_ops_per_second):
    at_limit = _cluster(
        _plan_for_shape(
            "cache.r7g.large",
            write_ops_per_second=write_ops_per_second,
            durable=True,
            lua_write_percent=lua_write_percent,
        )
    )
    beyond_limit = _cluster(
        _plan_for_shape(
            "cache.r7g.large",
            write_ops_per_second=write_ops_per_second + 1,
            durable=True,
            lua_write_percent=lua_write_percent,
        )
    )

    assert at_limit.cluster_params["valkey.shards"] == 1
    assert beyond_limit.cluster_params["valkey.shards"] == 2
    assert at_limit.cluster_params["valkey.simple_ops_per_second_per_node"] == 100_000
    assert at_limit.cluster_params["valkey.lua_ops_per_second_per_node"] == 50_000
    assert (
        at_limit.cluster_params["valkey.write_capacity_ops_per_second"]
        == write_ops_per_second
    )


def test_read_throughput_is_not_affected_by_durability():
    durable = _cluster(
        _plan_for_shape("cache.r7g.large", read_ops_per_second=500_000, durable=True)
    )
    ephemeral = _cluster(
        _plan_for_shape("cache.r7g.large", read_ops_per_second=500_000, durable=False)
    )

    assert durable.cluster_params["valkey.read_ops_per_second_per_node"] == 700_000
    assert (
        durable.cluster_params["valkey.read_ops_per_second_per_node"]
        == ephemeral.cluster_params["valkey.read_ops_per_second_per_node"]
    )
    assert (
        durable.cluster_params["valkey.cpu_capacity_units_required"]
        == (ephemeral.cluster_params["valkey.cpu_capacity_units_required"])
    )


def test_cluster_shape_for_50k_lua_writes_and_500k_reads():
    cluster = _cluster(
        _plan_for_shape(
            "cache.r7g.large",
            write_ops_per_second=50_000,
            read_ops_per_second=500_000,
            durable=False,
            lua_write_percent=1,
        )
    )

    assert cluster.count == 2
    assert cluster.cluster_params["valkey.shards"] == 1
    assert cluster.cluster_params["valkey.read_replicas_per_shard"] == 1
    assert cluster.cluster_params["valkey.cpu_capacity_units_required"] == (
        pytest.approx(50_000 / 50_000 + 500_000 / 700_000)
    )


def test_writes_only_scale_by_adding_shards():
    cluster = _cluster(
        _plan_for_shape(
            "cache.r7g.large", write_ops_per_second=1_500_000, durable=False
        )
    )

    assert cluster.cluster_params["valkey.shards"] == 3
    assert cluster.cluster_params["valkey.read_replicas_per_shard"] == 0


def test_durable_clusters_have_at_least_one_read_replica():
    durable = _cluster(_plan_for_shape("cache.r7g.large", durable=True))
    ephemeral = _cluster(_plan_for_shape("cache.r7g.large", durable=False))

    assert durable.cluster_params["valkey.read_replicas_per_shard"] == 1
    assert ephemeral.cluster_params["valkey.read_replicas_per_shard"] == 0


def test_sync_durability_surcharge_applies_only_to_durable_clusters():
    hourly_surcharge = {
        "cache.r7g.large": 0.01,
        "cache.r7g.xlarge": 0.02,
    }
    durable = _plan_for_shape(
        "cache.r7g.large",
        durable=True,
        sync_durability_surcharge=hourly_surcharge,
    )
    ephemeral = _plan_for_shape(
        "cache.r7g.large",
        durable=False,
        sync_durability_surcharge=hourly_surcharge,
    )

    assert durable is not None
    assert ephemeral is not None
    assert durable.candidate_clusters.annual_costs["valkey.sync-durability"] == Decimal(
        "175.20"
    )
    assert "valkey.sync-durability" not in ephemeral.candidate_clusters.annual_costs


def test_read_replica_limit_forces_additional_shards():
    at_replica_limit = _cluster(
        _plan_for_shape(
            "cache.r7g.large",
            read_ops_per_second=6 * 700_000,
            durable=True,
        )
    )
    beyond_replica_limit = _cluster(
        _plan_for_shape(
            "cache.r7g.large",
            read_ops_per_second=6 * 700_000 + 1,
            durable=True,
        )
    )

    assert at_replica_limit.count == 6
    assert at_replica_limit.cluster_params["valkey.shards"] == 1
    assert at_replica_limit.cluster_params["valkey.read_replicas_per_shard"] == 5
    assert beyond_replica_limit.count == 8
    assert beyond_replica_limit.cluster_params["valkey.shards"] == 2
    assert beyond_replica_limit.cluster_params["valkey.read_replicas_per_shard"] == 3


def test_network_throughput_increases_topology_size():
    cluster = _cluster(
        _plan_for_shape(
            "cache.r7g.large",
            read_ops_per_second=1_000,
            item_size_bytes=1_000_000,
            state_size_gib=1,
            durable=False,
        )
    )

    assert cluster.count == 9
    assert cluster.cluster_params["valkey.shards"] == 3
    assert cluster.cluster_params["valkey.read_replicas_per_shard"] == 2
    assert cluster.cluster_params["valkey.network_mbps_required"] == 8_000
    assert cluster.cluster_params["valkey.network_mbps_capacity"] >= 8_000


def test_uniform_reads_require_uniform_replica_layers():
    cluster = _cluster(
        _plan_for_shape(
            "cache.r7g.large",
            read_ops_per_second=3_400_000,
            write_ops_per_second=1_500_000,
            state_size_gib=1,
            durable=True,
        )
    )

    assert cluster.count == 30
    assert cluster.cluster_params["valkey.shards"] == 15
    assert cluster.cluster_params["valkey.read_replicas_per_shard"] == 1
    assert cluster.cluster_params["valkey.assumes_uniform_key_distribution"] is True


def test_default_node_quota_rejects_unprovisionable_cluster():
    desires = CapacityDesires(
        query_pattern=QueryPattern(
            estimated_write_per_second=certain_int(5_000_000),
        ),
        data_shape=DataShape(estimated_state_size_gib=certain_int(1)),
    )

    assert not planner.plan_certain(
        model_name="org.netflix.valkey",
        region="us-east-1",
        desires=desires,
        instance_families=["cache.r7g"],
    )
    plans = planner.plan_certain(
        model_name="org.netflix.valkey",
        region="us-east-1",
        desires=desires,
        instance_families=["cache.r7g"],
        extra_model_arguments={"valkey.max_nodes_per_cluster": 100},
    )
    assert plans[0].candidate_clusters.regional[0].count == 100


def test_storage_and_throughput_dominated_shapes():
    durability = FixedInterval(low=100, mid=100, high=100)
    storage_plans = planner.plan_certain(
        model_name="org.netflix.valkey",
        region="us-east-1",
        desires=CapacityDesires(
            data_shape=DataShape(
                estimated_state_size_gib=certain_int(300),
                durability_slo_order=durability,
            ),
        ),
    )
    throughput_plans = planner.plan_certain(
        model_name="org.netflix.valkey",
        region="us-east-1",
        desires=CapacityDesires(
            query_pattern=QueryPattern(
                estimated_read_per_second=certain_int(2_000_000),
                estimated_write_per_second=certain_int(2_000_000),
            ),
            data_shape=DataShape(
                estimated_state_size_gib=certain_int(1),
                durability_slo_order=durability,
            ),
        ),
    )

    storage_cluster = storage_plans[0].candidate_clusters.regional[0]
    throughput_cluster = throughput_plans[0].candidate_clusters.regional[0]
    assert storage_cluster.instance.name == "cache.r8g.2xlarge"
    assert storage_cluster.cluster_params["valkey.shards"] == 11
    assert throughput_cluster.instance.name == "cache.r7g.large"
    assert throughput_cluster.cluster_params["valkey.shards"] == 3
    assert throughput_cluster.count == 6


def test_planner_uses_reserved_elasticache_instance_price():
    plans = planner.plan_certain(
        model_name="org.netflix.valkey",
        region="us-east-1",
        desires=CapacityDesires(
            query_pattern=QueryPattern(
                estimated_read_per_second=certain_int(100_000),
                estimated_write_per_second=certain_int(100_000),
            ),
            data_shape=DataShape(estimated_state_size_gib=certain_int(1)),
        ),
        instance_families=["cache.r7g"],
    )

    assert plans
    cluster = plans[0].candidate_clusters.regional[0]
    assert cluster.instance.name == "cache.r7g.large"
    assert cluster.count == 2
    assert cluster.annual_cost == pytest.approx(2 * 690.64)
    assert plans[0].candidate_clusters.total_annual_cost == pytest.approx(1381.28)
