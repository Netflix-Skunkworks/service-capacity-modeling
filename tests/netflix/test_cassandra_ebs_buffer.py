import math

import pytest

from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.interface import Buffer
from service_capacity_modeling.interface import BufferComponent
from service_capacity_modeling.interface import Buffers
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import CurrentClusters
from service_capacity_modeling.interface import CurrentZoneClusterCapacity
from service_capacity_modeling.interface import certain_float
from service_capacity_modeling.interface import certain_int
from service_capacity_modeling.interface import DataShape
from service_capacity_modeling.interface import Interval
from service_capacity_modeling.interface import QueryPattern
from service_capacity_modeling.models.common import buffer_for_components
from service_capacity_modeling.models.common import EFFECTIVE_DISK_PER_NODE_GIB
from service_capacity_modeling.models.org.netflix.cassandra import (
    CASSANDRA_DISK_UTILIZATION_LIMIT,
)
from service_capacity_modeling.models.org.netflix.cassandra import (
    CASSANDRA_MAX_ATTACHED_DATA_PER_NODE_GIB,
)
from service_capacity_modeling.models.org.netflix.cassandra import (
    _with_disk_utilization_buffer,
)
from service_capacity_modeling.models.org.netflix.cassandra import (
    NflxCassandraArguments,
)
from service_capacity_modeling.models.org.netflix.cassandra import (
    NflxCassandraCapacityModel,
)
from tests.util import simple_drive


def _ebs_desires(
    buffers: Buffers | None = None, state_gib: int = 20_000
) -> CapacityDesires:
    return CapacityDesires(
        service_tier=1,
        query_pattern=QueryPattern(
            estimated_read_per_second=certain_int(1_000),
            estimated_write_per_second=certain_int(1_000),
            estimated_mean_read_latency_ms=certain_float(1),
            estimated_mean_write_latency_ms=certain_float(1),
        ),
        data_shape=DataShape(
            estimated_state_size_gib=certain_int(state_gib),
            estimated_compression_ratio=certain_float(1.0),
        ),
        buffers=buffers or Buffers(),
    )


def _plan_ebs(desires: CapacityDesires, **extra_model_arguments):
    return planner.plan_certain(
        model_name="org.netflix.cassandra",
        region="us-east-1",
        desires=desires,
        extra_model_arguments={
            "require_attached_disks": True,
            "require_local_disks": False,
            "copies_per_region": 3,
            "adaptive_storage_buffer": False,
            "cluster_size_mode": "unrestricted",
            **extra_model_arguments,
        },
    )[0].candidate_clusters.zonal[0]


def test_ebs_uses_direct_volume_target_without_adaptive_multiplier():
    result = _plan_ebs(_ebs_desires(), max_storage_buffer_ratio=4.0)

    assert result.count == 14
    assert result.attached_drives[0].size_gib == 2600
    assert result.cluster_params[EFFECTIVE_DISK_PER_NODE_GIB] == 2800
    assert result.cluster_params["cassandra.storage_buffer_ratio"] == pytest.approx(
        1 / 0.55, abs=0.01
    )
    assert result.cluster_params["cassandra.ebs_volume_strategy"] == "sampled_demand"


def test_ebs_default_allows_1_5_tib_data_without_heating_disk_utilization():
    args = NflxCassandraArguments.from_extra_model_arguments({})
    result = _plan_ebs(
        _ebs_desires(state_gib=15_360),
        adaptive_storage_buffer=True,
    )

    assert args.max_attached_data_per_node_gib == 1536
    assert CASSANDRA_MAX_ATTACHED_DATA_PER_NODE_GIB == 1536
    assert result.count == 10
    assert result.cluster_params[EFFECTIVE_DISK_PER_NODE_GIB] == 2800
    assert result.cluster_params["cassandra.storage_buffer_ratio"] == pytest.approx(
        1 / 0.55,
        abs=0.01,
    )
    data_per_node_gib = 15_360 / result.count
    assert data_per_node_gib == CASSANDRA_MAX_ATTACHED_DATA_PER_NODE_GIB
    assert data_per_node_gib / result.attached_drives[0].size_gib < 0.55


def test_ebs_default_density_limit_is_strict():
    result = _plan_ebs(
        _ebs_desires(state_gib=15_361),
        adaptive_storage_buffer=True,
    )

    assert result.count == 11
    assert 15_361 / result.count < CASSANDRA_MAX_ATTACHED_DATA_PER_NODE_GIB


def test_ebs_volume_target_replaces_explicit_storage_buffer():
    result = _plan_ebs(
        _ebs_desires(
            Buffers(
                desired={
                    "storage": Buffer(
                        ratio=3.0,
                        components=[BufferComponent.storage],
                    )
                }
            )
        )
    )

    assert result.count == 14
    assert result.attached_drives[0].size_gib == 2600
    assert result.cluster_params[EFFECTIVE_DISK_PER_NODE_GIB] == 2800
    assert result.cluster_params["cassandra.storage_buffer_ratio"] == pytest.approx(
        1 / 0.55,
        abs=0.01,
    )


def test_ebs_volume_target_respects_disk_utilization_cap_with_adaptive_storage():
    result = _plan_ebs(
        _ebs_desires(),
        adaptive_storage_buffer=True,
    )
    stricter = _plan_ebs(
        _ebs_desires(),
        adaptive_storage_buffer=True,
        max_disk_utilization=0.50,
    )

    assert result.cluster_params["cassandra.storage_buffer_ratio"] == pytest.approx(
        1 / 0.55,
        abs=0.01,
    )
    assert result.cluster_params[EFFECTIVE_DISK_PER_NODE_GIB] == 2800
    assert stricter.cluster_params["cassandra.storage_buffer_ratio"] == pytest.approx(
        1 / 0.50,
        abs=0.01,
    )
    assert stricter.cluster_params[EFFECTIVE_DISK_PER_NODE_GIB] == 3100


def test_disk_utilization_cap_does_not_weaken_default_buffer_fallback():
    original = _ebs_desires()
    desires = _with_disk_utilization_buffer(original, max_disk_utilization=0.55)

    disk_buffer = buffer_for_components(
        buffers=desires.buffers,
        components=[BufferComponent.disk],
    )

    assert disk_buffer.ratio == pytest.approx(1 / 0.55, abs=0.01)
    assert CASSANDRA_DISK_UTILIZATION_LIMIT not in original.buffers.desired


def test_max_disk_utilization_must_be_positive():
    with pytest.raises(ValueError, match="max_disk_utilization"):
        NflxCassandraArguments.from_extra_model_arguments({"max_disk_utilization": 0})


def test_provisioning_lifecycle_is_validated_and_in_model_schema():
    schema = NflxCassandraCapacityModel.extra_model_arguments_schema()
    assert "provisioning_lifecycle" in schema["properties"]
    with pytest.raises(ValueError, match="provisioning_lifecycle"):
        NflxCassandraArguments.from_extra_model_arguments(
            {"provisioning_lifecycle": "neww"}
        )


@pytest.mark.parametrize(
    "selected_model", ["org.netflix.cassandra", "org.netflix.key-value"]
)
def test_new_uncertain_ebs_purchase_uses_midpoint_while_regret_prices_sample(
    selected_model,
):
    desires = _ebs_desires(state_gib=9765)
    desires.data_shape.estimated_state_size_gib = Interval(
        low=3000, mid=9765.625, high=39062.5, confidence=0.98
    )
    desires.data_shape.estimated_compression_ratio = certain_float(3)
    result = planner.plan(
        model_name=selected_model,
        region="us-east-1",
        desires=desires,
        simulations=12,
        extra_model_arguments={
            "require_attached_disks": True,
            "copies_per_region": 3,
            "provisioning_lifecycle": "new",
        },
    )

    cluster = result.least_regret[0].candidate_clusters.zonal[0]
    allocation = cluster.cluster_params["cassandra.ebs_initial_allocation"]
    expected_gib = math.ceil((9765.625 / 3 / cluster.count) / 0.55 / 100) * 100
    assert cluster.cluster_params["cassandra.ebs_volume_strategy"] == "new_midpoint"
    assert cluster.attached_drives[0].size_gib == expected_gib
    assert allocation["sampled_requirement_volume_gib"] > 0
    assert (
        result.percentiles[95][0]
        .candidate_clusters.zonal[0]
        .attached_drives[0]
        .size_gib
        > expected_gib
    )


def test_empty_current_inventory_is_still_a_new_provisioning():
    desires = _ebs_desires(state_gib=1500)
    desires.current_clusters = CurrentClusters()
    desires.data_shape.estimated_state_size_gib = Interval(
        low=500, mid=1500, high=3000, confidence=0.98
    )

    result = planner.plan(
        model_name="org.netflix.cassandra",
        region="us-east-1",
        desires=desires,
        simulations=4,
        extra_model_arguments={
            "require_attached_disks": True,
            "copies_per_region": 3,
            "provisioning_lifecycle": "new",
        },
    )
    cluster = result.least_regret[0].candidate_clusters.zonal[0]
    assert cluster.cluster_params["cassandra.ebs_volume_strategy"] == "new_midpoint"
    assert "cassandra.ebs_initial_allocation" in cluster.cluster_params


def test_midpoint_purchase_respects_ebs_data_density_limit():
    cluster = _plan_ebs(
        _ebs_desires(state_gib=1000),
        initial_ebs_physical_state_gib=10_000,
        provisioning_lifecycle="new",
    )

    allocation = cluster.cluster_params["cassandra.ebs_initial_allocation"]
    assert cluster.count >= math.ceil(10_000 / CASSANDRA_MAX_ATTACHED_DATA_PER_NODE_GIB)
    assert (
        allocation["physical_data_per_node_gib"]
        <= CASSANDRA_MAX_ATTACHED_DATA_PER_NODE_GIB
    )


def test_existing_ebs_keeps_volume_floor_unless_rightsizing_is_requested():
    desires = _ebs_desires(state_gib=9765)
    desires.current_clusters = CurrentClusters(
        zonal=[
            CurrentZoneClusterCapacity(
                cluster_instance_name="r7a.4xlarge",
                cluster_instance_count=certain_int(12),
                cluster_drive=simple_drive(size_gib=2000),
                cpu_utilization=certain_float(1),
                disk_utilization_gib=certain_float(800),
                network_utilization_mbps=certain_float(1),
            )
        ]
    )
    existing = _plan_ebs(desires, required_cluster_size=12)
    rightsizing = _plan_ebs(
        desires, required_cluster_size=12, allow_ebs_volume_shrink=True
    )

    assert (
        existing.cluster_params["cassandra.ebs_volume_strategy"] == "existing_no_shrink"
    )
    assert existing.attached_drives[0].size_gib >= 2000
    assert (
        rightsizing.cluster_params["cassandra.ebs_volume_strategy"]
        == "existing_rightsize"
    )
    assert not _plan_ebs(_ebs_desires()).cluster_params.get(
        "cassandra.ebs_initial_allocation"
    )


def test_existing_ebs_keeps_volume_without_disk_usage_telemetry():
    desires = _ebs_desires(state_gib=9765)
    desires.current_clusters = CurrentClusters(
        zonal=[
            CurrentZoneClusterCapacity(
                cluster_instance_name="r7a.4xlarge",
                cluster_instance_count=certain_int(12),
                cluster_drive=simple_drive(size_gib=2000),
                cpu_utilization=certain_float(1),
                network_utilization_mbps=certain_float(1),
            )
        ]
    )

    result = _plan_ebs(desires, required_cluster_size=12)
    assert (
        result.cluster_params["cassandra.ebs_volume_strategy"] == "existing_no_shrink"
    )
    assert result.attached_drives[0].size_gib >= 2000


def test_new_ebs_regret_cost_view_does_not_mutate_delivered_volume():
    desires = _ebs_desires(state_gib=9765)
    plan = planner.plan_certain(
        model_name="org.netflix.cassandra",
        region="us-east-1",
        desires=desires,
        extra_model_arguments={
            "require_attached_disks": True,
            "copies_per_region": 3,
            "required_cluster_size": 12,
            "initial_ebs_physical_state_gib": 9765 / 3,
            "provisioning_lifecycle": "new",
        },
    )[0]
    purchase_cost = plan.candidate_clusters.total_annual_cost
    purchase_volume = plan.candidate_clusters.zonal[0].attached_drives[0].size_gib
    sampled = NflxCassandraCapacityModel.plan_for_regret(plan)

    assert sampled.candidate_clusters.total_annual_cost > purchase_cost
    assert (
        sampled.candidate_clusters.zonal[0].attached_drives[0].size_gib
        > purchase_volume
    )
    assert plan.candidate_clusters.total_annual_cost == purchase_cost
    assert (
        plan.candidate_clusters.zonal[0].attached_drives[0].size_gib == purchase_volume
    )


def test_below_midpoint_sample_uses_its_own_volume_cost_for_regret():
    plan = planner.plan_certain(
        model_name="org.netflix.cassandra",
        region="us-east-1",
        desires=_ebs_desires(state_gib=1000),
        extra_model_arguments={
            "require_attached_disks": True,
            "copies_per_region": 3,
            "required_cluster_size": 12,
            "initial_ebs_physical_state_gib": 9765 / 3,
            "provisioning_lifecycle": "new",
        },
    )[0]
    purchase = plan.candidate_clusters.zonal[0].attached_drives[0]
    sampled = NflxCassandraCapacityModel.plan_for_regret(plan)
    sample_drive = sampled.candidate_clusters.zonal[0].attached_drives[0]

    assert sample_drive.size_gib < purchase.size_gib
    assert sampled.candidate_clusters.total_annual_cost < (
        plan.candidate_clusters.total_annual_cost
    )
    assert (
        plan.candidate_clusters.zonal[0].attached_drives[0].size_gib
        == purchase.size_gib
    )


def test_initial_state_without_new_lifecycle_does_not_reduce_certain_plan():
    result = _plan_ebs(
        _ebs_desires(state_gib=9765),
        initial_ebs_physical_state_gib=100,
        required_cluster_size=12,
    )

    assert result.cluster_params["cassandra.ebs_volume_strategy"] == "sampled_demand"
    assert "cassandra.ebs_initial_allocation" not in result.cluster_params


def test_uncertain_regret_uses_sampled_volume_cost(monkeypatch):
    desires = _ebs_desires(state_gib=9765)
    desires.data_shape.estimated_state_size_gib = Interval(
        low=1000, mid=9765, high=39062, confidence=0.98
    )
    arguments = {
        "require_attached_disks": True,
        "copies_per_region": 3,
        "provisioning_lifecycle": "new",
    }

    with_sampled_cost = planner.plan(
        model_name="org.netflix.cassandra",
        region="us-east-1",
        desires=desires,
        simulations=6,
        extra_model_arguments=arguments,
    )
    monkeypatch.setattr(
        NflxCassandraCapacityModel,
        "plan_for_regret",
        staticmethod(lambda plan: plan),
    )
    with_purchase_cost = planner.plan(
        model_name="org.netflix.cassandra",
        region="us-east-1",
        desires=desires,
        simulations=6,
        extra_model_arguments=arguments,
    )

    def total_regret(result):
        return sum(
            regret
            for _, _, regret in result.explanation.regret_clusters_by_model[
                "org.netflix.cassandra"
            ]
        )

    assert total_regret(with_sampled_cost) != pytest.approx(
        total_regret(with_purchase_cost)
    )
