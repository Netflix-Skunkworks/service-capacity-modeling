"""Cassandra preference for attached storage over large ephemeral clusters."""

import pytest

from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.hardware import shapes
from service_capacity_modeling.interface import AccessPattern
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import CapacityRegretParameters
from service_capacity_modeling.interface import DataShape
from service_capacity_modeling.interface import Interval
from service_capacity_modeling.interface import QueryPattern
from service_capacity_modeling.models import RANK_PENALTIES
from service_capacity_modeling.models.org.netflix.cassandra import (
    _compute_penalties,
)
from service_capacity_modeling.models.org.netflix.cassandra import (
    NflxCassandraCapacityModel,
)


DATA_PER_NODE_THRESHOLD_GIB = 300
DATA_PER_NODE_CAP_GIB = 1024
EPHEMERAL_REGRET = 0.2


def _desires(
    state_gib: int = 4_000,
    reads_per_second: int = 1_000,
    writes_per_second: int = 1_000,
) -> CapacityDesires:
    return CapacityDesires(
        service_tier=1,
        query_pattern=QueryPattern(
            access_pattern=AccessPattern.throughput,
            estimated_read_per_second=Interval(
                low=reads_per_second // 2,
                mid=reads_per_second,
                high=reads_per_second * 2,
                confidence=0.98,
            ),
            estimated_write_per_second=Interval(
                low=writes_per_second // 2,
                mid=writes_per_second,
                high=writes_per_second * 2,
                confidence=0.98,
            ),
            estimated_mean_read_size_bytes=Interval(
                low=1024,
                mid=4096,
                high=65536,
                confidence=0.95,
            ),
            estimated_mean_write_size_bytes=Interval(
                low=128,
                mid=1024,
                high=4096,
                confidence=0.95,
            ),
        ),
        data_shape=DataShape(
            estimated_state_size_gib=Interval(
                low=state_gib // 2,
                mid=state_gib,
                high=state_gib * 2,
                confidence=0.98,
            ),
        ),
    )


def _cluster(plan):
    clusters = plan.candidate_clusters.zonal
    assert clusters
    return clusters[0]


def _data_per_node(plan):
    cluster = _cluster(plan)
    requirement = plan.requirements.zonal[0]
    buffer_ratio = cluster.cluster_params["cassandra.storage_buffer_ratio"]
    return requirement.disk_gib.mid / buffer_ratio / cluster.count


def _penalties(
    instance_name: str,
    data_per_node_gib: int,
    regret: float = EPHEMERAL_REGRET,
):
    return _compute_penalties(
        instance=shapes.instance(instance_name),
        large_instance_regret=0,
        data_per_node_gib=data_per_node_gib,
        ephemeral_maintenance_regret=regret,
        different_family_regret=0,
    )


@pytest.mark.parametrize(
    "data_per_node_gib,expected",
    [
        (
            DATA_PER_NODE_THRESHOLD_GIB - 1,
            {"ephemeral_maintenance": EPHEMERAL_REGRET},
        ),
        (
            DATA_PER_NODE_THRESHOLD_GIB,
            {"ephemeral_maintenance": EPHEMERAL_REGRET},
        ),
        (
            (DATA_PER_NODE_THRESHOLD_GIB + DATA_PER_NODE_CAP_GIB) // 2,
            {"ephemeral_maintenance": EPHEMERAL_REGRET * 1.25},
        ),
        (
            DATA_PER_NODE_CAP_GIB,
            {"ephemeral_maintenance": EPHEMERAL_REGRET * 1.5},
        ),
        (
            DATA_PER_NODE_CAP_GIB * 2,
            {"ephemeral_maintenance": EPHEMERAL_REGRET * 1.5},
        ),
    ],
    ids=[
        "below_threshold",
        "at_threshold",
        "halfway_to_cap",
        "at_cap",
        "above_cap",
    ],
)
def test_ephemeral_maintenance_penalty_has_baseline_growth_and_cap(
    data_per_node_gib, expected
):
    assert _penalties("m6id.2xlarge", data_per_node_gib) == expected


def test_ephemeral_maintenance_penalty_does_not_apply_to_ebs():
    assert not _penalties("m6a.2xlarge", DATA_PER_NODE_THRESHOLD_GIB * 2)


def test_ephemeral_maintenance_penalty_can_be_disabled():
    assert not _penalties("m6id.2xlarge", DATA_PER_NODE_THRESHOLD_GIB * 2, regret=0)


def test_ephemeral_maintenance_regret_prefers_ebs_by_default():
    plans = planner.plan_certain(
        model_name="org.netflix.cassandra",
        region="us-east-1",
        desires=_desires(state_gib=1_000),
        instance_families=["i3en", "m6a"],
        num_results=20,
        max_results_per_family=10,
    )

    cluster = _cluster(plans[0])
    assert cluster.instance.drive is None
    assert [drive.name for drive in cluster.attached_drives] == ["gp3"]


def test_disabling_ephemeral_maintenance_regret_restores_cost_ordering():
    plans = planner.plan_certain(
        model_name="org.netflix.cassandra",
        region="us-east-1",
        desires=_desires(state_gib=1_000),
        instance_families=["i3en", "m6a"],
        extra_model_arguments={"ephemeral_maintenance_regret": 0},
        num_results=20,
        max_results_per_family=10,
    )

    cluster = _cluster(plans[0])
    assert cluster.instance.drive is not None
    assert not cluster.attached_drives


def test_wide_ephemeral_cluster_carries_only_baseline_regret():
    plans = planner.plan_certain(
        model_name="org.netflix.cassandra",
        region="us-east-1",
        desires=_desires(
            state_gib=8_000,
            reads_per_second=400_000,
            writes_per_second=200_000,
        ),
        instance_families=["m6id"],
    )

    cluster = _cluster(plans[0])
    assert cluster.instance.drive is not None
    assert _data_per_node(plans[0]) < DATA_PER_NODE_THRESHOLD_GIB
    assert (
        cluster.cluster_params[RANK_PENALTIES]["ephemeral_maintenance"]
        == EPHEMERAL_REGRET
    )


def test_ephemeral_storage_remains_available_as_a_fallback():
    plans = planner.plan_certain(
        model_name="org.netflix.cassandra",
        region="us-east-1",
        desires=_desires(),
        instance_families=["i3en"],
    )

    cluster = _cluster(plans[0])
    assert cluster.instance.drive is not None
    assert _data_per_node(plans[0]) > DATA_PER_NODE_THRESHOLD_GIB
    assert cluster.cluster_params[RANK_PENALTIES]["ephemeral_maintenance"] > 0


def test_regret_cap_preserves_large_i7ie_pricing_advantage():
    plans = planner.plan_certain(
        model_name="org.netflix.cassandra",
        region="us-east-1",
        desires=_desires(state_gib=128_000),
        instance_families=["i7ie", "m7a"],
        num_results=20,
        max_results_per_family=10,
    )

    cluster = _cluster(plans[0])
    assert cluster.instance.family == "i7ie"
    assert cluster.cluster_params[RANK_PENALTIES]["ephemeral_maintenance"] == (
        EPHEMERAL_REGRET * 1.5
    )


def test_ephemeral_maintenance_regret_uses_compute_cost():
    proposed = planner.plan_certain(
        model_name="org.netflix.cassandra",
        region="us-east-1",
        desires=_desires(),
        instance_families=["i3en"],
        num_results=1,
    )[0]
    cluster = _cluster(proposed)
    penalty = cluster.cluster_params[RANK_PENALTIES]["ephemeral_maintenance"]
    compute_cost = sum(c.annual_cost for c in proposed.candidate_clusters.zonal)

    regrets = NflxCassandraCapacityModel.regret(
        regret_params=CapacityRegretParameters(),
        optimal_plan=proposed.model_copy(deep=True),
        proposed_plan=proposed,
    )

    assert regrets["ephemeral_maintenance"] == pytest.approx(penalty * compute_cost)
