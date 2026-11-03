import pytest

from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import CurrentClusters
from service_capacity_modeling.interface import CurrentZoneClusterCapacity
from service_capacity_modeling.interface import DataShape
from service_capacity_modeling.interface import QueryPattern
from service_capacity_modeling.interface import certain_float
from service_capacity_modeling.models.org.netflix.evcache import calculate_spread_cost


@pytest.mark.parametrize(
    "current_name,current_count,cpu,alternative,expected",
    [
        ("r7a.large", 8, 25, "r7a.xlarge", "r7a.large"),
        ("i7i.xlarge", 20, 100, "i7i.large", "i7i.xlarge"),
        ("r7a.xlarge", 8, 0.1, "r7a.large", "r7a.large"),
    ],
)
def test_evcache_same_family_size_preference(
    current_name, current_count, cpu, alternative, expected
):
    from service_capacity_modeling.interface import Lifecycle
    from service_capacity_modeling.models import RANK_PENALTIES

    desires = CapacityDesires(
        service_tier=1,
        query_pattern=QueryPattern(
            estimated_read_per_second=certain_float(1),
            estimated_write_per_second=certain_float(1),
        ),
        data_shape=DataShape(estimated_state_size_gib=certain_float(1)),
        current_clusters=CurrentClusters(
            zonal=[
                CurrentZoneClusterCapacity(
                    cluster_type="evcache",
                    cluster_instance_name=current_name,
                    cluster_instance_count=certain_float(current_count),
                    cpu_utilization=certain_float(cpu),
                    memory_utilization_gib=certain_float(1),
                    network_utilization_mbps=certain_float(1),
                )
            ]
        ),
    )
    options = {
        "model_name": "org.netflix.evcache",
        "region": "us-east-1",
        "desires": desires,
        "lifecycles": [Lifecycle.beta, Lifecycle.stable],
        "extra_model_arguments": {"different_family_regret": 0.0},
    }
    current = planner.plan_certain(**options, instance_families=[current_name])[0]
    resized = planner.plan_certain(**options, instance_families=[alternative])[0]
    selected = planner.plan_certain(
        **options, instance_families=[current_name, alternative]
    )[0]

    assert selected.candidate_clusters.zonal[0].instance.name == expected
    assert "size_migration" not in current.candidate_clusters.zonal[
        0
    ].cluster_params.get(RANK_PENALTIES, {})
    compute_cost = sum(c.annual_cost for c in resized.candidate_clusters.zonal)
    expected_penalties = {"size_migration": pytest.approx(compute_cost * 0.001)}
    spread_cost = calculate_spread_cost(resized.candidate_clusters.zonal[0].count)
    if spread_cost > 0:
        expected_penalties["under_spread"] = pytest.approx(spread_cost)
    assert (
        resized.candidate_clusters.zonal[0].cluster_params[RANK_PENALTIES]
        == expected_penalties
    )
    assert resized.rank == pytest.approx(
        resized.candidate_clusters.total_annual_cost
        + compute_cost * 0.001
        + spread_cost
    )
    assert resized.candidate_clusters.total_annual_cost == pytest.approx(compute_cost)
