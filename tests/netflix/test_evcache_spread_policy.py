import pytest

from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import certain_float
from service_capacity_modeling.interface import CurrentClusters
from service_capacity_modeling.interface import CurrentZoneClusterCapacity
from service_capacity_modeling.interface import DataShape
from service_capacity_modeling.interface import Interval
from service_capacity_modeling.interface import QueryPattern
from service_capacity_modeling.models.org.netflix.evcache import calculate_spread_cost


_small_cluster_desires = CapacityDesires(
    service_tier=1,
    query_pattern=QueryPattern(
        estimated_read_per_second=Interval(
            low=1000, mid=2000, high=3000, confidence=1.0
        ),
        estimated_write_per_second=Interval(low=100, mid=200, high=300, confidence=1.0),
        estimated_mean_write_size_bytes=Interval(
            low=100, mid=200, high=300, confidence=1.0
        ),
        estimated_mean_read_size_bytes=Interval(
            low=100, mid=200, high=300, confidence=1.0
        ),
    ),
    data_shape=DataShape(
        estimated_state_size_gib=Interval(low=1, mid=2, high=3, confidence=1.0),
    ),
)


@pytest.mark.parametrize("current_count", [1, 3, 9, 10, 11, 20])
@pytest.mark.parametrize("current_type", ["evcache", None])
def test_evcache_existing_cluster_spread_penalty_follows_current_size(
    current_count, current_type
):
    from service_capacity_modeling.models import RANK_PENALTIES

    desires = _small_cluster_desires.model_copy(deep=True)
    desires.service_tier = 0
    desires.current_clusters = CurrentClusters(
        zonal=[
            CurrentZoneClusterCapacity(
                cluster_type=current_type,
                cluster_instance_name="i4i.large",
                cluster_instance_count=certain_float(current_count),
                cpu_utilization=certain_float(1),
                memory_utilization_gib=certain_float(1),
            )
        ]
    )
    plans = planner.plan_certain(
        model_name="org.netflix.evcache",
        region="us-east-1",
        desires=desires,
        instance_families=["i4i.large", "i3en.xlarge"],
        num_results=2,
    )

    assert {p.candidate_clusters.zonal[0].instance.family for p in plans} == {
        "i4i",
        "i3en",
    }
    for plan in plans:
        cluster = plan.candidate_clusters.zonal[0]
        assert 1 <= cluster.count <= 10
        penalties = cluster.cluster_params.get(RANK_PENALTIES, {})
        if current_count <= 10:
            assert "under_spread" not in penalties
        else:
            assert penalties["under_spread"] == calculate_spread_cost(cluster.count)
        if cluster.instance.family == "i4i":
            assert "family_migration" not in penalties
        else:
            assert penalties["family_migration"] > 0
        assert plan.rank == pytest.approx(
            plan.candidate_clusters.total_annual_cost + sum(penalties.values())
        )


@pytest.mark.parametrize("current_kind", ["none", "other", "ambiguous"])
def test_evcache_without_identified_current_keeps_spread_penalty(current_kind):
    from service_capacity_modeling.models import RANK_PENALTIES

    desires = _small_cluster_desires.model_copy(deep=True)
    if current_kind != "none":
        current = CurrentZoneClusterCapacity(
            cluster_type="cassandra" if current_kind == "other" else None,
            cluster_instance_name="i4i.large",
            cluster_instance_count=certain_float(3),
        )
        zonal = [current]
        if current_kind == "ambiguous":
            zonal.append(
                current.model_copy(update={"cluster_instance_name": "i3en.xlarge"})
            )
        desires.current_clusters = CurrentClusters(zonal=zonal)
    plans = planner.plan_certain(
        model_name="org.netflix.evcache",
        region="us-east-1",
        desires=desires,
        instance_families=["i4i.large"],
    )

    assert plans
    for plan in plans:
        cluster = plan.candidate_clusters.zonal[0]
        assert cluster.count <= 10
        penalties = cluster.cluster_params[RANK_PENALTIES]
        assert penalties == {"under_spread": calculate_spread_cost(cluster.count)}
