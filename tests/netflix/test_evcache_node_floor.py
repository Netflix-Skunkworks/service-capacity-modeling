import pytest

from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.hardware import shapes
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import CurrentClusters
from service_capacity_modeling.interface import CurrentZoneClusterCapacity
from service_capacity_modeling.interface import DataShape
from service_capacity_modeling.interface import QueryPattern
from service_capacity_modeling.interface import certain_float


@pytest.mark.parametrize("tier", [0, 1, 2, 3])
@pytest.mark.parametrize("current_count", [1, 8, 20])
@pytest.mark.parametrize(
    "resource,usage",
    [
        ("cpu_utilization", 100),
        ("memory_utilization_gib", 12),
        ("network_utilization_mbps", 2000),
        ("disk_utilization_gib", 32),
    ],
)
def test_evcache_node_floor_allows_resource_expansion(
    tier, current_count, resource, usage
):
    current = CurrentZoneClusterCapacity(
        cluster_type="evcache",
        cluster_instance_name="i4i.large",
        cluster_instance_count=certain_float(current_count),
        cpu_utilization=certain_float(0.1),
        memory_utilization_gib=certain_float(0.1),
        network_utilization_mbps=certain_float(0.1),
        disk_utilization_gib=certain_float(0.1),
        cluster_drive=shapes.instance("i4i.large").drive,
    )
    setattr(current, resource, certain_float(usage))
    desires = CapacityDesires(
        service_tier=tier,
        query_pattern=QueryPattern(
            estimated_read_per_second=certain_float(1),
            estimated_write_per_second=certain_float(1),
        ),
        data_shape=DataShape(estimated_state_size_gib=certain_float(1)),
        current_clusters=CurrentClusters(zonal=[current]),
    )
    plans = planner.plan_certain(
        model_name="org.netflix.evcache",
        region="us-east-1",
        desires=desires,
        instance_families=["i4i.large"],
        extra_model_arguments={"max_local_data_per_node_gib": 8},
    )

    assert plans
    for plan in plans:
        for cluster in plan.candidate_clusters.zonal:
            assert cluster.count > current_count


@pytest.mark.parametrize(
    "cluster_type,current_count,expected",
    [(None, 8, 8), ("cassandra", 8, 2), ("evcache", 1.5, 2)],
)
def test_evcache_node_floor_requires_identified_current_capacity(
    cluster_type, current_count, expected
):
    desires = CapacityDesires(
        service_tier=2,
        query_pattern=QueryPattern(
            estimated_read_per_second=certain_float(1),
            estimated_write_per_second=certain_float(1),
        ),
        data_shape=DataShape(estimated_state_size_gib=certain_float(1)),
        current_clusters=CurrentClusters(
            zonal=[
                CurrentZoneClusterCapacity(
                    cluster_type=cluster_type,
                    cluster_instance_name="i4i.large",
                    cluster_instance_count=certain_float(current_count),
                    cpu_utilization=certain_float(0.1),
                    memory_utilization_gib=certain_float(0.1),
                    network_utilization_mbps=certain_float(0.1),
                )
            ]
        ),
    )
    plans = planner.plan_certain(
        model_name="org.netflix.evcache",
        region="us-east-1",
        desires=desires,
        instance_families=["i4i.large"],
    )

    assert plans
    for plan in plans:
        for cluster in plan.candidate_clusters.zonal:
            assert cluster.count == expected


@pytest.mark.parametrize("required_count,accepted", [(4, False), (8, True), (12, True)])
def test_evcache_required_size_must_satisfy_existing_node_floor(
    required_count, accepted
):
    desires = CapacityDesires(
        query_pattern=QueryPattern(
            estimated_read_per_second=certain_float(1),
            estimated_write_per_second=certain_float(1),
        ),
        data_shape=DataShape(estimated_state_size_gib=certain_float(1)),
        current_clusters=CurrentClusters(
            zonal=[
                CurrentZoneClusterCapacity(
                    cluster_type="evcache",
                    cluster_instance_name="i4i.large",
                    cluster_instance_count=certain_float(8),
                    cpu_utilization=certain_float(0.1),
                    memory_utilization_gib=certain_float(0.1),
                    network_utilization_mbps=certain_float(0.1),
                )
            ]
        ),
    )
    result = planner.plan_certain_explained(
        model_name="org.netflix.evcache",
        region="us-east-1",
        desires=desires,
        instance_families=["i4i.large"],
        extra_model_arguments={"required_cluster_size": required_count},
    )

    if accepted:
        assert result.plans
        assert all(
            c.count == required_count
            for plan in result.plans
            for c in plan.candidate_clusters.zonal
        )
    else:
        from service_capacity_modeling.interface import Bottleneck

        assert not result.plans
        assert any(
            excuse.bottleneck == Bottleneck.cluster_size
            for excuses in result.excuses_by_model.values()
            for excuse in excuses
        )


@pytest.mark.parametrize("legacy", [False, True])
def test_evcache_floor_uses_its_own_current_node_count(legacy):
    evcache = CurrentZoneClusterCapacity(
        cluster_type=None if legacy else "evcache",
        cluster_instance_name="i4i.large",
        cluster_instance_count=certain_float(8),
        cpu_utilization=certain_float(0.1),
        memory_utilization_gib=certain_float(0.1),
        network_utilization_mbps=certain_float(0.1),
    )
    zonal = [evcache.model_copy(deep=True), evcache, evcache.model_copy(deep=True)]
    if not legacy:
        zonal.insert(
            0,
            evcache.model_copy(
                update={
                    "cluster_type": "cassandra",
                    "cluster_instance_count": certain_float(100),
                }
            ),
        )
    plans = planner.plan_certain(
        model_name="org.netflix.evcache",
        region="us-east-1",
        desires=CapacityDesires(
            query_pattern=QueryPattern(
                estimated_read_per_second=certain_float(1),
                estimated_write_per_second=certain_float(1),
            ),
            data_shape=DataShape(estimated_state_size_gib=certain_float(1)),
            current_clusters=CurrentClusters(zonal=zonal),
        ),
        instance_families=["i4i.large"],
    )

    assert plans
    for plan in plans:
        assert [c.count for c in plan.candidate_clusters.zonal] == [8, 8, 8]
