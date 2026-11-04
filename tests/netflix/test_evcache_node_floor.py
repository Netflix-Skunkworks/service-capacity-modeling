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
@pytest.mark.parametrize(
    "resource,usage",
    [
        ("cpu_utilization", 100),
        ("memory_utilization_gib", 12),
        ("network_utilization_mbps", 2000),
        ("disk_utilization_gib", 32),
    ],
)
def test_evcache_single_node_can_expand_for_resource_demand(tier, resource, usage):
    current = CurrentZoneClusterCapacity(
        cluster_type="evcache",
        cluster_instance_name="i4i.large",
        cluster_instance_count=certain_float(1),
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
            assert cluster.count > 1


@pytest.mark.parametrize(
    "cluster_type,current_count,expected",
    [(None, 1, 1), ("cassandra", 1, 2), ("evcache", 1.5, 2)],
)
def test_evcache_single_node_exception_requires_identified_current_capacity(
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
