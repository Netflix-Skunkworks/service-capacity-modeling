from service_capacity_modeling.hardware import shapes
from service_capacity_modeling.interface import CapacityPlan
from service_capacity_modeling.interface import CapacityRequirement
from service_capacity_modeling.interface import certain_int
from service_capacity_modeling.interface import Clusters
from service_capacity_modeling.interface import Interval
from service_capacity_modeling.interface import RegionClusterCapacity
from service_capacity_modeling.interface import ZoneClusterCapacity
from service_capacity_modeling.models.common import merge_plan


def test_merge_plan():
    left_requirement = CapacityRequirement(
        core_reference_ghz=2.3,
        cpu_cores=Interval(low=10, mid=20, high=30, confidence=0.9),
        mem_gib=Interval(low=20, mid=100, high=200, confidence=0.9),
        network_mbps=Interval(low=1000, mid=2000, high=3000, confidence=0.9),
        disk_gib=Interval(low=40, mid=200, high=500, confidence=0.9),
    )
    right_requirement = CapacityRequirement(
        core_reference_ghz=2.3,
        cpu_cores=Interval(low=10, mid=20, high=30, confidence=0.9),
        mem_gib=Interval(low=20, mid=100, high=200, confidence=0.9),
        network_mbps=Interval(low=1000, mid=2000, high=3000, confidence=0.9),
        disk_gib=Interval(low=40, mid=200, high=500, confidence=0.9),
    )

    left_instance = shapes.region("us-east-1").instances["r5d.2xlarge"]
    right_instance = shapes.region("us-east-1").instances["m5.2xlarge"]

    left_plan = CapacityPlan(
        requirement=left_requirement,
        candidate_clusters=Clusters(
            total_annual_cost=certain_int(1234),
            zonal=[
                ZoneClusterCapacity(
                    cluster_type="left",
                    count=2,
                    instance=left_instance,
                    attached_drives=[],
                    annual_cost=1234,
                )
            ],
        ),
    )

    right_plan = CapacityPlan(
        requirement=right_requirement,
        candidate_clusters=Clusters(
            total_annual_cost=certain_int(1468),
            regional=[
                RegionClusterCapacity(
                    cluster_type="right",
                    count=2,
                    instance=right_instance,
                    attached_drives=[],
                    annual_cost=234,
                )
            ],
            zonal=[
                ZoneClusterCapacity(
                    cluster_type="right",
                    count=4,
                    instance=left_instance,
                    attached_drives=[],
                    annual_cost=1234,
                )
            ],
        ),
    )

    result = merge_plan(left_plan, right_plan)
    assert result is not None

    assert result.requirement.cpu_cores.mid == 40
    assert result.requirement.network_mbps.mid == 4000

    assert result.candidate_clusters.regional == right_plan.candidate_clusters.regional

    assert left_plan.candidate_clusters.zonal[0] in result.candidate_clusters.zonal
    assert right_plan.candidate_clusters.zonal[0] in result.candidate_clusters.zonal
