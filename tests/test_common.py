from decimal import Decimal

from service_capacity_modeling.hardware import shapes
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import CapacityPlan
from service_capacity_modeling.interface import CapacityRequirement
from service_capacity_modeling.interface import Clusters
from service_capacity_modeling.interface import Interval
from service_capacity_modeling.interface import QueryPattern
from service_capacity_modeling.interface import RegionClusterCapacity
from service_capacity_modeling.interface import RegionContext
from service_capacity_modeling.interface import Requirements
from service_capacity_modeling.interface import ZoneClusterCapacity
from service_capacity_modeling.models.common import merge_plan
from service_capacity_modeling.models.common import network_services


def test_merge_plan():
    left_requirement = CapacityRequirement(
        requirement_type="test",
        core_reference_ghz=2.3,
        cpu_cores=Interval(low=10, mid=20, high=30, confidence=0.98),
        mem_gib=Interval(low=20, mid=100, high=200, confidence=0.98),
        network_mbps=Interval(low=1000, mid=2000, high=3000, confidence=0.98),
        disk_gib=Interval(low=40, mid=200, high=500, confidence=0.98),
    )
    right_requirement = CapacityRequirement(
        requirement_type="test",
        core_reference_ghz=2.3,
        cpu_cores=Interval(low=10, mid=20, high=30, confidence=0.98),
        mem_gib=Interval(low=20, mid=100, high=200, confidence=0.98),
        network_mbps=Interval(low=1000, mid=2000, high=3000, confidence=0.98),
        disk_gib=Interval(low=40, mid=200, high=500, confidence=0.98),
    )

    left_instance = shapes.region("us-east-1").instances["r5d.2xlarge"]
    right_instance = shapes.region("us-east-1").instances["m5.2xlarge"]

    left_plan = CapacityPlan(
        requirements=Requirements(zonal=[left_requirement]),
        candidate_clusters=Clusters(
            annual_costs={"left-zonal": Decimal(1234)},
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
        requirements=Requirements(zonal=[right_requirement]),
        candidate_clusters=Clusters(
            annual_costs={"right-zonal": Decimal(1234), "right-regional": Decimal(234)},
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

    assert (
        result.requirements.zonal[0].cpu_cores.mid
        + result.requirements.zonal[1].cpu_cores.mid
        == 40
    )
    assert (
        result.requirements.zonal[0].network_mbps.mid
        + result.requirements.zonal[1].network_mbps.mid
        == 4000
    )

    assert result.candidate_clusters.regional == right_plan.candidate_clusters.regional

    assert left_plan.candidate_clusters.zonal[0] in result.candidate_clusters.zonal
    assert right_plan.candidate_clusters.zonal[0] in result.candidate_clusters.zonal
    assert "left-zonal" in result.candidate_clusters.annual_costs
    assert "right-zonal" in result.candidate_clusters.annual_costs
    assert result.candidate_clusters.total_annual_cost == (
        Decimal(1468) + Decimal(1234)
    )


def test_interval_scale():
    without_minmax = Interval(low=10, mid=100, high=1000)
    s = without_minmax.scale(3)
    assert s.low == 30
    assert s.mid == 300
    assert s.high == 3000
    assert s.minimum < s.low
    assert s.maximum > s.high

    with_minmax = Interval(
        low=10, mid=100, high=1000, minimum_value=1, maximum_value=1010
    )
    s = with_minmax.scale(3)
    assert s.low == 30
    assert s.mid == 300
    assert s.high == 3000
    assert s.minimum == 3
    assert s.maximum == 3030


def test_interval_offset():
    without_minmax = Interval(low=10, mid=100, high=1000)
    s = without_minmax.offset(3)
    assert s.low == 13
    assert s.mid == 103
    assert s.high == 1003
    assert s.minimum < s.low
    assert s.maximum > s.high

    with_minmax = Interval(
        low=10, mid=100, high=1000, minimum_value=1, maximum_value=1010
    )
    s = with_minmax.offset(3)
    assert s.low == 13
    assert s.mid == 103
    assert s.high == 1003
    assert s.minimum == 4
    assert s.maximum == 1013


def test_network_services():
    desires = CapacityDesires(
        service_tier=1,
        query_pattern=QueryPattern(
            estimated_write_per_second=Interval(
                low=1000, mid=10000, high=100000, confidence=0.98
            ),
            estimated_mean_write_size_bytes=Interval(
                low=128, mid=256, high=1024, confidence=0.98
            ),
        ),
    )
    hardware = shapes.region("us-east-1")
    region_context = RegionContext(
        services={n: s.copy(deep=True) for n, s in hardware.services.items()},
        num_regions=4,
        zones_in_region=3,
    )
    ns = network_services(
        "test", context=region_context, desires=desires, copies_per_region=3
    )
    cost_by_service = {}
    for service in ns:
        cost_by_service[service.service_type] = service.annual_cost

    assert 3 * 1500 < cost_by_service["test.net.inter.region"] < 3 * 1500 + 100
    assert 2 * 1500 < cost_by_service["test.net.intra.region"] < 2 * 1500 + 100
