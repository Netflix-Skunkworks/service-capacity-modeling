from decimal import Decimal

from service_capacity_modeling.hardware import shapes
from service_capacity_modeling.interface import Buffer
from service_capacity_modeling.interface import BufferComponent
from service_capacity_modeling.interface import BufferIntent
from service_capacity_modeling.interface import Buffers
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import CapacityPlan
from service_capacity_modeling.interface import CapacityRequirement
from service_capacity_modeling.interface import certain_float
from service_capacity_modeling.interface import Clusters
from service_capacity_modeling.interface import CurrentClusters
from service_capacity_modeling.interface import CurrentZoneClusterCapacity
from service_capacity_modeling.interface import default_reference_shape
from service_capacity_modeling.interface import Interval
from service_capacity_modeling.interface import QueryPattern
from service_capacity_modeling.interface import RegionClusterCapacity
from service_capacity_modeling.interface import RegionContext
from service_capacity_modeling.interface import Requirements
from service_capacity_modeling.interface import ZoneClusterCapacity
from service_capacity_modeling.models.common import get_cores_from_current_capacity
from service_capacity_modeling.models.common import get_disk_from_current_capacity
from service_capacity_modeling.models.common import merge_plan
from service_capacity_modeling.models.common import network_services
from service_capacity_modeling.models.common import normalize_cores
from service_capacity_modeling.models.common import sqrt_staffed_cores


def test_merge_plan():
    left_requirement = CapacityRequirement(
        requirement_type="test",
        reference_shape=default_reference_shape,
        cpu_cores=Interval(low=10, mid=20, high=30, confidence=0.98),
        mem_gib=Interval(low=20, mid=100, high=200, confidence=0.98),
        network_mbps=Interval(low=1000, mid=2000, high=3000, confidence=0.98),
        disk_gib=Interval(low=40, mid=200, high=500, confidence=0.98),
    )
    right_requirement = CapacityRequirement(
        requirement_type="test",
        reference_shape=default_reference_shape,
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
        services={n: s.model_copy(deep=True) for n, s in hardware.services.items()},
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
    assert 2 * 4 * 1500 < cost_by_service["test.net.intra.region"] < 2 * 4 * 1500 + 100


def test_different_tier_qos():
    tiers = (3, 2, 1, 0)
    prev_cores = 0
    for tier in tiers:
        desires = CapacityDesires(
            service_tier=tier,
            query_pattern=QueryPattern(
                estimated_read_per_second=Interval(
                    low=1000, mid=10000, high=100000, confidence=0.98
                ),
                estimated_write_per_second=Interval(
                    low=1000, mid=10000, high=100000, confidence=0.98
                ),
            ),
        )
        cores = sqrt_staffed_cores(desires)
        assert cores >= prev_cores
        prev_cores = cores


def test_normalize_cores():
    m5xl = shapes.region("us-east-1").instances["m5.xlarge"]
    r5xl = shapes.region("us-east-1").instances["r5.xlarge"]

    m6id = shapes.region("us-east-1").instances["m6id.xlarge"]
    i4ixl = shapes.region("us-east-1").instances["i4i.xlarge"]

    m7axl = shapes.region("us-east-1").instances["m7a.xlarge"]

    # Same generation should be the same
    assert normalize_cores(16, m5xl, r5xl) == 16
    assert normalize_cores(16, m6id, i4ixl) == 16

    # New generation should be higher
    assert 18 <= normalize_cores(16, m5xl, m6id) < 22
    assert 23 <= normalize_cores(16, m6id, m7axl) < 27
    assert 26 <= normalize_cores(16, m5xl, m7axl) < 30

    # All of these computers are much faster than the reference
    for shape in (m5xl, r5xl, m6id, i4ixl):
        assert normalize_cores(16, shape, default_reference_shape) < 13


def test_normalize_cores_6_7():
    """Note this is rather fragile to the exact math of normalize_cores

    If that method changes we'll need to change these assertions
    """
    m6ixl = shapes.region("us-east-1").instances["m6i.xlarge"]
    m7axl = shapes.region("us-east-1").instances["m7a.xlarge"]

    assert 10 == normalize_cores(16, m6ixl, default_reference_shape)
    assert 7 == normalize_cores(16, m7axl, default_reference_shape)


current_cluster = CurrentClusters(
    zonal=[
        CurrentZoneClusterCapacity(
            cluster_instance_name="i4i.2xlarge",
            cluster_instance=shapes.region("us-east-1").instances["i4i.2xlarge"],
            cluster_instance_count=Interval(low=8, mid=8, high=8, confidence=1),
            cpu_utilization=Interval(
                low=10.12, mid=13.2, high=14.194801291058118, confidence=1
            ),
            memory_utilization_gib=certain_float(4.0),
            network_utilization_mbps=certain_float(32.0),
            disk_utilization_gib=certain_float(20),
        )
    ]
)

buffers = Buffers(
    desired={
        "background": Buffer(
            ratio=2.0,
            intent=BufferIntent.desired,
            components=[BufferComponent.cpu, BufferComponent.network, "background"],
        ),
        "compute": Buffer(
            ratio=1.5, intent=BufferIntent.desired, components=[BufferComponent.compute]
        ),
        "storage": Buffer(
            ratio=4, intent=BufferIntent.desired, components=[BufferComponent.storage]
        ),
    }
)


def test_get_cores_with_buffer_scale():
    buffers_copy = buffers.model_copy(deep=True)
    buffers_copy.derived = {
        "compute": Buffer(
            ratio=4, intent=BufferIntent.scale, components=[BufferComponent.compute]
        )
    }
    i3_2xlarge = shapes.region("us-east-1").instances["i3.2xlarge"]
    needed_cores = get_cores_from_current_capacity(
        current_cluster.zonal[0], buffers_copy, i3_2xlarge
    )
    assert needed_cores == 141


def test_get_cores_with_buffer_desired():
    i3_2xlarge = shapes.region("us-east-1").instances["i3.2xlarge"]
    needed_cores = get_cores_from_current_capacity(
        current_cluster.zonal[0], buffers, i3_2xlarge
    )
    assert needed_cores == 35


def test_get_cores_with_buffer_preserve():
    buffers_copy = buffers.model_copy(deep=True)
    buffers_copy.derived = {
        "compute": Buffer(
            intent=BufferIntent.preserve, components=[BufferComponent.compute]
        )
    }
    i3_2xlarge = shapes.region("us-east-1").instances["i3.2xlarge"]
    needed_cores = get_cores_from_current_capacity(
        current_cluster.zonal[0], buffers_copy, i3_2xlarge
    )
    assert needed_cores == 64


def test_get_disk_with_buffer_desired():
    needed_disk = get_disk_from_current_capacity(current_cluster.zonal[0], buffers)
    assert needed_disk == 640


def test_get_disk_with_buffer_scale():
    current_cluster_copy = current_cluster.model_copy(deep=True)
    current_cluster_copy.zonal[0].disk_utilization_gib = certain_float(150)
    buffers_copy = buffers.model_copy(deep=True)
    buffers_copy.derived = {
        "storage": Buffer(
            ratio=8, intent=BufferIntent.scale, components=[BufferComponent.disk]
        )
    }
    needed_disk = get_disk_from_current_capacity(
        current_cluster_copy.zonal[0], buffers_copy
    )
    assert needed_disk == 38400


def test_get_disk_with_buffer_preserve():
    buffers_copy = buffers.model_copy(deep=True)
    buffers_copy.derived = {
        "storage": Buffer(
            intent=BufferIntent.preserve, components=[BufferComponent.disk]
        )
    }
    needed_disk = get_disk_from_current_capacity(current_cluster.zonal[0], buffers_copy)
    assert needed_disk == 13968
