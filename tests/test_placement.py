from service_capacity_modeling.hardware import shapes
from service_capacity_modeling.interface import Bottleneck
from service_capacity_modeling.interface import NodeCountConstraint
from service_capacity_modeling.models.common import compute_stateful_zone
from service_capacity_modeling.models.common import cpu_to_ncu
from service_capacity_modeling.models.common import stateful_node_allocatable
from service_capacity_modeling.models.placement import NodeAllocatable
from service_capacity_modeling.models.placement import estimate_zone_nodes
from service_capacity_modeling.models.placement import Unschedulable
from service_capacity_modeling.models.placement import ZoneNodeEstimate
from service_capacity_modeling.models.placement import ZoneRequirement

CONTAINER = NodeAllocatable(ncu=8, memory_gib=32, network_mbps=10_000, disk_gib=500)


def test_each_resource_asks_for_its_own_node_count():
    estimate = estimate_zone_nodes(
        ZoneRequirement(ncu=20, memory_gib=100, network_mbps=25_000, disk_gib=1_600),
        CONTAINER,
    )

    assert isinstance(estimate, ZoneNodeEstimate)
    by_type = estimate.node_counts.required_nodes_by_type
    assert by_type[NodeCountConstraint.cpu] == 3
    assert by_type[NodeCountConstraint.memory] == 4
    assert by_type[NodeCountConstraint.network] == 3
    assert by_type[NodeCountConstraint.disk_capacity] == 4
    assert estimate.count == 4


def test_cluster_size_rounds_before_the_minimum_applies():
    def pow2(n: int) -> int:
        return 1 << max(0, n - 1).bit_length()

    requirement = ZoneRequirement(ncu=40)  # 5 nodes of 8 NCUs

    rounded = estimate_zone_nodes(requirement, CONTAINER, cluster_size=pow2)
    floored = estimate_zone_nodes(
        requirement, CONTAINER, cluster_size=pow2, min_count=12
    )

    assert isinstance(rounded, ZoneNodeEstimate) and rounded.count == 8
    assert isinstance(floored, ZoneNodeEstimate) and floored.count == 12


def test_write_buffer_is_a_per_node_floor():
    node = CONTAINER.model_copy(update={"write_buffer_gib": 2})

    estimate = estimate_zone_nodes(ZoneRequirement(ncu=1, write_buffer_gib=9), node)

    assert isinstance(estimate, ZoneNodeEstimate) and estimate.count == 5


def test_disk_on_a_node_without_disk_is_unschedulable():
    diskless = CONTAINER.model_copy(update={"disk_gib": 0})

    estimate = estimate_zone_nodes(ZoneRequirement(ncu=1, disk_gib=10), diskless)

    assert isinstance(estimate, Unschedulable)
    assert estimate.insufficient == [Bottleneck.disk_capacity]


def test_a_node_missing_several_resources_names_each():
    bare = NodeAllocatable(ncu=8)

    estimate = estimate_zone_nodes(
        ZoneRequirement(ncu=1, memory_gib=4, network_mbps=100, disk_gib=10), bare
    )

    assert isinstance(estimate, Unschedulable)
    assert estimate.insufficient == [
        Bottleneck.memory,
        Bottleneck.network,
        Bottleneck.disk_capacity,
    ]
    assert estimate.reason == (
        "The node has no memory or network or disk_capacity for the requirement"
    )


def test_ec2_allocatable_matches_compute_stateful_zone():
    """One NCU requirement estimate on an EC2 node equals the zone the existing
    core-based entry point sizes."""
    i4i = shapes.region("us-east-1").instances["i4i.2xlarge"]
    assert i4i.drive is not None

    estimate = estimate_zone_nodes(
        ZoneRequirement(ncu=cpu_to_ncu(552, i4i), memory_gib=300, disk_gib=20_000),
        stateful_node_allocatable(i4i),
    )
    zone = compute_stateful_zone(
        instance=i4i,
        drive=shapes.region("us-east-1").drives["gp3"],
        needed_cores=552,
        needed_disk_gib=20_000,
        needed_memory_gib=300,
        needed_network_mbps=0,
    )

    assert isinstance(estimate, ZoneNodeEstimate)
    assert estimate.count == zone.count == 69
    assert estimate.node_counts.model_dump(mode="json") == zone.cluster_params
