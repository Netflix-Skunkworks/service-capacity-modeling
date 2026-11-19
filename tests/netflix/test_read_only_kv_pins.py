"""Read-only KV plans at a pinned layout or node count."""

from typing import Any
from typing import Dict

import pytest
from pydantic import ValidationError

from service_capacity_modeling.hardware import shapes
from service_capacity_modeling.interface import AccessPattern
from service_capacity_modeling.interface import Bottleneck
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import CapacityPlan
from service_capacity_modeling.interface import certain_float
from service_capacity_modeling.interface import certain_int
from service_capacity_modeling.interface import DataShape
from service_capacity_modeling.interface import Excuse
from service_capacity_modeling.interface import QueryPattern
from service_capacity_modeling.models.placement import DoesNotFit
from service_capacity_modeling.models.placement import NodeResources
from service_capacity_modeling.models.placement import place
from service_capacity_modeling.models.placement import Placement
from service_capacity_modeling.models.org.netflix.read_only_kv import (
    _estimate_read_only_kv_cluster,
)
from service_capacity_modeling.models.org.netflix.read_only_kv import MIN_NODES
from service_capacity_modeling.models.org.netflix.read_only_kv import (
    NflxReadOnlyKVArguments,
)
from service_capacity_modeling.models.org.netflix.read_only_kv import (
    NflxReadOnlyKVCapacityModel,
)
from service_capacity_modeling.models.org.netflix.read_only_kv import (
    read_only_kv_requirement,
)
from service_capacity_modeling.models.org.netflix.read_only_kv import (
    read_only_kv_node,
)

HARDWARE = shapes.region("us-east-1")
DESIRES = CapacityDesires(
    service_tier=1,
    query_pattern=QueryPattern(
        access_pattern=AccessPattern.latency,
        estimated_read_per_second=certain_int(50_000),
        estimated_write_per_second=certain_int(0),
        estimated_mean_read_latency_ms=certain_float(2.0),
        estimated_mean_write_latency_ms=certain_float(0),
        estimated_mean_read_size_bytes=certain_int(2048),
    ),
    data_shape=DataShape(estimated_state_size_gib=certain_int(1000)),
)
LAYOUT_KEYS = (
    "read-only-kv.partitions_per_node",
    "read-only-kv.replica_count",
    "read-only-kv.nodes_for_one_copy",
)


def _plan(instance: str, desires: CapacityDesires = DESIRES, **pins: Any) -> Any:
    args = NflxReadOnlyKVArguments(total_num_partitions=12, **pins)
    return _estimate_read_only_kv_cluster(HARDWARE.instances[instance], desires, args)


def _layout(plan: CapacityPlan) -> Dict[str, Any]:
    cluster = plan.candidate_clusters.regional[0]
    return {
        "count": cluster.count,
        **{k: cluster.cluster_params[k] for k in LAYOUT_KEYS},
    }


def test_pinning_a_plans_own_layout_reproduces_the_plan():
    planned = 0
    for name, instance in HARDWARE.instances.items():
        searched = _plan(name)
        if not isinstance(searched, CapacityPlan):
            continue
        planned += 1
        layout = _layout(searched)
        pinned = _plan(
            name,
            required_cluster_size=layout["count"],
            partitions_per_node=layout["read-only-kv.partitions_per_node"],
            replica_count=layout["read-only-kv.replica_count"],
        )
        assert isinstance(pinned, CapacityPlan), (name, pinned)
        assert pinned.model_dump() == searched.model_dump(), instance.name
    assert planned > 10


def test_a_shape_that_cannot_host_a_pinned_layout_is_excused():
    """i3en.6xlarge's dense layout does not fit m6id.2xlarge's disk, and its
    copies are too few for m6id.2xlarge's CPU."""
    dense = _layout(_plan("i3en.6xlarge"))

    excuse = _plan(
        "m6id.2xlarge",
        partitions_per_node=dense["read-only-kv.partitions_per_node"],
        replica_count=dense["read-only-kv.replica_count"],
    )

    assert isinstance(excuse, Excuse)
    assert excuse.bottleneck == Bottleneck.disk_capacity
    assert excuse.context == {"bottlenecks": ["disk_capacity", "cpu"]}


def test_a_required_count_names_the_short_resource():
    """4 nodes of m6id.2xlarge cannot serve 63 cores at any layout."""
    excuse = _plan("m6id.2xlarge", required_cluster_size=4, needed_cores=63)

    assert isinstance(excuse, Excuse)
    assert excuse.reason == "Cannot host 4 nodes"
    assert excuse.bottleneck == Bottleneck.cpu


def test_a_required_count_uses_the_densest_layout_within_the_cap():
    """A caller can cap partitions per node, so a deploy never puts more data
    on a node than the plan it came from."""
    small = DESIRES.model_copy(deep=True)
    small.data_shape.estimated_state_size_gib = certain_int(48)

    def placed(**pins: Any) -> tuple:
        plan = _plan("r6id.4xlarge", small, required_cluster_size=6, **pins)
        assert isinstance(plan, CapacityPlan), plan
        layout = _layout(plan)
        assert layout["count"] == 6
        return (
            layout["read-only-kv.partitions_per_node"],
            layout["read-only-kv.replica_count"],
        )

    assert placed(needed_cores=63) == (12, 6)
    assert placed(needed_cores=63, max_partitions_per_node=6) == (6, 3)


def test_a_layout_pin_needs_both_density_and_replicas():
    with pytest.raises(ValidationError):
        NflxReadOnlyKVArguments(total_num_partitions=12, partitions_per_node=4)


def test_a_container_size_is_placed_from_the_same_demand():
    """A node that is not an EC2 shape, such as a container, places the same
    requirement; one too small for a partition says which resource is short."""
    desires = DESIRES.merge_with(
        NflxReadOnlyKVCapacityModel.default_desires(DESIRES, {})
    )
    args = NflxReadOnlyKVArguments(total_num_partitions=12)
    requirement = read_only_kv_requirement(desires, args)
    assert requirement is not None

    limits = {"min_count": MIN_NODES, "max_count": args.max_regional_size}

    container = NodeResources(ncu=12, disk_gib=600, memory_gib=40)
    placed = place(requirement, container, **limits)
    assert isinstance(placed, Placement)
    assert requirement.partitions is not None and placed.partitions is not None
    per_partition_gib = requirement.partitions.disk_gib_per_partition
    assert placed.partitions.partitions_per_node * per_partition_gib <= 600
    assert placed.partitions.cpu_utilization <= 1

    too_small = place(requirement, NodeResources(ncu=12, disk_gib=50), **limits)
    assert isinstance(too_small, DoesNotFit)
    assert too_small.bottlenecks == [Bottleneck.disk_capacity]


def test_an_ec2_node_places_like_the_model_plans():
    desires = DESIRES.merge_with(
        NflxReadOnlyKVCapacityModel.default_desires(DESIRES, {})
    )
    args = NflxReadOnlyKVArguments(total_num_partitions=12)
    requirement = read_only_kv_requirement(desires, args)
    assert requirement is not None
    compared = 0
    for name, instance in HARDWARE.instances.items():
        node = read_only_kv_node(instance, desires, args)
        plan = _estimate_read_only_kv_cluster(instance, desires, args)
        if node is None or not isinstance(plan, CapacityPlan):
            continue
        compared += 1
        placed = place(
            requirement, node, min_count=MIN_NODES, max_count=args.max_regional_size
        )
        assert isinstance(placed, Placement), name
        assert placed.partitions is not None, name
        assert _layout(plan) == {
            "count": placed.node_count,
            "read-only-kv.partitions_per_node": placed.partitions.partitions_per_node,
            "read-only-kv.replica_count": placed.partitions.replica_count,
            "read-only-kv.nodes_for_one_copy": placed.partitions.nodes_for_one_copy,
        }, name
    assert compared > 10


def test_a_cpu_need_on_a_whole_number_of_cores_plans_no_extra_node():
    """966 NCUs is exactly 552 i4i.2xlarge vCPUs (1.75 NCUs each), so 69
    nodes of 8. Converting the need to the shape's vCPUs first gives
    552.0000000000001, which rounded up to 553 vCPUs and 70 nodes."""
    args = NflxReadOnlyKVArguments(
        total_num_partitions=1, min_replica_count=1, needed_cores=966
    )
    plan = _estimate_read_only_kv_cluster(
        HARDWARE.instances["i4i.2xlarge"], DESIRES, args
    )
    assert isinstance(plan, CapacityPlan)
    assert plan.candidate_clusters.regional[0].count == 69
    assert plan.requirements.regional[0].cpu_cores.mid == 552
