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
from service_capacity_modeling.models.org.netflix.read_only_kv import (
    _estimate_read_only_kv_cluster,
)
from service_capacity_modeling.models.org.netflix.read_only_kv import (
    NflxReadOnlyKVArguments,
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
    """i3en.6xlarge's dense layout does not fit m6id.2xlarge's disk."""
    dense = _layout(_plan("i3en.6xlarge"))

    excuse = _plan(
        "m6id.2xlarge",
        partitions_per_node=dense["read-only-kv.partitions_per_node"],
        replica_count=dense["read-only-kv.replica_count"],
    )

    assert isinstance(excuse, Excuse)
    assert excuse.bottleneck == Bottleneck.disk_capacity


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
