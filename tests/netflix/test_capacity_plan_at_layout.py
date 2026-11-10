"""Contract tests for models that can run mixed instance types at one layout."""

from typing import Any
from typing import Dict
from typing import Hashable
from typing import List
from typing import Tuple

import pytest

from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.interface import Buffer
from service_capacity_modeling.interface import BufferComponent
from service_capacity_modeling.interface import Buffers
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import CapacityPlan
from service_capacity_modeling.interface import DataShape
from service_capacity_modeling.interface import Drive
from service_capacity_modeling.interface import Instance
from service_capacity_modeling.interface import QueryPattern
from service_capacity_modeling.interface import RegionContext
from service_capacity_modeling.interface import certain_int
from service_capacity_modeling.models import CapacityModel

REGION = "us-east-1"

CASSANDRA = CapacityDesires(
    service_tier=1,
    query_pattern=QueryPattern(
        estimated_read_per_second=certain_int(60_000),
        estimated_write_per_second=certain_int(60_000),
    ),
    data_shape=DataShape(estimated_state_size_gib=certain_int(4_000)),
)
EVCACHE = CapacityDesires(
    service_tier=1,
    query_pattern=QueryPattern(
        estimated_read_per_second=certain_int(500_000),
        estimated_write_per_second=certain_int(50_000),
        estimated_mean_read_size_bytes=certain_int(2_000),
        estimated_mean_write_size_bytes=certain_int(2_000),
    ),
    data_shape=DataShape(estimated_state_size_gib=certain_int(600)),
)
EVCACHE_IN_MEMORY = CapacityDesires(
    service_tier=1,
    query_pattern=QueryPattern(
        estimated_read_per_second=certain_int(100_000),
        estimated_write_per_second=certain_int(10_000),
    ),
    data_shape=DataShape(estimated_state_size_gib=certain_int(10)),
)
ELASTICSEARCH = CapacityDesires(
    service_tier=1,
    query_pattern=QueryPattern(
        estimated_read_per_second=certain_int(10_000),
        estimated_write_per_second=certain_int(5_000),
    ),
    data_shape=DataShape(estimated_state_size_gib=certain_int(2_000)),
)
READ_ONLY_KV = CapacityDesires(
    service_tier=1,
    query_pattern=QueryPattern(estimated_read_per_second=certain_int(50_000)),
    data_shape=DataShape(estimated_state_size_gib=certain_int(1_000)),
)
READ_ONLY_KV_IN_MEMORY = READ_ONLY_KV.model_copy(
    update={
        "buffers": Buffers(
            desired={
                "sst-memory": Buffer(ratio=1.2, components=[BufferComponent.memory])
            }
        )
    }
)

# Every model that overrides shared_layouts needs at least one workload here.
WORKLOADS: List[Tuple[str, CapacityDesires, Dict[str, Any]]] = [
    ("org.netflix.cassandra", CASSANDRA, {}),
    ("org.netflix.cassandra", CASSANDRA, {"require_attached_disks": True}),
    ("org.netflix.cassandra", CASSANDRA, {"cluster_size_mode": "doubling"}),
    ("org.netflix.evcache", EVCACHE, {}),
    ("org.netflix.evcache", EVCACHE_IN_MEMORY, {}),
    ("org.netflix.elasticsearch.node", ELASTICSEARCH, {}),
    ("org.netflix.read-only-kv", READ_ONLY_KV, {"total_num_partitions": 48}),
    (
        "org.netflix.read-only-kv",
        READ_ONLY_KV_IN_MEMORY,
        {"total_num_partitions": 48, "sst_memory_overhead_ratio": 0.05},
    ),
]

Scenario = Tuple[Instance, Drive, RegionContext, CapacityPlan]


def _scenarios(
    model: CapacityModel, desires: CapacityDesires, args: Dict[str, Any]
) -> List[Scenario]:
    merged = desires.merge_with(model.default_desires(desires, args))
    scenarios = []
    for instance, drive, context in planner.generate_scenarios(
        model, REGION, merged, 3, None, None, None
    ):
        plan = model.capacity_plan(
            instance=instance,
            drive=drive,
            context=context,
            desires=merged,
            extra_model_arguments=args,
        )
        if isinstance(plan, CapacityPlan):
            scenarios.append((instance, drive, context, plan))
    return scenarios


def _layout_of(plan: CapacityPlan) -> Hashable:
    clusters = plan.candidate_clusters
    if clusters.zonal:
        return clusters.zonal[0].count
    params = clusters.regional[0].cluster_params
    return (
        params["read-only-kv.nodes_for_one_copy"],
        params["read-only-kv.replica_count"],
    )


def _planned_cluster(plan: CapacityPlan):
    clusters = plan.candidate_clusters
    return (clusters.zonal or clusters.regional)[0]


def test_every_model_with_layouts_has_a_workload():
    with_layouts = {
        name
        for name, model in planner.models.items()
        if type(model).shared_layouts is not CapacityModel.shared_layouts
    }
    assert with_layouts == {name for name, _, _ in WORKLOADS}


@pytest.mark.parametrize("name, desires, args", WORKLOADS)
def test_own_layout_reproduces_the_plan(name, desires, args):
    """A type planned at the layout its own plan uses gets that plan back, and
    the model lists that layout."""
    model = planner.models[name]
    scenarios = _scenarios(model, desires, args)
    assert scenarios
    layouts = model.shared_layouts([s[3] for s in scenarios], args)
    assert layouts is not None
    merged = desires.merge_with(model.default_desires(desires, args))
    for instance, drive, context, plan in scenarios:
        layout = _layout_of(plan)
        assert layout in layouts
        at_layout = model.capacity_plan_at_layout(
            instance, drive, context, merged, args, layout
        )
        assert isinstance(at_layout, CapacityPlan), instance.name
        own = _planned_cluster(plan)
        got = _planned_cluster(at_layout)
        assert got.count == own.count
        assert model.launch_config(got) == model.launch_config(own)
        assert (
            at_layout.candidate_clusters.annual_costs
            == plan.candidate_clusters.annual_costs
        )
        assert at_layout.rank == pytest.approx(plan.rank)


@pytest.mark.parametrize("name, desires, args", WORKLOADS)
def test_plans_at_a_layout_use_exactly_that_layout(name, desires, args):
    model = planner.models[name]
    scenarios = _scenarios(model, desires, args)
    layouts = model.shared_layouts([s[3] for s in scenarios], args)
    assert layouts
    merged = desires.merge_with(model.default_desires(desires, args))
    planned = 0
    for layout in layouts[:6]:
        for instance, drive, context, _ in scenarios:
            at_layout = model.capacity_plan_at_layout(
                instance, drive, context, merged, args, layout
            )
            if isinstance(at_layout, CapacityPlan):
                planned += 1
                assert _layout_of(at_layout) == layout
    assert planned
