"""Tests for read-only KV interchangeable capacity plans."""

import itertools
from typing import Dict
from typing import Tuple

import pytest

from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.interface import AccessPattern
from service_capacity_modeling.interface import Buffer
from service_capacity_modeling.interface import BufferComponent
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import CapacityPlan
from service_capacity_modeling.interface import certain_float
from service_capacity_modeling.interface import certain_int
from service_capacity_modeling.interface import DataShape
from service_capacity_modeling.interface import QueryPattern
from service_capacity_modeling.models.interchangeable import InterchangeableOptions
from service_capacity_modeling.models.interchangeable import InterchangeablePlans
from service_capacity_modeling.models.interchangeable import plan_interchangeable
from service_capacity_modeling.models.org.netflix.interchangeable_models import (
    INTERCHANGEABLE_MODELS,
)
from service_capacity_modeling.models.org.netflix.partition_aware_algorithm import (
    search_for_min_nodes,
)
from service_capacity_modeling.models.org.netflix.read_only_kv import (
    NflxReadOnlyKVCapacityModel,
)
from service_capacity_modeling.models.org.netflix.read_only_kv_interchangeable import (
    ReadOnlyKVInterchangeable,
)

MODEL_NAME = "org.netflix.read-only-kv"
REGION = "us-east-1"
ARGS = {"total_num_partitions": 48}
FAMILIES = ["m6id", "r5d", "r6id", "i4i"]


def _desires(rps: int, state_gib: int, memory_buffer: bool) -> CapacityDesires:
    buffers = NflxReadOnlyKVCapacityModel.default_buffers()
    if memory_buffer:
        # An explicit memory buffer turns on SST memory placement.
        buffers.desired["memory_reserve"] = Buffer(
            ratio=1.2, components=[BufferComponent.memory]
        )
    return CapacityDesires(
        service_tier=1,
        query_pattern=QueryPattern(
            access_pattern=AccessPattern.latency,
            estimated_read_per_second=certain_int(rps),
            estimated_write_per_second=certain_int(0),
            estimated_mean_read_latency_ms=certain_float(2.0),
            estimated_mean_write_latency_ms=certain_float(0),
            estimated_mean_read_size_bytes=certain_int(2048),
        ),
        data_shape=DataShape(estimated_state_size_gib=certain_int(state_gib)),
        buffers=buffers,
    )


DISK_ONLY = _desires(rps=50_000, state_gib=1_000, memory_buffer=False)
MEMORY_PLACEMENT = _desires(rps=50_000, state_gib=13_000, memory_buffer=True)
WORKLOADS = pytest.mark.parametrize(
    "desires", [DISK_ONLY, MEMORY_PLACEMENT], ids=["disk-only", "memory-placement"]
)


def _model(
    desires: CapacityDesires,
) -> Tuple[Dict[str, CapacityPlan], ReadOnlyKVInterchangeable]:
    candidates: Dict[str, CapacityPlan] = {}
    for plan in planner.plan_certain(
        model_name=MODEL_NAME,
        region=REGION,
        desires=desires,
        extra_model_arguments=ARGS,
        num_results=10_000,
        max_results_per_family=10_000,
    ):
        candidates.setdefault(plan.candidate_clusters.regional[0].instance.name, plan)
    return candidates, ReadOnlyKVInterchangeable(candidates, desires, ARGS)


def _plans(desires: CapacityDesires, max_fallback_types: int) -> InterchangeablePlans:
    return plan_interchangeable(
        planner,
        model_name=MODEL_NAME,
        model_factory=INTERCHANGEABLE_MODELS[MODEL_NAME],
        region=REGION,
        desires=desires,
        extra_model_arguments=ARGS,
        options=InterchangeableOptions(max_fallback_types=max_fallback_types),
        instance_families=FAMILIES,
        num_results=10,
    )


@WORKLOADS
def test_each_type_alone_gets_back_the_planners_layout(desires):
    candidates, model = _model(desires)
    layouts = model.layouts()
    assert candidates
    for name, plan in candidates.items():
        layout = next(layout for layout in layouts if model.serves(name, layout))
        resized = model.resize(plan, layout, [name])
        cluster = plan.candidate_clusters.regional[0]
        assert resized.candidate_clusters.regional[0].count == cluster.count
        for param in ("replica_count", "partitions_per_node", "nodes_for_one_copy"):
            key = f"read-only-kv.{param}"
            assert (
                resized.candidate_clusters.regional[0].cluster_params[key]
                == cluster.cluster_params[key]
            ), (name, param)
        assert resized.rank == pytest.approx(plan.rank)


def test_memory_workload_caps_partitions_by_memory():
    _, model = _model(MEMORY_PLACEMENT)
    assert all(p.memory_per_partition_gib is not None for p in model.problems.values())


@WORKLOADS
def test_layout_holds_on_every_listed_type(desires):
    _, model = _model(desires)
    result = _plans(desires, max_fallback_types=2)
    assert result.plans
    for interchangeable in result.plans:
        cluster = interchangeable.plan.candidate_clusters.regional[0]
        params = cluster.cluster_params
        ppn = params["read-only-kv.partitions_per_node"]
        rf = params["read-only-kv.replica_count"]
        n1 = params["read-only-kv.nodes_for_one_copy"]
        names = [
            interchangeable.preferred_instance_type,
            *interchangeable.fallback_instance_types,
        ]
        families = {planner.instance(n, region=REGION).family for n in names}
        assert len(families) == len(names)
        assert cluster.count >= interchangeable.single_type_node_count
        for name in names:
            # Checked against the planner's own limits for a fleet of only
            # this type, not against the adapter's serves().
            problem = model.problems[name]
            limits = search_for_min_nodes(problem)
            assert limits is not None
            assert ppn <= limits.max_partitions_per_node_by_disk
            if limits.max_partitions_per_node_by_memory is not None:
                assert ppn <= limits.max_partitions_per_node_by_memory
            assert ppn * n1 >= problem.n_partitions
            assert n1 * rf * problem.cpu_per_node >= problem.cpu_needed


@pytest.mark.parametrize("max_fallback_types", [1, 2])
def test_fallbacks_add_the_fewest_nodes_any_family_set_could(max_fallback_types):
    candidates, model = _model(DISK_ONLY)
    layouts = model.layouts()
    family = {
        n: p.candidate_clusters.regional[0].instance.family
        for n, p in candidates.items()
    }
    by_name = {
        p.preferred_instance_type: p
        for p in plan_interchangeable(
            planner,
            model_name=MODEL_NAME,
            model_factory=INTERCHANGEABLE_MODELS[MODEL_NAME],
            region=REGION,
            desires=DISK_ONLY,
            extra_model_arguments=ARGS,
            options=InterchangeableOptions(max_fallback_types=max_fallback_types),
            num_results=10_000,
            max_results_per_family=10_000,
        ).plans
    }
    served = {
        n: {i for i, layout in enumerate(layouts) if model.serves(n, layout)}
        for n in candidates
    }
    nodes = [max(n1 * rf, 2) for n1, rf in layouts]
    for name, chosen in by_name.items():
        if len(chosen.fallback_instance_types) < max_fallback_types:
            continue
        others = [n for n in candidates if family[n] != family[name]]
        fewest = min(
            nodes[i]
            for combo in itertools.combinations(others, max_fallback_types)
            if len({family[n] for n in combo}) == max_fallback_types
            for i in served[name].intersection(*(served[n] for n in combo))
        )
        assert chosen.plan.candidate_clusters.regional[0].count == fewest, name


def test_one_fallback_per_family_up_to_the_limit():
    for interchangeable in _plans(DISK_ONLY, max_fallback_types=3).plans:
        assert len(interchangeable.fallback_instance_types) == 3


def test_resized_plan_ranks_on_its_new_cost():
    for interchangeable in _plans(DISK_ONLY, max_fallback_types=2).plans:
        clusters = interchangeable.plan.candidate_clusters
        assert interchangeable.plan.rank >= clusters.total_annual_cost


def test_reports_dominance_between_candidates():
    edges = _plans(DISK_ONLY, max_fallback_types=2).dominance_edges
    assert edges
    assert all(edge.weaker != edge.stronger for edge in edges)
