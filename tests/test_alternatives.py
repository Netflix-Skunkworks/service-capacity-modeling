"""Alternative instance types: other types that run a plan's cluster at the
same node count and layout, and callers that pin the count themselves."""

import pytest

from service_capacity_modeling.capacity_planner import PlannerArguments
from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.interface import AccessPattern
from service_capacity_modeling.interface import AlternativeOptions
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import certain_float
from service_capacity_modeling.interface import certain_int
from service_capacity_modeling.interface import DataShape
from service_capacity_modeling.interface import default_reference_shape
from service_capacity_modeling.interface import QueryPattern
from service_capacity_modeling.models.common import normalize_cores_float
from service_capacity_modeling.models.org.netflix.read_only_kv import (
    NflxReadOnlyKVCapacityModel,
)

READ_ONLY_KV = CapacityDesires(
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
READ_ONLY_KV_ARGS = {"total_num_partitions": 12}
CASSANDRA = CapacityDesires(
    service_tier=1,
    query_pattern=QueryPattern(
        estimated_read_per_second=certain_int(100_000),
        estimated_write_per_second=certain_int(100_000),
        estimated_mean_read_latency_ms=certain_float(0.5),
        estimated_mean_write_latency_ms=certain_float(0.4),
    ),
    data_shape=DataShape(estimated_state_size_gib=certain_int(10)),
)
WITH_ALTERNATIVES = PlannerArguments(alternatives=AlternativeOptions())
LAYOUT = ("read-only-kv.partitions_per_node", "read-only-kv.replica_count")


def _kv_plans(**extra):
    return planner.plan_certain(
        model_name="org.netflix.read-only-kv",
        region="us-east-1",
        desires=READ_ONLY_KV,
        extra_model_arguments={**READ_ONLY_KV_ARGS, **extra},
        num_results=1000,
        planner_arguments=PlannerArguments(max_results_per_family=100),
    )


def test_alternatives_are_off_by_default():
    plan = planner.plan_certain(
        model_name="org.netflix.read-only-kv",
        region="us-east-1",
        desires=READ_ONLY_KV,
        extra_model_arguments=READ_ONLY_KV_ARGS,
    )[0]

    assert not plan.candidate_clusters.regional[0].alternatives


def test_read_only_kv_alternatives_run_the_same_partition_layout():
    plan = planner.plan_certain(
        model_name="org.netflix.read-only-kv",
        region="us-east-1",
        desires=READ_ONLY_KV,
        extra_model_arguments=READ_ONLY_KV_ARGS,
        planner_arguments=WITH_ALTERNATIVES,
    )[0]
    main = plan.candidate_clusters.regional[0]
    layout = [main.cluster_params[k] for k in LAYOUT]
    pinned = NflxReadOnlyKVCapacityModel.same_layout_arguments(main)
    hosts = {
        c.instance.name: c
        for c in (p.candidate_clusters.regional[0] for p in _kv_plans(**pinned))
    }

    assert main.alternatives
    for alt in main.alternatives:
        assert alt.family != main.instance.family
        cluster = hosts[alt.name]
        assert cluster.count == main.count
        assert [cluster.cluster_params[k] for k in LAYOUT] == layout


def test_other_models_get_alternatives_at_the_same_count():
    plan = planner.plan_certain(
        model_name="org.netflix.cassandra",
        region="us-east-1",
        desires=CASSANDRA,
        planner_arguments=WITH_ALTERNATIVES,
    )[0]
    zones = plan.candidate_clusters.zonal
    primary = zones[0]
    families = [a.family for a in primary.alternatives]

    assert primary.alternatives
    assert all(z.alternatives == primary.alternatives for z in zones)
    assert primary.instance.family not in families
    assert len(set(families)) == len(families)

    allowed = [families[-1]]
    only = planner.plan_certain(
        model_name="org.netflix.cassandra",
        region="us-east-1",
        desires=CASSANDRA,
        planner_arguments=PlannerArguments(
            alternatives=AlternativeOptions(families=allowed)
        ),
    )[0].candidate_clusters.zonal[0]
    assert [a.family for a in only.alternatives] == allowed


def test_alternatives_are_certain_planning_only():
    with pytest.raises(ValueError, match="certain planning only"):
        planner.plan(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=CASSANDRA,
            planner_arguments=WITH_ALTERNATIVES,
        )


def test_a_caller_can_hold_the_count_and_bring_its_own_cpu():
    """How an autoscaler holding the node count fixed calls SCM: pin the
    count, give its own CPU need, and rank the plans it gets back."""
    count = _kv_plans()[0].candidate_clusters.regional[0].count

    plans = _kv_plans(required_cluster_size=count, needed_cores=140)

    assert len(plans) > 5
    for plan in plans:
        cluster = plan.candidate_clusters.regional[0]
        assert cluster.count == count
        reference_cores = normalize_cores_float(
            cluster.instance.cpu * count,
            target_shape=default_reference_shape,
            reference_shape=cluster.instance,
        )
        assert reference_cores >= 140, cluster.instance.name
