"""Tests for instance fallback plans built on NodeCountAdapter."""

import pytest

from service_capacity_modeling.capacity_planner import PlannerArguments, planner
from service_capacity_modeling.interface import (
    CapacityDesires,
    DataShape,
    QueryPattern,
    certain_int,
)
from service_capacity_modeling.models.instance_fallback import (
    InstanceFallbackOptions,
    plan_with_instance_fallbacks,
)
from service_capacity_modeling.models.org.netflix.instance_fallback_adapters import (
    INSTANCE_FALLBACK_ADAPTERS,
)

REGION = "us-east-1"
CASSANDRA = "org.netflix.cassandra"
DESIRES = {
    CASSANDRA: CapacityDesires(
        service_tier=1,
        query_pattern=QueryPattern(
            estimated_read_per_second=certain_int(60_000),
            estimated_write_per_second=certain_int(60_000),
        ),
        data_shape=DataShape(estimated_state_size_gib=certain_int(4_000)),
    ),
    "org.netflix.evcache": CapacityDesires(
        service_tier=1,
        query_pattern=QueryPattern(
            estimated_read_per_second=certain_int(500_000),
            estimated_write_per_second=certain_int(50_000),
            estimated_mean_read_size_bytes=certain_int(2_000),
            estimated_mean_write_size_bytes=certain_int(2_000),
        ),
        data_shape=DataShape(estimated_state_size_gib=certain_int(600)),
    ),
    "org.netflix.elasticsearch.node": CapacityDesires(
        service_tier=1,
        query_pattern=QueryPattern(
            estimated_read_per_second=certain_int(10_000),
            estimated_write_per_second=certain_int(5_000),
        ),
        data_shape=DataShape(estimated_state_size_gib=certain_int(2_000)),
    ),
}


def _singles(model, args, desires):
    """Every single-type plan, as plan_with_instance_fallbacks sees them."""
    candidates = {}
    for plan in planner.plan_certain(
        model_name=model,
        region=REGION,
        desires=desires,
        extra_model_arguments=args,
        num_results=10_000,
        planner_arguments=PlannerArguments(
            max_results_per_family=10_000, preferred_family_penalty=0
        ),
    ):
        candidates.setdefault(plan.candidate_clusters.zonal[0].instance.name, plan)
    return candidates


def _plans(model, args, desires, **options):
    return plan_with_instance_fallbacks(
        planner,
        model,
        INSTANCE_FALLBACK_ADAPTERS[model],
        REGION,
        desires,
        args,
        InstanceFallbackOptions(**options),
        num_results=100,
        max_results_per_family=100,
    ).plans


@pytest.mark.parametrize(
    "model, args",
    [
        (CASSANDRA, {}),
        (CASSANDRA, {"cluster_size_mode": "doubling"}),
        ("org.netflix.evcache", {}),
        ("org.netflix.elasticsearch.node", {}),
    ],
)
def test_single_type_at_own_count_keeps_the_planner_plan(model, args):
    candidates = _singles(model, args, DESIRES[model])
    adapter = INSTANCE_FALLBACK_ADAPTERS[model](candidates, DESIRES[model], args)
    layouts = adapter.layouts()
    assert candidates
    for name, plan in candidates.items():
        layout = next(layout for layout in layouts if adapter.serves(name, layout))
        resized = adapter.resize(plan, layout, [name])
        assert resized.candidate_clusters == plan.candidate_clusters
        assert resized.rank == pytest.approx(plan.rank)


@pytest.mark.parametrize("model", list(DESIRES))
def test_every_listed_type_fits_the_shared_count(model):
    desires = DESIRES[model]
    singles = _singles(model, {}, desires)
    plans = _plans(model, {}, desires)
    assert any(len(p.fallback_instance_types) == 3 for p in plans)
    for item in plans:
        zones = item.plan.candidate_clusters.zonal
        count = zones[0].count
        assert all(zone.count == count for zone in zones)
        for name in [item.preferred_instance_type, *item.fallback_instance_types]:
            assert singles[name].candidate_clusters.zonal[0].count <= count
        own = singles[item.preferred_instance_type]
        added = (
            item.plan.candidate_clusters.total_annual_cost
            - own.candidate_clusters.total_annual_cost
        )
        assert item.plan.rank == pytest.approx(own.rank + added)


def test_cassandra_accepts_every_listed_type_at_the_shared_count():
    """Forcing Cassandra to the shared count confirms each listed type serves
    it with the same configuration, independently of the adapter."""
    desires = DESIRES[CASSANDRA]
    plans = _plans(
        CASSANDRA,
        {},
        desires,
        # A loose budget lets plans grow to reach more families.
        min_fallback_types=0,
        max_fallback_cost_increase=1.0,
    )
    assert any(
        p.plan.candidate_clusters.zonal[0].count > p.single_type_node_count
        for p in plans
        if p.fallback_instance_types
    )
    assert any(
        p.plan.candidate_clusters.zonal[0].attached_drives
        for p in plans
        if p.fallback_instance_types
    )
    forced = {}
    for item in plans:
        cluster = item.plan.candidate_clusters.zonal[0]
        if cluster.count not in forced:
            forced[cluster.count] = _singles(
                CASSANDRA, {"required_cluster_size": cluster.count}, desires
            )
        for name in [item.preferred_instance_type, *item.fallback_instance_types]:
            checked = forced[cluster.count][name].candidate_clusters.zonal[0]
            assert checked.count == cluster.count
            for key in ("cassandra.heap.gib", "cassandra.compaction.min_threshold"):
                assert checked.cluster_params[key] == cluster.cluster_params[key]
            if cluster.attached_drives:
                # At more nodes the model sizes smaller volumes; the plan keeps
                # the preferred type's, which still fit.
                assert checked.attached_drives
                assert (
                    checked.attached_drives[0].size_gib
                    <= cluster.attached_drives[0].size_gib
                )
            else:
                assert not checked.attached_drives
