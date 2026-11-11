"""Alternative instances planned through PlannerArguments.alternative_instances."""

import pytest

from service_capacity_modeling.capacity_planner import PlannerArguments
from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.interface import Bottleneck
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import CapacityPlan
from service_capacity_modeling.interface import DataShape
from service_capacity_modeling.interface import AlternativeInstanceOptions
from service_capacity_modeling.interface import QueryPattern
from service_capacity_modeling.interface import certain_int
from service_capacity_modeling.models.alternative_instances import planned_cluster
from service_capacity_modeling.tools.capture_baseline_costs import SCENARIOS
from tests.netflix.test_candidate_layouts import CASSANDRA
from tests.netflix.test_candidate_layouts import EVCACHE_IN_MEMORY
from tests.netflix.test_candidate_layouts import READ_ONLY_KV
from tests.netflix.test_candidate_layouts import layout_of
from tests.netflix.test_candidate_layouts import plan_at
from tests.netflix.test_candidate_layouts import plan_scenarios

REGION = "us-east-1"
ELASTICSEARCH = CapacityDesires(
    service_tier=1,
    query_pattern=QueryPattern(
        estimated_read_per_second=certain_int(10_000),
        estimated_write_per_second=certain_int(5_000),
    ),
    data_shape=DataShape(estimated_state_size_gib=certain_int(20_000)),
)
WORKLOADS = [
    ("org.netflix.cassandra", CASSANDRA, {}),
    ("org.netflix.cassandra", CASSANDRA, {"require_attached_disks": True}),
    ("org.netflix.evcache", EVCACHE_IN_MEMORY, {}),
    ("org.netflix.elasticsearch.node", ELASTICSEARCH, {}),
    ("org.netflix.read-only-kv", READ_ONLY_KV, {"total_num_partitions": 48}),
]


def _plan(model, desires, args, **options):
    return planner.plan_certain_explained(
        model_name=model,
        region=REGION,
        desires=desires,
        extra_model_arguments=args,
        num_results=20,
        planner_arguments=PlannerArguments(
            max_results_per_family=20,
            alternative_instances=AlternativeInstanceOptions(**options),
        ),
    )


def _cluster_cost(plan):
    clusters = plan.candidate_clusters
    return clusters.total_annual_cost - sum(s.annual_cost for s in clusters.services)


def _best(plans):
    return min(plans, key=lambda p: (p.rank, p.candidate_clusters.total_annual_cost))


def _own_costs(model, desires, args):
    """Each type's cluster cost on its own, as the planner picks it."""
    by_name = {}
    for *_, plan in plan_scenarios(model, desires, args):
        by_name.setdefault(planned_cluster(plan).instance.name, []).append(plan)
    return {name: _cluster_cost(_best(plans)) for name, plans in by_name.items()}


def _at_layout(model, desires, args, instance_name, layout):
    """The type's plan at ``layout``, or None if it cannot run it."""
    merged = desires.merge_with(model.default_desires(desires, args))
    plans = [
        plan_at(model, instance, drive, context, merged, args, layout)
        for instance, drive, context, _ in plan_scenarios(model, desires, args)
        if instance.name == instance_name
    ]
    plans = [p for p in plans if isinstance(p, CapacityPlan)]
    return _best(plans) if plans else None


@pytest.mark.parametrize("name, desires, args", WORKLOADS)
def test_alternatives_run_the_layout_cheapest_first_within_the_limits(
    name, desires, args
):
    model = planner.models[name]
    own_costs = _own_costs(model, desires, args)
    result = _plan(name, desires, args)
    plans = result.plans
    assert any(planned_cluster(p).alternative_instances for p in plans)
    for plan in plans:
        cluster = planned_cluster(plan)
        explained = result.alternative_plans_by_model[name][cluster.instance.name]
        own_cost = own_costs[cluster.instance.name]
        assert _cluster_cost(plan) <= own_cost * 1.2 + 1e-6
        families = {cluster.instance.family}
        costs = []
        for instance in cluster.alternative_instances:
            assert instance.family not in families
            families.add(instance.family)
            alternative = _at_layout(
                model, desires, args, instance.name, layout_of(plan)
            )
            assert alternative is not None, instance.name
            assert layout_of(alternative) == layout_of(plan)
            costs.append(_cluster_cost(alternative))
        assert costs == sorted(costs)
        assert all(c <= own_cost * 1.5 + 1e-6 for c in costs)
        # The explained alternative plans are the ones the cluster lists.
        assert [planned_cluster(p).instance for p in explained] == list(
            cluster.alternative_instances
        )
        assert [_cluster_cost(p) for p in explained] == pytest.approx(costs)


def test_the_node_cap_ignores_service_costs():
    """Cassandra's network and backup services do not scale with nodes, so the
    cap compares cluster costs: c5d.4xlarge cannot double its nodes."""
    desires = CapacityDesires(
        service_tier=1,
        query_pattern=QueryPattern(
            estimated_read_per_second=certain_int(100_000),
            estimated_write_per_second=certain_int(100_000),
        ),
        data_shape=DataShape(estimated_state_size_gib=certain_int(100)),
    )
    model = planner.models["org.netflix.cassandra"]
    own_costs = _own_costs(model, desires, {})
    plans = _plan("org.netflix.cassandra", desires, {}, min_alternatives=0).plans
    assert plans
    for plan in plans:
        own_cost = own_costs[planned_cluster(plan).instance.name]
        assert _cluster_cost(plan) <= own_cost * 1.2 + 1e-6


def test_alternatives_have_the_primary_attached_drives():
    """A local-disk primary never lists an EBS-only type, nor the reverse."""
    args = {"required_cluster_size": 13}
    model = planner.models["org.netflix.cassandra"]
    plans = planner.plan_certain(
        model_name="org.netflix.cassandra",
        region=REGION,
        desires=CASSANDRA,
        extra_model_arguments=args,
        instance_families=["r7a", "m6id"],
        planner_arguments=PlannerArguments(
            alternative_instances=AlternativeInstanceOptions(min_alternatives=0)
        ),
    )
    assert plans
    for plan in plans:
        cluster = planned_cluster(plan)
        for instance in cluster.alternative_instances:
            alternative = _at_layout(
                model, CASSANDRA, args, instance.name, layout_of(plan)
            )
            assert planned_cluster(alternative).attached_drives == (
                cluster.attached_drives
            )


def test_caller_layout_arguments_win():
    scenario = SCENARIOS["read_only_kv_medium"]
    args = {
        **scenario["extra_args"],
        "required_partitions_per_node": 2,
        "required_replica_count": 4,
    }
    plans = _plan(scenario["model"], scenario["desires"], args, min_alternatives=0)
    assert plans.plans
    for plan in plans.plans:
        assert layout_of(plan) == {
            "required_partitions_per_node": 2,
            "required_replica_count": 4,
        }


def test_a_composed_plan_needs_every_role():
    """Alone in the i4i family, data nodes have no alternatives, so no
    Elasticsearch plan is complete."""
    result = planner.plan_certain_explained(
        model_name="org.netflix.elasticsearch",
        region=REGION,
        desires=ELASTICSEARCH,
        instance_families=["i4i"],
        planner_arguments=PlannerArguments(
            alternative_instances=AlternativeInstanceOptions()
        ),
    )
    assert not result.plans
    assert result.excuses_by_model["org.netflix.elasticsearch.node"]


def test_grown_elasticsearch_plans_rank_by_cost():
    """Elasticsearch data plans have rank 0, so plans that grew to reach more
    families still order by cost."""
    plans = _plan(
        "org.netflix.elasticsearch.node",
        ELASTICSEARCH,
        {},
        min_alternatives=2,
        max_alternative_cost_increase=0.2,
    ).plans
    costs = [p.candidate_clusters.total_annual_cost for p in plans]
    assert len(plans) > 1
    assert costs == sorted(costs)


def test_a_type_that_cannot_reach_the_minimum_is_excused():
    result = _plan(
        "org.netflix.elasticsearch.node",
        ELASTICSEARCH,
        {},
        min_alternatives=5,
        max_alternatives=5,
        max_alternative_cost_increase=0.0,
    )
    excuses = result.excuses_by_model["org.netflix.elasticsearch.node"]
    excused = {e.instance for e in excuses if e.bottleneck == Bottleneck.cost}
    assert excused
    assert not excused & {planned_cluster(p).instance.name for p in result.plans}


def test_composed_elasticsearch_lists_alternatives_on_data_clusters_only():
    plans = _plan(
        "org.netflix.elasticsearch",
        ELASTICSEARCH,
        {},
        min_alternatives=0,
    ).plans
    assert plans
    for plan in plans:
        for cluster in plan.candidate_clusters.zonal:
            if cluster.cluster_type != "elasticsearch-data":
                assert not cluster.alternative_instances
    assert any(
        cluster.alternative_instances
        for plan in plans
        for cluster in plan.candidate_clusters.zonal
        if cluster.cluster_type == "elasticsearch-data"
    )


def test_models_without_layouts_are_unchanged():
    args = {"alternative_instances": AlternativeInstanceOptions()}
    desires = CapacityDesires(
        service_tier=1,
        query_pattern=QueryPattern(estimated_read_per_second=certain_int(10_000)),
    )
    with_alternatives = planner.plan_certain(
        model_name="org.netflix.stateless-java",
        region=REGION,
        desires=desires,
        planner_arguments=PlannerArguments(**args),
    )
    without = planner.plan_certain(
        model_name="org.netflix.stateless-java", region=REGION, desires=desires
    )
    assert with_alternatives == without


def test_uncertain_planning_rejects_alternative_instances():
    with pytest.raises(ValueError, match="certain planning only"):
        planner.plan(
            model_name="org.netflix.cassandra",
            region=REGION,
            desires=CASSANDRA,
            simulations=4,
            planner_arguments=PlannerArguments(
                alternative_instances=AlternativeInstanceOptions()
            ),
        )
