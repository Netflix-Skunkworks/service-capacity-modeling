"""Cassandra integration tests for the explainability API.

Tests here require a full plan_certain_explained() run against the Cassandra
model. Generic type tests (Excuse, FamilyGraph, FamilyTrait) live in
tests/test_explainability.py.
"""

import pytest

from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.explainability import (
    ExplainedPlans,
    ExplainedUncertainPlans,
    STATEFUL_DATASTORE_FAMILIES,
)
from service_capacity_modeling.interface import (
    Bottleneck,
    CapacityDesires,
    ExcuseSummary,
    DataShape,
    Excuse,
    QueryPattern,
    RegretPlanSummary,
    UncertainCapacityPlan,
    certain_float,
    certain_int,
)


EXTRA_MODEL_ARGS = {"require_local_disks": False}


small_workload = CapacityDesires(
    service_tier=1,
    query_pattern=QueryPattern(
        estimated_read_per_second=certain_int(1000),
        estimated_write_per_second=certain_int(1000),
        estimated_mean_read_latency_ms=certain_float(0.5),
        estimated_mean_write_latency_ms=certain_float(0.4),
    ),
    data_shape=DataShape(
        estimated_state_size_gib=certain_int(100),
    ),
)


@pytest.fixture(scope="module")
def explained_plans():
    """Shared fixture for plan_certain_explained() with small_workload."""
    return planner.plan_certain_explained(
        model_name="org.netflix.cassandra",
        region="us-east-1",
        desires=small_workload,
        extra_model_arguments=EXTRA_MODEL_ARGS,
    )


class TestCassandraFamilyGraph:
    """Test that Cassandra's plan_certain_explained() builds the right graph."""

    def test_preferred_families_are_graph_nodes(self, explained_plans):
        # With no current_cluster in desires, all nodes come from
        # STATEFUL_DATASTORE_FAMILIES (current_shape_families is empty)
        for fam in explained_plans.family_graph.traits:
            assert fam in STATEFUL_DATASTORE_FAMILIES

    def test_stateful_datastore_families_constant(self):
        assert len(STATEFUL_DATASTORE_FAMILIES) > 0
        assert "i4i" in STATEFUL_DATASTORE_FAMILIES
        assert "r6a" in STATEFUL_DATASTORE_FAMILIES

    def test_traits_are_derived(self, explained_plans):
        traits = explained_plans.family_graph.traits
        if "i4i" in traits:
            assert traits["i4i"].has_local_disk is True
            assert traits["i4i"].local_disk_gib_per_vcpu is not None
        if "r6a" in traits:
            assert traits["r6a"].has_local_disk is False

    def test_i4i_disk_bottleneck_suggests_alternatives(self, explained_plans):
        excuse = Excuse(
            instance="i4i.4xlarge",
            drive="gp3",
            reason="Cluster too large",
            bottleneck=Bottleneck.disk_capacity,
        )
        alts = explained_plans.family_graph.suggest_alternatives(excuse)
        to_families = {a.to_family for a in alts}
        assert "i3en" in to_families
        assert "r7a" in to_families


class TestPlanCertainExplained:
    """Test plan_certain_explained() returns excuses and plans."""

    def test_returns_explained_plans(self, explained_plans):
        assert isinstance(explained_plans, ExplainedPlans)
        assert len(explained_plans.plans) > 0
        assert len(explained_plans.excuses_by_model) > 0

    def test_excuses_have_structured_fields(self, explained_plans):
        excuses = [e for es in explained_plans.excuses_by_model.values() for e in es]
        for excuse in excuses:
            assert excuse.instance
            assert excuse.drive
            assert excuse.reason

    def test_excuses_use_bottleneck_enum(self, explained_plans):
        excuses = [e for es in explained_plans.excuses_by_model.values() for e in es]
        typed_excuses = [e for e in excuses if e.bottleneck is not None]
        assert len(typed_excuses) > 0
        for excuse in typed_excuses:
            assert isinstance(excuse.bottleneck, Bottleneck)

    def test_family_graph_is_populated(self, explained_plans):
        assert len(explained_plans.family_graph.traits) > 0
        assert len(explained_plans.family_graph.edges) > 0


class TestPlanExplained:
    """Test that plan_explained() returns excuses, family graph, and plan."""

    @pytest.fixture(scope="class")
    def explained_uncertain(self):
        return planner.plan_explained(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=small_workload,
            simulations=2,
            extra_model_arguments=EXTRA_MODEL_ARGS,
        )

    def test_returns_explained_uncertain_plans(self, explained_uncertain):
        assert isinstance(explained_uncertain, ExplainedUncertainPlans)

    def test_plan_is_uncertain_capacity_plan(self, explained_uncertain):
        assert isinstance(explained_uncertain.plan, UncertainCapacityPlan)
        assert len(explained_uncertain.plan.least_regret) > 0

    def test_excuses_populated(self, explained_uncertain):
        assert len(explained_uncertain.excuses_by_model) > 0
        excuses = [
            e for es in explained_uncertain.excuses_by_model.values() for e in es
        ]
        assert all(isinstance(e, Excuse) for e in excuses)

    def test_family_graph_populated(self, explained_uncertain):
        assert len(explained_uncertain.family_graph.traits) > 0
        assert len(explained_uncertain.family_graph.edges) > 0

    def test_explanation_has_excuses_by_model(self, explained_uncertain):
        assert explained_uncertain.plan.explanation.excuses_by_model
        excuses_flat = [
            e
            for es in explained_uncertain.plan.explanation.excuses_by_model.values()
            for e in es
        ]
        assert len(excuses_flat) > 0

    def test_explanation_has_regret_clusters(self, explained_uncertain):
        assert explained_uncertain.plan.explanation.regret_clusters_by_model

    def test_explained_uncertain_has_least_regret_summaries(self, explained_uncertain):
        assert len(explained_uncertain.least_regret_summaries) == len(
            explained_uncertain.plan.least_regret
        )
        assert all(
            isinstance(summary, RegretPlanSummary)
            for summary in explained_uncertain.least_regret_summaries
        )

    def test_explained_uncertain_has_counted_excuses(self, explained_uncertain):
        assert explained_uncertain.excuse_summary
        assert all(
            isinstance(excuse, ExcuseSummary)
            for excuse in explained_uncertain.excuse_summary
        )
        assert (
            max(
                excuse.occurrence_count for excuse in explained_uncertain.excuse_summary
            )
            >= 1
        )
        assert (
            max(excuse.sample_count for excuse in explained_uncertain.excuse_summary)
            >= 1
        )
        assert explained_uncertain.excuse_summary[0].example_samples

    def test_plan_always_has_excuses(self, explained_uncertain):
        """Excuses are always populated — no explain flag needed."""
        assert explained_uncertain.plan.explanation.excuses_by_model
