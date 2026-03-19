"""Cassandra integration tests for the explainability API.

Tests here require a full plan_certain_explained() run against the Cassandra
model. Generic type tests (Excuse, FamilyGraph, FamilyTrait) live in
tests/test_explainability.py.
"""

import pytest

from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.explainability import (
    ExplainedPlans,
    STATEFUL_DATASTORE_FAMILIES,
)
from service_capacity_modeling.interface import (
    Bottleneck,
    CapacityDesires,
    DataShape,
    Excuse,
    QueryPattern,
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
        assert len(explained_plans.excuses) > 0

    def test_excuses_have_structured_fields(self, explained_plans):
        for excuse in explained_plans.excuses:
            assert excuse.instance
            assert excuse.drive
            assert excuse.reason

    def test_excuses_use_bottleneck_enum(self, explained_plans):
        typed_excuses = [e for e in explained_plans.excuses if e.bottleneck is not None]
        assert len(typed_excuses) > 0
        for excuse in typed_excuses:
            assert isinstance(excuse.bottleneck, Bottleneck)

    def test_family_graph_is_populated(self, explained_plans):
        assert len(explained_plans.family_graph.traits) > 0
        assert len(explained_plans.family_graph.edges) > 0


class TestPlanExplainFlag:
    """Test that plan(explain=True) populates excuses_by_model."""

    def test_plan_explain_includes_excuses(self):
        result = planner.plan(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=small_workload,
            simulations=2,
            explain=True,
            extra_model_arguments=EXTRA_MODEL_ARGS,
        )
        assert result.explanation.excuses_by_model, (
            "explain=True should produce excuses for a real workload"
        )
        excuses_flat = [
            e for es in result.explanation.excuses_by_model.values() for e in es
        ]
        assert len(excuses_flat) > 0
        assert all(isinstance(e, Excuse) for e in excuses_flat)

    def test_plan_explain_false_has_no_excuses(self):
        result = planner.plan(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=small_workload,
            simulations=2,
            explain=False,
            extra_model_arguments=EXTRA_MODEL_ARGS,
        )
        assert not result.explanation.excuses_by_model
