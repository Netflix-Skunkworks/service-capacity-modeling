"""Tests for explainability: Excuses, FamilyGraph, ExplainedPlans."""

import pytest

from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.hardware import shapes
from service_capacity_modeling.explainability import (
    ExplainedPlans,
    FamilyEdge,
    FamilyGraph,
    FamilyTrait,
    STATEFUL_DATASTORE_FAMILIES,
)
from service_capacity_modeling.interface import (
    Bottleneck,
    CapacityDesires,
    DataShape,
    Excuse,
    ExcuseTag,
    QueryPattern,
    certain_float,
    certain_int,
)
from service_capacity_modeling.models.org.netflix.cassandra import (
    _compute_excuse_tags,
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


class TestExcuseModel:
    """Test the Excuse data model."""

    def test_excuse_basic_fields(self):
        excuse = Excuse(
            instance="r6a.2xlarge",
            drive="gp3",
            reason="Cluster too large: 128 nodes > max 64",
            bottleneck=Bottleneck.disk_capacity,
        )
        assert excuse.instance == "r6a.2xlarge"
        assert excuse.drive == "gp3"
        assert excuse.bottleneck == Bottleneck.disk_capacity
        assert not excuse.tags
        assert not excuse.context

    def test_excuse_with_context(self):
        excuse = Excuse(
            instance="i4i.2xlarge",
            drive="gp3",
            reason="Requires attached disks but i4i has local drives",
            context={"instance_drive": "local_nvme", "require_attached_disks": True},
            bottleneck=Bottleneck.drive_type,
            tags=["different_family"],
        )
        assert excuse.context["require_attached_disks"] is True
        assert "different_family" in excuse.tags

    def test_excuse_serialization_excludes_unset(self):
        excuse = Excuse(
            instance="r6a.xlarge",
            drive="gp3",
            reason="Instance too small",
        )
        data = excuse.model_dump()
        assert "bottleneck" not in data
        assert data["instance"] == "r6a.xlarge"


class TestFamilyTrait:
    """Test FamilyTrait.from_instance() derivation."""

    def test_from_instance_local_disk(self):
        hardware = shapes.region("us-east-1")
        inst = hardware.instances["i4i.8xlarge"]
        trait = FamilyTrait.from_instance(inst)
        assert trait.family == "i4i"
        assert trait.has_local_disk is True
        assert trait.memory_gib_per_vcpu == pytest.approx(7.63, abs=0.1)
        assert trait.local_disk_gib_per_vcpu is not None
        assert trait.local_disk_gib_per_vcpu > 200

    def test_from_instance_ebs(self):
        hardware = shapes.region("us-east-1")
        # r6a is EBS-only (no local drive)
        r6a_inst = None
        for inst in hardware.instances.values():
            if inst.family == "r6a":
                r6a_inst = inst
                break
        assert r6a_inst is not None
        trait = FamilyTrait.from_instance(r6a_inst)
        assert trait.family == "r6a"
        assert trait.has_local_disk is False
        assert trait.local_disk_gib_per_vcpu is None
        assert trait.drive_type is None
        assert trait.memory_gib_per_vcpu > 7.0

    def test_ratios_constant_across_sizes(self):
        hardware = shapes.region("us-east-1")
        i4i_traits = [
            FamilyTrait.from_instance(inst)
            for inst in hardware.instances.values()
            if inst.family == "i4i" and inst.cpu >= 4
        ]
        assert len(i4i_traits) >= 2
        ratios = {t.memory_gib_per_vcpu for t in i4i_traits}
        assert len(ratios) == 1


class TestFamilyGraph:
    """Test the FamilyGraph and suggest_alternatives."""

    def test_suggest_alternatives_finds_edges(self):
        graph = FamilyGraph(
            edges=[
                FamilyEdge(
                    from_family="i4i",
                    to_family="i3en",
                    trade_off="4x disk/node",
                    improves=[Bottleneck.disk_capacity],
                    degrades=[Bottleneck.disk_iops],
                ),
                FamilyEdge(
                    from_family="i4i",
                    to_family="r7a",
                    trade_off="EBS, unlimited disk",
                    improves=[Bottleneck.disk_capacity, Bottleneck.memory],
                    degrades=[Bottleneck.disk_iops],
                ),
            ],
        )
        excuse = Excuse(
            instance="i4i.2xlarge",
            drive="gp3",
            reason="Cluster too large",
            bottleneck=Bottleneck.disk_capacity,
        )
        alts = graph.suggest_alternatives(excuse)
        assert len(alts) == 2
        assert {a.to_family for a in alts} == {"i3en", "r7a"}

    def test_suggest_alternatives_no_bottleneck(self):
        graph = FamilyGraph(
            edges=[
                FamilyEdge(
                    from_family="i4i",
                    to_family="i3en",
                    trade_off="x",
                    improves=[Bottleneck.disk_capacity],
                ),
            ],
        )
        excuse = Excuse(instance="i4i.2xlarge", drive="gp3", reason="test")
        assert graph.suggest_alternatives(excuse) == []

    def test_suggest_alternatives_no_matching_edges(self):
        graph = FamilyGraph(
            edges=[
                FamilyEdge(
                    from_family="m6id",
                    to_family="m7a",
                    trade_off="x",
                    improves=[Bottleneck.disk_capacity],
                ),
            ],
        )
        excuse = Excuse(
            instance="i4i.2xlarge",
            drive="gp3",
            reason="test",
            bottleneck=Bottleneck.disk_capacity,
        )
        assert graph.suggest_alternatives(excuse) == []

    def test_empty_graph(self):
        graph = FamilyGraph()
        assert not graph.traits
        assert not graph.edges


@pytest.mark.parametrize(
    "excuse_inst, current_inst, expected_tags",
    [
        ("r6a.2xlarge", "r6a.2xlarge", [ExcuseTag.current_shape]),
        ("r6a.4xlarge", "r6a.2xlarge", [ExcuseTag.same_family, ExcuseTag.size_up]),
        ("r6a.xlarge", "r6a.2xlarge", [ExcuseTag.same_family, ExcuseTag.size_down]),
        ("i4i.2xlarge", "r6a.2xlarge", [ExcuseTag.different_family]),
        ("r6a.2xlarge", None, []),
    ],
)
def test_compute_excuse_tags(excuse_inst, current_inst, expected_tags):
    assert _compute_excuse_tags(excuse_inst, current_inst) == expected_tags


class TestAWSFamilyGraph:
    """Test that plan_certain_explained() auto-builds the M×N family graph."""

    def test_preferred_families_are_graph_nodes(self, explained_plans):
        # With no current_cluster in desires, all nodes come from
        # STATEFUL_DATASTORE_FAMILIES (current_shape_families is empty)
        for fam in explained_plans.family_graph.traits:
            assert fam in STATEFUL_DATASTORE_FAMILIES

    def test_stateful_datastore_families_are_library_default(self):
        # Sanity-check the library-level constant exists and is non-empty
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

    def test_edges_use_bottleneck_enum(self, explained_plans):
        for edge in explained_plans.family_graph.edges:
            for b in edge.improves + edge.degrades:
                assert isinstance(b, Bottleneck)

    def test_graph_is_m_times_n(self, explained_plans):
        graph = explained_plans.family_graph
        n = len(graph.traits)
        assert len(graph.edges) == n * (n - 1)

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


class TestFamilyGraphWithoutExcuses:
    """Family graph should be populated even when no excuses are generated."""

    def test_build_family_graph_no_excuses_returns_populated_graph(self):
        """FamilyGraph.build() with empty excuses uses preferred_families as base."""
        hardware = shapes.region("us-east-1")
        graph = FamilyGraph.build(
            excuses=[],
            hardware=hardware,
            preferred_families=STATEFUL_DATASTORE_FAMILIES,
        )
        assert len(graph.traits) > 0, (
            "Graph should be populated from preferred_families even with no excuses"
        )
        assert "i4i" in graph.traits
        assert "r6a" in graph.traits

    def test_build_family_graph_none_preferred_gives_empty_graph(self):
        """preferred_families=None → empty base, no families imposed on the model."""
        hardware = shapes.region("us-east-1")
        graph = FamilyGraph.build(
            excuses=[],
            hardware=hardware,
            preferred_families=None,
        )
        assert len(graph.traits) == 0
        assert len(graph.edges) == 0
