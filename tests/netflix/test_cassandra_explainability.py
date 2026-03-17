"""Tests for the Explainability mixin: Excuses, FamilyGraph, and ExplainedPlans."""

import pytest

from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.hardware import shapes
from service_capacity_modeling.explainability import (
    ExplainedPlans,
    FamilyEdge,
    FamilyGraph,
    FamilyTrait,
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
from service_capacity_modeling.models.org.netflix.cassandra import (
    NflxCassandraCapacityModel,
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
        ("r6a.2xlarge", "r6a.2xlarge", ["current_shape"]),
        ("r6a.4xlarge", "r6a.2xlarge", ["same_family", "size_up"]),
        ("r6a.xlarge", "r6a.2xlarge", ["same_family", "size_down"]),
        ("i4i.2xlarge", "r6a.2xlarge", ["different_family"]),
        ("r6a.2xlarge", None, []),
    ],
)
def test_compute_excuse_tags(excuse_inst, current_inst, expected_tags):
    assert _compute_excuse_tags(excuse_inst, current_inst) == expected_tags


class TestCassandraFamilyGraph:
    """Test the Cassandra model's family graph with derived traits."""

    @pytest.fixture(scope="class")
    def graph(self):
        hardware = shapes.region("us-east-1")
        return NflxCassandraCapacityModel.family_graph(hardware)

    def test_family_graph_has_expected_families(self, graph):
        expected = {"i4i", "m6id", "i3en", "r5d", "r6a", "m7a", "r7a"}
        assert set(graph.traits.keys()) == expected

    def test_traits_are_derived(self, graph):
        i4i = graph.traits["i4i"]
        assert i4i.has_local_disk is True
        assert i4i.memory_gib_per_vcpu > 0
        assert i4i.local_disk_gib_per_vcpu is not None
        r6a = graph.traits["r6a"]
        assert r6a.has_local_disk is False
        assert r6a.local_disk_gib_per_vcpu is None

    def test_family_graph_has_edges(self, graph):
        assert len(graph.edges) > 0
        known = set(graph.traits.keys())
        for edge in graph.edges:
            assert edge.from_family in known
            assert edge.to_family in known

    def test_edges_use_bottleneck_enum(self, graph):
        for edge in graph.edges:
            for b in edge.improves:
                assert isinstance(b, Bottleneck)
            for b in edge.degrades:
                assert isinstance(b, Bottleneck)

    def test_i4i_disk_bottleneck_suggests_alternatives(self, graph):
        excuse = Excuse(
            instance="i4i.4xlarge",
            drive="gp3",
            reason="Cluster too large",
            bottleneck=Bottleneck.disk_capacity,
        )
        alts = graph.suggest_alternatives(excuse)
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

    def test_no_unsupported_drive_excuses(self, explained_plans):
        """With allowed_cloud_drives=('gp3',), no aurora/io2/gp2 excuses."""
        bad_drives = {"aurora", "io2", "gp2"}
        for excuse in explained_plans.excuses:
            assert excuse.drive not in bad_drives

    def test_family_graph_is_populated(self, explained_plans):
        assert len(explained_plans.family_graph.traits) > 0
        assert len(explained_plans.family_graph.edges) > 0

    def test_excuses_filtered_to_relevant_families(self, explained_plans):
        """All excuses should be from families in the model's family graph."""
        relevant = set(explained_plans.family_graph.traits.keys())
        for excuse in explained_plans.excuses:
            family = excuse.instance.rsplit(".", 1)[0]
            assert family in relevant, (
                f"Excuse for {excuse.instance} is from family {family} "
                f"which is not in relevant families {relevant}"
            )

    def test_excuse_count_is_reasonable(self, explained_plans):
        """With gp3-only + family filtering, should be <50 excuses."""
        assert len(explained_plans.excuses) < 50


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
        assert hasattr(result.explanation, "excuses_by_model")
        if result.explanation.excuses_by_model:
            for model_name, excuses in result.explanation.excuses_by_model.items():
                assert isinstance(model_name, str)
                for excuse in excuses:
                    assert isinstance(excuse, Excuse)

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
