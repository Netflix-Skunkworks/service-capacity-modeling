"""Tests for the Explainability mixin: Excuses, FamilyGraph, and ExplainedPlans."""

import pytest

from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.interface import (
    CapacityDesires,
    DataShape,
    Excuse,
    ExplainedPlans,
    FamilyEdge,
    FamilyGraph,
    FamilyTrait,
    QueryPattern,
    certain_float,
    certain_int,
)
from service_capacity_modeling.models.org.netflix.cassandra import (
    NflxCassandraCapacityModel,
    _compute_excuse_tags,
)


EXTRA_MODEL_ARGS = {"require_local_disks": False}


# A workload that produces excuses: tiny data + low QPS means small instances
# are rejected for being too small, and large clusters are rejected for other
# families.
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
            bottleneck="disk",
        )
        assert excuse.instance == "r6a.2xlarge"
        assert excuse.drive == "gp3"
        assert excuse.bottleneck == "disk"
        assert not excuse.tags
        assert not excuse.context

    def test_excuse_with_context(self):
        excuse = Excuse(
            instance="i4i.2xlarge",
            drive="gp3",
            reason="Requires attached disks but i4i has local drives",
            context={"instance_drive": "local_nvme", "require_attached_disks": True},
            bottleneck="drive_type",
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
        # bottleneck was not set, should not appear
        assert "bottleneck" not in data
        # Required fields should be present
        assert data["instance"] == "r6a.xlarge"


class TestFamilyGraph:
    """Test the FamilyGraph and suggest_alternatives."""

    def test_suggest_alternatives_finds_edges(self):
        graph = FamilyGraph(
            families={
                "i4i": FamilyTrait(family="i4i", storage_type="local_nvme"),
                "i3en": FamilyTrait(family="i3en", storage_type="local_nvme"),
                "r7a": FamilyTrait(family="r7a", storage_type="ebs"),
            },
            edges=[
                FamilyEdge(
                    from_family="i4i",
                    to_family="i3en",
                    trade_off="4x disk/node",
                    improves=["disk_capacity"],
                    degrades=["iops_per_gib"],
                ),
                FamilyEdge(
                    from_family="i4i",
                    to_family="r7a",
                    trade_off="EBS, unlimited disk",
                    improves=["disk_capacity", "memory"],
                    degrades=["iops_latency"],
                ),
            ],
        )
        excuse = Excuse(
            instance="i4i.2xlarge",
            drive="gp3",
            reason="Cluster too large",
            bottleneck="disk_capacity",
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
                    improves=["disk"],
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
                    improves=["disk_capacity"],
                ),
            ],
        )
        excuse = Excuse(
            instance="i4i.2xlarge",
            drive="gp3",
            reason="test",
            bottleneck="disk_capacity",
        )
        # No edges from i4i
        assert graph.suggest_alternatives(excuse) == []

    def test_empty_graph(self):
        graph = FamilyGraph()
        assert not graph.families
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
    """Test the Cassandra model's family graph."""

    def test_family_graph_has_expected_families(self):
        graph = NflxCassandraCapacityModel.family_graph()
        expected = {"i4i", "m6id", "i3en", "r5d", "r6a", "m7a", "r7a"}
        assert set(graph.families.keys()) == expected

    def test_family_graph_has_edges(self):
        graph = NflxCassandraCapacityModel.family_graph()
        assert len(graph.edges) > 0
        # All edges reference known families
        known = set(graph.families.keys())
        for edge in graph.edges:
            assert edge.from_family in known, f"Unknown from_family: {edge.from_family}"
            assert edge.to_family in known, f"Unknown to_family: {edge.to_family}"

    def test_i4i_disk_bottleneck_suggests_alternatives(self):
        graph = NflxCassandraCapacityModel.family_graph()
        excuse = Excuse(
            instance="i4i.4xlarge",
            drive="gp3",
            reason="Cluster too large",
            bottleneck="disk_capacity",
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
            assert excuse.instance, "Excuse must have an instance"
            assert excuse.drive, "Excuse must have a drive"
            assert excuse.reason, "Excuse must have a reason"

    def test_excuses_include_drive_type_rejections(self, explained_plans):
        drive_type_excuses = [
            e for e in explained_plans.excuses if e.bottleneck == "drive_type"
        ]
        assert len(drive_type_excuses) > 0

    def test_family_graph_is_populated(self, explained_plans):
        assert len(explained_plans.family_graph.families) > 0
        assert len(explained_plans.family_graph.edges) > 0


class TestExplainedPlansRender:
    """Test the markdown render method."""

    def test_render_empty_excuses(self):
        ep = ExplainedPlans(plans=[])
        assert ep.render() == ""

    def test_render_groups_by_tag(self):
        excuses = [
            Excuse(
                instance="r6a.2xlarge",
                drive="gp3",
                reason="Cluster too large",
                tags=["current_shape"],
                bottleneck="disk",
            ),
            Excuse(
                instance="r6a.4xlarge",
                drive="gp3",
                reason="Still too large",
                tags=["same_family", "size_up"],
            ),
            Excuse(
                instance="i4i.2xlarge",
                drive="gp3",
                reason="Has local drives",
                tags=["different_family"],
                bottleneck="drive_type",
            ),
        ]
        graph = FamilyGraph(
            edges=[
                FamilyEdge(
                    from_family="r6a",
                    to_family="r7a",
                    trade_off="Newer gen",
                    improves=["disk", "generation"],
                ),
            ],
        )
        ep = ExplainedPlans(plans=[], excuses=excuses, family_graph=graph)
        md = ep.render()

        assert "## Why shapes were rejected" in md
        assert "### Current shape" in md
        assert "### Same family" in md
        assert "### Different families" in md
        assert "### Family trade-off map" in md
        assert "r6a.2xlarge" in md
        assert "Bottleneck: disk" in md

    def test_render_with_live_data(self, explained_plans):
        """Render from actual planner output."""
        md = explained_plans.render()
        if explained_plans.excuses:
            assert "## Why shapes were rejected" in md


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
