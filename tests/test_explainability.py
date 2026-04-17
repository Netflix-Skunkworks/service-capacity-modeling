"""Tests for core explainability types: Excuse, FamilyTrait, FamilyEdge, FamilyGraph.

These tests verify library-level contracts with no model-specific logic.
Model integration tests live in tests/netflix/test_<model>_explainability.py.
"""

import pytest

from service_capacity_modeling.explainability import (
    count_excuses,
    deduplicate_excuses,
    FamilyEdge,
    FamilyGraph,
    FamilyTrait,
    STATEFUL_DATASTORE_FAMILIES,
)
from service_capacity_modeling.hardware import shapes
from service_capacity_modeling.interface import (
    Bottleneck,
    CountedExcuse,
    Excuse,
    ExcuseTag,
)
from service_capacity_modeling.models.utils import compute_excuse_tags


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

    def test_deduplicate_preserves_distinct_bottlenecks_and_tags(self):
        excuses = [
            Excuse(
                instance="r6a.xlarge",
                drive="gp3",
                reason="too small",
                bottleneck=Bottleneck.cpu,
                tags=[ExcuseTag.same_family],
            ),
            Excuse(
                instance="r6a.xlarge",
                drive="gp3",
                reason="too small",
                bottleneck=Bottleneck.memory,
                tags=[ExcuseTag.same_family],
            ),
            Excuse(
                instance="r6a.xlarge",
                drive="gp3",
                reason="too small",
                bottleneck=Bottleneck.cpu,
                tags=[ExcuseTag.size_up],
            ),
        ]
        deduped = deduplicate_excuses(excuses)
        assert len(deduped) == 3

    def test_count_excuses_merges_same_identity_but_drops_conflicting_context(self):
        excuses = [
            Excuse(
                instance="r6a.xlarge",
                drive="gp3",
                reason="too small",
                bottleneck=Bottleneck.cpu,
                tags=[ExcuseTag.same_family],
                context={"needed_cores": 12},
            ),
            Excuse(
                instance="r6a.xlarge",
                drive="gp3",
                reason="too small",
                bottleneck=Bottleneck.cpu,
                tags=[ExcuseTag.same_family],
                context={"needed_cores": 16},
            ),
        ]
        counted = count_excuses(excuses)
        assert len(counted) == 1
        assert isinstance(counted[0], CountedExcuse)
        assert counted[0].occurrence_count == 2
        assert counted[0].world_count == 2
        assert counted[0].context == {}
        assert len(counted[0].example_worlds) == 2


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
        r6a_inst = next(
            inst for inst in hardware.instances.values() if inst.family == "r6a"
        )
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
    """Test FamilyGraph.suggest_alternatives."""

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
    assert compute_excuse_tags(excuse_inst, current_inst) == expected_tags


class TestFamilyGraphBuild:
    """Test FamilyGraph.build() construction from hardware data."""

    def test_no_excuses_uses_preferred_families(self):
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

    def test_none_preferred_gives_empty_graph(self):
        """preferred_families=None → empty base, no families imposed on the model."""
        hardware = shapes.region("us-east-1")
        graph = FamilyGraph.build(
            excuses=[],
            hardware=hardware,
            preferred_families=None,
        )
        assert len(graph.traits) == 0
        assert len(graph.edges) == 0

    def test_m_times_n_edges(self):
        """Graph has exactly n*(n-1) directed edges for n families."""
        hardware = shapes.region("us-east-1")
        graph = FamilyGraph.build(
            excuses=[],
            hardware=hardware,
            preferred_families=STATEFUL_DATASTORE_FAMILIES,
        )
        n = len(graph.traits)
        assert len(graph.edges) == n * (n - 1)

    def test_edges_use_bottleneck_enum(self):
        hardware = shapes.region("us-east-1")
        graph = FamilyGraph.build(
            excuses=[],
            hardware=hardware,
            preferred_families=STATEFUL_DATASTORE_FAMILIES,
        )
        for edge in graph.edges:
            for b in edge.improves + edge.degrades:
                assert isinstance(b, Bottleneck)
