"""Tests for core explainability types.

These tests verify library-level contracts with no model-specific logic.
Model integration tests live in tests/netflix/test_<model>_explainability.py.
"""

import pytest

from service_capacity_modeling.explainability import (
    count_sample_excuses,
    deduplicate_excuses,
    SampledExcuse,
)
from service_capacity_modeling.interface import (
    Bottleneck,
    ExcuseSummary,
    Excuse,
    ExcuseTag,
    SampleRef,
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
            Excuse(
                instance="r6a.xlarge",
                drive="gp3",
                reason="too small",
                bottleneck=Bottleneck.cpu,
                tags=[ExcuseTag.same_family],
            ),
        ]
        counted = count_sample_excuses(
            [
                SampledExcuse(
                    source_model="model-a",
                    model_sample=SampleRef(sample_id="model-1", sample_label="first"),
                    excuse=excuses[0],
                ),
                SampledExcuse(
                    source_model="model-a",
                    model_sample=SampleRef(sample_id="model-2", sample_label="second"),
                    excuse=excuses[1],
                ),
                SampledExcuse(
                    source_model="model-a",
                    model_sample=SampleRef(sample_id="model-3", sample_label="third"),
                    excuse=excuses[2],
                ),
            ]
        )
        assert len(counted) == 1
        assert isinstance(counted[0], ExcuseSummary)
        assert counted[0].source_model == "model-a"
        assert counted[0].occurrence_count == 3
        assert counted[0].sample_count == 3
        assert counted[0].context == {}
        assert [sample.sample_id for sample in counted[0].example_samples] == [
            "model-1",
            "model-2",
            "model-3",
        ]

    def test_count_excuses_keeps_source_models_separate(self):
        excuse = Excuse(
            instance="r6a.xlarge",
            drive="gp3",
            reason="too small",
            bottleneck=Bottleneck.cpu,
        )
        counted = count_sample_excuses(
            [
                SampledExcuse(
                    source_model="model-a",
                    model_sample=SampleRef(sample_id="model-a-1", sample_label="a"),
                    excuse=excuse,
                ),
                SampledExcuse(
                    source_model="model-b",
                    model_sample=SampleRef(sample_id="model-b-1", sample_label="b"),
                    excuse=excuse,
                ),
            ]
        )

        assert [summary.source_model for summary in counted] == ["model-a", "model-b"]
        assert [summary.example_samples[0].sample_id for summary in counted] == [
            "model-a-1",
            "model-b-1",
        ]


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
