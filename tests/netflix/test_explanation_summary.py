"""Tests for explanation_summary.summarize()."""

from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.explanation_summary import summarize
from service_capacity_modeling.interface import (
    CapacityDesires,
    CurrentClusters,
    CurrentZoneClusterCapacity,
    DataShape,
    Interval,
    QueryPattern,
    AccessPattern,
    certain_float,
    certain_int,
)
from service_capacity_modeling.models.plan_comparison import compare_plans


EXTRA = {"required_cluster_size": 64, "require_local_disks": False}

large_workload = CapacityDesires(
    service_tier=0,
    query_pattern=QueryPattern(
        access_pattern=AccessPattern.latency,
        estimated_read_per_second=Interval(
            low=318550, mid=828926, high=1380915, confidence=0.98
        ),
        estimated_write_per_second=Interval(
            low=140728, mid=385944, high=659975, confidence=0.98
        ),
    ),
    data_shape=DataShape(
        estimated_state_size_gib=Interval(
            low=26639, mid=40988, high=54667, confidence=0.98
        ),
    ),
    current_clusters=CurrentClusters(
        zonal=[
            CurrentZoneClusterCapacity(
                cluster_instance_name="m6id.12xlarge",
                cluster_instance_count=Interval(low=63, mid=64, high=64, confidence=1),
                cluster_type="cassandra",
                cpu_utilization=Interval(low=4.5, mid=7.97, high=13.04, confidence=1),
                network_utilization_mbps=Interval(
                    low=4.63, mid=12.95, high=63.7, confidence=1
                ),
                disk_utilization_gib=Interval(
                    low=416, mid=640, high=854, confidence=0.98
                ),
            ),
        ]
        * 3,
    ),
)


class TestSummarize:
    def test_headline_has_savings(self):
        explained = planner.plan_certain_explained(
            "org.netflix.cassandra",
            "us-east-1",
            large_workload,
            extra_model_arguments=EXTRA,
        )
        baseline = planner.extract_baseline_plan(
            "org.netflix.cassandra",
            "us-east-1",
            large_workload,
            extra_model_arguments=EXTRA,
        )
        comparison = compare_plans(baseline, explained.plans[0])
        summary = summarize(explained, baseline=baseline, comparison=comparison)

        assert "current" in summary["headline"]
        assert "recommended" in summary["headline"]
        assert summary["headline"]["annual_savings"] > 0

    def test_resources_have_status(self):
        explained = planner.plan_certain_explained(
            "org.netflix.cassandra",
            "us-east-1",
            large_workload,
            extra_model_arguments=EXTRA,
        )
        baseline = planner.extract_baseline_plan(
            "org.netflix.cassandra",
            "us-east-1",
            large_workload,
            extra_model_arguments=EXTRA,
        )
        comparison = compare_plans(baseline, explained.plans[0])
        summary = summarize(explained, baseline=baseline, comparison=comparison)

        assert len(summary["resources"]) > 0
        for r in summary["resources"]:
            assert r["status"] in ("ok", "over_provisioned", "under_provisioned")
            assert "name" in r
            assert "ratio" in r

    def test_same_family_rejections(self):
        explained = planner.plan_certain_explained(
            "org.netflix.cassandra",
            "us-east-1",
            large_workload,
            extra_model_arguments=EXTRA,
        )
        summary = summarize(explained)

        for rej in summary["same_family_rejections"]:
            assert "instance" in rej
            assert "reason" in rej

    def test_summary_without_baseline(self):
        explained = planner.plan_certain_explained(
            "org.netflix.cassandra",
            "us-east-1",
            CapacityDesires(
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
            ),
            extra_model_arguments={"require_local_disks": False},
        )
        summary = summarize(explained)

        assert "recommended" in summary["headline"]
        assert "annual_savings" not in summary["headline"]
        assert summary["resources"] == []

    def test_json_serializable(self):
        import json

        explained = planner.plan_certain_explained(
            "org.netflix.cassandra",
            "us-east-1",
            large_workload,
            extra_model_arguments=EXTRA,
        )
        baseline = planner.extract_baseline_plan(
            "org.netflix.cassandra",
            "us-east-1",
            large_workload,
            extra_model_arguments=EXTRA,
        )
        comparison = compare_plans(baseline, explained.plans[0])
        summary = summarize(explained, baseline=baseline, comparison=comparison)

        # Must be JSON-serializable without custom encoders
        json_str = json.dumps(summary)
        assert len(json_str) > 100
