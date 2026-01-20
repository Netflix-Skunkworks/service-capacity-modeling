"""
Regression tests to ensure cost calculations remain stable.

These tests compare actual capacity planning outputs against frozen baselines
stored in service_capacity_modeling/tools/data/baseline_costs.json. Any significant
deviation (>1%) indicates a potential regression that needs investigation.

Usage:
    # Run regression tests (will fail if costs drift from baseline)
    tox -- tests/netflix/test_cost_regression.py

    # Update baseline when costs intentionally change
    tox -e capture-baseline
"""

import json
from importlib import resources
from typing import Any

import pytest

from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.interface import (
    CapacityDesires,
    certain_float,
    certain_int,
    CurrentClusters,
    CurrentZoneClusterCapacity,
    DataShape,
    Interval,
    QueryPattern,
)
from service_capacity_modeling import tools as scm_tools


# ============================================================================
# Test Scenarios - must match tools/capture_baseline_costs.py
# ============================================================================

RDS_SMALL_TIER1 = CapacityDesires(
    service_tier=1,
    query_pattern=QueryPattern(
        estimated_read_per_second=certain_int(200),
        estimated_write_per_second=certain_int(100),
        estimated_mean_read_latency_ms=certain_float(10),
        estimated_mean_write_latency_ms=certain_float(10),
    ),
    data_shape=DataShape(estimated_state_size_gib=certain_int(50)),
)

RDS_TIER3 = CapacityDesires(
    service_tier=3,
    query_pattern=QueryPattern(
        estimated_read_per_second=certain_int(200),
        estimated_write_per_second=certain_int(100),
        estimated_mean_read_latency_ms=certain_float(20),
        estimated_mean_write_latency_ms=certain_float(20),
    ),
    data_shape=DataShape(estimated_state_size_gib=certain_int(200)),
)

AURORA_SMALL_TIER1 = CapacityDesires(
    service_tier=1,
    query_pattern=QueryPattern(
        estimated_read_per_second=certain_int(100),
        estimated_write_per_second=certain_int(100),
        estimated_mean_read_latency_ms=certain_float(10),
        estimated_mean_write_latency_ms=certain_float(10),
    ),
    data_shape=DataShape(estimated_state_size_gib=certain_int(50)),
)

AURORA_TIER3 = CapacityDesires(
    service_tier=3,
    query_pattern=QueryPattern(
        estimated_read_per_second=certain_int(200),
        estimated_write_per_second=certain_int(100),
        estimated_mean_read_latency_ms=certain_float(10),
        estimated_mean_write_latency_ms=certain_float(10),
    ),
    data_shape=DataShape(estimated_state_size_gib=certain_int(200)),
)

CASSANDRA_SMALL_HIGH_QPS = CapacityDesires(
    service_tier=1,
    query_pattern=QueryPattern(
        estimated_read_per_second=certain_int(100_000),
        estimated_write_per_second=certain_int(100_000),
        estimated_mean_read_latency_ms=certain_float(0.5),
        estimated_mean_write_latency_ms=certain_float(0.4),
    ),
    data_shape=DataShape(estimated_state_size_gib=certain_int(10)),
)

CASSANDRA_HIGH_WRITES = CapacityDesires(
    service_tier=1,
    query_pattern=QueryPattern(
        estimated_read_per_second=certain_int(10_000),
        estimated_write_per_second=certain_int(500_000),
    ),
    data_shape=DataShape(estimated_state_size_gib=certain_int(300)),
)

# Kafka - throughput-based
THROUGHPUT = 100 * 1024 * 1024  # 100 MiB/s
KAFKA_THROUGHPUT = CapacityDesires(
    service_tier=1,
    query_pattern=QueryPattern(
        estimated_read_per_second=Interval(low=1, mid=2, high=2, confidence=0.98),
        estimated_write_per_second=Interval(low=1, mid=1, high=1, confidence=0.98),
        estimated_mean_write_size_bytes=Interval(
            low=THROUGHPUT, mid=THROUGHPUT, high=THROUGHPUT * 2, confidence=0.98
        ),
    ),
)

# EVCache scenarios
EVCACHE_TINY = CapacityDesires(
    service_tier=1,
    query_pattern=QueryPattern(
        estimated_read_per_second=certain_int(1_000),
        estimated_write_per_second=certain_int(100),
        estimated_mean_read_latency_ms=certain_float(1.0),
    ),
    data_shape=DataShape(
        estimated_state_size_gib=certain_int(1),
        estimated_state_item_count=Interval(
            low=10_000, mid=100_000, high=200_000, confidence=0.98
        ),
    ),
)

EVCACHE_SMALL = CapacityDesires(
    service_tier=1,
    query_pattern=QueryPattern(
        estimated_read_per_second=certain_int(100_000),
        estimated_write_per_second=certain_int(10_000),
        estimated_mean_read_latency_ms=certain_float(1.0),
    ),
    data_shape=DataShape(
        estimated_state_size_gib=certain_int(10),
        estimated_state_item_count=Interval(
            low=1_000_000, mid=10_000_000, high=20_000_000, confidence=0.98
        ),
    ),
)

EVCACHE_LARGE = CapacityDesires(
    service_tier=1,
    query_pattern=QueryPattern(
        estimated_read_per_second=certain_int(500_000),
        estimated_write_per_second=certain_int(50_000),
        estimated_mean_read_latency_ms=certain_float(1.0),
    ),
    data_shape=DataShape(
        estimated_state_size_gib=certain_int(500),
        estimated_state_item_count=Interval(
            low=10_000_000, mid=100_000_000, high=200_000_000, confidence=0.98
        ),
    ),
)

SCENARIOS: dict[str, dict[str, Any]] = {
    "rds_small_tier1": {
        "model": "org.netflix.rds",
        "region": "us-east-1",
        "desires": RDS_SMALL_TIER1,
        "extra_args": None,
    },
    "rds_tier3": {
        "model": "org.netflix.rds",
        "region": "us-east-1",
        "desires": RDS_TIER3,
        "extra_args": None,
    },
    "aurora_small_tier1": {
        "model": "org.netflix.aurora",
        "region": "us-east-1",
        "desires": AURORA_SMALL_TIER1,
        "extra_args": None,
    },
    "aurora_tier3": {
        "model": "org.netflix.aurora",
        "region": "us-east-1",
        "desires": AURORA_TIER3,
        "extra_args": None,
    },
    "cassandra_small_high_qps_local": {
        "model": "org.netflix.cassandra",
        "region": "us-east-1",
        "desires": CASSANDRA_SMALL_HIGH_QPS,
        "extra_args": {"require_local_disks": True, "copies_per_region": 3},
    },
    "cassandra_high_writes_ebs": {
        "model": "org.netflix.cassandra",
        "region": "us-east-1",
        "desires": CASSANDRA_HIGH_WRITES,
        "extra_args": {"require_local_disks": False, "copies_per_region": 2},
    },
    "kafka_100mib_throughput": {
        "model": "org.netflix.kafka",
        "region": "us-east-1",
        "desires": KAFKA_THROUGHPUT,
        "extra_args": {"require_local_disks": False},
    },
    "evcache_tiny_with_spread": {
        "model": "org.netflix.evcache",
        "region": "us-east-1",
        "desires": EVCACHE_TINY,
        "extra_args": {"cross_region_replication": "none"},
    },
    "evcache_small_no_replication": {
        "model": "org.netflix.evcache",
        "region": "us-east-1",
        "desires": EVCACHE_SMALL,
        "extra_args": {"cross_region_replication": "none"},
    },
    "evcache_large_with_replication": {
        "model": "org.netflix.evcache",
        "region": "us-east-1",
        "desires": EVCACHE_LARGE,
        "extra_args": {"cross_region_replication": "sets", "copies_per_region": 2},
    },
}


def load_baseline() -> dict[str, dict[str, Any]]:
    """Load baseline costs from package resources."""
    try:
        baseline_file = resources.files(scm_tools).joinpath(
            "data", "baseline_costs.json"
        )
        content = baseline_file.read_text(encoding="utf-8")
        baselines = json.loads(content)
        return {
            b["scenario"]: b for b in baselines if "scenario" in b and "error" not in b
        }
    except FileNotFoundError:
        return {}


@pytest.fixture(scope="session")
def baseline_costs() -> dict[str, dict[str, Any]]:
    """Load baseline costs from JSON file."""
    return load_baseline()


class TestBaselineDrift:
    """Test that actual costs match baseline within tolerance."""

    @pytest.mark.parametrize("scenario_name", list(SCENARIOS.keys()))
    def test_baseline_drift(
        self,
        scenario_name: str,
        baseline_costs: dict[str, dict[str, Any]],
    ) -> None:
        """Test that scenario costs match baseline within 1% tolerance."""
        scenario = SCENARIOS[scenario_name]

        if scenario_name not in baseline_costs:
            pytest.fail(
                f"Scenario '{scenario_name}' not in baseline. "
                "Run: tox -e capture-baseline"
            )

        baseline = baseline_costs[scenario_name]
        baseline_total = baseline["total_annual_cost"]
        baseline_keys = set(baseline["annual_costs"].keys())

        # Run plan_certain to get actual costs
        cap_plans = planner.plan_certain(
            model_name=scenario["model"],
            region=scenario["region"],
            desires=scenario["desires"],
            num_results=1,
            extra_model_arguments=scenario["extra_args"] or {},
        )

        assert cap_plans, f"No capacity plans generated for {scenario_name}"
        cap_plan = cap_plans[0]

        actual_total = float(cap_plan.candidate_clusters.total_annual_cost)
        actual_keys = set(cap_plan.candidate_clusters.annual_costs.keys())

        assert actual_total == pytest.approx(baseline_total, rel=0.01), (
            f"Total cost drift for {scenario_name}: "
            f"baseline=${baseline_total:,.2f}, actual=${actual_total:,.2f}, "
            f"diff={((actual_total - baseline_total) / baseline_total) * 100:+.2f}%. "
            "If expected, update baseline: tox -e capture-baseline"
        )

        added = actual_keys - baseline_keys
        removed = baseline_keys - actual_keys
        assert baseline_keys == actual_keys, (
            f"Cost keys changed for {scenario_name}: added={added}, removed={removed}. "
            "If expected, update baseline: tox -e capture-baseline"
        )


# ============================================================================
# extract_baseline_plan cost comparison tests
# ============================================================================

# CostAwareModel scenarios (Cassandra, Kafka, EVCache) can use extract_baseline_plan
COST_AWARE_SCENARIOS = {
    name: scenario
    for name, scenario in SCENARIOS.items()
    if scenario["model"]
    in ("org.netflix.cassandra", "org.netflix.kafka", "org.netflix.evcache")
}


def _clusters_to_current(cap_plan) -> CurrentClusters:
    """Convert plan's candidate_clusters to CurrentClusters format.

    Takes the zonal clusters from a capacity plan and converts them to
    CurrentZoneClusterCapacity format for use with extract_baseline_plan.
    """
    current_zonal = []
    for cluster in cap_plan.candidate_clusters.zonal:
        current_zonal.append(
            CurrentZoneClusterCapacity(
                cluster_instance_name=cluster.instance.name,
                cluster_instance=None,  # Will be resolved from catalog
                cluster_instance_count=Interval(
                    low=cluster.count,
                    mid=cluster.count,
                    high=cluster.count,
                    confidence=1.0,
                ),
                # Include drive if attached
                cluster_drive=cluster.attached_drives[0]
                if cluster.attached_drives
                else None,
            )
        )
    return CurrentClusters(zonal=current_zonal)


class TestExtractBaselinePlanCostMatch:
    """Test that extract_baseline_plan costs match plan_certain costs.

    For CostAwareModel models (Cassandra, Kafka, EVCache), the baseline
    extraction should produce the same costs as plan_certain when given
    the same cluster configuration.
    """

    @pytest.mark.parametrize("scenario_name", list(COST_AWARE_SCENARIOS.keys()))
    def test_baseline_total_cost_matches_planner(self, scenario_name: str) -> None:
        """Baseline total_annual_cost should match planner exactly."""
        scenario = COST_AWARE_SCENARIOS[scenario_name]

        # Get recommended plan
        cap_plans = planner.plan_certain(
            model_name=scenario["model"],
            region=scenario["region"],
            desires=scenario["desires"],
            num_results=1,
            extra_model_arguments=scenario["extra_args"] or {},
        )

        if not cap_plans:
            pytest.skip(f"No capacity plans generated for {scenario_name}")

        recommended = cap_plans[0]

        # Convert to current clusters and extract baseline
        current_clusters = _clusters_to_current(recommended)
        desires_with_current = scenario["desires"].model_copy(deep=True)
        desires_with_current.current_clusters = current_clusters

        baseline = planner.extract_baseline_plan(
            model_name=scenario["model"],
            region=scenario["region"],
            desires=desires_with_current,
            extra_model_arguments=scenario["extra_args"] or {},
        )

        # Compare total annual cost
        recommended_total = recommended.candidate_clusters.total_annual_cost
        baseline_total = baseline.candidate_clusters.total_annual_cost

        assert baseline_total == pytest.approx(recommended_total, rel=0.01), (
            f"Total cost mismatch for {scenario_name}: "
            f"recommended=${recommended_total:,.2f}, "
            f"baseline=${baseline_total:,.2f}\n"
            f"Breakdown - recommended: {recommended.candidate_clusters.annual_costs}\n"
            f"Breakdown - baseline: {baseline.candidate_clusters.annual_costs}"
        )

    @pytest.mark.parametrize("scenario_name", list(COST_AWARE_SCENARIOS.keys()))
    def test_baseline_cost_keys_match_planner(self, scenario_name: str) -> None:
        """Baseline extraction should produce same cost keys as planner."""
        scenario = COST_AWARE_SCENARIOS[scenario_name]

        # Get recommended plan
        cap_plans = planner.plan_certain(
            model_name=scenario["model"],
            region=scenario["region"],
            desires=scenario["desires"],
            num_results=1,
            extra_model_arguments=scenario["extra_args"] or {},
        )

        if not cap_plans:
            pytest.skip(f"No capacity plans generated for {scenario_name}")

        recommended = cap_plans[0]

        # Convert to current clusters and extract baseline
        current_clusters = _clusters_to_current(recommended)
        desires_with_current = scenario["desires"].model_copy(deep=True)
        desires_with_current.current_clusters = current_clusters

        baseline = planner.extract_baseline_plan(
            model_name=scenario["model"],
            region=scenario["region"],
            desires=desires_with_current,
            extra_model_arguments=scenario["extra_args"] or {},
        )

        # Compare cost breakdown keys
        recommended_keys = set(recommended.candidate_clusters.annual_costs.keys())
        baseline_keys = set(baseline.candidate_clusters.annual_costs.keys())

        assert baseline_keys == recommended_keys, (
            f"Cost keys differ for {scenario_name}: "
            f"recommended={recommended_keys}, baseline={baseline_keys}"
        )
