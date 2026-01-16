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
        "extra_args": {"require_local_disks": True},
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


def capture_scenario_costs(scenario_name: str) -> dict[str, Any]:
    """Run planner and capture costs for a scenario."""
    scenario = SCENARIOS[scenario_name]
    try:
        cap_plans = planner.plan_certain(
            model_name=scenario["model"],
            region=scenario["region"],
            desires=scenario["desires"],
            num_results=1,
            extra_model_arguments=scenario["extra_args"] or {},
        )

        if not cap_plans:
            return {"error": "No capacity plans generated", "scenario": scenario_name}

        cap_plan = cap_plans[0]
        clusters = cap_plan.candidate_clusters

        result: dict[str, Any] = {
            "scenario": scenario_name,
            "model": scenario["model"],
            "region": scenario["region"],
            "service_tier": scenario["desires"].service_tier,
            "annual_costs": {k: float(v) for k, v in clusters.annual_costs.items()},
            "total_annual_cost": float(clusters.total_annual_cost),
        }

        return result
    except (ValueError, KeyError, AttributeError) as e:
        return {"error": str(e), "scenario": scenario_name}


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
        actual = capture_scenario_costs(scenario_name)

        if "error" in actual:
            pytest.skip(f"Scenario failed: {actual['error']}")

        if scenario_name not in baseline_costs:
            pytest.fail(
                f"Scenario '{scenario_name}' not in baseline. "
                "Run: tox -e capture-baseline"
            )

        baseline = baseline_costs[scenario_name]
        baseline_total = baseline["total_annual_cost"]
        actual_total = actual["total_annual_cost"]

        assert actual_total == pytest.approx(baseline_total, rel=0.01), (
            f"Total cost drift for {scenario_name}: "
            f"baseline=${baseline_total:,.2f}, actual=${actual_total:,.2f}, "
            f"diff={((actual_total - baseline_total) / baseline_total) * 100:+.2f}%. "
            "If expected, update baseline: tox -e capture-baseline"
        )

        # Compare cost breakdown keys
        baseline_keys = set(baseline["annual_costs"].keys())
        actual_keys = set(actual["annual_costs"].keys())

        added = actual_keys - baseline_keys
        removed = baseline_keys - actual_keys
        assert baseline_keys == actual_keys, (
            f"Cost keys changed for {scenario_name}: added={added}, removed={removed}. "
            "If expected, update baseline: tox -e capture-baseline"
        )
