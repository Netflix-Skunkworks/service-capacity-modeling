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
from service_capacity_modeling import tools as scm_tools
from service_capacity_modeling.tools.capture_baseline_costs import SCENARIOS


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

        # Inline plan_certain call for cleaner test
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
