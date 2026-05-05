"""Regression tests for stochastic planner snapshots."""

import json
from importlib import resources
from typing import Any

import pytest

from service_capacity_modeling import tools as scm_tools
from service_capacity_modeling.tools.capture_baseline_costs import (
    UNCERTAIN_SCENARIOS,
    capture_uncertain,
)

_BASELINE_HELP = (
    "To fix: tox -e capture-baseline\nTo auto-update on commit: pre-commit install"
)


def load_baseline() -> dict[str, dict[str, Any]]:
    """Load uncertain planner baselines from package resources."""
    try:
        baseline_file = resources.files(scm_tools).joinpath(
            "data", "baseline_uncertain.json"
        )
        content = baseline_file.read_text(encoding="utf-8")
        baselines = json.loads(content)
        return {b["scenario"]: b for b in baselines if "scenario" in b}
    except FileNotFoundError:
        return {}


@pytest.fixture(scope="session")
def uncertain_baselines() -> dict[str, dict[str, Any]]:
    return load_baseline()


def _assert_cluster_matches(
    actual: dict[str, Any], expected: dict[str, Any], scenario_name: str
) -> None:
    assert actual["cluster_type"] == expected["cluster_type"], scenario_name
    assert actual["deployment"] == expected["deployment"], scenario_name
    assert actual["instance"] == expected["instance"], scenario_name
    assert actual["count"] == expected["count"], scenario_name
    assert actual.get("attached_drives", []) == expected.get("attached_drives", []), (
        scenario_name
    )
    assert actual.get("cluster_params", {}) == expected.get("cluster_params", {}), (
        f"cluster_params drift for {scenario_name}.\n{_BASELINE_HELP}"
    )
    assert actual["annual_cost"] == pytest.approx(expected["annual_cost"], rel=0.01), (
        f"cluster annual_cost drift for {scenario_name}: "
        f"baseline=${expected['annual_cost']:,.2f}, "
        f"actual=${actual['annual_cost']:,.2f}.\n"
        f"{_BASELINE_HELP}"
    )


def _assert_candidate_matches(
    actual: dict[str, Any], expected: dict[str, Any], scenario_name: str, label: str
) -> None:
    assert actual["total_annual_cost"] == pytest.approx(
        expected["total_annual_cost"], rel=0.01
    ), (
        f"{label} total cost drift for {scenario_name}: "
        f"baseline=${expected['total_annual_cost']:,.2f}, "
        f"actual=${actual['total_annual_cost']:,.2f}.\n{_BASELINE_HELP}"
    )
    assert set(actual["annual_costs"].keys()) == set(expected["annual_costs"].keys()), (
        f"{label} annual_cost keys changed for {scenario_name}: "
        f"baseline={set(expected['annual_costs'].keys())}, "
        f"actual={set(actual['annual_costs'].keys())}.\n{_BASELINE_HELP}"
    )
    for key, expected_cost in expected["annual_costs"].items():
        assert actual["annual_costs"][key] == pytest.approx(expected_cost, rel=0.01), (
            f"{label} annual_cost bucket '{key}' drift for {scenario_name}: "
            f"baseline=${expected_cost:,.2f}, "
            f"actual=${actual['annual_costs'][key]:,.2f}.\n"
            f"{_BASELINE_HELP}"
        )

    assert len(actual["clusters"]) == len(expected["clusters"]), (
        f"{label} cluster count drift for {scenario_name}: "
        f"baseline={len(expected['clusters'])}, actual={len(actual['clusters'])}.\n"
        f"{_BASELINE_HELP}"
    )
    for actual_cluster, expected_cluster in zip(
        actual["clusters"], expected["clusters"]
    ):
        _assert_cluster_matches(actual_cluster, expected_cluster, scenario_name)


def _assert_plan_sequence_matches(
    actual: list[dict[str, Any]],
    expected: list[dict[str, Any]],
    scenario_name: str,
    label: str,
) -> None:
    assert len(actual) == len(expected), (
        f"{label} plan count drift for {scenario_name}: "
        f"baseline={len(expected)}, actual={len(actual)}.\n{_BASELINE_HELP}"
    )
    for idx, (actual_plan, expected_plan) in enumerate(zip(actual, expected)):
        _assert_candidate_matches(
            actual_plan, expected_plan, scenario_name, f"{label}[{idx}]"
        )


class TestUncertainBaselineDrift:
    @pytest.mark.parametrize("scenario_name", list(UNCERTAIN_SCENARIOS.keys()))
    def test_uncertain_baseline_drift(
        self,
        scenario_name: str,
        uncertain_baselines: dict[str, dict[str, Any]],
    ) -> None:
        if scenario_name not in uncertain_baselines:
            pytest.fail(
                "Scenario "
                f"'{scenario_name}' not in uncertain baseline.\n{_BASELINE_HELP}"
            )

        scenario = UNCERTAIN_SCENARIOS[scenario_name]
        baseline = uncertain_baselines[scenario_name]
        actual = capture_uncertain(
            model_name=scenario["model"],
            region=scenario["region"],
            desires=scenario["desires"],
            extra_args=scenario["extra_args"],
            scenario_name=scenario_name,
            simulations=baseline["simulations"],
            num_results=baseline["num_results"],
        )

        assert "error" not in actual, (
            f"Unexpected error for {scenario_name}: {actual['error']}\n{_BASELINE_HELP}"
        )
        assert baseline["simulations"] == actual["simulations"]
        assert baseline["num_results"] == actual["num_results"]

        _assert_plan_sequence_matches(
            actual["least_regret"],
            baseline["least_regret"],
            scenario_name,
            "least_regret",
        )
        _assert_plan_sequence_matches(
            actual["mean"],
            baseline["mean"],
            scenario_name,
            "mean",
        )

        assert set(actual["percentiles"].keys()) == set(baseline["percentiles"].keys())
        for percentile, expected in baseline["percentiles"].items():
            _assert_plan_sequence_matches(
                actual["percentiles"][percentile],
                expected,
                scenario_name,
                f"percentiles[{percentile}]",
            )
