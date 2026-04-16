"""Regression tests for uncertain explained snapshots."""

import json
from importlib import resources
from typing import Any

import pytest

from service_capacity_modeling import tools as scm_tools
from service_capacity_modeling.tools.capture_baseline_costs import (
    UNCERTAIN_EXPLAINED_SCENARIOS,
    capture_uncertain_explained,
)

_BASELINE_HELP = (
    "To fix: tox -e capture-baseline\nTo auto-update on commit: pre-commit install"
)


def load_baseline() -> dict[str, dict[str, Any]]:
    """Load explained uncertain baselines from package resources."""
    try:
        baseline_file = resources.files(scm_tools).joinpath(
            "data", "baseline_uncertain_explained.json"
        )
        content = baseline_file.read_text(encoding="utf-8")
        baselines = json.loads(content)
        return {b["scenario"]: b for b in baselines if "scenario" in b}
    except FileNotFoundError:
        return {}


@pytest.fixture(scope="session")
def explained_baselines() -> dict[str, dict[str, Any]]:
    return load_baseline()


def _assert_cost_blocks_match(
    actual: dict[str, Any], expected: dict[str, Any], scenario_name: str, label: str
) -> None:
    assert actual["plan"]["total_annual_cost"] == pytest.approx(
        expected["plan"]["total_annual_cost"], rel=0.01
    ), f"{label} plan cost drift for {scenario_name}.\n{_BASELINE_HELP}"
    assert len(actual["plan"]["clusters"]) == len(expected["plan"]["clusters"]), (
        f"{label} cluster count drift for {scenario_name}.\n{_BASELINE_HELP}"
    )
    for actual_cluster, expected_cluster in zip(
        actual["plan"]["clusters"], expected["plan"]["clusters"]
    ):
        for field in ("cluster_type", "deployment", "instance", "count"):
            assert actual_cluster[field] == expected_cluster[field], (
                f"{label} cluster field '{field}' drift for {scenario_name}.\n"
                f"{_BASELINE_HELP}"
            )
        assert actual_cluster.get("attached_drives", []) == expected_cluster.get(
            "attached_drives", []
        ), f"{label} attached drives drift for {scenario_name}.\n{_BASELINE_HELP}"
        assert actual_cluster.get("cluster_params", {}) == expected_cluster.get(
            "cluster_params", {}
        ), f"{label} cluster_params drift for {scenario_name}.\n{_BASELINE_HELP}"
        assert actual_cluster["annual_cost"] == pytest.approx(
            expected_cluster["annual_cost"], rel=0.01
        ), f"{label} cluster annual_cost drift for {scenario_name}.\n{_BASELINE_HELP}"
    assert (
        actual["plan"]["annual_costs"].keys() == expected["plan"]["annual_costs"].keys()
    ), f"{label} annual_cost keys drift for {scenario_name}.\n{_BASELINE_HELP}"
    for key, expected_cost in expected["plan"]["annual_costs"].items():
        assert actual["plan"]["annual_costs"][key] == pytest.approx(
            expected_cost, rel=0.01
        ), (
            f"{label} annual_cost bucket '{key}' drift for {scenario_name}.\n"
            f"{_BASELINE_HELP}"
        )


def _assert_regret_summary_match(
    actual: dict[str, Any], expected: dict[str, Any], scenario_name: str, label: str
) -> None:
    _assert_cost_blocks_match(actual, expected, scenario_name, label)
    assert actual["equivalent_plan_count"] == expected["equivalent_plan_count"]
    for field in (
        "selected_total_regret",
        "min_total_regret",
        "max_total_regret",
        "mean_total_regret",
    ):
        assert actual[field] == pytest.approx(expected[field], rel=0.01), (
            f"{label} {field} drift for {scenario_name}.\n{_BASELINE_HELP}"
        )
    assert actual["selected_regret_components"] == pytest.approx(
        expected["selected_regret_components"], rel=0.01
    ), (
        f"{label} selected regret components drift for {scenario_name}.\n"
        f"{_BASELINE_HELP}"
    )
    assert actual["mean_regret_components"] == pytest.approx(
        expected["mean_regret_components"], rel=0.01
    ), f"{label} mean regret components drift for {scenario_name}.\n{_BASELINE_HELP}"
    assert (
        actual["selected_regret_components_by_model"].keys()
        == expected["selected_regret_components_by_model"].keys()
    ), (
        f"{label} selected per-model regret keys drift for {scenario_name}.\n"
        f"{_BASELINE_HELP}"
    )
    for model_name, expected_components in expected[
        "selected_regret_components_by_model"
    ].items():
        assert actual["selected_regret_components_by_model"][
            model_name
        ] == pytest.approx(expected_components, rel=0.01), (
            f"{label} selected per-model regret drift for "
            f"{scenario_name}/{model_name}.\n{_BASELINE_HELP}"
        )
    assert (
        actual["mean_regret_components_by_model"].keys()
        == expected["mean_regret_components_by_model"].keys()
    ), (
        f"{label} mean per-model regret keys drift for {scenario_name}.\n"
        f"{_BASELINE_HELP}"
    )
    for model_name, expected_components in expected[
        "mean_regret_components_by_model"
    ].items():
        assert actual["mean_regret_components_by_model"][model_name] == pytest.approx(
            expected_components, rel=0.01
        ), (
            f"{label} mean per-model regret drift for "
            f"{scenario_name}/{model_name}.\n{_BASELINE_HELP}"
        )
    assert actual["representative_models"] == expected["representative_models"]


class TestUncertainExplainedBaselineDrift:
    @pytest.mark.parametrize(
        "scenario_name", list(UNCERTAIN_EXPLAINED_SCENARIOS.keys())
    )
    def test_uncertain_explained_baseline_drift(
        self,
        scenario_name: str,
        explained_baselines: dict[str, dict[str, Any]],
    ) -> None:
        if scenario_name not in explained_baselines:
            pytest.fail(
                f"Scenario '{scenario_name}' not in uncertain explained baseline.\n"
                f"{_BASELINE_HELP}"
            )

        scenario = UNCERTAIN_EXPLAINED_SCENARIOS[scenario_name]
        baseline = explained_baselines[scenario_name]
        actual = capture_uncertain_explained(
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

        assert len(actual["least_regret_summaries"]) == len(
            baseline["least_regret_summaries"]
        ), f"least_regret_summaries length drift for {scenario_name}.\n{_BASELINE_HELP}"
        for idx, (actual_summary, expected_summary) in enumerate(
            zip(actual["least_regret_summaries"], baseline["least_regret_summaries"])
        ):
            _assert_regret_summary_match(
                actual_summary,
                expected_summary,
                scenario_name,
                f"least_regret_summaries[{idx}]",
            )

        assert actual["excuse_summary"] == baseline["excuse_summary"], (
            f"excuse_summary drift for {scenario_name}.\n{_BASELINE_HELP}"
        )
