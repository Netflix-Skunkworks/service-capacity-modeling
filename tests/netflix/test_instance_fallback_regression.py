"""Regression snapshot for the experimental instance fallback planner."""

import json
from importlib import resources
from typing import Any

import pytest

from service_capacity_modeling import tools as scm_tools
from service_capacity_modeling.tools.capture_baseline_costs import (
    INSTANCE_FALLBACK_SCENARIOS,
    capture_instance_fallbacks,
)

_BASELINE_HELP = (
    "Review the change in fallback types, counts, or cost, then refresh with: "
    "tox -e capture-baseline"
)


@pytest.fixture(scope="session")
def baselines() -> dict[str, dict[str, Any]]:
    content = (
        resources.files(scm_tools)
        .joinpath("data", "baseline_instance_fallback.json")
        .read_text(encoding="utf-8")
    )
    return {b["scenario"]: b for b in json.loads(content)}


def _shape(plan: dict[str, Any]) -> dict[str, Any]:
    """Everything but costs, which are compared with a tolerance."""
    return {
        "preferred_instance_type": plan["preferred_instance_type"],
        "fallback_instance_types": plan["fallback_instance_types"],
        "single_type_node_count": plan["single_type_node_count"],
        "clusters": [
            {k: v for k, v in cluster.items() if k != "annual_cost"}
            for cluster in plan["clusters"]
        ],
        "cost_keys": sorted(plan["annual_costs"]),
    }


@pytest.mark.parametrize("scenario_name", list(INSTANCE_FALLBACK_SCENARIOS))
def test_instance_fallback_baseline(
    scenario_name: str, baselines: dict[str, dict[str, Any]]
) -> None:
    assert scenario_name in baselines, _BASELINE_HELP
    scenario = INSTANCE_FALLBACK_SCENARIOS[scenario_name]
    actual = capture_instance_fallbacks(
        scenario["model"],
        scenario["region"],
        scenario["desires"],
        scenario["extra_args"],
        scenario_name,
    )
    expected = baselines[scenario_name]
    assert "error" not in actual, actual.get("error")
    assert expected["plans"], "a baseline with no plans checks nothing"

    assert [_shape(p) for p in actual["plans"]] == [
        _shape(p) for p in expected["plans"]
    ], _BASELINE_HELP
    for actual_plan, expected_plan in zip(actual["plans"], expected["plans"]):
        assert actual_plan["total_annual_cost"] == pytest.approx(
            expected_plan["total_annual_cost"], rel=0.01
        ), _BASELINE_HELP
