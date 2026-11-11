"""Regression snapshot for the alternative instance planner."""

import json
from importlib import resources
from typing import Any

import pytest

from service_capacity_modeling import tools as scm_tools
from service_capacity_modeling.tools.capture_baseline_costs import (
    ALTERNATIVE_INSTANCE_SCENARIOS,
    capture_alternative_instances,
)

_BASELINE_HELP = (
    "Review the change in alternative types, counts, or cost, then refresh with: "
    "tox -e capture-baseline"
)


@pytest.fixture(scope="session")
def baselines() -> dict[str, dict[str, Any]]:
    content = (
        resources.files(scm_tools)
        .joinpath("data", "baseline_alternative_instances.json")
        .read_text(encoding="utf-8")
    )
    return {b["scenario"]: b for b in json.loads(content)}


def _shape(plan: dict[str, Any]) -> dict[str, Any]:
    """Everything but costs, which are compared with a tolerance."""
    return {
        "primary_instance_type": plan["primary_instance_type"],
        "alternative_instances": plan["alternative_instances"],
        "clusters": [
            {k: v for k, v in cluster.items() if k != "annual_cost"}
            for cluster in plan["clusters"]
        ],
    }


@pytest.mark.parametrize("scenario_name", list(ALTERNATIVE_INSTANCE_SCENARIOS))
def test_alternative_instances_baseline(
    scenario_name: str, baselines: dict[str, dict[str, Any]]
) -> None:
    assert scenario_name in baselines, _BASELINE_HELP
    scenario = ALTERNATIVE_INSTANCE_SCENARIOS[scenario_name]
    actual = capture_alternative_instances(
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
