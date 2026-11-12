"""Regression snapshot for alternative instance types on both planning paths."""

import json
from importlib import resources
from typing import Any

import pytest

from service_capacity_modeling import tools as scm_tools
from service_capacity_modeling.tools.capture_baseline_costs import (
    ALTERNATIVE_SCENARIOS,
    capture_alternatives,
)

_BASELINE_HELP = (
    "Review the change in alternative types, counts, or cost, then refresh with: "
    "tox -e capture-baseline"
)


@pytest.fixture(scope="session")
def baselines() -> dict[str, dict[str, Any]]:
    content = (
        resources.files(scm_tools)
        .joinpath("data", "baseline_alternatives.json")
        .read_text(encoding="utf-8")
    )
    return {b["scenario"]: b for b in json.loads(content)}


def _shape(plan: dict[str, Any]) -> dict[str, Any]:
    """Everything but costs, which are compared with a tolerance."""
    return {
        "primary_instance_type": plan["primary_instance_type"],
        "alternatives": [
            {k: v for k, v in alt.items() if k != "annual_cost"}
            for alt in plan["alternatives"]
        ],
        "clusters": [
            {k: v for k, v in cluster.items() if k != "annual_cost"}
            for cluster in plan["clusters"]
        ],
    }


def _costs(plan: dict[str, Any]) -> list[float]:
    return [
        plan["total_annual_cost"],
        *(a["annual_cost"] for a in plan["alternatives"]),
    ]


@pytest.mark.parametrize("scenario_name", list(ALTERNATIVE_SCENARIOS))
def test_alternatives_baseline(
    scenario_name: str, baselines: dict[str, dict[str, Any]]
) -> None:
    assert scenario_name in baselines, _BASELINE_HELP
    scenario = ALTERNATIVE_SCENARIOS[scenario_name]
    actual = capture_alternatives(
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
        assert _costs(actual_plan) == pytest.approx(_costs(expected_plan), rel=0.01), (
            _BASELINE_HELP
        )
