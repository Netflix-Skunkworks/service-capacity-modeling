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

Recommended Setup:
    # Install pre-commit hooks to auto-update baselines on commit
    pre-commit install

    With hooks installed, the baseline capture runs automatically during commit.
    If you're seeing test failures due to baseline drift, run:
        pre-commit install && tox -e capture-baseline
"""

import json
from importlib import resources
from typing import Any

import pytest

from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.interface import (
    CapacityDesires,
    CapacityPlan,
    certain_int,
    CurrentClusters,
    CurrentRegionClusterCapacity,
    CurrentZoneClusterCapacity,
)
from service_capacity_modeling.models import CostAwareModel
from service_capacity_modeling.models.plan_comparison import (
    compare_plans,
    ComparisonStrategy,
    exact_match,
    gte,
    ignore_resource,
    ResourceTolerances,
)
from service_capacity_modeling import tools as scm_tools
from service_capacity_modeling.tools.capture_baseline_costs import SCENARIOS

# Help message for baseline-related test failures
_BASELINE_HELP = (
    "To fix: tox -e capture-baseline\nTo auto-update on commit: pre-commit install"
)


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


def _get_single_drive(attached_drives, cluster_type: str):
    """Extract the single attached drive, or None if empty.

    Currently no capacity models use multiple drives per node (RAID is not
    implemented). This helper enforces that assumption and will fail loudly
    if a model starts returning multiple drives.
    """
    if len(attached_drives) > 1:
        raise ValueError(
            f"Cluster type '{cluster_type}' has {len(attached_drives)} drives; "
            "expected 0 or 1"
        )
    return attached_drives[0] if attached_drives else None


def _clusters_to_current(cap_plan: CapacityPlan) -> CurrentClusters:
    """Convert plan's candidate_clusters to CurrentClusters format.

    Handles both zonal and regional clusters for composite models.
    Preserves cluster_type for proper routing in extract_baseline_plan.
    """
    current_zonal = []
    for cluster in cap_plan.candidate_clusters.zonal:
        current_zonal.append(
            CurrentZoneClusterCapacity(
                cluster_instance_name=cluster.instance.name,
                cluster_instance_count=certain_int(cluster.count),
                cluster_drive=_get_single_drive(
                    cluster.attached_drives, cluster.cluster_type
                ),
                cluster_type=cluster.cluster_type,
            )
        )

    current_regional = []
    for cluster in cap_plan.candidate_clusters.regional:
        current_regional.append(
            CurrentRegionClusterCapacity(
                cluster_instance_name=cluster.instance.name,
                cluster_instance_count=certain_int(cluster.count),
                cluster_drive=_get_single_drive(
                    cluster.attached_drives, cluster.cluster_type
                ),
                cluster_type=cluster.cluster_type,
            )
        )

    return CurrentClusters(zonal=current_zonal, regional=current_regional)


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
                f"Scenario '{scenario_name}' not in baseline.\n{_BASELINE_HELP}"
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
            f"diff={((actual_total - baseline_total) / baseline_total) * 100:+.2f}%.\n"
            f"{_BASELINE_HELP}"
        )

        added = actual_keys - baseline_keys
        removed = baseline_keys - actual_keys
        assert baseline_keys == actual_keys, (
            f"Cost keys changed for {scenario_name}: "
            f"added={added}, removed={removed}.\n{_BASELINE_HELP}"
        )

        # For CostAwareModel scenarios, also verify extract_baseline_plan
        # Composite models (like Key-Value) now work via unified routing through
        # _sub_models() DAG - the planner automatically routes costs to each model
        model = planner.models[scenario["model"]]
        if isinstance(model, CostAwareModel):
            # Clusters should satisfy the model's own requirements
            self_check = compare_plans(
                cap_plan,
                cap_plan,
                strategy=ComparisonStrategy.requirements,
                tolerances=ResourceTolerances(
                    default=gte(1.0),
                    annual_cost=exact_match(),
                ),
            )
            assert self_check.is_equivalent, (
                f"Recommendation self-consistency failed for {scenario_name}: "
                + "; ".join(str(d) for d in self_check.get_out_of_tolerance())
            )

            current_clusters = _clusters_to_current(cap_plan)
            desires_with_current = scenario["desires"].model_copy(deep=True)
            desires_with_current.current_clusters = current_clusters

            extracted = planner.extract_baseline_plan(
                model_name=scenario["model"],
                region=scenario["region"],
                desires=desires_with_current,
                extra_model_arguments=scenario["extra_args"] or {},
            )

            extracted_total = float(extracted.candidate_clusters.total_annual_cost)
            assert extracted_total == pytest.approx(baseline_total, rel=0.01), (
                f"extract_baseline_plan cost drift for {scenario_name}: "
                f"baseline=${baseline_total:,.2f}, extracted=${extracted_total:,.2f}"
            )

            # Verify each cost bucket matches the original plan
            original_costs = cap_plan.candidate_clusters.annual_costs
            extracted_costs = extracted.candidate_clusters.annual_costs
            assert set(extracted_costs.keys()) == set(original_costs.keys()), (
                f"extract_baseline_plan cost keys mismatch for {scenario_name}: "
                f"original={set(original_costs.keys())}, "
                f"extracted={set(extracted_costs.keys())}"
            )
            for key in original_costs:
                assert float(extracted_costs[key]) == pytest.approx(
                    float(original_costs[key]), rel=0.01
                ), (
                    f"extract_baseline_plan bucket '{key}' drift for {scenario_name}: "
                    f"original=${float(original_costs[key]):,.2f}, "
                    f"extracted=${float(extracted_costs[key]):,.2f}"
                )

            # Extracted baseline's clusters should satisfy original requirements
            roundtrip = compare_plans(
                extracted,
                cap_plan,
                strategy=ComparisonStrategy.requirements,
                tolerances=ResourceTolerances(
                    default=gte(1.0),
                    annual_cost=ignore_resource(),
                ),
            )
            assert roundtrip.is_equivalent, (
                f"Baseline round-trip failed for {scenario_name}: "
                + "; ".join(str(d) for d in roundtrip.get_out_of_tolerance())
            )


class TestExtractBaselinePlanValidation:
    """Tests for extract_baseline_plan input validation and edge cases."""

    def test_error_current_clusters_none(self):
        """Error when current_clusters is None."""
        desires = CapacityDesires()
        with pytest.raises(ValueError, match="current_clusters is None"):
            planner.extract_baseline_plan(
                model_name="org.netflix.cassandra",
                region="us-east-1",
                desires=desires,
            )

    def test_error_empty_clusters(self):
        """Error when no zonal or regional clusters defined."""
        desires = CapacityDesires(current_clusters=CurrentClusters())
        with pytest.raises(ValueError, match="no zonal or regional"):
            planner.extract_baseline_plan(
                model_name="org.netflix.cassandra",
                region="us-east-1",
                desires=desires,
            )

    def test_error_invalid_model(self):
        """Error when model_name doesn't exist."""
        current = CurrentZoneClusterCapacity(
            cluster_instance_name="m5.xlarge",
            cluster_instance=None,
            cluster_instance_count=certain_int(3),
        )
        desires = CapacityDesires(current_clusters=CurrentClusters(zonal=[current]))
        with pytest.raises(ValueError, match="does not exist"):
            planner.extract_baseline_plan(
                model_name="org.netflix.nonexistent",
                region="us-east-1",
                desires=desires,
            )

    def test_error_invalid_instance_name(self):
        """Error when instance name doesn't exist in hardware catalog."""
        current = CurrentZoneClusterCapacity(
            cluster_instance_name="nonexistent-instance-type",
            cluster_instance=None,
            cluster_instance_count=certain_int(3),
        )
        desires = CapacityDesires(current_clusters=CurrentClusters(zonal=[current]))
        with pytest.raises(ValueError, match="nonexistent-instance-type"):
            planner.extract_baseline_plan(
                model_name="org.netflix.cassandra",
                region="us-east-1",
                desires=desires,
            )

    def test_uses_model_service_name(self):
        """Uses model's service_name for requirement_type."""
        current = CurrentZoneClusterCapacity(
            cluster_instance_name="m5.xlarge",
            cluster_instance=None,
            cluster_instance_count=certain_int(3),
            cluster_type="cassandra",
        )
        desires = CapacityDesires(current_clusters=CurrentClusters(zonal=[current]))

        baseline = planner.extract_baseline_plan(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=desires,
        )

        assert "cassandra" in baseline.requirements.zonal[0].requirement_type

    def test_instance_resolved_from_hardware_catalog(self):
        """Instance is resolved from hardware catalog when not provided."""
        current = CurrentZoneClusterCapacity(
            cluster_instance_name="m5.xlarge",
            cluster_instance=None,
            cluster_instance_count=certain_int(4),
            cluster_type="cassandra",
        )
        desires = CapacityDesires(current_clusters=CurrentClusters(zonal=[current]))

        baseline = planner.extract_baseline_plan(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=desires,
        )

        assert baseline.candidate_clusters.zonal[0].count == 4
        assert baseline.candidate_clusters.zonal[0].instance.name == "m5.xlarge"
        assert baseline.candidate_clusters.zonal[0].instance.cpu == 4
