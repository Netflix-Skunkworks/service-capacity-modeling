"""Tests for different_family_regret: prefer current instance family.

When a cluster runs on m6id, migrating to c6id requires paying on-demand
prices (reservations are family-specific). The different_family_regret
(default 10%) penalises cross-family plans so savings must exceed the
threshold to justify switching. c6id is ~13% cheaper than m6id for this
workload, so the default 10% allows the switch while 15% blocks it.
"""

import pytest
from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.interface import (
    CapacityDesires,
    CurrentClusters,
    CurrentZoneClusterCapacity,
    DataShape,
    Interval,
    QueryPattern,
)


def _desires(current_instance=None):
    """Build desires, optionally with a current cluster."""
    base = {
        "service_tier": 1,
        "query_pattern": QueryPattern(
            estimated_read_per_second=Interval(
                low=4000, mid=400_000, high=400_000, confidence=0.98
            ),
            estimated_write_per_second=Interval(
                low=4000, mid=200_000, high=200_000, confidence=0.98
            ),
        ),
        "data_shape": DataShape(
            estimated_state_size_gib=Interval(
                low=1000, mid=2000, high=3000, confidence=0.98
            ),
        ),
    }
    if current_instance:
        base["current_clusters"] = CurrentClusters(
            zonal=[
                CurrentZoneClusterCapacity(
                    cluster_instance_name=current_instance,
                    cluster_instance_count=Interval(
                        low=16, mid=16, high=16, confidence=1.0
                    ),
                    cpu_utilization=Interval(low=10, mid=40, high=60, confidence=0.98),
                )
            ]
        )
    return CapacityDesires(**base)


def _first_family(desires, extra_model_arguments, families=None):
    """Return the top-ranked instance family."""
    if families is None:
        families = ["m6id", "c6id"]
    plans = planner.plan_certain(
        model_name="org.netflix.cassandra",
        region="us-east-1",
        desires=desires,
        extra_model_arguments=extra_model_arguments,
        instance_families=families,
        num_results=20,
        max_results_per_family=10,
    )
    assert plans, "planner returned no plans"
    return plans[0].candidate_clusters.zonal[0].instance.family


@pytest.mark.parametrize(
    "current,extra_args,expected_first",
    [
        pytest.param(
            "m6id.8xlarge",
            {"different_family_regret": 0.15},
            "m6id",
            id="high_regret_keeps_m6id",
        ),
        pytest.param("m6id.8xlarge", {}, "c6id", id="default_allows_savings"),
        pytest.param(
            "m6id.8xlarge",
            {"different_family_regret": 0},
            "c6id",
            id="disabled_pure_cost",
        ),
        pytest.param(None, {}, "c6id", id="new_provisioning"),
        pytest.param(
            "c6id.8xlarge",
            {"different_family_regret": 0.15},
            "c6id",
            id="reverse_keeps_c6id",
        ),
    ],
)
def test_family_ordering(current, extra_args, expected_first):
    desires = _desires(current)
    result = _first_family(desires, extra_args)
    assert result == expected_first, f"expected {expected_first} first, got {result}"


def test_combined_family_and_large_instance_penalties():
    """Both penalties should compose: cross-family plans carry family_migration."""
    desires = _desires("m6id.8xlarge")
    plans = planner.plan_certain(
        model_name="org.netflix.cassandra",
        region="us-east-1",
        desires=desires,
        extra_model_arguments={"different_family_regret": 0.15},
        instance_families=["m6id", "c6id"],
    )
    for p in plans:
        family = p.candidate_clusters.zonal[0].instance.family
        penalties = p.candidate_clusters.zonal[0].cluster_params.get(
            "rank_penalties", {}
        )
        if family != "m6id" and penalties:
            assert "family_migration" in penalties, (
                f"Expected family_migration penalty on {family} plan"
            )
