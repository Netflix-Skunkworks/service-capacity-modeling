"""Tests for family selection penalties.

Two penalties interact:
1. Cassandra-level different_family_regret (default 10%) — penalises
   cross-family plans relative to the *current* cluster's family.
   Configured via extra_model_arguments["different_family_regret"].
2. Planner-level preferred_family penalty (15%, _PREFERRED_FAMILY_RANK_PENALTY)
   — applied by the planner to any plan whose instance family is not in
   model.preferred_families(). Models declare their preferred set; the
   planner enforces the bias uniformly (no per-model code needed).

c6id is not in CASSANDRA_PREFERRED_FAMILIES, so it always carries the 15%
planner penalty. c6id is ~18% cheaper per-CPU than m6id. Combined:
  - default (10% model + 15% planner = 25% total): m6id wins
  - disabled model penalty (0% + 15% planner = 15% total): c6id still wins
    (18% savings > 15% penalty)
  - high model penalty (25% + 15% = 40%): m6id wins decisively
"""

import pytest
from service_capacity_modeling.capacity_planner import planner, PlannerArguments
from service_capacity_modeling.interface import (
    CapacityDesires,
    CurrentClusters,
    CurrentZoneClusterCapacity,
    DataShape,
    Interval,
    QueryPattern,
)
from service_capacity_modeling.models.org.netflix.cassandra import (
    NflxCassandraArguments,
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
            {"different_family_regret": 0.25},
            "m6id",
            id="high_regret_keeps_m6id",
        ),
        pytest.param("m6id.8xlarge", {}, "m6id", id="preferred_penalty_keeps_m6id"),
        pytest.param(
            "m6id.8xlarge",
            {"different_family_regret": 0},
            "c6id",
            id="disabled_pure_cost",
        ),
        pytest.param(None, {}, "c6id", id="new_provisioning"),
        pytest.param(
            "c6id.8xlarge",
            {"different_family_regret": 0.25},
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
        extra_model_arguments={"different_family_regret": 0.25},
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


def test_planner_arguments_zero_penalty_pure_cost():
    """PlannerArguments.preferred_family_penalty=0 removes family bias.

    With both penalties disabled (different_family_regret=0 AND
    preferred_family_penalty=0), c6id wins as the cheaper option.
    """
    desires = _desires("m6id.8xlarge")
    plans = planner.plan_certain(
        model_name="org.netflix.cassandra",
        region="us-east-1",
        desires=desires,
        extra_model_arguments={"different_family_regret": 0},
        instance_families=["m6id", "c6id"],
        num_results=20,
        max_results_per_family=10,
        planner_arguments=PlannerArguments(preferred_family_penalty=0.0),
    )
    assert plans, "planner returned no plans"
    assert plans[0].candidate_clusters.zonal[0].instance.family == "c6id", (
        "With preferred_family_penalty=0 and different_family_regret=0, "
        "c6id should rank first as the cheaper option"
    )


def test_planner_arguments_max_results_per_family():
    """PlannerArguments.max_results_per_family limits plans per instance family."""
    from collections import Counter

    desires = _desires(None)
    plans = planner.plan_certain(
        model_name="org.netflix.cassandra",
        region="us-east-1",
        desires=desires,
        extra_model_arguments={"require_local_disks": False},
        instance_families=["i4i", "r6a"],
        num_results=20,
        planner_arguments=PlannerArguments(max_results_per_family=3),
    )
    assert plans, "planner returned no plans"
    family_counts = Counter(
        p.candidate_clusters.zonal[0].instance.family for p in plans
    )
    assert all(count <= 3 for count in family_counts.values()), (
        f"Each family should appear at most 3 times, got: {dict(family_counts)}"
    )


def test_non_preferred_family_regret_is_planner_responsibility():
    """Finding 1: non_preferred_family_regret must not be a Cassandra argument.

    The preferred-family rank penalty belongs in the planner (via preferred_families()),
    not in each model's argument list. After this change, NflxCassandraArguments
    should not expose non_preferred_family_regret.
    """
    fields = dict(NflxCassandraArguments.model_fields)
    assert "non_preferred_family_regret" not in fields, (
        "non_preferred_family_regret should be removed from Cassandra args; "
        "the planner applies this penalty via model.preferred_families()"
    )
