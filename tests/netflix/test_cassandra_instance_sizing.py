"""Tests for Cassandra instance-sizing regret behavior.

Workload: 350k reads/s, 30k writes/s, 500 GiB state, require_local_disks=False.

Without any regret penalties, the planner sorts by (rank, cost) and AWS pricing
rounding makes larger instances (24xlarge) appear marginally cheaper -- so they win.
This is the problem that large_instance_regret (PR #230) and same_family_regret
(PR #207) fix.

This file establishes the unpenalized baseline. The regret test files verify the fix:
- test_cassandra_large_instance_regret.py -- large_instance_regret flips ordering
- test_cassandra_family_migration.py -- same_family_regret penalizes family switches
"""

from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.interface import (
    CapacityDesires,
    DataShape,
    Interval,
    QueryPattern,
)


# Typical mid-size Cassandra workload -- deterministic (no uncertainty)
DESIRES = CapacityDesires(
    service_tier=1,
    query_pattern=QueryPattern(
        estimated_read_per_second=Interval(
            low=350_000, mid=350_000, high=350_000, confidence=0.98
        ),
        estimated_write_per_second=Interval(
            low=30_000, mid=30_000, high=30_000, confidence=0.98
        ),
    ),
    data_shape=DataShape(
        estimated_state_size_gib=Interval(low=500, mid=500, high=500, confidence=0.98),
        estimated_compression_ratio=Interval(low=1, mid=1, high=1, confidence=1),
    ),
)

NO_REGRET = {
    "require_local_disks": False,
    "large_instance_regret": 0,
    "same_family_regret": 0,
}


class TestUnpenalizedBaseline:
    """Without regret penalties, AWS pricing rounding makes larger instances win."""

    def test_unpenalized_prefers_large_instance(self):
        """With all regrets disabled, pricing rounding favors large instances."""
        cap_plans = planner.plan_certain(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=DESIRES,
            extra_model_arguments=NO_REGRET,
        )

        assert cap_plans, "No capacity plans generated"
        result = cap_plans[0].candidate_clusters.zonal[0]

        # Without regret: large instances win due to AWS 3yr pricing rounding
        instance_size = result.instance.name.split(".")[-1]
        assert instance_size in ("12xlarge", "16xlarge", "24xlarge", "32xlarge"), (
            f"Expected large instance without regret, got {result.instance.name}"
        )
