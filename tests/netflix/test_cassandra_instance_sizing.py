"""Tests for Cassandra instance-sizing regret behavior.

Workload: 350k reads/s, 30k writes/s, 500 GiB state, require_local_disks=False.

Without any regret penalties, the planner sorts by (rank, cost) and picks the
marginally cheapest option. With the m8id family available (Granite Rapids +
local NVMe), that unpenalized pick is now m8id.2xlarge; before m8id existed,
AWS 3yr pricing rounding made large older-gen instances (24xlarge) win here.
Either way, the unpenalized pick is what large_instance_regret (PR #230) and
different_family_regret (PR #207) exist to correct.

This file establishes the unpenalized baseline. The regret test files verify the fix:
- test_cassandra_large_instance_regret.py -- large_instance_regret flips ordering
- test_cassandra_family_migration.py -- different_family_regret penalizes switches
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
    "different_family_regret": 0,
}


class TestUnpenalizedBaseline:
    """Without regret penalties, the (rank, cost) sort picks the marginally
    cheapest option -- now the newest-generation m8id family."""

    def test_unpenalized_prefers_cheapest_newest_family(self):
        """With all regrets disabled, m8id (Granite Rapids + local NVMe) is the
        marginally cheapest pick, so it wins the unpenalized (rank, cost) sort.

        Before the m8id family existed, AWS 3yr pricing rounding made large
        older-gen instances (12xlarge/24xlarge) win here; the regret tests still
        verify that penalties correct whatever the unpenalized pick is.
        """
        cap_plans = planner.plan_certain(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=DESIRES,
            extra_model_arguments=NO_REGRET,
        )

        assert cap_plans, "No capacity plans generated"
        result = cap_plans[0].candidate_clusters.zonal[0]

        # Without regret the newest, marginally-cheapest family wins the sort.
        assert result.instance.name == "m8id.2xlarge", (
            f"Expected m8id.2xlarge without regret, got {result.instance.name}"
        )
