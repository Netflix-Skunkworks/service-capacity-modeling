"""Baseline tests for Cassandra unconstrained instance sizing.

With require_local_disks=False and no required_cluster_size, the planner
picks the cheapest viable option across all families. For a typical
350k read / 30k write workload with 500 GiB state, the top plan is a
≤8xlarge instance (currently c6id.4xlarge).

The large_instance_regret parameter (defined but not yet active) will
add a within-family penalty for instances above 8xlarge in a subsequent
change.
"""

from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.interface import (
    CapacityDesires,
    CurrentClusters,
    CurrentZoneClusterCapacity,
    DataShape,
    Interval,
    QueryPattern,
    certain_int,
)
from service_capacity_modeling.models.plan_comparison import compare_plans
from service_capacity_modeling.tools.capture_baseline_costs import SCENARIOS


scenario = SCENARIOS["cassandra_vertical_baseline"]


class TestCassandraInstanceSizingBaseline:
    """Baseline tests for unconstrained Cassandra instance sizing."""

    def test_deterministic_prefers_small_instance(self):
        """plan_certain() picks ≤8xlarge when unconstrained."""
        cap_plans = planner.plan_certain(
            model_name=scenario["model"],
            region=scenario["region"],
            desires=scenario["desires"],
            extra_model_arguments=scenario["extra_args"],
        )

        assert cap_plans, "No capacity plans generated"
        cap_plan = cap_plans[0]
        result = cap_plan.candidate_clusters.zonal[0]

        # Unconstrained: cheapest family/size wins — currently ≤8xlarge
        instance_size = result.instance.name.split(".")[-1]
        assert instance_size in ("2xlarge", "4xlarge", "8xlarge"), (
            f"Expected ≤8xlarge unconstrained, got {result.instance.name}"
        )

    def test_least_regret_with_pinned_cluster_size(self):
        """plan() with required_cluster_size=2 picks large (capacity constraint)."""
        uncertain_desires = CapacityDesires(
            service_tier=1,
            query_pattern=QueryPattern(
                estimated_read_per_second=Interval(
                    low=200_000, mid=350_000, high=500_000, confidence=0.98
                ),
                estimated_write_per_second=Interval(
                    low=20_000, mid=30_000, high=40_000, confidence=0.98
                ),
            ),
            data_shape=DataShape(
                estimated_state_size_gib=Interval(
                    low=300, mid=500, high=700, confidence=0.98
                ),
                estimated_compression_ratio=Interval(
                    low=1, mid=1, high=1, confidence=1
                ),
            ),
        )

        result = planner.plan(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=uncertain_desires,
            extra_model_arguments={
                "require_local_disks": False,
                "required_cluster_size": 2,
            },
        )

        lr = result.least_regret[0]
        cluster = lr.candidate_clusters.zonal[0]

        # Pinned at 2 nodes — must pick large instance to fit workload
        assert cluster.count == 2
        instance_size = cluster.instance.name.split(".")[-1]
        assert instance_size in ("16xlarge", "24xlarge"), (
            f"Expected large instance, got {cluster.instance.name}"
        )

    def test_baseline_comparison(self):
        """compare_plans() round-trip works with unconstrained recommendation."""
        cap_plans = planner.plan_certain(
            model_name=scenario["model"],
            region=scenario["region"],
            desires=scenario["desires"],
            extra_model_arguments=scenario["extra_args"],
        )
        assert cap_plans
        rec = cap_plans[0]
        rec_cluster = rec.candidate_clusters.zonal[0]

        # Build current deployment from recommendation (round-trip)
        current = CurrentZoneClusterCapacity(
            cluster_instance_name=rec_cluster.instance.name,
            cluster_instance_count=certain_int(rec_cluster.count),
            cluster_type="cassandra",
        )
        desires_with_current = scenario["desires"].model_copy(deep=True)
        desires_with_current.current_clusters = CurrentClusters(zonal=[current])

        # Extract baseline and compare
        baseline = planner.extract_baseline_plan(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=desires_with_current,
            extra_model_arguments=scenario["extra_args"],
        )
        comparison = compare_plans(baseline, rec)

        # Unconstrained top pick should be ≤8xlarge
        instance_size = rec_cluster.instance.name.split(".")[-1]
        assert instance_size in ("2xlarge", "4xlarge", "8xlarge"), (
            f"Expected ≤8xlarge unconstrained, got {rec_cluster.instance.name}"
        )
        assert comparison.cpu.ratio > 0
