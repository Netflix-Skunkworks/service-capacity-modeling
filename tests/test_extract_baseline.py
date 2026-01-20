"""
Tests for extract_baseline_plan functionality.

These tests verify that extract_baseline_plan produces costs consistent with
the capacity planner when given the same cluster configuration.

The approach:
1. Run plan_certain() to get a recommended plan
2. Convert those clusters to CurrentClusters format
3. Run extract_baseline_plan() with those current_clusters
4. Compare the costs - they should match

Only models with CostAwareModel mixin are supported: Cassandra, EVCache, Kafka.
"""

import pytest

from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.interface import (
    CapacityDesires,
    CurrentClusters,
    CurrentZoneClusterCapacity,
    Interval,
    certain_float,
    certain_int,
    DataShape,
    QueryPattern,
)


# ============================================================================
# Test Scenarios - subset of cost regression scenarios for CostAwareModel
# ============================================================================

CASSANDRA_SMALL_HIGH_QPS = CapacityDesires(
    service_tier=1,
    query_pattern=QueryPattern(
        estimated_read_per_second=certain_int(100_000),
        estimated_write_per_second=certain_int(100_000),
        estimated_mean_read_latency_ms=certain_float(0.5),
        estimated_mean_write_latency_ms=certain_float(0.4),
    ),
    data_shape=DataShape(estimated_state_size_gib=certain_int(10)),
)

CASSANDRA_HIGH_WRITES = CapacityDesires(
    service_tier=1,
    query_pattern=QueryPattern(
        estimated_read_per_second=certain_int(10_000),
        estimated_write_per_second=certain_int(500_000),
    ),
    data_shape=DataShape(estimated_state_size_gib=certain_int(300)),
)

THROUGHPUT = 100 * 1024 * 1024  # 100 MiB/s
KAFKA_THROUGHPUT = CapacityDesires(
    service_tier=1,
    query_pattern=QueryPattern(
        estimated_read_per_second=Interval(low=1, mid=2, high=2, confidence=0.98),
        estimated_write_per_second=Interval(low=1, mid=1, high=1, confidence=0.98),
        estimated_mean_write_size_bytes=Interval(
            low=THROUGHPUT, mid=THROUGHPUT, high=THROUGHPUT * 2, confidence=0.98
        ),
    ),
)

EVCACHE_SMALL = CapacityDesires(
    service_tier=1,
    query_pattern=QueryPattern(
        estimated_read_per_second=certain_int(100_000),
        estimated_write_per_second=certain_int(10_000),
        estimated_mean_read_latency_ms=certain_float(1.0),
    ),
    data_shape=DataShape(
        estimated_state_size_gib=certain_int(10),
        estimated_state_item_count=Interval(
            low=1_000_000, mid=10_000_000, high=20_000_000, confidence=0.98
        ),
    ),
)

# Scenarios for extract_baseline_plan cost comparison tests
# Only CostAwareModel models: Cassandra, Kafka, EVCache
BASELINE_SCENARIOS = {
    "cassandra_small_high_qps_local": {
        "model": "org.netflix.cassandra",
        "region": "us-east-1",
        "desires": CASSANDRA_SMALL_HIGH_QPS,
        "extra_args": {"require_local_disks": True, "copies_per_region": 3},
    },
    "cassandra_high_writes_ebs": {
        "model": "org.netflix.cassandra",
        "region": "us-east-1",
        "desires": CASSANDRA_HIGH_WRITES,
        "extra_args": {"require_local_disks": False, "copies_per_region": 2},
    },
    "kafka_100mib_throughput": {
        "model": "org.netflix.kafka",
        "region": "us-east-1",
        "desires": KAFKA_THROUGHPUT,
        "extra_args": {"require_local_disks": False},
    },
    "evcache_small_no_replication": {
        "model": "org.netflix.evcache",
        "region": "us-east-1",
        "desires": EVCACHE_SMALL,
        "extra_args": {"cross_region_replication": "none"},
    },
}


def _clusters_to_current(cap_plan):
    """Convert plan's candidate_clusters to CurrentClusters format.

    Takes the zonal clusters from a capacity plan and converts them to
    CurrentZoneClusterCapacity format for use with extract_baseline_plan.
    """
    current_zonal = []
    for cluster in cap_plan.candidate_clusters.zonal:
        current_zonal.append(
            CurrentZoneClusterCapacity(
                cluster_instance_name=cluster.instance.name,
                cluster_instance=None,  # Will be resolved from catalog
                cluster_instance_count=Interval(
                    low=cluster.count,
                    mid=cluster.count,
                    high=cluster.count,
                    confidence=1.0,
                ),
                # Include drive if attached
                cluster_drive=cluster.attached_drives[0]
                if cluster.attached_drives
                else None,
            )
        )
    return CurrentClusters(zonal=current_zonal)


class TestExtractBaselinePlanCostRegression:
    """Test that extract_baseline_plan costs match plan_certain costs.

    This validates that when you:
    1. Run plan_certain() to get a recommended plan
    2. Extract a baseline from those same clusters

    Cluster infrastructure costs should match exactly. Service costs (network,
    backup) may differ because they depend on the requirement calculation which
    differs between capacity_plan() and extract_baseline_plan().
    """

    @pytest.mark.parametrize("scenario_name", list(BASELINE_SCENARIOS.keys()))
    def test_baseline_cluster_cost_matches_planner(self, scenario_name: str) -> None:
        """Baseline cluster infrastructure cost should match planner exactly."""
        scenario = BASELINE_SCENARIOS[scenario_name]

        # Step 1: Get recommended plan from planner
        cap_plans = planner.plan_certain(
            model_name=scenario["model"],
            region=scenario["region"],
            desires=scenario["desires"],
            num_results=1,
            extra_model_arguments=scenario["extra_args"] or {},
        )

        if not cap_plans:
            pytest.skip(f"No capacity plans generated for {scenario_name}")

        recommended = cap_plans[0]

        # Step 2: Convert recommended clusters to CurrentClusters
        current_clusters = _clusters_to_current(recommended)

        # Step 3: Create desires with current_clusters populated
        desires_with_current = scenario["desires"].model_copy(deep=True)
        desires_with_current.current_clusters = current_clusters

        # Step 4: Extract baseline from current clusters
        baseline = planner.extract_baseline_plan(
            model_name=scenario["model"],
            region=scenario["region"],
            desires=desires_with_current,
            extra_model_arguments=scenario["extra_args"] or {},
        )

        # Step 5: Compare cluster infrastructure costs (zonal-clusters key)
        # Service costs (network, backup) may differ due to requirement calc
        service_type = scenario["model"].split(".")[-1]
        cluster_key = f"{service_type}.zonal-clusters"

        recommended_cluster_cost = float(
            recommended.candidate_clusters.annual_costs.get(cluster_key, 0)
        )
        baseline_cluster_cost = float(
            baseline.candidate_clusters.annual_costs.get(cluster_key, 0)
        )

        assert baseline_cluster_cost == pytest.approx(
            recommended_cluster_cost, rel=0.01
        ), (
            f"Cluster cost mismatch for {scenario_name}: "
            f"recommended=${recommended_cluster_cost:,.2f}, "
            f"baseline=${baseline_cluster_cost:,.2f}"
        )

    @pytest.mark.parametrize("scenario_name", list(BASELINE_SCENARIOS.keys()))
    def test_baseline_cost_keys_match_planner(self, scenario_name: str) -> None:
        """Baseline extraction should produce same cost keys as planner."""
        scenario = BASELINE_SCENARIOS[scenario_name]

        # Get recommended plan
        cap_plans = planner.plan_certain(
            model_name=scenario["model"],
            region=scenario["region"],
            desires=scenario["desires"],
            num_results=1,
            extra_model_arguments=scenario["extra_args"] or {},
        )

        if not cap_plans:
            pytest.skip(f"No capacity plans generated for {scenario_name}")

        recommended = cap_plans[0]

        # Convert to current clusters and extract baseline
        current_clusters = _clusters_to_current(recommended)
        desires_with_current = scenario["desires"].model_copy(deep=True)
        desires_with_current.current_clusters = current_clusters

        baseline = planner.extract_baseline_plan(
            model_name=scenario["model"],
            region=scenario["region"],
            desires=desires_with_current,
            extra_model_arguments=scenario["extra_args"] or {},
        )

        # Compare cost breakdown keys
        recommended_keys = set(recommended.candidate_clusters.annual_costs.keys())
        baseline_keys = set(baseline.candidate_clusters.annual_costs.keys())

        assert baseline_keys == recommended_keys, (
            f"Cost keys differ for {scenario_name}: "
            f"recommended={recommended_keys}, baseline={baseline_keys}"
        )

    @pytest.mark.parametrize("scenario_name", list(BASELINE_SCENARIOS.keys()))
    def test_baseline_total_cost_matches_planner(self, scenario_name: str) -> None:
        """Baseline total_annual_cost should match planner exactly."""
        scenario = BASELINE_SCENARIOS[scenario_name]

        # Get recommended plan
        cap_plans = planner.plan_certain(
            model_name=scenario["model"],
            region=scenario["region"],
            desires=scenario["desires"],
            num_results=1,
            extra_model_arguments=scenario["extra_args"] or {},
        )

        if not cap_plans:
            pytest.skip(f"No capacity plans generated for {scenario_name}")

        recommended = cap_plans[0]

        # Convert to current clusters and extract baseline
        current_clusters = _clusters_to_current(recommended)
        desires_with_current = scenario["desires"].model_copy(deep=True)
        desires_with_current.current_clusters = current_clusters

        baseline = planner.extract_baseline_plan(
            model_name=scenario["model"],
            region=scenario["region"],
            desires=desires_with_current,
            extra_model_arguments=scenario["extra_args"] or {},
        )

        # Compare total annual cost
        recommended_total = recommended.candidate_clusters.total_annual_cost
        baseline_total = baseline.candidate_clusters.total_annual_cost

        assert baseline_total == pytest.approx(recommended_total, rel=0.01), (
            f"Total cost mismatch for {scenario_name}: "
            f"recommended=${recommended_total:,.2f}, "
            f"baseline=${baseline_total:,.2f}\n"
            f"Breakdown - recommended: {recommended.candidate_clusters.annual_costs}\n"
            f"Breakdown - baseline: {baseline.candidate_clusters.annual_costs}"
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
            cluster_instance_count=Interval(low=3, mid=3, high=3, confidence=1.0),
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
            cluster_instance_count=Interval(low=3, mid=3, high=3, confidence=1.0),
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
            cluster_instance_count=Interval(low=3, mid=3, high=3, confidence=1.0),
        )
        desires = CapacityDesires(current_clusters=CurrentClusters(zonal=[current]))

        baseline = planner.extract_baseline_plan(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=desires,
            extra_model_arguments={"copies_per_region": 3},
        )

        assert "cassandra" in baseline.requirements.zonal[0].requirement_type

    def test_instance_resolved_from_hardware_catalog(self):
        """Instance is resolved from hardware catalog when not provided."""
        current = CurrentZoneClusterCapacity(
            cluster_instance_name="m5.xlarge",
            cluster_instance=None,
            cluster_instance_count=Interval(low=4, mid=4, high=4, confidence=1.0),
        )
        desires = CapacityDesires(current_clusters=CurrentClusters(zonal=[current]))

        baseline = planner.extract_baseline_plan(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=desires,
            extra_model_arguments={"copies_per_region": 3},
        )

        # m5.xlarge has 4 vCPUs
        assert baseline.candidate_clusters.zonal[0].count == 4
        assert baseline.candidate_clusters.zonal[0].instance.name == "m5.xlarge"
        assert baseline.candidate_clusters.zonal[0].instance.cpu == 4
