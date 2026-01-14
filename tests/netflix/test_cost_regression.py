"""
Regression tests to ensure cost calculations remain stable.

These tests freeze the current annual_cost outputs from capacity planning
to prevent regressions when refactoring cost calculation logic.

The baseline costs were captured from the main branch before any changes
to cost calculation structure. Any significant deviation (>1%) from these
baselines indicates a potential regression that needs investigation.
"""

import pytest

from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.interface import (
    CapacityDesires,
    certain_float,
    certain_int,
    DataShape,
    Interval,
    QueryPattern,
)


# ============================================================================
# Test Fixtures - CapacityDesires for each scenario
# ============================================================================

# RDS Scenarios
RDS_SMALL_TIER1 = CapacityDesires(
    service_tier=1,
    query_pattern=QueryPattern(
        estimated_read_per_second=certain_int(200),
        estimated_write_per_second=certain_int(100),
        estimated_mean_read_latency_ms=certain_float(10),
        estimated_mean_write_latency_ms=certain_float(10),
    ),
    data_shape=DataShape(
        estimated_state_size_gib=certain_int(50),
    ),
)

RDS_TIER3 = CapacityDesires(
    service_tier=3,
    query_pattern=QueryPattern(
        estimated_read_per_second=certain_int(200),
        estimated_write_per_second=certain_int(100),
        estimated_mean_read_latency_ms=certain_float(20),
        estimated_mean_write_latency_ms=certain_float(20),
    ),
    data_shape=DataShape(
        estimated_state_size_gib=certain_int(200),
    ),
)

# Aurora Scenarios
AURORA_SMALL_TIER1 = CapacityDesires(
    service_tier=1,
    query_pattern=QueryPattern(
        estimated_read_per_second=certain_int(100),
        estimated_write_per_second=certain_int(100),
        estimated_mean_read_latency_ms=certain_float(10),
        estimated_mean_write_latency_ms=certain_float(10),
    ),
    data_shape=DataShape(
        estimated_state_size_gib=certain_int(50),
    ),
)

AURORA_TIER3 = CapacityDesires(
    service_tier=3,
    query_pattern=QueryPattern(
        estimated_read_per_second=certain_int(200),
        estimated_write_per_second=certain_int(100),
        estimated_mean_read_latency_ms=certain_float(10),
        estimated_mean_write_latency_ms=certain_float(10),
    ),
    data_shape=DataShape(
        estimated_state_size_gib=certain_int(200),
    ),
)

# Cassandra Scenarios
CASSANDRA_SMALL_HIGH_QPS = CapacityDesires(
    service_tier=1,
    query_pattern=QueryPattern(
        estimated_read_per_second=certain_int(100_000),
        estimated_write_per_second=certain_int(100_000),
        estimated_mean_read_latency_ms=certain_float(0.5),
        estimated_mean_write_latency_ms=certain_float(0.4),
    ),
    data_shape=DataShape(
        estimated_state_size_gib=certain_int(10),
    ),
)

CASSANDRA_HIGH_WRITES = CapacityDesires(
    service_tier=1,
    query_pattern=QueryPattern(
        estimated_read_per_second=certain_int(10_000),
        estimated_write_per_second=certain_int(500_000),
    ),
    data_shape=DataShape(
        estimated_state_size_gib=certain_int(300),
    ),
)

# Kafka Scenarios
KAFKA_SIMPLE = CapacityDesires(
    service_tier=1,
    query_pattern=QueryPattern(
        estimated_read_per_second=certain_int(10000),
        estimated_write_per_second=certain_int(10000),
        estimated_mean_write_size_bytes=certain_int(512),
    ),
    data_shape=DataShape(
        estimated_state_size_gib=certain_int(100),
    ),
)

# EVCache Scenarios
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

EVCACHE_LARGE = CapacityDesires(
    service_tier=1,
    query_pattern=QueryPattern(
        estimated_read_per_second=certain_int(500_000),
        estimated_write_per_second=certain_int(50_000),
        estimated_mean_read_latency_ms=certain_float(1.0),
    ),
    data_shape=DataShape(
        estimated_state_size_gib=certain_int(500),
        estimated_state_item_count=Interval(
            low=10_000_000, mid=100_000_000, high=200_000_000, confidence=0.98
        ),
    ),
)


# ============================================================================
# RDS Cost Regression Tests
# ============================================================================


class TestRDSCostRegression:
    """Regression tests for RDS cost calculations."""

    def test_rds_small_tier1_total_cost(self):
        """Test RDS small tier 1 total annual cost."""
        cap_plan = planner.plan_certain(
            model_name="org.netflix.rds",
            region="us-east-1",
            desires=RDS_SMALL_TIER1,
        )[0]

        clusters = cap_plan.candidate_clusters
        assert float(clusters.total_annual_cost) == pytest.approx(2673.34, rel=0.01)

    def test_rds_small_tier1_cost_breakdown(self):
        """Test RDS small tier 1 cost breakdown keys."""
        cap_plan = planner.plan_certain(
            model_name="org.netflix.rds",
            region="us-east-1",
            desires=RDS_SMALL_TIER1,
        )[0]

        clusters = cap_plan.candidate_clusters
        assert "rds-cluster.regional-clusters" in clusters.annual_costs
        assert float(clusters.annual_costs["rds-cluster.regional-clusters"]) == pytest.approx(
            2673.34, rel=0.01
        )

    def test_rds_tier3_total_cost(self):
        """Test RDS tier 3 total annual cost."""
        cap_plan = planner.plan_certain(
            model_name="org.netflix.rds",
            region="us-east-1",
            desires=RDS_TIER3,
        )[0]

        clusters = cap_plan.candidate_clusters
        assert float(clusters.total_annual_cost) == pytest.approx(3608.00, rel=0.01)


# ============================================================================
# Aurora Cost Regression Tests
# ============================================================================


class TestAuroraCostRegression:
    """Regression tests for Aurora cost calculations."""

    def test_aurora_small_tier1_total_cost(self):
        """Test Aurora small tier 1 total annual cost."""
        cap_plan = planner.plan_certain(
            model_name="org.netflix.aurora",
            region="us-east-1",
            desires=AURORA_SMALL_TIER1,
        )[0]

        clusters = cap_plan.candidate_clusters
        assert float(clusters.total_annual_cost) == pytest.approx(7362.98, rel=0.01)

    def test_aurora_small_tier1_cost_breakdown(self):
        """Test Aurora small tier 1 cost breakdown keys.

        Aurora costs are broken down into:
        - aurora-cluster.regional-clusters: instance + storage costs
        - aurora-cluster.io: IO operation costs (separate for baseline extraction parity)
        """
        cap_plan = planner.plan_certain(
            model_name="org.netflix.aurora",
            region="us-east-1",
            desires=AURORA_SMALL_TIER1,
        )[0]

        clusters = cap_plan.candidate_clusters
        # Infrastructure costs (instance + storage)
        assert "aurora-cluster.regional-clusters" in clusters.annual_costs
        assert float(
            clusters.annual_costs["aurora-cluster.regional-clusters"]
        ) == pytest.approx(7158.00, rel=0.01)

        # IO costs (separate for baseline extraction parity)
        assert "aurora-cluster.io" in clusters.annual_costs
        assert float(
            clusters.annual_costs["aurora-cluster.io"]
        ) == pytest.approx(204.98, rel=0.01)

    def test_aurora_tier3_total_cost(self):
        """Test Aurora tier 3 total annual cost."""
        cap_plan = planner.plan_certain(
            model_name="org.netflix.aurora",
            region="us-east-1",
            desires=AURORA_TIER3,
        )[0]

        clusters = cap_plan.candidate_clusters
        assert float(clusters.total_annual_cost) == pytest.approx(4162.13, rel=0.01)


# ============================================================================
# Cassandra Cost Regression Tests
# ============================================================================


class TestCassandraCostRegression:
    """Regression tests for Cassandra cost calculations."""

    def test_cassandra_small_high_qps_total_cost(self):
        """Test Cassandra small high QPS total annual cost."""
        cap_plan = planner.plan_certain(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=CASSANDRA_SMALL_HIGH_QPS,
            extra_model_arguments={"require_local_disks": True},
        )[0]

        clusters = cap_plan.candidate_clusters
        assert float(clusters.total_annual_cost) == pytest.approx(149891.40, rel=0.01)

    def test_cassandra_small_high_qps_cost_breakdown(self):
        """Test Cassandra small high QPS has all cost keys."""
        cap_plan = planner.plan_certain(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=CASSANDRA_SMALL_HIGH_QPS,
            extra_model_arguments={"require_local_disks": True},
        )[0]

        clusters = cap_plan.candidate_clusters

        # Should have zonal cluster costs
        assert "cassandra.zonal-clusters" in clusters.annual_costs
        assert float(clusters.annual_costs["cassandra.zonal-clusters"]) == pytest.approx(
            29508.0, rel=0.01
        )

        # Should have network costs (inter-region and intra-region)
        assert "cassandra.net.inter.region" in clusters.annual_costs
        assert "cassandra.net.intra.region" in clusters.annual_costs

        # Should have backup costs
        assert "cassandra.backup.s3-standard" in clusters.annual_costs

    def test_cassandra_high_writes_ebs_total_cost(self):
        """Test Cassandra high writes with EBS total annual cost."""
        cap_plan = planner.plan_certain(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=CASSANDRA_HIGH_WRITES,
            extra_model_arguments={"require_local_disks": False, "copies_per_region": 2},
        )[0]

        clusters = cap_plan.candidate_clusters
        assert float(clusters.total_annual_cost) == pytest.approx(445286.82, rel=0.01)


# ============================================================================
# Kafka Cost Regression Tests
# ============================================================================


class TestKafkaCostRegression:
    """Regression tests for Kafka cost calculations."""

    def test_kafka_simple_total_cost(self):
        """Test Kafka simple scenario total annual cost."""
        cap_plan = planner.plan_certain(
            model_name="org.netflix.kafka",
            region="us-east-1",
            desires=KAFKA_SIMPLE,
            extra_model_arguments={"require_local_disks": False},
        )[0]

        clusters = cap_plan.candidate_clusters
        assert float(clusters.total_annual_cost) == pytest.approx(148486.02, rel=0.01)

    def test_kafka_simple_cost_breakdown(self):
        """Test Kafka simple scenario has correct cost keys."""
        cap_plan = planner.plan_certain(
            model_name="org.netflix.kafka",
            region="us-east-1",
            desires=KAFKA_SIMPLE,
            extra_model_arguments={"require_local_disks": False},
        )[0]

        clusters = cap_plan.candidate_clusters

        # Kafka should only have zonal cluster costs (no service costs)
        assert "kafka.zonal-clusters" in clusters.annual_costs
        assert float(clusters.annual_costs["kafka.zonal-clusters"]) == pytest.approx(
            148486.02, rel=0.01
        )

        # Should not have network or backup costs
        assert not any("net" in k for k in clusters.annual_costs.keys())
        assert not any("backup" in k for k in clusters.annual_costs.keys())


# ============================================================================
# EVCache Cost Regression Tests
# ============================================================================


class TestEVCacheCostRegression:
    """Regression tests for EVCache cost calculations."""

    def test_evcache_small_no_replication_total_cost(self):
        """Test EVCache small without replication total annual cost."""
        cap_plan = planner.plan_certain(
            model_name="org.netflix.evcache",
            region="us-east-1",
            desires=EVCACHE_SMALL,
            extra_model_arguments={"cross_region_replication": "none"},
        )[0]

        clusters = cap_plan.candidate_clusters
        assert float(clusters.total_annual_cost) == pytest.approx(31265.13, rel=0.01)

    def test_evcache_small_cost_breakdown(self):
        """Test EVCache small has correct cost keys."""
        cap_plan = planner.plan_certain(
            model_name="org.netflix.evcache",
            region="us-east-1",
            desires=EVCACHE_SMALL,
            extra_model_arguments={"cross_region_replication": "none"},
        )[0]

        clusters = cap_plan.candidate_clusters

        # Should have zonal cluster costs
        assert "evcache.zonal-clusters" in clusters.annual_costs
        assert float(clusters.annual_costs["evcache.zonal-clusters"]) == pytest.approx(
            31265.13, rel=0.01
        )

        # Should have spread cost (may be 0.0 for larger clusters)
        assert "evcache.spread.cost" in clusters.annual_costs

        # Should not have network costs when replication is none
        assert not any("net" in k for k in clusters.annual_costs.keys())

    def test_evcache_large_with_replication_total_cost(self):
        """Test EVCache large with replication total annual cost."""
        cap_plan = planner.plan_certain(
            model_name="org.netflix.evcache",
            region="us-east-1",
            desires=EVCACHE_LARGE,
            extra_model_arguments={
                "cross_region_replication": "sets",
                "copies_per_region": 2,
            },
        )[0]

        clusters = cap_plan.candidate_clusters
        assert float(clusters.total_annual_cost) == pytest.approx(175441.83, rel=0.01)

    def test_evcache_large_with_replication_cost_breakdown(self):
        """Test EVCache large with replication has network costs."""
        cap_plan = planner.plan_certain(
            model_name="org.netflix.evcache",
            region="us-east-1",
            desires=EVCACHE_LARGE,
            extra_model_arguments={
                "cross_region_replication": "sets",
                "copies_per_region": 2,
            },
        )[0]

        clusters = cap_plan.candidate_clusters

        # Should have zonal cluster costs
        assert "evcache.zonal-clusters" in clusters.annual_costs

        # Should have spread cost
        assert "evcache.spread.cost" in clusters.annual_costs

        # Should have network costs when replication is enabled
        assert "evcache.net.inter.region" in clusters.annual_costs
        assert "evcache.net.intra.region" in clusters.annual_costs


# ============================================================================
# Cross-Service Consistency Tests
# ============================================================================


class TestCostKeyConsistency:
    """Tests to verify cost key naming conventions are consistent."""

    def test_regional_services_use_regional_key(self):
        """Verify regional services (RDS, Aurora) use '.regional-clusters' key."""
        for model_name, desires in [
            ("org.netflix.rds", RDS_SMALL_TIER1),
            ("org.netflix.aurora", AURORA_SMALL_TIER1),
        ]:
            cap_plan = planner.plan_certain(
                model_name=model_name,
                region="us-east-1",
                desires=desires,
            )[0]

            clusters = cap_plan.candidate_clusters
            assert any(
                "regional-clusters" in k for k in clusters.annual_costs.keys()
            ), f"{model_name} should have regional-clusters cost key"

    def test_zonal_services_use_zonal_key(self):
        """Verify zonal services (Cassandra, Kafka, EVCache) use '.zonal-clusters' key."""
        scenarios = [
            (
                "org.netflix.cassandra",
                CASSANDRA_SMALL_HIGH_QPS,
                {"require_local_disks": True},
            ),
            ("org.netflix.kafka", KAFKA_SIMPLE, {"require_local_disks": False}),
            (
                "org.netflix.evcache",
                EVCACHE_SMALL,
                {"cross_region_replication": "none"},
            ),
        ]

        for model_name, desires, extra_args in scenarios:
            cap_plan = planner.plan_certain(
                model_name=model_name,
                region="us-east-1",
                desires=desires,
                extra_model_arguments=extra_args,
            )[0]

            clusters = cap_plan.candidate_clusters
            assert any(
                "zonal-clusters" in k for k in clusters.annual_costs.keys()
            ), f"{model_name} should have zonal-clusters cost key"

    def test_total_annual_cost_matches_sum(self):
        """Verify total_annual_cost equals sum of all cost components."""
        scenarios = [
            ("org.netflix.rds", RDS_SMALL_TIER1, {}),
            ("org.netflix.aurora", AURORA_SMALL_TIER1, {}),
            (
                "org.netflix.cassandra",
                CASSANDRA_SMALL_HIGH_QPS,
                {"require_local_disks": True},
            ),
            ("org.netflix.kafka", KAFKA_SIMPLE, {"require_local_disks": False}),
            (
                "org.netflix.evcache",
                EVCACHE_SMALL,
                {"cross_region_replication": "none"},
            ),
        ]

        for model_name, desires, extra_args in scenarios:
            cap_plan = planner.plan_certain(
                model_name=model_name,
                region="us-east-1",
                desires=desires,
                extra_model_arguments=extra_args,
            )[0]

            clusters = cap_plan.candidate_clusters
            sum_of_costs = sum(float(v) for v in clusters.annual_costs.values())

            assert float(clusters.total_annual_cost) == pytest.approx(
                sum_of_costs, rel=0.001
            ), f"{model_name}: total_annual_cost should equal sum of annual_costs"
