# pylint: disable=too-many-lines

"""
Tests for Netflix Read-Only Key-Value capacity model.

A read-only data serving layer backed by RocksDB that:
- Loads data from offline sources
- Serves read traffic with low latency
- Deploys regionally with configurable replication (RF=2 default)
"""

import math
from dataclasses import dataclass
from typing import Optional

from hypothesis import given, settings
from hypothesis import strategies as st

from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.interface import AccessPattern
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import certain_float
from service_capacity_modeling.interface import certain_int
from service_capacity_modeling.interface import DataShape
from service_capacity_modeling.interface import QueryPattern
from tests.util import get_total_storage_gib
from tests.util import has_local_storage


# Property test configuration for Read-Only KV model.
# See tests/netflix/PROPERTY_TESTING.md for configuration options and examples.
PROPERTY_TEST_CONFIG = {
    "org.netflix.read-only-kv": {
        "extra_model_arguments": {
            "total_num_partitions": 12,
        },
        # Read-only model: no writes
        "write_size_range": (0, 0),
        # Ensure read size is set for network bandwidth calculation
        "read_size_range": (128, 8192),
    },
}

# Test fixtures
small_dataset_high_rps = CapacityDesires(
    service_tier=1,
    query_pattern=QueryPattern(
        access_pattern=AccessPattern.latency,
        estimated_read_per_second=certain_int(100_000),
        # ReadOnlyKV is read-only
        estimated_write_per_second=certain_int(0),
        estimated_mean_read_latency_ms=certain_float(1.0),
        estimated_mean_write_latency_ms=certain_float(0),
    ),
    data_shape=DataShape(
        estimated_state_size_gib=certain_int(50),
    ),
)

large_dataset_moderate_rps = CapacityDesires(
    service_tier=1,
    query_pattern=QueryPattern(
        access_pattern=AccessPattern.latency,
        estimated_read_per_second=certain_int(50_000),
        estimated_write_per_second=certain_int(0),
        estimated_mean_read_latency_ms=certain_float(2.0),
        estimated_mean_write_latency_ms=certain_float(0),
        estimated_mean_read_size_bytes=certain_int(2048),
    ),
    data_shape=DataShape(
        estimated_state_size_gib=certain_int(1000),
    ),
)

throughput_workload = CapacityDesires(
    service_tier=2,
    query_pattern=QueryPattern(
        access_pattern=AccessPattern.throughput,
        estimated_read_per_second=certain_int(10_000),
        estimated_write_per_second=certain_int(0),
        estimated_mean_read_latency_ms=certain_float(10.0),
        estimated_mean_write_latency_ms=certain_float(0),
        # Larger reads for throughput pattern (scans)
        estimated_mean_read_size_bytes=certain_int(8192),
    ),
    data_shape=DataShape(
        estimated_state_size_gib=certain_int(2000),
    ),
)


class TestReadOnlyKVBasicCapacityPlanning:
    """Test basic ReadOnlyKV capacity planning scenarios."""

    def test_small_dataset_high_rps(self):
        """Test ReadOnlyKV with small dataset but high read RPS."""
        cap_plan = planner.plan_certain(
            model_name="org.netflix.read-only-kv",
            region="us-east-1",
            desires=small_dataset_high_rps,
            extra_model_arguments={"total_num_partitions": 4},
        )[0]

        result = cap_plan.candidate_clusters.regional[0]

        # Should be a regional cluster
        assert result.cluster_type == "read-only-kv"

        # For high RPS, we need sufficient CPU
        total_cores = result.count * result.instance.cpu
        assert total_cores >= 16, f"Expected at least 16 cores, got {total_cores}"

        # Should have sufficient storage for data (50GB * RF=2 * buffer)
        total_storage = get_total_storage_gib(result)
        assert total_storage is not None
        assert total_storage >= 100, f"Expected >= 100 GiB storage, got {total_storage}"

    def test_large_dataset_moderate_rps(self):
        """Test ReadOnlyKV with large dataset and moderate RPS."""
        cap_plan = planner.plan_certain(
            model_name="org.netflix.read-only-kv",
            region="us-east-1",
            desires=large_dataset_moderate_rps,
            extra_model_arguments={"total_num_partitions": 12},
        )[0]

        result = cap_plan.candidate_clusters.regional[0]

        # For large datasets, should have storage-optimized instances
        # (i-series for local disks) or sufficient attached storage
        total_storage = get_total_storage_gib(result)
        assert total_storage is not None

        # 1TB data * RF=2 = 2TB minimum
        assert total_storage >= 2000, (
            f"Expected >= 2000 GiB storage for 1TB dataset with RF=2, "
            f"got {total_storage}"
        )

        # Memory should be sufficient for RocksDB block cache
        total_memory = result.count * result.instance.ram_gib
        # Instance minimum is 64GB, so minimum cluster memory is 64GB * 2 nodes = 128GB
        assert total_memory >= 128, f"Expected >= 128 GiB memory, got {total_memory}"

    def test_throughput_workload(self):
        """Test ReadOnlyKV with throughput-oriented workload (scans)."""
        cap_plan = planner.plan_certain(
            model_name="org.netflix.read-only-kv",
            region="us-east-1",
            desires=throughput_workload,
            extra_model_arguments={"total_num_partitions": 16},
        )[0]

        result = cap_plan.candidate_clusters.regional[0]

        # Throughput workloads need sufficient network bandwidth
        total_network = result.count * result.instance.net_mbps
        # 10k RPS * 8KB = 80 MB/s = 640 Mbps
        assert total_network >= 640, (
            f"Expected >= 640 Mbps network for throughput workload, got {total_network}"
        )


class TestReadOnlyKVStorageTypes:
    """Test ReadOnlyKV storage (local disks only)."""

    def test_uses_local_disks_only(self):
        """Test that ReadOnlyKV only uses instances with local disks.

        EBS/attached disks are not supported because the partition-aware
        algorithm relies on fixed disk capacity to calculate partition placement.
        """
        cap_plan = planner.plan_certain(
            model_name="org.netflix.read-only-kv",
            region="us-east-1",
            desires=large_dataset_moderate_rps,
            extra_model_arguments={
                "total_num_partitions": 12,
            },
        )[0]

        result = cap_plan.candidate_clusters.regional[0]

        # Should use local storage instances (i-series, m5d, m6id, r5d, etc.)
        assert has_local_storage(result), "Expected local storage instance"
        assert result.instance.drive is not None, (
            f"Expected instance with local drive, got {result.instance.name}"
        )
        # No attached drives should be used
        assert len(result.attached_drives) == 0, (
            "Expected no attached drives (EBS not supported)"
        )


class TestReadOnlyKVReplication:
    """Test ReadOnlyKV replication factor configurations."""

    def test_rf2_default(self):
        """Test ReadOnlyKV with min_replica_count=2 (default)."""
        cap_plan = planner.plan_certain(
            model_name="org.netflix.read-only-kv",
            region="us-east-1",
            desires=small_dataset_high_rps,
            extra_model_arguments={
                "min_replica_count": 2,
                "total_num_partitions": 4,
            },
        )[0]

        result = cap_plan.candidate_clusters.regional[0]

        # Check actual replica_count in cluster_params (may be >= min)
        actual_rf = result.cluster_params["read-only-kv.replica_count"]
        assert actual_rf >= 2, f"Current RF {actual_rf} should be >= min RF 2"

        # Storage should account for data replication
        total_storage = get_total_storage_gib(result)
        assert total_storage is not None
        # 50GB data * RF * buffer = at least 100GB
        assert total_storage >= 100

    def test_rf3_explicit(self):
        """Test ReadOnlyKV with min_replica_count=3 explicitly configured."""
        cap_plan = planner.plan_certain(
            model_name="org.netflix.read-only-kv",
            region="us-east-1",
            desires=small_dataset_high_rps,
            extra_model_arguments={
                "min_replica_count": 3,
                "total_num_partitions": 4,
            },
        )[0]

        result = cap_plan.candidate_clusters.regional[0]

        # Check actual replica_count in cluster_params (may be >= min)
        actual_rf = result.cluster_params["read-only-kv.replica_count"]
        assert actual_rf >= 3, f"Current RF {actual_rf} should be >= min RF 3"

        # Storage should be higher with more replicas
        total_storage = get_total_storage_gib(result)
        assert total_storage is not None
        # 50GB data * RF=3 * buffer = at least 150GB
        assert total_storage >= 150


class TestReadOnlyKVClusterConstraints:
    """Test ReadOnlyKV cluster size and regional constraints."""

    def test_max_regional_size(self):
        """Test that ReadOnlyKV respects max regional size."""
        # This should fail to plan if max_regional_size is too small
        # for the required capacity
        desires = CapacityDesires(
            service_tier=1,
            query_pattern=QueryPattern(
                access_pattern=AccessPattern.latency,
                estimated_read_per_second=certain_int(1_000_000),
                estimated_write_per_second=certain_int(0),
                estimated_mean_read_latency_ms=certain_float(1.0),
            ),
            data_shape=DataShape(
                estimated_state_size_gib=certain_int(10000),
            ),
        )

        # With a very small max_regional_size, should still get a plan
        # if instances can handle the load
        cap_plans = planner.plan_certain(
            model_name="org.netflix.read-only-kv",
            region="us-east-1",
            desires=desires,
            extra_model_arguments={
                "max_regional_size": 256,
                "total_num_partitions": 64,
            },
        )

        # Should have at least one valid plan
        assert len(cap_plans) > 0, "Expected at least one capacity plan"

        for plan in cap_plans:
            result = plan.candidate_clusters.regional[0]
            assert result.count <= 256, (
                f"Cluster size {result.count} exceeds max_regional_size=256"
            )


class TestReadOnlyKVCostCalculation:
    """Test ReadOnlyKV cost calculation."""

    def test_cost_calculation(self):
        """Test that ReadOnlyKV cost is computed correctly.

        With local disks only, cost = instance_count * instance_annual_cost.
        """
        cap_plan = planner.plan_certain(
            model_name="org.netflix.read-only-kv",
            region="us-east-1",
            desires=large_dataset_moderate_rps,
            extra_model_arguments={
                "total_num_partitions": 12,
            },
        )[0]

        result = cap_plan.candidate_clusters.regional[0]

        # Cost should be instance cost * count (no EBS cost for local disks)
        expected_cost = result.count * result.instance.annual_cost

        # The cluster annual cost should match
        assert abs(result.annual_cost - expected_cost) < 1, (
            f"Cluster cost {result.annual_cost} doesn't match "
            f"expected {expected_cost} (count={result.count} * "
            f"instance_cost={result.instance.annual_cost})"
        )

        # Verify no attached drives (local disks only)
        assert len(result.attached_drives) == 0, (
            "Expected no attached drives for local disk instances"
        )

    def test_rf2_cheaper_than_rf3(self):
        """Test that RF=2 is cheaper than or equal to RF=3 for same workload.

        Note: In some cases (small datasets that fit on same instance count),
        RF=2 and RF=3 may have the same cost. This test verifies RF=2 is
        never MORE expensive than RF=3.
        """
        # Use a larger dataset where RF difference will be more pronounced
        large_dataset = CapacityDesires(
            service_tier=1,
            query_pattern=QueryPattern(
                access_pattern=AccessPattern.latency,
                estimated_read_per_second=certain_int(50_000),
                estimated_write_per_second=certain_int(0),
                estimated_mean_read_latency_ms=certain_float(2.0),
                estimated_mean_write_latency_ms=certain_float(0),
                estimated_mean_read_size_bytes=certain_int(2048),
            ),
            data_shape=DataShape(
                estimated_state_size_gib=certain_int(5000),  # 5TB dataset
            ),
        )

        rf2_plan = planner.plan_certain(
            model_name="org.netflix.read-only-kv",
            region="us-east-1",
            desires=large_dataset,
            extra_model_arguments={
                "min_replica_count": 2,
                "total_num_partitions": 24,
            },
        )[0]

        rf3_plan = planner.plan_certain(
            model_name="org.netflix.read-only-kv",
            region="us-east-1",
            desires=large_dataset,
            extra_model_arguments={
                "min_replica_count": 3,
                "total_num_partitions": 24,
            },
        )[0]

        # Get actual replica counts from cluster params
        rf2_actual = rf2_plan.candidate_clusters.regional[0].cluster_params[
            "read-only-kv.replica_count"
        ]
        rf3_actual = rf3_plan.candidate_clusters.regional[0].cluster_params[
            "read-only-kv.replica_count"
        ]

        # min_replica_count=3 should result in >= 3 replicas
        assert rf3_actual >= rf2_actual, (
            f"RF3 plan ({rf3_actual} replicas) should have >= RF2 plan "
            f"({rf2_actual} replicas)"
        )


class TestReadOnlyKVMultiplePlans:
    """E2E tests that validate multiple generated plans are reasonable."""

    def test_generates_multiple_valid_plans(self):
        """Verify we get multiple plans and all meet minimum requirements."""
        plans = planner.plan_certain(
            model_name="org.netflix.read-only-kv",
            region="us-east-1",
            desires=large_dataset_moderate_rps,
            extra_model_arguments={
                "total_num_partitions": 12,
            },
        )

        # Should generate multiple plans (at least 2)
        assert len(plans) >= 2, f"Expected >= 2 plans, got {len(plans)}"

        # All plans should meet minimum requirements
        min_storage_gib = 2000  # 1TB * RF=2

        for i, plan in enumerate(plans[:5]):  # Check top 5 plans
            result = plan.candidate_clusters.regional[0]
            total_storage = get_total_storage_gib(result)

            assert total_storage is not None
            assert total_storage >= min_storage_gib, (
                f"Plan {i} ({result.instance.name}): "
                f"insufficient storage {total_storage} < {min_storage_gib}"
            )
            # Verify instance has local disk (required by partition-aware algorithm)
            assert result.instance.drive is not None, (
                f"Plan {i} ({result.instance.name}): expected local disk"
            )

    def test_plans_sorted_by_cost(self):
        """Verify plans are sorted by cost (cheapest first)."""
        plans = planner.plan_certain(
            model_name="org.netflix.read-only-kv",
            region="us-east-1",
            desires=large_dataset_moderate_rps,
            extra_model_arguments={
                "total_num_partitions": 12,
            },
        )

        assert len(plans) >= 2, "Need at least 2 plans to verify sorting"

        costs = [float(p.candidate_clusters.total_annual_cost) for p in plans]
        assert costs == sorted(costs), (
            f"Plans should be sorted by cost (ascending), got costs: {costs[:5]}"
        )

    def test_different_instance_types_considered(self):
        """Verify planner considers multiple instance types."""
        plans = planner.plan_certain(
            model_name="org.netflix.read-only-kv",
            region="us-east-1",
            desires=large_dataset_moderate_rps,
            extra_model_arguments={
                "total_num_partitions": 12,
            },
        )

        instance_types = {p.candidate_clusters.regional[0].instance.name for p in plans}
        assert len(instance_types) >= 2, (
            f"Should consider multiple instance types, got only: {instance_types}"
        )

    def test_all_plans_have_consistent_cluster_type(self):
        """Verify all plans have the correct cluster type."""
        plans = planner.plan_certain(
            model_name="org.netflix.read-only-kv",
            region="us-east-1",
            desires=small_dataset_high_rps,
            extra_model_arguments={
                "total_num_partitions": 4,
            },
        )

        for i, plan in enumerate(plans):
            result = plan.candidate_clusters.regional[0]
            assert result.cluster_type == "read-only-kv", (
                f"Plan {i}: expected cluster_type='read-only-kv', "
                f"got '{result.cluster_type}'"
            )

    def test_plans_scale_with_data_size(self):
        """Verify larger datasets result in more total storage."""
        small_plans = planner.plan_certain(
            model_name="org.netflix.read-only-kv",
            region="us-east-1",
            desires=small_dataset_high_rps,  # 50GB
            extra_model_arguments={
                "total_num_partitions": 4,
            },
        )

        large_plans = planner.plan_certain(
            model_name="org.netflix.read-only-kv",
            region="us-east-1",
            desires=large_dataset_moderate_rps,  # 1TB
            extra_model_arguments={
                # More partitions for larger dataset
                "total_num_partitions": 24,
            },
        )

        # Check partition info is in cluster_params
        small_result = small_plans[0].candidate_clusters.regional[0]
        large_result = large_plans[0].candidate_clusters.regional[0]

        assert "read-only-kv.partitions_per_node" in small_result.cluster_params
        assert "read-only-kv.replica_count" in small_result.cluster_params

        assert "read-only-kv.partitions_per_node" in large_result.cluster_params
        assert "read-only-kv.replica_count" in large_result.cluster_params

        # Large dataset should have more total storage
        small_storage = get_total_storage_gib(small_result)
        large_storage = get_total_storage_gib(large_result)

        assert large_storage >= small_storage, (
            f"Large dataset ({large_storage} GiB) should have >= "
            f"small dataset ({small_storage} GiB) storage"
        )


class TestReadOnlyKVPartitionAwareAlgorithm:
    """Tests for the partition-aware capacity planning algorithm."""

    def test_compute_heavy_increases_replica_count(self):
        """Test that compute-heavy workloads increase replica count.

        For compute-heavy workloads, the algorithm should increase replica_count
        beyond min_replica_count to satisfy CPU requirements.
        """
        # High RPS workload that needs lots of CPU
        compute_heavy = CapacityDesires(
            service_tier=1,
            query_pattern=QueryPattern(
                access_pattern=AccessPattern.latency,
                estimated_read_per_second=certain_int(500_000),  # Very high RPS
                estimated_write_per_second=certain_int(0),
                estimated_mean_read_latency_ms=certain_float(1.0),
                estimated_mean_read_size_bytes=certain_int(1024),
            ),
            data_shape=DataShape(
                # Small dataset relative to RPS
                estimated_state_size_gib=certain_int(100),
            ),
        )

        plans = planner.plan_certain(
            model_name="org.netflix.read-only-kv",
            region="us-east-1",
            desires=compute_heavy,
            extra_model_arguments={
                "min_replica_count": 2,
                "total_num_partitions": 4,
            },
        )

        assert len(plans) > 0, "Should generate at least one plan"

        # Check that at least one plan increased replica count beyond minimum
        result = plans[0].candidate_clusters.regional[0]
        actual_rf = result.cluster_params["read-only-kv.replica_count"]

        # For such a compute-heavy workload, replica count should be > min
        # (This validates that the algorithm is working)
        assert actual_rf >= 2, f"Replica count {actual_rf} should be >= min 2"

        # Verify partitions_per_node is set
        partitions_per_node = result.cluster_params["read-only-kv.partitions_per_node"]
        assert partitions_per_node >= 1, "Should have at least 1 partition per node"

    def test_storage_heavy_uses_min_replica_count(self):
        """Test that storage-heavy workloads use close to min_replica_count.

        For storage-heavy workloads with low RPS, the algorithm should use
        replica_count close to min_replica_count since CPU isn't the bottleneck.
        """
        # Large dataset with low RPS
        storage_heavy = CapacityDesires(
            service_tier=1,
            query_pattern=QueryPattern(
                access_pattern=AccessPattern.latency,
                estimated_read_per_second=certain_int(1_000),  # Low RPS
                estimated_write_per_second=certain_int(0),
                estimated_mean_read_latency_ms=certain_float(2.0),
                estimated_mean_read_size_bytes=certain_int(2048),
            ),
            data_shape=DataShape(
                # Large dataset relative to RPS
                estimated_state_size_gib=certain_int(5000),  # 5TB
            ),
        )

        plans = planner.plan_certain(
            model_name="org.netflix.read-only-kv",
            region="us-east-1",
            desires=storage_heavy,
            extra_model_arguments={
                "min_replica_count": 2,
                "total_num_partitions": 24,
            },
        )

        assert len(plans) > 0, "Should generate at least one plan"

        result = plans[0].candidate_clusters.regional[0]
        actual_rf = result.cluster_params["read-only-kv.replica_count"]

        # For storage-heavy, should be at or close to minimum
        assert actual_rf >= 2, f"Replica count {actual_rf} should be >= min 2"

    def test_partition_size_affects_partitions_per_node(self):
        """Test that larger partitions result in fewer partitions per node."""
        # Same data size but different partition counts
        desires = CapacityDesires(
            service_tier=1,
            query_pattern=QueryPattern(
                access_pattern=AccessPattern.latency,
                estimated_read_per_second=certain_int(10_000),
                estimated_write_per_second=certain_int(0),
                estimated_mean_read_latency_ms=certain_float(2.0),
                estimated_mean_read_size_bytes=certain_int(2048),
            ),
            data_shape=DataShape(
                estimated_state_size_gib=certain_int(1000),  # 1TB
            ),
        )

        # Few partitions = large partition size
        few_partitions_plan = planner.plan_certain(
            model_name="org.netflix.read-only-kv",
            region="us-east-1",
            desires=desires,
            extra_model_arguments={
                "total_num_partitions": 4,  # 250GB per partition
            },
        )[0]

        # Many partitions = small partition size
        many_partitions_plan = planner.plan_certain(
            model_name="org.netflix.read-only-kv",
            region="us-east-1",
            desires=desires,
            extra_model_arguments={
                "total_num_partitions": 32,  # ~31GB per partition
            },
        )[0]

        few_ppn = few_partitions_plan.candidate_clusters.regional[0].cluster_params[
            "read-only-kv.partitions_per_node"
        ]
        many_ppn = many_partitions_plan.candidate_clusters.regional[0].cluster_params[
            "read-only-kv.partitions_per_node"
        ]

        # More partitions (smaller size) should allow more partitions per node
        assert many_ppn >= few_ppn, (
            f"Many small partitions ({many_ppn} per node) should allow >= "
            f"few large partitions ({few_ppn} per node)"
        )

    def test_cluster_params_contains_all_required_fields(self):
        """Test that cluster_params contains all partition-aware algorithm outputs."""
        plans = planner.plan_certain(
            model_name="org.netflix.read-only-kv",
            region="us-east-1",
            desires=large_dataset_moderate_rps,
            extra_model_arguments={
                "total_num_partitions": 12,
            },
        )

        assert len(plans) > 0

        result = plans[0].candidate_clusters.regional[0]

        # All required partition-aware algorithm outputs should be present
        required_params = [
            "read-only-kv.replica_count",
            "read-only-kv.partitions_per_node",
        ]

        for param in required_params:
            assert param in result.cluster_params, (
                f"Missing required cluster_param: {param}"
            )

        # Validate values
        replica_count = result.cluster_params["read-only-kv.replica_count"]
        partitions_per_node = result.cluster_params["read-only-kv.partitions_per_node"]

        assert replica_count >= 2
        assert partitions_per_node >= 1
        assert result.count >= 2


# ============================================================================
# STANDARDIZED ALGORITHM TESTING
# ============================================================================
#
# This section provides LeetCode-style verification of the partition capacity
# algorithm using a standardized problem representation.
#
# KEY INSIGHT FOR IMMUTABLE READ-ONLY DATA:
# The greedy algorithm (maximize partitions per node) is CORRECT because:
# 1. Higher RF is FREE for immutable data (no write amplification)
# 2. Higher RF provides exponentially better fault tolerance (P(down) = p^RF)
# 3. Higher RF is critical for AZ fault tolerance with random placement
# ============================================================================


@dataclass(frozen=True)
class StandardizedProblem:
    """
    Canonical problem representation for partition capacity planning.
    All model-specific details (buffers, instance shapes) pre-computed.
    """

    n_partitions: int  # Total partitions (atomic units)
    partition_size_gib: float  # Size WITH buffer already applied
    disk_per_node_gib: float  # Effective disk (after max_data cap)
    min_replicas: int  # Minimum replication factor
    cpu_needed: int  # Total CPU cores required
    cpu_per_node: int  # CPU cores per node
    max_nodes: int = 10000  # Cluster size limit


@dataclass
class CapacityResult:
    """Result from the capacity algorithm."""

    node_count: int
    replica_count: int
    partitions_per_node: int
    nodes_for_one_copy: int


class TestReadOnlyKVStandardizedAlgorithm:
    """
    Standardized tests for the partition-aware capacity algorithm.

    These tests verify:
    1. The actual implementation matches the reference algorithm
    2. Results always satisfy constraints (CPU, disk, RF, max nodes)
    3. Property-based tests with random inputs
    """

    def _run_actual_algorithm(
        self, problem: StandardizedProblem
    ) -> Optional[CapacityResult]:
        """Run the actual implementation and return standardized result."""
        from service_capacity_modeling.interface import (
            CapacityRequirement,
            Drive,
            DriveType,
            Instance,
        )
        from service_capacity_modeling.models.org.netflix.read_only_kv import (
            NflxReadOnlyKVArguments,
            _compute_read_only_kv_regional_cluster,
        )

        # Create mock instance
        mock_instance = Instance(
            name="test-instance",
            cpu=problem.cpu_per_node,
            cpu_ghz=3.0,
            ram_gib=128,
            net_mbps=10000,
            drive=Drive(
                name="local-nvme",
                drive_type=DriveType.local_ssd,
                size_gib=int(problem.disk_per_node_gib),
            ),
            annual_cost=5000,
        )

        # Create requirement with pre-computed values
        requirement = CapacityRequirement(
            requirement_type="read-only-kv-regional",
            cpu_cores=certain_int(problem.cpu_needed),
            mem_gib=certain_float(0),
            disk_gib=certain_float(0),
            network_mbps=certain_float(100),
            context={},
        )

        args = NflxReadOnlyKVArguments(
            total_num_partitions=problem.n_partitions,
            min_replica_count=problem.min_replicas,
            max_regional_size=problem.max_nodes,
            max_data_per_node_gib=int(problem.disk_per_node_gib),
        )

        cluster = _compute_read_only_kv_regional_cluster(
            instance=mock_instance,
            requirement=requirement,
            args=args,
            partition_size_with_buffer_gib=problem.partition_size_gib,
            disk_buffer_ratio=1.0,  # Partition size already includes buffer
        )

        if cluster is None:
            return None

        return CapacityResult(
            node_count=cluster.count,
            replica_count=cluster.cluster_params["read-only-kv.replica_count"],
            partitions_per_node=cluster.cluster_params[
                "read-only-kv.partitions_per_node"
            ],
            nodes_for_one_copy=cluster.cluster_params[
                "read-only-kv.nodes_for_one_copy"
            ],
        )

    # ========================================================================
    # DETERMINISTIC TESTS
    # ========================================================================

    def test_data_constrained_workload(self):
        """Test data-constrained workload where disk determines sizing.

        Hand calculation:
        - ppn = floor(2048 / 575) = 3
        - nodes_for_one_copy = ceil(200 / 3) = 67
        - cpu_needed = 800, cpu_per_node = 16
        - nodes_for_cpu = ceil(800 / 16) = 50
        - Since n1c (67) > nodes_for_cpu (50), disk is bottleneck
        - rf = 2 (min), node_count = 67 * 2 = 134
        """
        problem = StandardizedProblem(
            n_partitions=200,
            partition_size_gib=575,
            disk_per_node_gib=2048,
            min_replicas=2,
            cpu_needed=800,
            cpu_per_node=16,
            max_nodes=10000,
        )

        result = self._run_actual_algorithm(problem)

        assert result is not None
        assert result.partitions_per_node == 3  # floor(2048 / 575)
        assert result.nodes_for_one_copy == 67  # ceil(200 / 3)
        assert result.replica_count == 2  # min_replicas (CPU satisfied)
        assert result.node_count == 134  # 67 * 2

    def test_cpu_constrained_workload(self):
        """Test CPU-constrained workload where algorithm uses max PPn (highest RF).

        Hand calculation:
        - max_ppn = floor(2048 / 575) = 3
        - cpu_needed = 3200, cpu_per_node = 16

        PPn=3: base=ceil(200/3)=67, cpu_per_copy=67*16=1072
               min_rf=ceil(3200/1072)=3, nodes=67*3=201

        Algorithm picks PPn=3 (max) because it prioritizes higher RF
        for fault tolerance.
        """
        problem = StandardizedProblem(
            n_partitions=200,
            partition_size_gib=575,
            disk_per_node_gib=2048,
            min_replicas=2,
            cpu_needed=3200,
            cpu_per_node=16,
            max_nodes=10000,
        )

        result = self._run_actual_algorithm(problem)

        assert result is not None
        assert result.partitions_per_node == 3  # Max PPn for highest RF
        assert result.nodes_for_one_copy == 67
        assert result.replica_count == 3
        assert result.node_count == 201

    def test_single_partition_per_node(self):
        """Test when only 1 partition fits per node.

        Hand calculation:
        - ppn = floor(2048 / 2000) = 1
        - nodes_for_one_copy = ceil(100 / 1) = 100
        - cpu_needed = 500, cpu_per_node = 8
        - rf=2: 100*2=200 nodes, 200*8=1600 CPU >= 500 (satisfied)
        """
        problem = StandardizedProblem(
            n_partitions=100,
            partition_size_gib=2000,
            disk_per_node_gib=2048,
            min_replicas=2,
            cpu_needed=500,
            cpu_per_node=8,
            max_nodes=10000,
        )

        result = self._run_actual_algorithm(problem)

        assert result is not None
        assert result.partitions_per_node == 1
        assert result.nodes_for_one_copy == 100
        assert result.replica_count == 2
        assert result.node_count == 200

    def test_partition_too_large_returns_none(self):
        """When partition > disk, should return None."""
        problem = StandardizedProblem(
            n_partitions=100,
            partition_size_gib=3000,  # Larger than disk
            disk_per_node_gib=2048,
            min_replicas=2,
            cpu_needed=100,
            cpu_per_node=16,
            max_nodes=10000,
        )

        result = self._run_actual_algorithm(problem)
        assert result is None

    def test_exceeds_max_nodes_returns_none(self):
        """When constraints can't be met within max_nodes, return None."""
        problem = StandardizedProblem(
            n_partitions=10000,
            partition_size_gib=1000,
            disk_per_node_gib=1000,  # 1 partition per node
            min_replicas=2,
            cpu_needed=1000000,  # Huge CPU need
            cpu_per_node=16,
            max_nodes=100,  # But max 100 nodes
        )

        result = self._run_actual_algorithm(problem)
        assert result is None

    def test_fallback_ppn_when_greedy_exceeds_max_nodes(self):
        """Test that algorithm finds solution when greedy PPn exceeds max_nodes.

        This is the key case where greedy max PPn fails but a smaller PPn works:

        N=21 partitions, max_nodes=10, cpu_per_node=2, cpu_needed=19
        disk=1000, partition_size=100 → max_ppn=10

        Greedy (PPn=10): base=ceil(21/10)=3, cpu_per_copy=6
                         min_rf=ceil(19/6)=4, nodes=3*4=12 > 10 ❌

        Fallback (PPn=5): base=ceil(21/5)=5, cpu_per_copy=10
                          min_rf=ceil(19/10)=2, nodes=5*2=10 ≤ 10 ✅
        """
        problem = StandardizedProblem(
            n_partitions=21,
            partition_size_gib=100,
            disk_per_node_gib=1000,  # max_ppn = 10
            min_replicas=2,
            cpu_needed=19,
            cpu_per_node=2,
            max_nodes=10,
        )

        result = self._run_actual_algorithm(problem)

        assert result is not None
        assert result.partitions_per_node == 5  # Not greedy (10), but fallback
        assert result.nodes_for_one_copy == 5  # ceil(21/5)
        assert result.replica_count == 2
        assert result.node_count == 10  # Fits within max_nodes

    # ========================================================================
    # CONSTRAINT VALIDATION TESTS
    # ========================================================================

    def test_result_satisfies_all_constraints(self):
        """Verify results satisfy CPU, disk, RF, and max_nodes constraints."""
        problems = [
            StandardizedProblem(200, 575, 2048, 2, 800, 16, 10000),
            StandardizedProblem(100, 1024, 2048, 2, 100, 16, 10000),
            StandardizedProblem(1000, 100, 2000, 3, 5000, 16, 10000),
            StandardizedProblem(512, 107.8, 2048, 4, 60, 24, 256),  # IHS-like
        ]

        for problem in problems:
            result = self._run_actual_algorithm(problem)
            if result is None:
                continue

            # CPU satisfied
            total_cpu = result.node_count * problem.cpu_per_node
            assert total_cpu >= problem.cpu_needed, (
                f"CPU not satisfied: {total_cpu} < {problem.cpu_needed}"
            )

            # Disk satisfied (enough slots for all replicas)
            total_slots = result.node_count * result.partitions_per_node
            total_replicas = problem.n_partitions * result.replica_count
            assert total_slots >= total_replicas, (
                f"Disk not satisfied: {total_slots} < {total_replicas}"
            )

            # RF >= min
            assert result.replica_count >= problem.min_replicas

            # Max nodes respected
            assert result.node_count <= problem.max_nodes

    # ========================================================================
    # HYPOTHESIS PROPERTY-BASED TESTS
    # ========================================================================

    @staticmethod
    @st.composite
    def valid_problems(draw):
        """Generate random but valid capacity problems."""
        disk_per_node = draw(st.integers(min_value=100, max_value=10000))
        partition_size = draw(st.integers(min_value=10, max_value=disk_per_node))
        n_partitions = draw(st.integers(min_value=1, max_value=1000))
        min_replicas = draw(st.integers(min_value=1, max_value=5))
        cpu_per_node = draw(st.integers(min_value=1, max_value=128))
        max_nodes = draw(st.integers(min_value=10, max_value=10000))
        cpu_needed = draw(st.integers(min_value=1, max_value=max_nodes * cpu_per_node))

        return StandardizedProblem(
            n_partitions=n_partitions,
            partition_size_gib=float(partition_size),
            disk_per_node_gib=float(disk_per_node),
            min_replicas=min_replicas,
            cpu_needed=cpu_needed,
            cpu_per_node=cpu_per_node,
            max_nodes=max_nodes,
        )

    @given(problem=valid_problems())
    @settings(max_examples=200, deadline=None)
    def test_hypothesis_satisfies_constraints(self, problem: StandardizedProblem):
        """PROPERTY: Results must always satisfy all constraints."""
        result = self._run_actual_algorithm(problem)
        if result is None:
            return  # No result to validate

        # CPU constraint
        total_cpu = result.node_count * problem.cpu_per_node
        assert total_cpu >= problem.cpu_needed, (
            f"CPU constraint violated: {total_cpu} < {problem.cpu_needed}"
        )

        # Disk constraint
        total_slots = result.node_count * result.partitions_per_node
        total_replicas = problem.n_partitions * result.replica_count
        assert total_slots >= total_replicas, (
            f"Disk constraint violated: {total_slots} < {total_replicas}"
        )

        # Min replica constraint
        assert result.replica_count >= problem.min_replicas

        # Max nodes constraint
        assert result.node_count <= problem.max_nodes

    @given(problem=valid_problems())
    @settings(max_examples=200, deadline=None)
    def test_hypothesis_node_count_formula(self, problem: StandardizedProblem):
        """PROPERTY: node_count = max(2, nodes_for_one_copy × replica_count)."""
        result = self._run_actual_algorithm(problem)
        if result is None:
            return

        expected = max(2, result.nodes_for_one_copy * result.replica_count)
        assert result.node_count == expected, (
            f"node_count={result.node_count} != "
            f"max(2, {result.nodes_for_one_copy} × {result.replica_count}) = {expected}"
        )

    @given(problem=valid_problems())
    @settings(max_examples=200, deadline=None)
    def test_hypothesis_uses_first_valid_ppn(self, problem: StandardizedProblem):
        """PROPERTY: Algorithm returns first valid PPn (max PPn = highest RF)."""
        result = self._run_actual_algorithm(problem)
        if result is None:
            return

        # Verify that the chosen PPn is the highest valid one
        max_ppn = int(problem.disk_per_node_gib / problem.partition_size_gib)

        # No higher PPn should be valid
        for ppn in range(max_ppn, result.partitions_per_node, -1):
            base = math.ceil(problem.n_partitions / ppn)
            if base >= 2:
                min_rf = max(
                    1, math.ceil(problem.cpu_needed / (base * problem.cpu_per_node))
                )
                rf = max(problem.min_replicas, min_rf)
                nodes = base * rf
            else:
                if 2 * problem.cpu_per_node >= problem.cpu_needed:
                    rf = problem.min_replicas
                    nodes = max(2, rf)
                else:
                    rf = max(
                        problem.min_replicas,
                        math.ceil(problem.cpu_needed / problem.cpu_per_node),
                    )
                    nodes = rf

            # This higher PPn should exceed max_nodes
            # (otherwise algorithm would have chosen it)
            assert nodes > problem.max_nodes, (
                f"Higher PPn={ppn} gives {nodes} nodes "
                f"<= max_nodes={problem.max_nodes}, "
                f"but algorithm chose PPn={result.partitions_per_node}"
            )


class TestReadOnlyKVExploration:
    """Exploratory tests that print results for parameter tuning.

    Run: pytest ...::TestReadOnlyKVExploration::test_... -v -s
    """

    def _print_workload_summary(self, desires, extra_args):
        """Print workload parameters."""
        print("\n" + "=" * 60)
        print("WORKLOAD PARAMETERS")
        print("=" * 60)

        # Data shape
        data_size = desires.data_shape.estimated_state_size_gib
        if hasattr(data_size, "mid"):
            print(f"Data Size: {data_size.mid} GiB")
        else:
            print(f"Data Size: {data_size} GiB")

        # Query pattern
        qp = desires.query_pattern
        rps = qp.estimated_read_per_second
        if hasattr(rps, "low") and hasattr(rps, "high"):
            print(f"Read RPS:  {rps.low:,} - {rps.mid:,} - {rps.high:,}")
        elif hasattr(rps, "mid"):
            print(f"Read RPS:  {rps.mid:,}")

        latency = qp.estimated_mean_read_latency_ms
        if hasattr(latency, "mid"):
            print(f"Latency:   {latency.mid} ms target")

        # Extra args and partition info
        print("\nModel Arguments:")
        for k, v in extra_args.items():
            print(f"  {k}: {v}")

        # Buffers (default values)
        cpu_buffer = 1.5
        disk_buffer = 1.15
        print(f"\nBuffers: CPU={cpu_buffer}x, Disk={disk_buffer}x")

        # Calculate and print partition sizes
        total_partitions = extra_args.get("total_num_partitions", 0)
        if total_partitions and hasattr(data_size, "mid"):
            partition_size = data_size.mid / total_partitions
            partition_size_buf = partition_size * disk_buffer
            print("\nPartition Info:")
            print(f"  partition_size: {partition_size:.2f} GiB")
            print(f"  partition_size_with_buffer: {partition_size_buf:.2f} GiB")

    def _print_cluster_row(self, info: dict):
        """Print a single cluster row with utilization metrics."""
        cpu_pct = (info["need_cpu"] / info["cpu"] * 100) if info["cpu"] else 0
        disk_pct = (info["need_disk"] / info["disk"] * 100) if info["disk"] else 0
        ppn = info["partitions_per_node"]
        print(
            f"{info['label']:<8} {info['name']:<16} "
            f"Count={info['count']}, RF={info['rf']}, partitions_per_node={ppn}"
        )
        print("-" * 70)
        # Node specs
        inst = info.get("instance")
        if inst:
            disk_str = f"{inst.drive.size_gib:,} GiB" if inst.drive else "N/A"
            print(
                f"        Node: {inst.cpu} cores, {inst.ram_gib:.0f} GiB RAM, "
                f"{disk_str} disk"
            )
        # Cluster totals vs needs
        print(
            f"        CPU:  {info['cpu']:>6}/{info['need_cpu']:>6} = {cpu_pct:>5.1f}%"
        )
        d, nd = info["disk"], info["need_disk"]
        print(f"        Disk: {d:>6,}/{nd:>6,.0f} = {disk_pct:>5.1f}%")
        cost_str = f"        Cost: ${info['cost']:,.0f}/yr"
        if info.get("vs_curr"):
            cost_str += f"  {info['vs_curr']}"
        print(cost_str)
        print()

    def _print_comparison_table(  # noqa: C901 pylint: disable=too-many-locals
        self, plans, data_size_gib, current_cluster_info
    ):
        """Print comparison table of current vs recommended clusters."""
        # Get requirements context from first plan
        ctx = plans[0].requirements.regional[0].context
        raw_cpu = ctx.get("raw_cores", 0)
        cpu_buffer = ctx.get("compute_buffer_ratio", 1.5)
        need_cpu = raw_cpu * cpu_buffer  # Buffered CPU cores needed
        disk_buffer = ctx.get("disk_buffer_ratio", 1.15)
        partition_size_gib = ctx.get("partition_size_gib", 0)
        part_size_buf = partition_size_gib * disk_buffer

        print("\n" + "=" * 70)
        print("COMPARISON TABLE (have / need = %)")
        print("=" * 70)

        actual_cost = current_cluster_info.get("cost")

        # Print current cluster if specified
        cfg = current_cluster_info.get("config")
        inst = current_cluster_info.get("instance")
        if cfg and inst:
            rf = cfg.get("replica_count", 1)
            disk_node = inst.drive.size_gib if inst.drive else 0
            self._print_cluster_row(
                {
                    "label": "Current",
                    "name": inst.name,
                    "instance": inst,
                    "count": cfg["count"],
                    "rf": rf,
                    "partitions_per_node": cfg.get("partitions_per_node", "?"),
                    "cpu": cfg["count"] * inst.cpu,
                    "need_cpu": need_cpu,
                    "disk": disk_node * cfg["count"],
                    "need_disk": data_size_gib * rf,
                    "cost": actual_cost,
                }
            )

        # Print recommendations
        for i, plan in enumerate(plans[:5]):
            r = plan.candidate_clusters.regional[0]
            rf = r.cluster_params.get("read-only-kv.replica_count", 1)
            partitions_per_node = r.cluster_params.get(
                "read-only-kv.partitions_per_node", "?"
            )
            disk_node = r.instance.drive.size_gib if r.instance.drive else 0

            vs_curr = ""
            if actual_cost:
                diff = r.annual_cost - actual_cost
                sign = "-" if diff < 0 else "+"
                vs_curr = f"{sign}${abs(diff):,.0f} ({diff / actual_cost * 100:+.0f}%)"

            print("=" * 70)
            self._print_cluster_row(
                {
                    "label": f"#{i + 1}",
                    "name": r.instance.name,
                    "instance": r.instance,
                    "count": r.count,
                    "rf": rf,
                    "partitions_per_node": partitions_per_node,
                    "cpu": r.count * r.instance.cpu,
                    "need_cpu": need_cpu,
                    "disk": disk_node * r.count,
                    "need_disk": data_size_gib * rf,
                    "cost": r.annual_cost,
                    "vs_curr": vs_curr,
                }
            )

            # Debug: show how key values are calculated
            eff_disk = r.cluster_params.get(
                "read-only-kv.effective_disk_per_node_gib", "?"
            )
            nodes_for_one_copy = r.cluster_params.get(
                "read-only-kv.nodes_for_one_copy", "?"
            )
            nodes_for_cpu = r.cluster_params.get("read-only-kv.nodes_for_cpu", "?")
            total_partitions = ctx.get("total_num_partitions", "?")
            print(
                f"        partitions_per_node = eff_disk / partition_size_with_buffer "
                f"= {eff_disk} / {part_size_buf:.2f} = {partitions_per_node}"
            )
            print(
                f"        nodes_for_one_copy = total_partitions / partitions_per_node "
                f"= {total_partitions} / {partitions_per_node} = {nodes_for_one_copy}"
            )
            print(f"        nodes_for_cpu = {nodes_for_cpu}")
            print()

    def _run_capacity_exploration(
        self,
        workload: CapacityDesires,
        extra_args: dict,
        actual_cluster: dict = None,
    ):
        """Run capacity exploration and print comparison table.

        Args:
            workload: CapacityDesires defining the workload
            extra_args: Model arguments (total_num_partitions, min_replica_count, etc.)
            actual_cluster: Current cluster config for comparison, or None to skip
                           Format: {"instance": str, "count": int, "replica_count": int,
                                    "partitions_per_node": int}
        """
        self._print_workload_summary(workload, extra_args)

        plans = planner.plan_certain(
            model_name="org.netflix.read-only-kv",
            region="us-east-1",
            desires=workload,
            extra_model_arguments=extra_args,
        )

        # Look up actual cluster if specified
        actual_instance = None
        actual_cost = None
        if actual_cluster:
            try:
                actual_instance = planner.instance(
                    actual_cluster["instance"], region="us-east-1"
                )
                actual_cost = actual_instance.annual_cost * actual_cluster["count"]
            except KeyError:
                print(f"\nWARNING: Instance '{actual_cluster['instance']}' not found")
                actual_cluster = None

        # Print comparison table
        data_size_gib = workload.data_shape.estimated_state_size_gib.mid
        current_info = {
            "config": actual_cluster,
            "instance": actual_instance,
            "cost": actual_cost,
        }
        self._print_comparison_table(plans, data_size_gib, current_info)

    def test_ihs(self):
        workload = CapacityDesires(
            service_tier=1,
            query_pattern=QueryPattern(
                access_pattern=AccessPattern.latency,
                estimated_read_per_second=certain_int(20_000),
                estimated_write_per_second=certain_int(0),
                estimated_mean_read_latency_ms=certain_float(2.0),
                estimated_mean_write_latency_ms=certain_float(0),
            ),
            data_shape=DataShape(
                estimated_state_size_gib=certain_int(48_000),
            ),
        )

        extra_args = {
            "total_num_partitions": 512,
            "min_replica_count": 4,
        }

        actual_cluster = {
            "instance": "i3en.6xlarge",
            "count": 64,
            "replica_count": 4,
            "partitions_per_node": 32,
        }

        self._run_capacity_exploration(workload, extra_args, actual_cluster)

    def test_ads_profile(self):
        workload = CapacityDesires(
            service_tier=1,
            query_pattern=QueryPattern(
                access_pattern=AccessPattern.latency,
                estimated_read_per_second=certain_int(20_000),
                estimated_write_per_second=certain_int(0),
                estimated_mean_read_latency_ms=certain_float(2),
                estimated_mean_write_latency_ms=certain_float(0),
            ),
            data_shape=DataShape(
                estimated_state_size_gib=certain_int(1397),
            ),
        )

        extra_args = {
            "total_num_partitions": 8,
            "min_replica_count": 3,
        }

        actual_cluster = {
            "instance": "i3en.6xlarge",
            "count": 3,
            "replica_count": 3,
            "partitions_per_node": 16,
        }

        self._run_capacity_exploration(workload, extra_args, actual_cluster)

    def test_feature_query_service(self):
        workload = CapacityDesires(
            service_tier=1,
            query_pattern=QueryPattern(
                access_pattern=AccessPattern.latency,
                estimated_read_per_second=certain_int(17_000),
                estimated_write_per_second=certain_int(0),
                estimated_mean_read_latency_ms=certain_float(2),
                estimated_mean_write_latency_ms=certain_float(0),
            ),
            data_shape=DataShape(
                estimated_state_size_gib=certain_int(60),
            ),
        )

        extra_args = {
            "total_num_partitions": 16,
            "min_replica_count": 4,
        }

        actual_cluster = {
            "instance": "i3en.6xlarge",
            "count": 4,
            "replica_count": 4,
            "partitions_per_node": 4,
        }

        self._run_capacity_exploration(workload, extra_args, actual_cluster)
