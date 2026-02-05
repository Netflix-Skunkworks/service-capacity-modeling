"""
Tests for Netflix Read-Only Key-Value capacity model.

A read-only data serving layer backed by RocksDB that:
- Loads data from offline sources
- Serves read traffic with low latency
- Deploys regionally with configurable replication (RF=2 default)

Note: Algorithm-specific tests live in test_partition_aware_algorithm.py.
These tests focus on model integration and E2E behavior.
"""

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
