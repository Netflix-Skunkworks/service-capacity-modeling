"""
Tests for Netflix Read-Only Key-Value capacity model.

A read-only data serving layer backed by RocksDB that:
- Loads data from offline sources
- Serves read traffic with low latency
- Deploys regionally with configurable replication (RF=2 default)
"""

from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.interface import AccessPattern
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import certain_float
from service_capacity_modeling.interface import certain_int
from service_capacity_modeling.interface import DataShape
from service_capacity_modeling.interface import QueryPattern
from tests.util import get_total_storage_gib
from tests.util import has_attached_storage
from tests.util import has_local_storage


# Property test configuration for Read-Only KV model.
# See tests/netflix/PROPERTY_TESTING.md for configuration options and examples.
PROPERTY_TEST_CONFIG = {
    "org.netflix.read-only-kv": {
        # Exclude from universal property tests because:
        # 1. Model requires estimated_mean_read_size_bytes for network bandwidth
        # 2. merge_with uses write_size_bytes (0 for read-only) causing div by 0
        "exclude_from_universal_tests": True,
        "extra_model_arguments": {"require_local_disks": False},
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
        estimated_state_item_count=certain_int(50_000_000),
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
        estimated_state_item_count=certain_int(1_000_000_000),
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
        estimated_state_item_count=certain_int(500_000_000),
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
            extra_model_arguments={},
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
            extra_model_arguments={},
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
        # At least 10% of data size for block cache
        assert total_memory >= 200, (
            f"Expected >= 200 GiB memory for block cache, got {total_memory}"
        )

    def test_throughput_workload(self):
        """Test ReadOnlyKV with throughput-oriented workload (scans)."""
        cap_plan = planner.plan_certain(
            model_name="org.netflix.read-only-kv",
            region="us-east-1",
            desires=throughput_workload,
            extra_model_arguments={},
        )[0]

        result = cap_plan.candidate_clusters.regional[0]

        # Throughput workloads need sufficient network bandwidth
        total_network = result.count * result.instance.net_mbps
        # 10k RPS * 8KB = 80 MB/s = 640 Mbps
        assert total_network >= 640, (
            f"Expected >= 640 Mbps network for throughput workload, got {total_network}"
        )


class TestReadOnlyKVStorageTypes:
    """Test ReadOnlyKV with different storage configurations."""

    def test_local_disks_required(self):
        """Test ReadOnlyKV with local disks required (default behavior)."""
        cap_plan = planner.plan_certain(
            model_name="org.netflix.read-only-kv",
            region="us-east-1",
            desires=large_dataset_moderate_rps,
            extra_model_arguments={
                "require_local_disks": True,
                "require_attached_disks": False,
            },
        )[0]

        result = cap_plan.candidate_clusters.regional[0]

        # Should use local storage instances (i-series, m5d, m6id, etc.)
        assert has_local_storage(result), (
            "Expected local storage with require_local_disks=True"
        )
        # Local disk instances include i-series, m5d, m6id, r5d, etc.
        # Just verify it has local storage via the instance.drive check
        assert result.instance.drive is not None, (
            f"Expected instance with local drive, got {result.instance.name}"
        )

    def test_attached_disks_ebs(self):
        """Test ReadOnlyKV with EBS attached disks."""
        cap_plan = planner.plan_certain(
            model_name="org.netflix.read-only-kv",
            region="us-east-1",
            desires=large_dataset_moderate_rps,
            extra_model_arguments={
                "require_local_disks": False,
                "require_attached_disks": True,
            },
        )[0]

        result = cap_plan.candidate_clusters.regional[0]

        # Should use attached storage (EBS)
        assert has_attached_storage(result), (
            "Expected attached storage with require_attached_disks=True"
        )
        assert result.attached_drives[0].name in ("gp2", "gp3"), (
            f"Expected gp2/gp3 drive, got {result.attached_drives[0].name}"
        )


class TestReadOnlyKVReplication:
    """Test ReadOnlyKV replication factor configurations."""

    def test_rf2_default(self):
        """Test ReadOnlyKV with RF=2 (default)."""
        cap_plan = planner.plan_certain(
            model_name="org.netflix.read-only-kv",
            region="us-east-1",
            desires=small_dataset_high_rps,
            extra_model_arguments={"replica_count": 2},
        )[0]

        result = cap_plan.candidate_clusters.regional[0]
        requirement = cap_plan.requirements.regional[0]

        # Check RF is recorded in context
        assert requirement.context["replica_count"] == 2

        # Storage should account for RF=2
        total_storage = get_total_storage_gib(result)
        assert total_storage is not None
        # 50GB data * RF=2 * buffer = at least 100GB
        assert total_storage >= 100

    def test_rf3_explicit(self):
        """Test ReadOnlyKV with RF=3 explicitly configured."""
        cap_plan = planner.plan_certain(
            model_name="org.netflix.read-only-kv",
            region="us-east-1",
            desires=small_dataset_high_rps,
            extra_model_arguments={"replica_count": 3},
        )[0]

        result = cap_plan.candidate_clusters.regional[0]
        requirement = cap_plan.requirements.regional[0]

        # Check RF is recorded in context
        assert requirement.context["replica_count"] == 3

        # Storage should be higher with RF=3
        total_storage = get_total_storage_gib(result)
        assert total_storage is not None
        # 50GB data * RF=3 * buffer = at least 150GB
        assert total_storage >= 150


class TestReadOnlyKVMemoryConfiguration:
    """Test ReadOnlyKV RocksDB memory configurations."""

    def test_high_block_cache_percent(self):
        """Test ReadOnlyKV with higher block cache percentage."""
        cap_plan = planner.plan_certain(
            model_name="org.netflix.read-only-kv",
            region="us-east-1",
            desires=large_dataset_moderate_rps,
            extra_model_arguments={
                "rocksdb_block_cache_percent": 0.5,  # 50% cache
            },
        )[0]

        result = cap_plan.candidate_clusters.regional[0]
        requirement = cap_plan.requirements.regional[0]

        # Check block cache percent is recorded
        assert requirement.context["block_cache_percent"] == 0.5

        # Higher cache should result in more memory
        total_memory = result.count * result.instance.ram_gib
        # 1TB data * 50% = 500GB minimum block cache
        assert total_memory >= 500, (
            f"Expected >= 500 GiB memory for 50% block cache, got {total_memory}"
        )

    def test_low_block_cache_percent(self):
        """Test ReadOnlyKV with lower block cache percentage."""
        cap_plan = planner.plan_certain(
            model_name="org.netflix.read-only-kv",
            region="us-east-1",
            desires=large_dataset_moderate_rps,
            extra_model_arguments={
                "rocksdb_block_cache_percent": 0.1,  # 10% cache
            },
        )[0]

        requirement = cap_plan.requirements.regional[0]

        # Check block cache percent is recorded
        assert requirement.context["block_cache_percent"] == 0.1


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
                estimated_state_item_count=certain_int(10_000_000_000),
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
                "require_local_disks": False,
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

    def test_cost_includes_storage(self):
        """Test that ReadOnlyKV cost includes storage costs."""
        cap_plan = planner.plan_certain(
            model_name="org.netflix.read-only-kv",
            region="us-east-1",
            desires=large_dataset_moderate_rps,
            extra_model_arguments={
                "require_local_disks": False,
                "require_attached_disks": True,
            },
        )[0]

        result = cap_plan.candidate_clusters.regional[0]

        # Cost should include both compute and storage
        compute_cost = result.count * result.instance.annual_cost
        storage_cost = (
            result.attached_drives[0].annual_cost * result.count
            if result.attached_drives
            else 0
        )
        expected_total = compute_cost + storage_cost

        # The cluster annual cost should match
        assert abs(result.annual_cost - expected_total) < 1, (
            f"Cluster cost {result.annual_cost} doesn't match "
            f"compute({compute_cost}) + storage({storage_cost}) = {expected_total}"
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
                estimated_state_item_count=certain_int(5_000_000_000),
            ),
        )

        rf2_plan = planner.plan_certain(
            model_name="org.netflix.read-only-kv",
            region="us-east-1",
            desires=large_dataset,
            extra_model_arguments={"replica_count": 2, "require_local_disks": False},
        )[0]

        rf3_plan = planner.plan_certain(
            model_name="org.netflix.read-only-kv",
            region="us-east-1",
            desires=large_dataset,
            extra_model_arguments={"replica_count": 3, "require_local_disks": False},
        )[0]

        rf2_cost = float(rf2_plan.candidate_clusters.total_annual_cost)
        rf3_cost = float(rf3_plan.candidate_clusters.total_annual_cost)

        # RF=2 should be cheaper than or equal to RF=3
        assert rf2_cost <= rf3_cost, (
            f"RF=2 cost ({rf2_cost}) should be <= RF=3 cost ({rf3_cost})"
        )

        # Also verify the storage difference reflects RF
        rf2_storage = rf2_plan.requirements.regional[0].disk_gib.mid
        rf3_storage = rf3_plan.requirements.regional[0].disk_gib.mid
        # RF=3 should require ~50% more storage than RF=2
        assert rf3_storage > rf2_storage, (
            f"RF=3 storage ({rf3_storage}) should be > RF=2 storage ({rf2_storage})"
        )


class TestReadOnlyKVMultiplePlans:
    """E2E tests that validate multiple generated plans are reasonable."""

    def test_generates_multiple_valid_plans(self):
        """Verify we get multiple plans and all meet minimum requirements."""
        plans = planner.plan_certain(
            model_name="org.netflix.read-only-kv",
            region="us-east-1",
            desires=large_dataset_moderate_rps,
            extra_model_arguments={"require_local_disks": False},
        )

        # Should generate multiple plans (at least 2)
        assert len(plans) >= 2, f"Expected >= 2 plans, got {len(plans)}"

        # All plans should meet minimum requirements
        min_storage_gib = 2000  # 1TB * RF=2
        min_memory_gib = 200  # ~20% of data for block cache

        for i, plan in enumerate(plans[:5]):  # Check top 5 plans
            result = plan.candidate_clusters.regional[0]
            total_storage = get_total_storage_gib(result)
            total_memory = result.count * result.instance.ram_gib

            assert total_storage is not None
            assert total_storage >= min_storage_gib, (
                f"Plan {i} ({result.instance.name}): "
                f"insufficient storage {total_storage} < {min_storage_gib}"
            )
            assert total_memory >= min_memory_gib, (
                f"Plan {i} ({result.instance.name}): "
                f"insufficient memory {total_memory} < {min_memory_gib}"
            )

    def test_plans_sorted_by_cost(self):
        """Verify plans are sorted by cost (cheapest first)."""
        plans = planner.plan_certain(
            model_name="org.netflix.read-only-kv",
            region="us-east-1",
            desires=large_dataset_moderate_rps,
            extra_model_arguments={"require_local_disks": False},
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
            extra_model_arguments={"require_local_disks": False},
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
            extra_model_arguments={"require_local_disks": False},
        )

        for i, plan in enumerate(plans):
            result = plan.candidate_clusters.regional[0]
            assert result.cluster_type == "read-only-kv", (
                f"Plan {i}: expected cluster_type='read-only-kv', "
                f"got '{result.cluster_type}'"
            )

    def test_plans_scale_with_data_size(self):
        """Verify larger datasets result in more resources across all plans."""
        small_plans = planner.plan_certain(
            model_name="org.netflix.read-only-kv",
            region="us-east-1",
            desires=small_dataset_high_rps,  # 50GB
            extra_model_arguments={"require_local_disks": False},
        )

        large_plans = planner.plan_certain(
            model_name="org.netflix.read-only-kv",
            region="us-east-1",
            desires=large_dataset_moderate_rps,  # 1TB
            extra_model_arguments={"require_local_disks": False},
        )

        # Best plan for large dataset should cost more than best for small
        small_best_cost = float(small_plans[0].candidate_clusters.total_annual_cost)
        large_best_cost = float(large_plans[0].candidate_clusters.total_annual_cost)

        assert large_best_cost > small_best_cost, (
            f"Large dataset ({large_best_cost}) should cost more than "
            f"small dataset ({small_best_cost})"
        )

        # Storage requirement should scale with data size
        small_storage = small_plans[0].requirements.regional[0].disk_gib.mid
        large_storage = large_plans[0].requirements.regional[0].disk_gib.mid

        # 1TB is 20x larger than 50GB, so storage should be significantly more
        assert large_storage > small_storage * 10, (
            f"Large storage ({large_storage}) should be >> small ({small_storage})"
        )
