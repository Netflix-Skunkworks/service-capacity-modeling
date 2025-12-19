"""Tests for plan comparison utility"""

import pytest
from decimal import Decimal

from service_capacity_modeling.interface import (
    CapacityDesires,
    CapacityPlan,
    CapacityRequirement,
    Clusters,
    CurrentClusters,
    CurrentZoneClusterCapacity,
    CurrentRegionClusterCapacity,
    Drive,
    Instance,
    Interval,
    Lifecycle,
    Platform,
    Requirements,
    ZoneClusterCapacity,
    certain_float,
    certain_int,
)
from service_capacity_modeling.models.plan_comparison import (
    allow_over_provisioning,
    compare_plans,
    ComparisonResult,
    extract_baseline_plan,
    ignore_resource,
    ResourceTolerances,
    ResourceType,
    strict_tolerance,
    symmetric_tolerance,
    Tolerance,
    tolerance,
)


def _create_instance(name: str = "test.xlarge") -> Instance:
    """Helper to create a test instance"""
    return Instance(
        name=name,
        cpu=8,
        cpu_ghz=2.4,
        cpu_ipc_scale=1.0,
        ram_gib=32,
        net_mbps=10000,
        lifecycle=Lifecycle.stable,
        platforms=[Platform.amd64],
    )


def _create_plan(
    cpu_cores: int = 100,
    mem_gib: float = 200.0,
    disk_gib: float = 1000.0,
    network_mbps: float = 5000.0,
    annual_cost: float = 10000.0,
    *,
    instance_name: str = "test.xlarge",
) -> CapacityPlan:
    """Helper to create test plans with specified resources"""
    instance = _create_instance(instance_name)

    requirement = CapacityRequirement(
        requirement_type="test",
        cpu_cores=certain_int(cpu_cores),
        mem_gib=certain_float(mem_gib),
        disk_gib=certain_float(disk_gib),
        network_mbps=certain_float(network_mbps),
    )

    cluster = ZoneClusterCapacity(
        cluster_type="test",
        count=10,
        instance=instance,
        annual_cost=annual_cost,
    )

    return CapacityPlan(
        requirements=Requirements(zonal=[requirement]),
        candidate_clusters=Clusters(
            annual_costs={"test": Decimal(str(annual_cost))},
            zonal=[cluster],
        ),
    )


# -----------------------------------------------------------------------------
# Tolerance class tests
# -----------------------------------------------------------------------------


class TestTolerance:
    """Tests for Tolerance class"""

    def test_tolerance_contains_within_bounds(self):
        """'in' returns True for values within bounds"""
        t = Tolerance(lower=-0.10, upper=0.20)
        assert 0.0 in t  # Exact match
        assert -0.05 in t  # 5% under
        assert 0.15 in t  # 15% over
        assert -0.10 in t  # At lower boundary
        assert 0.20 in t  # At upper boundary

    def test_tolerance_contains_outside_bounds(self):
        """'in' returns False for values outside bounds"""
        t = Tolerance(lower=-0.10, upper=0.20)
        assert -0.15 not in t  # Too far under
        assert 0.25 not in t  # Too far over

    def test_tolerance_frozen(self):
        """Tolerance should be frozen (hashable)"""
        t1 = Tolerance(lower=-0.10, upper=0.20)
        t2 = Tolerance(lower=-0.10, upper=0.20)
        # Frozen models are hashable and equal hashes for equal values
        assert hash(t1) == hash(t2)
        assert t1 == t2

    def test_tolerance_asymmetric_bounds(self):
        """Tolerance can have different lower/upper bounds"""
        # Must be over-provisioned by 15-30%
        t = Tolerance(lower=0.15, upper=0.30)
        assert 0.10 not in t  # Not enough over
        assert 0.20 in t  # Within range
        assert 0.35 not in t  # Too much over


class TestToleranceHelpers:
    """Tests for tolerance helper functions"""

    def test_tolerance_explicit(self):
        """tolerance() creates explicit bounds"""
        t = tolerance(-0.05, 0.20)
        assert t.lower == -0.05
        assert t.upper == 0.20

    def test_symmetric_tolerance(self):
        """symmetric_tolerance() creates equal bounds in both directions"""
        t = symmetric_tolerance(0.10)
        assert t.lower == -0.10
        assert t.upper == 0.10

    def test_allow_over_provisioning(self):
        """allow_over_provisioning() creates infinite lower bound"""
        t = allow_over_provisioning(0.10)
        assert t.lower == float("-inf")
        assert t.upper == 0.10
        assert -100.0 in t  # Baseline can be arbitrarily over-provisioned
        assert 0.05 in t  # Comparison can be slightly over baseline
        assert 0.15 not in t  # But not too much over

    def test_allow_over_provisioning_default(self):
        """allow_over_provisioning() defaults to 10% under-provisioning limit"""
        t = allow_over_provisioning()
        assert t.lower == float("-inf")
        assert t.upper == 0.10

    def test_strict_tolerance(self):
        """strict_tolerance() requires near-exact match (float epsilon)"""
        t = strict_tolerance()
        # Uses float epsilon (1e-9) for bounds
        assert t.lower == pytest.approx(-1e-9)
        assert t.upper == pytest.approx(1e-9)
        assert 0.0 in t
        assert 1e-10 in t  # Within epsilon
        assert 0.01 not in t  # 1% is way outside
        assert -0.01 not in t

    def test_ignore_resource(self):
        """ignore_resource() accepts any value"""
        t = ignore_resource()
        assert t.lower == float("-inf")
        assert t.upper == float("inf")
        assert 0.0 in t
        assert 100.0 in t
        assert -100.0 in t

    def test_helpers_are_cached(self):
        """Helper functions should return cached instances"""
        t1 = symmetric_tolerance(0.10)
        t2 = symmetric_tolerance(0.10)
        assert t1 is t2


# -----------------------------------------------------------------------------
# ResourceTolerances tests
# -----------------------------------------------------------------------------


class TestResourceTolerances:
    """Tests for per-resource tolerance configuration"""

    def test_default_fallback(self):
        """get_tolerance falls back to default when not specified"""
        tolerances = ResourceTolerances(cpu=symmetric_tolerance(0.05))
        default = symmetric_tolerance(0.10)

        # CPU has specific tolerance
        assert tolerances.get_tolerance(ResourceType.cpus, default).lower == -0.05
        # Memory falls back to default
        assert tolerances.get_tolerance(ResourceType.mem_gib, default).lower == -0.10

    def test_all_resources_configurable(self):
        """All resource types can be configured"""
        tolerances = ResourceTolerances(
            cost=tolerance(-0.01, 0.01),
            cpu=tolerance(-0.02, 0.02),
            memory=tolerance(-0.03, 0.03),
            disk=tolerance(-0.04, 0.04),
            network=tolerance(-0.05, 0.05),
        )
        default = strict_tolerance()

        assert tolerances.get_tolerance(ResourceType.cost, default).lower == -0.01
        assert tolerances.get_tolerance(ResourceType.cpus, default).lower == -0.02
        assert tolerances.get_tolerance(ResourceType.mem_gib, default).lower == -0.03
        assert tolerances.get_tolerance(ResourceType.disk_gib, default).lower == -0.04
        assert (
            tolerances.get_tolerance(ResourceType.network_mbps, default).lower == -0.05
        )


# -----------------------------------------------------------------------------
# ComparisonResult tests
# -----------------------------------------------------------------------------


class TestComparisonResult:
    """Tests for ComparisonResult enum"""

    def test_comparison_result_values(self):
        """ComparisonResult should have expected values"""
        assert ComparisonResult.gt.value == "gt"
        assert ComparisonResult.lt.value == "lt"
        assert ComparisonResult.equivalent.value == "equivalent"


# -----------------------------------------------------------------------------
# compare_plans tests - equivalent
# -----------------------------------------------------------------------------


class TestComparePlansEquivalent:
    """Test cases where plans should be equivalent"""

    def test_identical_plans(self):
        """Identical plans should be equivalent"""
        plan = _create_plan()
        result = compare_plans(plan, plan)
        assert result.is_equivalent
        # All differences should be within tolerance
        assert all(d.is_equivalent for d in result.differences)

    def test_within_default_tolerance(self):
        """Plans within default tolerance (unlimited under, 10% over) are equivalent"""
        baseline = _create_plan(cpu_cores=100)
        comparison = _create_plan(cpu_cores=105)  # 5% over - within 10% limit
        result = compare_plans(baseline, comparison)
        assert result.is_equivalent

    def test_under_provisioned_default(self):
        """Under-provisioned (baseline bigger) should be okay with default tolerance"""
        baseline = _create_plan(cpu_cores=100, mem_gib=200)
        comparison = _create_plan(cpu_cores=50, mem_gib=100)  # 50% lower - OK
        result = compare_plans(baseline, comparison)
        assert result.is_equivalent

    def test_within_symmetric_tolerance(self):
        """Plans within symmetric tolerance should be equivalent"""
        baseline = _create_plan(cpu_cores=100)
        comparison = _create_plan(cpu_cores=105)  # 5% higher
        result = compare_plans(
            baseline, comparison, default_tolerance=symmetric_tolerance(0.10)
        )
        assert result.is_equivalent

    def test_at_upper_boundary(self):
        """Exactly at upper boundary (10% over) should be equivalent"""
        baseline = _create_plan(cpu_cores=100)
        comparison = _create_plan(cpu_cores=110)  # Exactly 10% over
        result = compare_plans(baseline, comparison)
        assert result.is_equivalent

    def test_all_resources_within_tolerance(self):
        """Multiple resources slightly different but all within tolerance"""
        baseline = _create_plan(
            cpu_cores=100, mem_gib=200, disk_gib=1000, network_mbps=5000
        )
        comparison = _create_plan(
            cpu_cores=105,
            mem_gib=210,
            disk_gib=1050,
            network_mbps=5250,  # All ~5% over (within 10% limit)
        )
        result = compare_plans(baseline, comparison)
        assert result.is_equivalent


# -----------------------------------------------------------------------------
# compare_plans tests - different
# -----------------------------------------------------------------------------


class TestCompareplansDifferent:
    """Test cases where plans should be different (over-provisioned beyond 10%)"""

    def test_over_provisioned_cpu(self):
        """Over-provisioned CPU beyond 10% tolerance should be flagged"""
        baseline = _create_plan(cpu_cores=100)
        comparison = _create_plan(cpu_cores=120)  # 20% over
        result = compare_plans(baseline, comparison)

        assert not result.is_equivalent
        assert result.cpu.comparison_result == ComparisonResult.gt
        assert not result.cpu.is_equivalent

    def test_over_provisioned_memory(self):
        """Over-provisioned memory beyond 10% should be flagged"""
        baseline = _create_plan(mem_gib=200)
        comparison = _create_plan(mem_gib=250)  # 25% over
        result = compare_plans(baseline, comparison)

        assert not result.is_equivalent
        assert result.memory.comparison_result == ComparisonResult.gt

    def test_over_provisioned_disk(self):
        """Over-provisioned disk beyond 10% should be flagged"""
        baseline = _create_plan(disk_gib=1000)
        comparison = _create_plan(disk_gib=1200)  # 20% over
        result = compare_plans(baseline, comparison)

        assert not result.is_equivalent
        assert result.disk.comparison_result == ComparisonResult.gt

    def test_over_provisioned_network(self):
        """Over-provisioned network beyond 10% should be flagged"""
        baseline = _create_plan(network_mbps=5000)
        comparison = _create_plan(network_mbps=7000)  # 40% over
        result = compare_plans(baseline, comparison)

        assert not result.is_equivalent
        assert result.network.comparison_result == ComparisonResult.gt

    def test_over_provisioned_cost(self):
        """Higher cost beyond 10% tolerance should be flagged"""
        baseline = _create_plan(annual_cost=10000)
        comparison = _create_plan(annual_cost=13000)  # 30% higher cost
        result = compare_plans(baseline, comparison)

        assert not result.is_equivalent
        assert result.cost.comparison_result == ComparisonResult.gt

    def test_under_provisioned_with_symmetric_tolerance(self):
        """Under-provisioned should be flagged with symmetric tolerance"""
        baseline = _create_plan(cpu_cores=100)
        comparison = _create_plan(cpu_cores=50)  # 50% lower
        result = compare_plans(
            baseline, comparison, default_tolerance=symmetric_tolerance(0.10)
        )

        assert not result.is_equivalent
        assert result.cpu.comparison_result == ComparisonResult.lt

    def test_multiple_out_of_tolerance(self):
        """Multiple resources out of tolerance should all be flagged"""
        baseline = _create_plan(cpu_cores=100, mem_gib=200, disk_gib=1000)
        comparison = _create_plan(
            cpu_cores=130, mem_gib=260, disk_gib=1300
        )  # All 30% over
        result = compare_plans(baseline, comparison)
        assert not result.is_equivalent
        assert len(result.get_out_of_tolerance()) >= 3

    def test_get_out_of_tolerance(self):
        """get_out_of_tolerance() should return only problematic resources"""
        baseline = _create_plan(cpu_cores=100, mem_gib=200)
        comparison = _create_plan(cpu_cores=120, mem_gib=200)  # Only CPU over
        result = compare_plans(baseline, comparison)

        out_of_tolerance = result.get_out_of_tolerance()
        assert len(out_of_tolerance) == 1
        assert out_of_tolerance[0].resource == ResourceType.cpus


# -----------------------------------------------------------------------------
# Per-resource tolerance tests
# -----------------------------------------------------------------------------


class TestPerResourceTolerances:
    """Tests for per-resource tolerance configuration"""

    def test_different_tolerances_per_resource(self):
        """Different resources can have different tolerances"""
        baseline = _create_plan(cpu_cores=100, mem_gib=200)
        comparison = _create_plan(cpu_cores=85, mem_gib=170)  # 15% lower each

        tolerances = ResourceTolerances(
            cpu=symmetric_tolerance(0.20),  # 20% tolerance for CPU
            memory=symmetric_tolerance(0.10),  # 10% tolerance for memory
        )
        result = compare_plans(baseline, comparison, tolerances=tolerances)

        # CPU should be within tolerance (15% under, 20% allowed)
        assert result.cpu.is_equivalent

        # Memory should be out of tolerance (15% under, only 10% allowed)
        assert not result.memory.is_equivalent

    def test_strict_cpu_lenient_cost(self):
        """Strict CPU tolerance with lenient cost tolerance"""
        baseline = _create_plan(cpu_cores=100, annual_cost=10000)
        comparison = _create_plan(
            cpu_cores=103, annual_cost=8000
        )  # 3% more CPU, 20% less cost

        tolerances = ResourceTolerances(
            cpu=symmetric_tolerance(0.02),  # Only 2% allowed
            cost=allow_over_provisioning(0.05),  # Baseline can be huge, 5% over allowed
        )
        result = compare_plans(baseline, comparison, tolerances=tolerances)

        # CPU should be out of tolerance (3% over, only 2% allowed)
        assert not result.cpu.is_equivalent
        # Cost should be within tolerance (under is always allowed)
        assert result.cost.is_equivalent


# -----------------------------------------------------------------------------
# Explanation tests
# -----------------------------------------------------------------------------


class TestComparePlansExplanations:
    """Test that __str__ explanations are meaningful"""

    def test_explanation_contains_direction(self):
        """Explanation should indicate under or over provisioned"""
        baseline = _create_plan(cpu_cores=100)
        comparison = _create_plan(cpu_cores=130)  # 30% over
        result = compare_plans(baseline, comparison)

        assert "over" in str(result.cpu)

    def test_explanation_contains_percentage(self):
        """Explanation should contain the percentage difference"""
        baseline = _create_plan(cpu_cores=100)
        comparison = _create_plan(cpu_cores=130)  # 30% higher
        result = compare_plans(baseline, comparison)

        assert "30.0%" in str(result.cpu)

    def test_explanation_within_tolerance(self):
        """Explanation for within-tolerance should indicate that"""
        baseline = _create_plan(cpu_cores=100)
        comparison = _create_plan(cpu_cores=95)  # 5% lower, within tolerance
        result = compare_plans(baseline, comparison)

        assert "within tolerance" in str(result.cpu)

    def test_explanation_over_provisioned(self):
        """Explanation should indicate over-provisioned when appropriate"""
        baseline = _create_plan(cpu_cores=100)
        comparison = _create_plan(cpu_cores=150)  # 50% higher
        result = compare_plans(
            baseline, comparison, default_tolerance=symmetric_tolerance(0.10)
        )

        assert "over" in str(result.cpu)


# -----------------------------------------------------------------------------
# Context tests
# -----------------------------------------------------------------------------


class TestComparePlansContext:
    """Test context information in results"""

    def test_context_contains_default_tolerance(self):
        """Context should include the default tolerance used"""
        tol = symmetric_tolerance(0.15)
        result = compare_plans(_create_plan(), _create_plan(), default_tolerance=tol)
        assert result.context["default_tolerance"]["lower"] == -0.15
        assert result.context["default_tolerance"]["upper"] == 0.15

    def test_context_contains_costs(self):
        """Context should include baseline and comparison costs"""
        result = compare_plans(
            _create_plan(annual_cost=10000),
            _create_plan(annual_cost=12000),
        )
        assert result.context["baseline_cost"] == 10000
        assert result.context["comparison_cost"] == 12000


# -----------------------------------------------------------------------------
# Edge case tests
# -----------------------------------------------------------------------------


class TestComparePlansEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_zero_baseline_resource_over(self):
        """Zero baseline with comparison over should be gt and out of tolerance"""
        baseline = _create_plan(network_mbps=0)
        comparison = _create_plan(network_mbps=1000)
        result = compare_plans(baseline, comparison)

        # With default allow_over_provisioning, infinite over is out of tolerance
        assert result.network.comparison_result == ComparisonResult.gt
        assert not result.network.is_equivalent

    def test_zero_both_resources(self):
        """Both zero should be equivalent"""
        baseline = _create_plan(network_mbps=0)
        comparison = _create_plan(network_mbps=0)
        result = compare_plans(baseline, comparison)

        assert result.network.comparison_result == ComparisonResult.equivalent
        assert result.network.is_equivalent

    def test_strict_tolerance_catches_small_differences(self):
        """Strict tolerance should catch small differences"""
        baseline = _create_plan(cpu_cores=100)
        comparison = _create_plan(cpu_cores=99)  # 1% lower
        result = compare_plans(
            baseline, comparison, default_tolerance=strict_tolerance()
        )
        assert not result.is_equivalent

    def test_lenient_tolerance_accepts_large_differences(self):
        """Lenient tolerance should accept larger differences"""
        baseline = _create_plan(cpu_cores=100)
        comparison = _create_plan(cpu_cores=80)  # 20% lower
        result = compare_plans(
            baseline, comparison, default_tolerance=symmetric_tolerance(0.25)
        )
        assert result.is_equivalent

    def test_difference_percent_calculation(self):
        """Verify difference percentage is calculated correctly"""
        baseline = _create_plan(cpu_cores=100)
        comparison = _create_plan(cpu_cores=75)  # 25% lower
        result = compare_plans(baseline, comparison)

        assert result.cpu.difference_percent == pytest.approx(-0.25, rel=0.01)
        assert result.cpu.baseline_value == 100
        assert result.cpu.comparison_value == 75

    def test_returns_all_differences(self):
        """compare_plans should return differences for ALL resources"""
        result = compare_plans(_create_plan(), _create_plan())
        # Should have 5 differences: cost, cpus, mem_gib, disk_gib, network_mbps
        assert len(result.differences) == 5
        resources = {d.resource for d in result.differences}
        assert resources == {
            ResourceType.cost,
            ResourceType.cpus,
            ResourceType.mem_gib,
            ResourceType.disk_gib,
            ResourceType.network_mbps,
        }

    def test_tolerance_with_unusual_bounds(self):
        """Tolerance can require over-provisioning"""
        baseline = _create_plan(cpu_cores=100)
        comparison = _create_plan(cpu_cores=115)  # 15% over

        # Require 10-20% over-provisioning for CPU only
        tolerances = ResourceTolerances(cpu=tolerance(0.10, 0.20))
        result = compare_plans(baseline, comparison, tolerances=tolerances)

        # CPU should be within its unusual tolerance
        assert result.cpu.is_equivalent
        assert (
            result.cpu.comparison_result == ComparisonResult.gt
        )  # It's over-provisioned

        # 5% over should fail for this unusual tolerance
        comparison_low = _create_plan(cpu_cores=105)
        result_low = compare_plans(baseline, comparison_low, tolerances=tolerances)
        assert not result_low.cpu.is_equivalent  # 5% is below required 10%


# -----------------------------------------------------------------------------
# extract_baseline_plan tests
# -----------------------------------------------------------------------------


def _create_instance_with_local_drive(  # pylint: disable=too-many-positional-arguments
    name: str = "i3.xlarge",
    cpu: int = 4,
    ram_gib: float = 30.5,
    net_mbps: float = 10000,
    drive_size_gib: int = 950,
    annual_cost: float = 2500.0,
) -> Instance:
    """Helper to create an instance with local (NVMe) storage"""
    local_drive = Drive(name="local-nvme", size_gib=drive_size_gib)
    return Instance(
        name=name,
        cpu=cpu,
        cpu_ghz=2.4,
        cpu_ipc_scale=1.0,
        ram_gib=ram_gib,
        net_mbps=net_mbps,
        drive=local_drive,
        annual_cost=annual_cost,
        lifecycle=Lifecycle.stable,
        platforms=[Platform.amd64],
    )


def _create_instance_without_local_drive(
    name: str = "m5.xlarge",
    cpu: int = 4,
    ram_gib: float = 16.0,
    net_mbps: float = 10000,
    annual_cost: float = 2000.0,
) -> Instance:
    """Helper to create an instance without local storage"""
    return Instance(
        name=name,
        cpu=cpu,
        cpu_ghz=2.4,
        cpu_ipc_scale=1.0,
        ram_gib=ram_gib,
        net_mbps=net_mbps,
        drive=None,
        annual_cost=annual_cost,
        lifecycle=Lifecycle.stable,
        platforms=[Platform.amd64],
    )


def _create_attached_drive(
    name: str = "gp3",
    size_gib: int = 500,
    annual_cost_per_gib: float = 1.0,
) -> Drive:
    """Helper to create an attached drive (e.g., EBS).

    Note: Drive.annual_cost is a computed property that calculates
    size_gib * annual_cost_per_gib + iops_costs. We set annual_cost_per_gib
    instead of annual_cost directly.
    """
    return Drive(name=name, size_gib=size_gib, annual_cost_per_gib=annual_cost_per_gib)


class TestExtractBaselinePlan:
    """Tests for extract_baseline_plan helper"""

    def test_zonal_cluster_with_local_storage(self):
        """Extract baseline from zonal cluster with local NVMe storage"""
        instance = _create_instance_with_local_drive(
            cpu=4, ram_gib=30.5, net_mbps=10000, drive_size_gib=950, annual_cost=2500.0
        )
        current = CurrentZoneClusterCapacity(
            cluster_instance_name="i3.xlarge",
            cluster_instance=instance,
            cluster_instance_count=Interval(low=8, mid=8, high=8, confidence=1.0),
        )
        desires = CapacityDesires(current_clusters=CurrentClusters(zonal=[current]))

        baseline = extract_baseline_plan(desires, region="us-east-1")

        # Verify baseline was created correctly
        assert len(baseline.requirements.zonal) == 1
        assert len(baseline.requirements.regional) == 0

        req = baseline.requirements.zonal[0]
        # 4 CPU * 8 nodes = 32 raw cores, normalized to default_reference_shape
        # (instance cpu_ghz=2.4 vs reference cpu_ghz=2.3 → ~34 normalized cores)
        assert req.cpu_cores.mid == 34
        # 30.5 GiB * 8 nodes = 244 GiB
        assert req.mem_gib.mid == 244.0
        # 950 GiB local drive * 8 nodes = 7600 GiB
        assert req.disk_gib.mid == 7600.0
        # 10000 Mbps * 8 nodes = 80000 Mbps
        assert req.network_mbps.mid == 80000.0

        # Cost = instance.annual_cost * count (no attached drive)
        expected_cost = 2500.0 * 8  # $20,000
        assert baseline.candidate_clusters.total_annual_cost == expected_cost

    def test_zonal_cluster_with_attached_storage(self):
        """Extract baseline from zonal cluster with attached EBS storage"""
        instance = _create_instance_without_local_drive(
            cpu=4, ram_gib=16.0, net_mbps=10000, annual_cost=2000.0
        )
        # Create a drive with name "gp3" which exists in hardware shapes
        attached_drive = Drive(name="gp3", size_gib=500)
        current = CurrentZoneClusterCapacity(
            cluster_instance_name="m5.xlarge",
            cluster_instance=instance,
            cluster_drive=attached_drive,
            cluster_instance_count=Interval(low=6, mid=6, high=6, confidence=1.0),
        )
        desires = CapacityDesires(current_clusters=CurrentClusters(zonal=[current]))

        baseline = extract_baseline_plan(desires, region="us-east-1")

        req = baseline.requirements.zonal[0]
        # 500 GiB attached * 6 nodes = 3000 GiB
        assert req.disk_gib.mid == 3000.0

        # Cost includes instance + drive pricing from hardware shapes
        # Instance: $2000 * 6 = $12,000
        # Drive: gp3 pricing from shapes * 500 GiB * 6 nodes
        instance_cost = 2000.0 * 6
        assert baseline.candidate_clusters.total_annual_cost >= instance_cost

        # Attached drives should be in candidate_clusters with pricing from shapes
        assert len(baseline.candidate_clusters.zonal[0].attached_drives) == 1
        assert baseline.candidate_clusters.zonal[0].attached_drives[0].size_gib == 500

    def test_regional_cluster(self):
        """Extract baseline from regional cluster"""
        instance = _create_instance_without_local_drive(cpu=8, ram_gib=32.0)
        current = CurrentRegionClusterCapacity(
            cluster_instance_name="m5.2xlarge",
            cluster_instance=instance,
            cluster_instance_count=Interval(low=10, mid=10, high=10, confidence=1.0),
        )
        desires = CapacityDesires(current_clusters=CurrentClusters(regional=[current]))

        baseline = extract_baseline_plan(desires, region="us-east-1")

        # Should be regional requirements, not zonal
        assert len(baseline.requirements.zonal) == 0
        assert len(baseline.requirements.regional) == 1
        assert len(baseline.candidate_clusters.regional) == 1
        assert len(baseline.candidate_clusters.zonal) == 0

        req = baseline.requirements.regional[0]
        # 8 CPU * 10 nodes = 80 raw cores, normalized → 84
        assert req.cpu_cores.mid == 84

    def test_error_when_current_clusters_is_none(self):
        """Should raise ValueError when current_clusters is None"""
        desires = CapacityDesires()  # No current_clusters

        with pytest.raises(ValueError, match="current_clusters is None"):
            extract_baseline_plan(desires, region="us-east-1")

    def test_error_when_current_clusters_is_empty(self):
        """Should raise ValueError when current_clusters is empty"""
        desires = CapacityDesires(current_clusters=CurrentClusters())

        with pytest.raises(ValueError, match="no zonal or regional"):
            extract_baseline_plan(desires, region="us-east-1")

    def test_error_when_both_zonal_and_regional(self):
        """Should raise ValueError when both zonal and regional clusters exist"""
        instance = _create_instance_without_local_drive()
        count = Interval(low=3, mid=3, high=3, confidence=1.0)
        desires = CapacityDesires(
            current_clusters=CurrentClusters(
                zonal=[
                    CurrentZoneClusterCapacity(
                        cluster_instance_name="test",
                        cluster_instance=instance,
                        cluster_instance_count=count,
                    )
                ],
                regional=[
                    CurrentRegionClusterCapacity(
                        cluster_instance_name="test",
                        cluster_instance=instance,
                        cluster_instance_count=count,
                    )
                ],
            )
        )

        with pytest.raises(ValueError, match="both zonal and regional"):
            extract_baseline_plan(desires, region="us-east-1")

    def test_error_when_cluster_instance_is_none(self):
        """Should raise ValueError when cluster_instance is not resolved"""
        current = CurrentZoneClusterCapacity(
            cluster_instance_name="unresolved.xlarge",
            cluster_instance=None,  # Not resolved
            cluster_instance_count=Interval(low=3, mid=3, high=3, confidence=1.0),
        )
        desires = CapacityDesires(current_clusters=CurrentClusters(zonal=[current]))

        with pytest.raises(ValueError, match="cluster_instance is None"):
            extract_baseline_plan(desires, region="us-east-1")

    def test_error_when_drive_not_in_hardware_catalog(self):
        """Should raise ValueError when cluster_drive is not in regional shapes"""
        instance = _create_instance_without_local_drive()
        unknown_drive = Drive(name="unknown-drive-type", size_gib=500)

        current = CurrentZoneClusterCapacity(
            cluster_instance_name="m5.xlarge",
            cluster_instance=instance,
            cluster_drive=unknown_drive,  # Drive not in hardware catalog
            cluster_instance_count=Interval(low=3, mid=3, high=3, confidence=1.0),
        )
        desires = CapacityDesires(current_clusters=CurrentClusters(zonal=[current]))

        with pytest.raises(ValueError, match="Cannot price drive 'unknown-drive-type'"):
            extract_baseline_plan(desires, region="us-east-1")

    def test_no_storage_returns_zero_disk(self):
        """Instance without any storage should return 0 disk"""
        instance = Instance(
            name="compute-only",
            cpu=4,
            cpu_ghz=2.4,
            cpu_ipc_scale=1.0,
            ram_gib=16.0,
            net_mbps=10000,
            drive=None,  # No local storage
            annual_cost=1000.0,
            lifecycle=Lifecycle.stable,
            platforms=[Platform.amd64],
        )
        current = CurrentZoneClusterCapacity(
            cluster_instance_name="compute-only",
            cluster_instance=instance,
            cluster_drive=None,  # No attached storage either
            cluster_instance_count=Interval(low=5, mid=5, high=5, confidence=1.0),
        )
        desires = CapacityDesires(current_clusters=CurrentClusters(zonal=[current]))

        baseline = extract_baseline_plan(desires, region="us-east-1")

        assert baseline.requirements.zonal[0].disk_gib.mid == 0.0

    def test_attached_storage_takes_precedence_over_local(self):
        """When both attached and local storage exist, attached should be used"""
        instance = _create_instance_with_local_drive(drive_size_gib=950)
        # Use "gp3" which exists in hardware shapes
        attached_drive = Drive(name="gp3", size_gib=2000)

        current = CurrentZoneClusterCapacity(
            cluster_instance_name="i3.xlarge",
            cluster_instance=instance,
            cluster_drive=attached_drive,  # Has both local and attached
            cluster_instance_count=Interval(low=3, mid=3, high=3, confidence=1.0),
        )
        desires = CapacityDesires(current_clusters=CurrentClusters(zonal=[current]))

        baseline = extract_baseline_plan(desires, region="us-east-1")

        # Should use attached storage (2000 GiB), not local (950 GiB)
        assert baseline.requirements.zonal[0].disk_gib.mid == 6000.0  # 2000 * 3

    def test_integration_with_compare_plans(self):
        """Full integration test: extract and compare plans"""
        # Current deployment: 8x i3.xlarge with local storage
        instance = _create_instance_with_local_drive(
            cpu=4, ram_gib=30.5, net_mbps=10000, drive_size_gib=950, annual_cost=2500.0
        )
        current = CurrentZoneClusterCapacity(
            cluster_instance_name="i3.xlarge",
            cluster_instance=instance,
            cluster_instance_count=Interval(low=8, mid=8, high=8, confidence=1.0),
        )
        desires = CapacityDesires(current_clusters=CurrentClusters(zonal=[current]))

        # Recommended: slightly more CPU, same memory, similar disk
        # Baseline cost: $2500 * 8 = $20,000
        recommended = _create_plan(
            cpu_cores=35,  # vs 32 current (9% more)
            mem_gib=250,  # vs 244 current (2.5% more)
            disk_gib=7500,  # vs 7600 current (1.3% less)
            network_mbps=85000,  # vs 80000 current (6% more)
            annual_cost=21000,  # vs 20000 current (5% more)
        )

        baseline = extract_baseline_plan(desires, region="us-east-1")

        # Cost is now calculated, so we can compare directly
        result = compare_plans(baseline, recommended)

        # Should be equivalent with default tolerances
        assert result.is_equivalent

        # Verify comparison result is correct for resources
        assert result.cpu.comparison_result == ComparisonResult.gt  # More CPU
        assert (
            result.disk.comparison_result == ComparisonResult.lt
        )  # Slightly less disk
        assert result.cost.comparison_result == ComparisonResult.gt  # Higher cost

    def test_integration_with_custom_tolerances(self):
        """Integration test with strict tolerances should detect differences"""
        instance = _create_instance_with_local_drive(
            cpu=4, ram_gib=30.5, annual_cost=2500.0
        )
        current = CurrentZoneClusterCapacity(
            cluster_instance_name="i3.xlarge",
            cluster_instance=instance,
            cluster_instance_count=Interval(low=8, mid=8, high=8, confidence=1.0),
        )
        desires = CapacityDesires(current_clusters=CurrentClusters(zonal=[current]))

        # Recommended with significantly different CPU
        # Baseline: 32 raw cores → 34 normalized cores
        # Recommended: 40 cores
        # Difference: (40 - 34) / 34 = 17.6%
        recommended = _create_plan(
            cpu_cores=40,
            mem_gib=244,  # Same as current
        )

        baseline = extract_baseline_plan(desires, region="us-east-1")

        # With strict 10% tolerance, the 17.6% CPU increase should be flagged
        tolerances = ResourceTolerances(cpu=symmetric_tolerance(0.10))
        result = compare_plans(baseline, recommended, tolerances=tolerances)

        assert not result.is_equivalent
        assert not result.cpu.is_equivalent
        assert result.cpu.comparison_result == ComparisonResult.gt
        # Difference is now based on normalized baseline (34 cores, not 32)
        assert result.cpu.difference_percent == pytest.approx(0.176, rel=0.01)
