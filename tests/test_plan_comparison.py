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
    compare_plans,
    extract_baseline_plan,
    gte,
    ignore_resource,
    lte,
    plus_or_minus,
    ResourceTolerances,
    ResourceType,
    exact_match,
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
        """'in' returns True for ratios within bounds"""
        t = Tolerance(lower=0.9, upper=1.2)
        assert 1.0 in t  # Exact match
        assert 0.95 in t  # 5% extra capacity
        assert 1.15 in t  # 15% over
        assert 0.9 in t  # At lower boundary
        assert 1.2 in t  # At upper boundary

    def test_tolerance_contains_outside_bounds(self):
        """'in' returns False for ratios outside bounds"""
        t = Tolerance(lower=0.9, upper=1.2)
        assert 0.85 not in t  # Too much extra
        assert 1.25 not in t  # Too much over

    def test_tolerance_frozen(self):
        """Tolerance should be frozen (hashable)"""
        t1 = Tolerance(lower=0.9, upper=1.2)
        t2 = Tolerance(lower=0.9, upper=1.2)
        # Frozen models are hashable and equal hashes for equal values
        assert hash(t1) == hash(t2)
        assert t1 == t2

    def test_tolerance_asymmetric_bounds(self):
        """Tolerance can have different lower/upper bounds"""
        # Require ratio between 1.15 and 1.30 (15-30% buffer)
        t = Tolerance(lower=1.15, upper=1.30)
        assert 1.10 not in t  # Not enough buffer
        assert 1.20 in t  # Within range
        assert 1.35 not in t  # Too much buffer


class TestToleranceHelpers:
    """Tests for tolerance helper functions"""

    def test_tolerance_explicit(self):
        """tolerance() creates explicit ratio bounds"""
        t = tolerance(0.95, 1.20)
        assert t.lower == 0.95
        assert t.upper == 1.20

    def test_plus_or_minus(self):
        """plus_or_minus() creates symmetric bounds around 1.0"""
        t = plus_or_minus(0.10)
        assert t.lower == 0.9
        assert t.upper == 1.1

    def test_lte(self):
        """lte() sets upper bound (ratio must be ≤ value)"""
        t = lte(1.1)
        assert t.lower == 0.0
        assert t.upper == 1.1

        # Baseline=100, comparison=110 → ratio = 1.1 → OK (at boundary)
        assert 1.1 in t
        # Baseline=100, comparison=111 → ratio = 1.11 → NOT OK
        assert 1.11 not in t
        # Baseline=100, comparison=90 → ratio = 0.9 → OK (extra capacity)
        assert 0.9 in t
        # Baseline=100, comparison=10 → ratio = 0.1 → OK (lots extra)
        assert 0.1 in t

    def test_gte(self):
        """gte() sets lower bound (ratio must be ≥ value)"""
        t = gte(0.9)
        assert t.lower == 0.9
        assert t.upper == float("inf")

        # Baseline=100, comparison=90 → ratio = 0.9 → OK (at boundary)
        assert 0.9 in t
        # Baseline=100, comparison=89 → ratio = 0.89 → NOT OK (too much extra)
        assert 0.89 not in t
        # Baseline=100, comparison=110 → ratio = 1.1 → OK (need more)
        assert 1.1 in t
        # Baseline=100, comparison=200 → ratio = 2.0 → OK (need lots more)
        assert 2.0 in t

    def test_exact_match(self):
        """exact_match() requires near-exact match (ratio ≈ 1.0)"""
        t = exact_match()
        # Uses float epsilon (1e-9) for bounds around 1.0
        assert t.lower == pytest.approx(1.0 - 1e-9)
        assert t.upper == pytest.approx(1.0 + 1e-9)
        assert 1.0 in t
        assert 1.0 + 1e-10 in t  # Within epsilon
        assert 1.01 not in t  # 1% is way outside
        assert 0.99 not in t

    def test_ignore_resource(self):
        """ignore_resource() accepts any ratio"""
        t = ignore_resource()
        assert t.lower == 0.0
        assert t.upper == float("inf")
        assert 1.0 in t
        assert 100.0 in t
        assert 0.01 in t

    def test_helpers_are_cached(self):
        """Helper functions should return cached instances"""
        t1 = plus_or_minus(0.10)
        t2 = plus_or_minus(0.10)
        assert t1 is t2


# -----------------------------------------------------------------------------
# ResourceTolerances tests
# -----------------------------------------------------------------------------


class TestResourceTolerances:
    """Tests for per-resource tolerance configuration"""

    def test_default_fallback(self):
        """get_tolerance falls back to default when not specified"""
        tolerances = ResourceTolerances(
            default=plus_or_minus(0.10),
            cpu=plus_or_minus(0.05),
        )

        # CPU has specific tolerance
        assert tolerances.get_tolerance(ResourceType.cpus).lower == 0.95
        # Memory falls back to default
        assert tolerances.get_tolerance(ResourceType.mem_gib).lower == 0.9

    def test_all_resources_configurable(self):
        """All resource types can be configured"""
        tolerances = ResourceTolerances(
            default=exact_match(),
            cost=tolerance(0.99, 1.01),
            cpu=tolerance(0.98, 1.02),
            memory=tolerance(0.97, 1.03),
            disk=tolerance(0.96, 1.04),
            network=tolerance(0.95, 1.05),
        )

        assert tolerances.get_tolerance(ResourceType.cost).lower == 0.99
        assert tolerances.get_tolerance(ResourceType.cpus).lower == 0.98
        assert tolerances.get_tolerance(ResourceType.mem_gib).lower == 0.97
        assert tolerances.get_tolerance(ResourceType.disk_gib).lower == 0.96
        assert tolerances.get_tolerance(ResourceType.network_mbps).lower == 0.95


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
        assert all(d.is_equivalent for d in result.differences.values())

    def test_within_default_tolerance(self):
        """Plans within default tolerance (ratio ≤ 1.1) are equivalent"""
        baseline = _create_plan(cpu_cores=100)
        # comparison=105 → ratio = 1.05, within 1.1 limit
        comparison = _create_plan(cpu_cores=105)
        result = compare_plans(baseline, comparison)
        assert result.is_equivalent

    def test_over_provisioned_default(self):
        """Over-provisioned (baseline larger) is allowed with default tolerance"""
        baseline = _create_plan(cpu_cores=200, mem_gib=400)
        # comparison=100 → ratio = 0.5, any extra capacity is OK
        comparison = _create_plan(cpu_cores=100, mem_gib=200)
        result = compare_plans(baseline, comparison)
        # Default tolerance allows unlimited extra capacity (lower=0.0)
        assert result.is_equivalent

    def test_within_symmetric_tolerance(self):
        """Plans within symmetric tolerance should be equivalent"""
        baseline = _create_plan(cpu_cores=105)
        # comparison=100 → ratio = 100/105 = 0.952, within ±10%
        comparison = _create_plan(cpu_cores=100)
        result = compare_plans(
            baseline,
            comparison,
            tolerances=ResourceTolerances(default=plus_or_minus(0.10)),
        )
        assert result.is_equivalent

    def test_at_upper_boundary(self):
        """Exactly at upper boundary (ratio = 1.1) should be equivalent"""
        baseline = _create_plan(cpu_cores=100)
        # comparison=110 → ratio = 1.1, exactly at upper bound
        comparison = _create_plan(cpu_cores=110)
        result = compare_plans(baseline, comparison)
        assert result.is_equivalent

    def test_all_resources_within_tolerance(self):
        """Multiple resources slightly exceeding baseline but within tolerance"""
        baseline = _create_plan(
            cpu_cores=100, mem_gib=200, disk_gib=1000, network_mbps=5000
        )
        # All ~5% higher → ratio = 1.05, within 1.1 limit
        comparison = _create_plan(
            cpu_cores=105,
            mem_gib=210,
            disk_gib=1050,
            network_mbps=5250,
        )
        result = compare_plans(baseline, comparison)
        assert result.is_equivalent


# -----------------------------------------------------------------------------
# compare_plans tests - different
# -----------------------------------------------------------------------------


class TestCompareplansDifferent:
    """Test cases where plans should be different (ratio exceeds 1.1)"""

    def test_under_provisioned_cpu(self):
        """Under-provisioned CPU beyond 10% tolerance should be flagged"""
        baseline = _create_plan(cpu_cores=100)
        # comparison=120 → ratio = 1.2, exceeds 1.1 limit
        comparison = _create_plan(cpu_cores=120)
        result = compare_plans(baseline, comparison)

        assert not result.is_equivalent
        assert result.cpu.is_under_provisioned
        assert not result.cpu.is_equivalent

    def test_under_provisioned_memory(self):
        """Under-provisioned memory beyond 10% should be flagged"""
        baseline = _create_plan(mem_gib=200)
        # comparison=250 → ratio = 1.25, exceeds 1.1 limit
        comparison = _create_plan(mem_gib=250)
        result = compare_plans(baseline, comparison)

        assert not result.is_equivalent
        assert result.memory.is_under_provisioned

    def test_under_provisioned_disk(self):
        """Under-provisioned disk beyond 10% should be flagged"""
        baseline = _create_plan(disk_gib=1000)
        # comparison=1200 → ratio = 1.2, exceeds 1.1 limit
        comparison = _create_plan(disk_gib=1200)
        result = compare_plans(baseline, comparison)

        assert not result.is_equivalent
        assert result.disk.is_under_provisioned

    def test_under_provisioned_network(self):
        """Under-provisioned network beyond 10% should be flagged"""
        baseline = _create_plan(network_mbps=5000)
        # comparison=7000 → ratio = 1.4, exceeds 1.1 limit
        comparison = _create_plan(network_mbps=7000)
        result = compare_plans(baseline, comparison)

        assert not result.is_equivalent
        assert result.network.is_under_provisioned

    def test_under_provisioned_cost(self):
        """Higher cost beyond 10% tolerance should be flagged"""
        baseline = _create_plan(annual_cost=10000)
        # comparison=13000 → ratio = 1.3, exceeds 1.1 limit
        comparison = _create_plan(annual_cost=13000)
        result = compare_plans(baseline, comparison)

        assert not result.is_equivalent
        assert result.cost.is_under_provisioned

    def test_over_provisioned_with_symmetric_tolerance(self):
        """Over-provisioned should be flagged with symmetric tolerance"""
        baseline = _create_plan(cpu_cores=100)
        # comparison=50 → ratio = 0.5, below 0.9 lower bound
        comparison = _create_plan(cpu_cores=50)
        result = compare_plans(
            baseline,
            comparison,
            tolerances=ResourceTolerances(default=plus_or_minus(0.10)),
        )

        assert not result.is_equivalent
        assert result.cpu.is_over_provisioned

    def test_multiple_out_of_tolerance(self):
        """Multiple resources out of tolerance should all be flagged"""
        baseline = _create_plan(cpu_cores=100, mem_gib=200, disk_gib=1000)
        # All ~30% more → ratio = 1.3, exceeds 1.1 limit
        comparison = _create_plan(cpu_cores=130, mem_gib=260, disk_gib=1300)
        result = compare_plans(baseline, comparison)
        assert not result.is_equivalent
        assert len(result.get_out_of_tolerance()) >= 3

    def test_get_out_of_tolerance(self):
        """get_out_of_tolerance() should return only problematic resources"""
        baseline = _create_plan(cpu_cores=100, mem_gib=200)
        # Only CPU exceeds (120/100 = 1.2 > 1.1), memory is same
        comparison = _create_plan(cpu_cores=120, mem_gib=200)
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
        # comparison has 15% less → ratio = 0.85
        comparison = _create_plan(cpu_cores=85, mem_gib=170)

        tolerances = ResourceTolerances(
            cpu=plus_or_minus(0.20),  # 20% tolerance for CPU (0.8 to 1.2)
            memory=plus_or_minus(0.10),  # 10% tolerance for memory (0.9 to 1.1)
        )
        result = compare_plans(baseline, comparison, tolerances=tolerances)

        # CPU should be within tolerance (0.85 ≥ 0.8)
        assert result.cpu.is_equivalent

        # Memory should be out of tolerance (0.85 < 0.9)
        assert not result.memory.is_equivalent

    def test_strict_cpu_lenient_cost(self):
        """Strict CPU tolerance with lenient cost tolerance"""
        baseline = _create_plan(cpu_cores=100, annual_cost=10000)
        # CPU: 103/100 = 1.03, Cost: 8000/10000 = 0.8
        comparison = _create_plan(cpu_cores=103, annual_cost=8000)

        tolerances = ResourceTolerances(
            cpu=plus_or_minus(0.02),  # Only 2% allowed (0.98 to 1.02)
            cost=lte(1.05),  # ratio ≤ 1.05
        )
        result = compare_plans(baseline, comparison, tolerances=tolerances)

        # CPU should be out of tolerance (1.03 > 1.02)
        assert not result.cpu.is_equivalent
        assert result.cpu.is_under_provisioned
        # Cost is 0.8 (have extra budget), which is OK with lte(1.05)
        assert result.cost.is_equivalent


# -----------------------------------------------------------------------------
# Explanation tests
# -----------------------------------------------------------------------------


class TestComparePlansExplanations:
    """Test that __str__ explanations are meaningful"""

    def test_explanation_contains_direction(self):
        """Explanation should indicate under or over provisioned"""
        baseline = _create_plan(cpu_cores=100)
        # comparison=130 → ratio = 1.3, exceeds symmetric tolerance
        comparison = _create_plan(cpu_cores=130)
        result = compare_plans(
            baseline,
            comparison,
            tolerances=ResourceTolerances(default=plus_or_minus(0.10)),
        )

        assert "under" in str(result.cpu)

    def test_explanation_contains_ratio(self):
        """Explanation should contain the ratio"""
        baseline = _create_plan(cpu_cores=100)
        # comparison=130 → ratio = 1.3
        comparison = _create_plan(cpu_cores=130)
        result = compare_plans(
            baseline,
            comparison,
            tolerances=ResourceTolerances(default=plus_or_minus(0.10)),
        )

        assert "1.30" in str(result.cpu)

    def test_explanation_within_tolerance(self):
        """Explanation for within-tolerance should indicate that"""
        baseline = _create_plan(cpu_cores=100)
        # comparison=105 → ratio = 1.05, within default 1.1 tolerance
        comparison = _create_plan(cpu_cores=105)
        result = compare_plans(baseline, comparison)

        assert "within tolerance" in str(result.cpu)

    def test_explanation_over_provisioned(self):
        """Explanation should indicate over-provisioned when appropriate"""
        baseline = _create_plan(cpu_cores=100)
        # comparison=50 → ratio = 0.5, below symmetric tolerance lower bound
        comparison = _create_plan(cpu_cores=50)
        result = compare_plans(
            baseline,
            comparison,
            tolerances=ResourceTolerances(default=plus_or_minus(0.10)),
        )

        assert "over" in str(result.cpu)


# -----------------------------------------------------------------------------
# Edge case tests
# -----------------------------------------------------------------------------


class TestComparePlansEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_zero_baseline_resource(self):
        """Zero baseline with non-zero comparison returns infinity"""
        baseline = _create_plan(network_mbps=0)
        comparison = _create_plan(network_mbps=1000)
        result = compare_plans(baseline, comparison)

        # baseline=0, comparison=1000 → ratio = inf
        # Default tolerance upper=1.1, so inf > 1.1 → under-provisioned
        assert result.network.ratio == float("inf")
        assert not result.network.is_equivalent
        assert result.network.is_under_provisioned

    def test_zero_both_resources(self):
        """Both zero should be equivalent (ratio = 1.0)"""
        baseline = _create_plan(network_mbps=0)
        comparison = _create_plan(network_mbps=0)
        result = compare_plans(baseline, comparison)

        assert result.network.ratio == 1.0
        assert not result.network.is_over_provisioned
        assert not result.network.is_under_provisioned
        assert result.network.is_equivalent

    def test_exact_match_catches_small_differences(self):
        """Strict tolerance should catch small differences"""
        baseline = _create_plan(cpu_cores=100)
        comparison = _create_plan(cpu_cores=99)  # ratio = 0.99
        result = compare_plans(
            baseline,
            comparison,
            tolerances=ResourceTolerances(default=exact_match()),
        )
        assert not result.is_equivalent

    def test_lenient_tolerance_accepts_large_differences(self):
        """Lenient tolerance should accept larger differences"""
        baseline = _create_plan(cpu_cores=100)
        comparison = _create_plan(cpu_cores=80)  # ratio = 0.8
        result = compare_plans(
            baseline,
            comparison,
            tolerances=ResourceTolerances(default=plus_or_minus(0.25)),
        )
        assert result.is_equivalent

    def test_ratio_calculation(self):
        """Verify ratio is calculated correctly"""
        baseline = _create_plan(cpu_cores=100)
        comparison = _create_plan(cpu_cores=75)
        result = compare_plans(baseline, comparison)

        # ratio = comparison / baseline = 75 / 100 = 0.75
        assert result.cpu.ratio == pytest.approx(0.75, rel=0.01)
        assert result.cpu.baseline_value == 100
        assert result.cpu.comparison_value == 75

    def test_returns_all_differences(self):
        """compare_plans should return differences for ALL resources"""
        result = compare_plans(_create_plan(), _create_plan())
        # Should have 5 differences: cost, cpus, mem_gib, disk_gib, network_mbps
        assert len(result.differences) == 5
        assert set(result.differences.keys()) == {
            ResourceType.cost,
            ResourceType.cpus,
            ResourceType.mem_gib,
            ResourceType.disk_gib,
            ResourceType.network_mbps,
        }

    def test_tolerance_with_unusual_bounds(self):
        """Tolerance can require extra capacity (10-20% buffer)"""
        baseline = _create_plan(cpu_cores=115)  # Have 15% more
        comparison = _create_plan(cpu_cores=100)  # Need 100

        # ratio = 100/115 = 0.87 (have 13% extra)
        # Require 10-20% buffer: ratio should be 0.80 to 0.90
        tolerances = ResourceTolerances(cpu=tolerance(0.80, 0.90))
        result = compare_plans(baseline, comparison, tolerances=tolerances)

        # CPU should be within its tolerance (0.87 is in [0.80, 0.90])
        assert result.cpu.is_equivalent
        # When within tolerance, not flagged as over/under
        assert not result.cpu.is_over_provisioned
        assert not result.cpu.is_under_provisioned

        # Only 5% extra should fail (ratio = 100/105 = 0.95, above 0.90)
        baseline_low = _create_plan(cpu_cores=105)  # Only 5% extra
        result_low = compare_plans(baseline_low, comparison, tolerances=tolerances)
        assert not result_low.cpu.is_equivalent
        assert result_low.cpu.is_under_provisioned  # ratio 0.95 > upper bound 0.90


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
        # Baseline (current): $2500 * 8 = $20,000, ~34 normalized CPU, 7600 disk
        recommended = _create_plan(
            cpu_cores=35,  # vs ~34 current → ratio = 1.03 (need slightly more)
            mem_gib=250,  # vs 244 current → ratio = 1.02 (need slightly more)
            disk_gib=7500,  # vs 7600 current → ratio = 0.99 (have extra)
            network_mbps=85000,  # vs 80000 current → ratio = 1.06 (need more)
            annual_cost=21000,  # vs 20000 current → ratio = 1.05 (costs more)
        )

        baseline = extract_baseline_plan(desires, region="us-east-1")

        # Cost is now calculated, so we can compare directly
        result = compare_plans(baseline, recommended)

        # Should be equivalent with default tolerances
        assert result.is_equivalent

        # When within tolerance, resources are not flagged as over/under
        assert not result.cpu.is_over_provisioned
        assert not result.cpu.is_under_provisioned
        # Verify the ratio directions:
        # ratio > 1.0 = comparison > baseline (need more)
        # ratio < 1.0 = comparison < baseline (have extra)
        assert result.disk.ratio < 1.0  # Have extra disk
        assert result.cpu.ratio > 1.0  # Need more CPU

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
        # Baseline: 32 raw cores → ~34 normalized cores
        # Recommended: 40 cores
        # Ratio: 40 / 34 = 1.176 (need ~18% more)
        recommended = _create_plan(
            cpu_cores=40,
            mem_gib=244,  # Same as current
        )

        baseline = extract_baseline_plan(desires, region="us-east-1")

        # With strict 10% tolerance, the 1.176 ratio should be flagged
        tolerances = ResourceTolerances(cpu=plus_or_minus(0.10))
        result = compare_plans(baseline, recommended, tolerances=tolerances)

        assert not result.is_equivalent
        assert not result.cpu.is_equivalent
        assert result.cpu.is_under_provisioned  # Need more CPU
        # Ratio is 40 / 34 ≈ 1.176
        assert result.cpu.ratio == pytest.approx(1.176, rel=0.05)
