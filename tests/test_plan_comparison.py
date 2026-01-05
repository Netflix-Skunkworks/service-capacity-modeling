"""Tests for plan comparison utility - minimal test suite."""

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
    exact_match,
    gte,
    ignore_resource,
    lte,
    plus_or_minus,
    ResourceComparison,
    ResourceTolerances,
    ResourceType,
    Tolerance,
    tolerance,
)


# -----------------------------------------------------------------------------
# Test helpers
# -----------------------------------------------------------------------------


def _create_instance(name: str = "test.xlarge") -> Instance:
    """Helper to create a test instance."""
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
) -> CapacityPlan:
    """Helper to create test plans with specified resources."""
    instance = _create_instance()
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


def _create_instance_with_local_drive(
    cpu: int = 4,
    ram_gib: float = 30.5,
    net_mbps: float = 10000,
    drive_size_gib: int = 950,
    annual_cost: float = 2500.0,
) -> Instance:
    """Helper to create an instance with local (NVMe) storage."""
    return Instance(
        name="i3.xlarge",
        cpu=cpu,
        cpu_ghz=2.4,
        cpu_ipc_scale=1.0,
        ram_gib=ram_gib,
        net_mbps=net_mbps,
        drive=Drive(name="local-nvme", size_gib=drive_size_gib),
        annual_cost=annual_cost,
        lifecycle=Lifecycle.stable,
        platforms=[Platform.amd64],
    )


def _create_instance_without_local_drive(
    cpu: int = 4,
    ram_gib: float = 16.0,
    net_mbps: float = 10000,
    annual_cost: float = 2000.0,
) -> Instance:
    """Helper to create an instance without local storage."""
    return Instance(
        name="m5.xlarge",
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


# -----------------------------------------------------------------------------
# Tolerance tests
# -----------------------------------------------------------------------------


class TestTolerance:
    def test_boundaries_are_inclusive(self):
        t = Tolerance(lower=0.9, upper=1.2)
        assert 0.9 in t
        assert 1.2 in t
        assert 0.89 not in t
        assert 1.21 not in t


class TestToleranceHelpers:
    """Tests for tolerance helper functions."""

    def test_tolerance(self):
        t = tolerance(0.95, 1.20)
        assert t.lower == 0.95
        assert t.upper == 1.20

    def test_plus_or_minus(self):
        t = plus_or_minus(0.10)
        assert t.lower == 0.9
        assert t.upper == 1.1

    def test_lte(self):
        t = lte(1.1)
        assert t.lower == 0.0
        assert t.upper == 1.1
        assert 0.5 in t
        assert 1.2 not in t

    def test_gte(self):
        t = gte(0.9)
        assert t.lower == 0.9
        assert t.upper == float("inf")
        assert 2.0 in t
        assert 0.8 not in t

    def test_exact_match(self):
        t = exact_match()
        assert 1.0 in t
        assert 0.99 not in t

    def test_ignore_resource(self):
        t = ignore_resource()
        assert 0.01 in t
        assert 100.0 in t


# -----------------------------------------------------------------------------
# ResourceTolerances tests
# -----------------------------------------------------------------------------


class TestResourceTolerances:
    """Tests for per-resource tolerance configuration."""

    def test_fallback_to_default(self):
        """get_tolerance falls back to default when not specified."""
        tolerances = ResourceTolerances(
            default=plus_or_minus(0.10),
            cpu=plus_or_minus(0.05),
        )
        # CPU has specific tolerance
        assert tolerances.get_tolerance(ResourceType.cpus).lower == 0.95
        # Memory falls back to default
        assert tolerances.get_tolerance(ResourceType.mem_gib).lower == 0.9


# -----------------------------------------------------------------------------
# ResourceComparison tests
# -----------------------------------------------------------------------------


class TestResourceComparison:
    """Tests for ResourceComparison ratio calculation and direction detection."""

    def test_ratio_calculation(self):
        """Ratio is calculated as comparison / baseline."""
        comp = ResourceComparison(
            resource=ResourceType.cpus,
            baseline_value=100,
            comparison_value=75,
            tolerance=lte(1.1),
        )
        assert comp.ratio == pytest.approx(0.75)
        assert comp.baseline_value == 100
        assert comp.comparison_value == 75

    def test_ratio_zero_baseline_nonzero_comparison(self):
        """Zero baseline with non-zero comparison returns infinity."""
        comp = ResourceComparison(
            resource=ResourceType.cpus,
            baseline_value=0,
            comparison_value=100,
            tolerance=lte(1.1),
        )
        assert comp.ratio == float("inf")
        assert comp.exceeds_upper_bound

    def test_ratio_both_zero(self):
        """Both zero returns ratio of 1.0 (exact match)."""
        comp = ResourceComparison(
            resource=ResourceType.cpus,
            baseline_value=0,
            comparison_value=0,
            tolerance=lte(1.1),
        )
        assert comp.ratio == 1.0
        assert comp.is_equivalent

    @pytest.mark.parametrize(
        "baseline,comparison,expected_bound",
        [
            (100, 120, "upper"),  # 1.2 > 1.1 upper bound
            (100, 130, "upper"),  # 1.3 > 1.1 upper bound
            (100, 50, "lower"),  # 0.5 < 0.9 lower bound (with symmetric)
        ],
    )
    def test_exceeds_bound_detection(self, baseline, comparison, expected_bound):
        tol = plus_or_minus(0.10) if expected_bound == "lower" else lte(1.1)
        comp = ResourceComparison(
            resource=ResourceType.cpus,
            baseline_value=baseline,
            comparison_value=comparison,
            tolerance=tol,
        )
        if expected_bound == "upper":
            assert comp.exceeds_upper_bound
            assert not comp.exceeds_lower_bound
        else:
            assert comp.exceeds_lower_bound
            assert not comp.exceeds_upper_bound

    def test_is_equivalent_when_within_tolerance(self):
        comp = ResourceComparison(
            resource=ResourceType.cpus,
            baseline_value=100,
            comparison_value=105,
            tolerance=lte(1.1),
        )
        assert comp.is_equivalent
        assert not comp.exceeds_upper_bound
        assert not comp.exceeds_lower_bound

    def test_str_shows_bound_exceeded(self):
        comp = ResourceComparison(
            resource=ResourceType.cpus,
            baseline_value=100,
            comparison_value=130,
            tolerance=plus_or_minus(0.10),
        )
        assert "exceeds upper bound" in str(comp)

        comp_lower = ResourceComparison(
            resource=ResourceType.cpus,
            baseline_value=100,
            comparison_value=50,
            tolerance=plus_or_minus(0.10),
        )
        assert "exceeds lower bound" in str(comp_lower)

    def test_str_shows_within_tolerance(self):
        """__str__ indicates when within tolerance."""
        comp = ResourceComparison(
            resource=ResourceType.cpus,
            baseline_value=100,
            comparison_value=105,
            tolerance=lte(1.1),
        )
        assert "within tolerance" in str(comp)


# -----------------------------------------------------------------------------
# compare_plans tests
# -----------------------------------------------------------------------------


class TestComparePlans:
    """Tests for compare_plans function."""

    def test_identical_plans_are_equivalent(self):
        """Identical plans should be equivalent."""
        plan = _create_plan()
        result = compare_plans(plan, plan)
        assert result.is_equivalent
        assert all(c.is_equivalent for c in result.comparisons.values())

    def test_within_tolerance_is_equivalent(self):
        """Plans within default tolerance are equivalent."""
        baseline = _create_plan(cpu_cores=100)
        comparison = _create_plan(cpu_cores=105)  # ratio = 1.05 < 1.1
        result = compare_plans(baseline, comparison)
        assert result.is_equivalent

    def test_exceeding_tolerance_is_not_equivalent(self):
        baseline = _create_plan(cpu_cores=100)
        comparison = _create_plan(cpu_cores=120)
        result = compare_plans(baseline, comparison)
        assert not result.is_equivalent
        assert result.cpu.exceeds_upper_bound

    @pytest.mark.parametrize(
        "resource_kwarg,accessor",
        [
            ("cpu_cores", "cpu"),
            ("mem_gib", "memory"),
            ("disk_gib", "disk"),
            ("network_mbps", "network"),
            ("annual_cost", "cost"),
        ],
    )
    def test_exceeds_upper_bound_for_each_resource(self, resource_kwarg, accessor):
        baseline_kwargs = {
            "cpu_cores": 100,
            "mem_gib": 200.0,
            "disk_gib": 1000.0,
            "network_mbps": 5000.0,
            "annual_cost": 10000.0,
        }
        comparison_kwargs = baseline_kwargs.copy()
        comparison_kwargs[resource_kwarg] = baseline_kwargs[resource_kwarg] * 1.3

        baseline = _create_plan(**baseline_kwargs)
        comparison = _create_plan(**comparison_kwargs)
        result = compare_plans(baseline, comparison)

        resource_result = getattr(result, accessor)
        assert resource_result.exceeds_upper_bound
        assert not resource_result.is_equivalent

    def test_exceeds_lower_bound_with_symmetric_tolerance(self):
        baseline = _create_plan(cpu_cores=100)
        comparison = _create_plan(cpu_cores=50)
        result = compare_plans(
            baseline,
            comparison,
            tolerances=ResourceTolerances(default=plus_or_minus(0.10)),
        )
        assert not result.is_equivalent
        assert result.cpu.exceeds_lower_bound

    def test_get_out_of_tolerance(self):
        """get_out_of_tolerance returns only problematic resources."""
        baseline = _create_plan(cpu_cores=100, mem_gib=200)
        comparison = _create_plan(cpu_cores=120, mem_gib=200)  # Only CPU exceeds
        result = compare_plans(baseline, comparison)

        out_of_tolerance = result.get_out_of_tolerance()
        assert len(out_of_tolerance) == 1
        assert out_of_tolerance[0].resource == ResourceType.cpus

    def test_returns_all_resource_types(self):
        """compare_plans returns comparisons for all 5 resource types."""
        result = compare_plans(_create_plan(), _create_plan())
        assert len(result.comparisons) == 5
        assert set(result.comparisons.keys()) == {
            ResourceType.cost,
            ResourceType.cpus,
            ResourceType.mem_gib,
            ResourceType.disk_gib,
            ResourceType.network_mbps,
        }

    def test_per_resource_tolerances(self):
        baseline = _create_plan(cpu_cores=100, mem_gib=200)
        comparison = _create_plan(cpu_cores=85, mem_gib=170)

        tolerances = ResourceTolerances(
            cpu=plus_or_minus(0.20),
            memory=plus_or_minus(0.10),
        )
        result = compare_plans(baseline, comparison, tolerances=tolerances)

        assert result.cpu.is_equivalent  # 0.85 >= 0.8
        assert not result.memory.is_equivalent  # 0.85 < 0.9
        assert result.memory.exceeds_lower_bound


# -----------------------------------------------------------------------------
# extract_baseline_plan tests
# -----------------------------------------------------------------------------


class TestExtractBaselinePlan:
    """Tests for extract_baseline_plan helper."""

    def test_zonal_cluster_extraction(self):
        """Extract baseline from zonal cluster."""
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

        assert len(baseline.requirements.zonal) == 1
        assert len(baseline.requirements.regional) == 0
        req = baseline.requirements.zonal[0]
        # 4 CPU * 8 nodes = 32 raw, normalized → ~34
        assert req.cpu_cores.mid == 34
        # 30.5 GiB * 8 = 244
        assert req.mem_gib.mid == 244.0
        # 950 * 8 = 7600
        assert req.disk_gib.mid == 7600.0

    def test_regional_cluster_extraction(self):
        """Extract baseline from regional cluster."""
        instance = _create_instance_without_local_drive(cpu=8, ram_gib=32.0)
        current = CurrentRegionClusterCapacity(
            cluster_instance_name="m5.2xlarge",
            cluster_instance=instance,
            cluster_instance_count=Interval(low=10, mid=10, high=10, confidence=1.0),
        )
        desires = CapacityDesires(current_clusters=CurrentClusters(regional=[current]))

        baseline = extract_baseline_plan(desires, region="us-east-1")

        assert len(baseline.requirements.zonal) == 0
        assert len(baseline.requirements.regional) == 1

    @pytest.mark.parametrize(
        "desires_factory,error_match",
        [
            (CapacityDesires, "current_clusters is None"),
            (
                lambda: CapacityDesires(current_clusters=CurrentClusters()),
                "no zonal or regional",
            ),
            (
                lambda: CapacityDesires(
                    current_clusters=CurrentClusters(
                        zonal=[
                            CurrentZoneClusterCapacity(
                                cluster_instance_name="test",
                                cluster_instance=None,
                                cluster_instance_count=Interval(
                                    low=3, mid=3, high=3, confidence=1.0
                                ),
                            )
                        ]
                    )
                ),
                "cluster_instance is None",
            ),
        ],
    )
    def test_error_cases(self, desires_factory, error_match):
        """Various error conditions raise ValueError."""
        with pytest.raises(ValueError, match=error_match):
            extract_baseline_plan(desires_factory(), region="us-east-1")

    def test_integration_extract_and_compare(self):
        """Full integration: extract baseline and compare with recommended."""
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

        # Recommended plan slightly different but within tolerance
        recommended = _create_plan(
            cpu_cores=35,  # vs ~34 → ratio ~1.03
            mem_gib=250,  # vs 244 → ratio ~1.02
            disk_gib=7500,  # vs 7600 → ratio ~0.99
            network_mbps=85000,
            annual_cost=21000,
        )

        result = compare_plans(baseline, recommended)
        assert result.is_equivalent
