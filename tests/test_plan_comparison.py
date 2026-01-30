"""Tests for plan comparison utility - minimal test suite."""

import pytest
from decimal import Decimal

from service_capacity_modeling.interface import (
    CapacityPlan,
    CapacityRequirement,
    Clusters,
    Drive,
    Instance,
    Lifecycle,
    Platform,
    Requirements,
    ZoneClusterCapacity,
    certain_float,
    certain_int,
)
from service_capacity_modeling.models.plan_comparison import (
    compare_plans,
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
    to_reference_cores,
)


# -----------------------------------------------------------------------------
# Test helpers
# -----------------------------------------------------------------------------


def _create_instance(
    name: str = "test.xlarge",
    cpu: int = 8,
    cpu_ghz: float = 2.3,  # Same as default_reference_shape for 1:1 normalization
    cpu_ipc_scale: float = 1.0,
) -> Instance:
    """Helper to create a test instance."""
    return Instance(
        name=name,
        cpu=cpu,
        cpu_ghz=cpu_ghz,
        cpu_ipc_scale=cpu_ipc_scale,
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
    """Helper to create test plans with specified resources.

    CPU is computed from candidate_clusters, so cpu_cores sets the instance CPU
    with count=1. The instance uses default_reference_shape IPC/GHz for 1:1
    normalization (cpu_cores directly equals reference-equivalent cores).
    """
    # Instance with same IPC/GHz as default_reference_shape for 1:1 normalization
    instance = _create_instance(cpu=cpu_cores)
    requirement = CapacityRequirement(
        requirement_type="test",
        cpu_cores=certain_int(cpu_cores),
        mem_gib=certain_float(mem_gib),
        disk_gib=certain_float(disk_gib),
        network_mbps=certain_float(network_mbps),
    )
    cluster = ZoneClusterCapacity(
        cluster_type="test",
        count=1,  # count=1 so instance.cpu equals total CPU
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
# to_reference_cores tests
# -----------------------------------------------------------------------------


class TestToReferenceCores:
    """Tests for to_reference_cores() utility function."""

    def test_same_as_reference_shape(self):
        """Instance matching reference shape returns cores unchanged.

        default_reference_shape: 2.3 GHz, IPC 1.0
        100 cores @ 2.3 GHz, IPC 1.0 → 100 × (2.3×1.0)/(2.3×1.0) = 100
        """
        instance = _create_instance(cpu=100, cpu_ghz=2.3, cpu_ipc_scale=1.0)
        assert to_reference_cores(100, instance) == 100.0

    def test_faster_instance_gives_more_reference_cores(self):
        """Faster instance (higher GHz) yields more reference-equivalent cores.

        100 cores @ 3.0 GHz, IPC 1.0 → 100 × (3.0×1.0)/(2.3×1.0) = 130.4
        """
        instance = _create_instance(cpu=100, cpu_ghz=3.0, cpu_ipc_scale=1.0)
        result = to_reference_cores(100, instance)
        assert 130.0 < result < 131.0  # 130.43

    def test_higher_ipc_gives_more_reference_cores(self):
        """Higher IPC yields more reference-equivalent cores.

        100 cores @ 2.3 GHz, IPC 2.0 → 100 × (2.3×2.0)/(2.3×1.0) = 200
        """
        instance = _create_instance(cpu=100, cpu_ghz=2.3, cpu_ipc_scale=2.0)
        assert to_reference_cores(100, instance) == 200.0

    def test_slower_instance_gives_fewer_reference_cores(self):
        """Slower instance (lower GHz or IPC) yields fewer reference-equivalent cores.

        100 cores @ 2.0 GHz, IPC 0.8 → 100 × (2.0×0.8)/(2.3×1.0) = 69.6
        """
        instance = _create_instance(cpu=100, cpu_ghz=2.0, cpu_ipc_scale=0.8)
        result = to_reference_cores(100, instance)
        assert 69.0 < result < 70.0  # 69.57

    def test_returns_float_not_int(self):
        """Returns float for precise ratio calculations (unlike normalize_cores).

        32 cores @ 2.4 GHz, IPC 1.0 → 32 × (2.4×1.0)/(2.3×1.0) = 33.39
        normalize_cores would return ceil(33.39) = 34, but we need 33.39
        """
        instance = _create_instance(cpu=32, cpu_ghz=2.4, cpu_ipc_scale=1.0)
        result = to_reference_cores(32, instance)
        assert isinstance(result, float)
        assert 33.3 < result < 33.5  # 33.39, not 34


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
        assert tolerances.get_tolerance(ResourceType.cpu).lower == 0.95
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
            resource=ResourceType.cpu,
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
            resource=ResourceType.cpu,
            baseline_value=0,
            comparison_value=100,
            tolerance=lte(1.1),
        )
        assert comp.ratio == float("inf")
        assert comp.exceeds_upper_bound

    def test_ratio_both_zero(self):
        """Both zero returns ratio of 1.0 (exact match)."""
        comp = ResourceComparison(
            resource=ResourceType.cpu,
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
            resource=ResourceType.cpu,
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
            resource=ResourceType.cpu,
            baseline_value=100,
            comparison_value=105,
            tolerance=lte(1.1),
        )
        assert comp.is_equivalent
        assert not comp.exceeds_upper_bound
        assert not comp.exceeds_lower_bound

    def test_str_shows_bound_exceeded(self):
        comp = ResourceComparison(
            resource=ResourceType.cpu,
            baseline_value=100,
            comparison_value=130,
            tolerance=plus_or_minus(0.10),
        )
        assert "exceeds upper bound" in str(comp)

        comp_lower = ResourceComparison(
            resource=ResourceType.cpu,
            baseline_value=100,
            comparison_value=50,
            tolerance=plus_or_minus(0.10),
        )
        assert "exceeds lower bound" in str(comp_lower)

    def test_str_shows_within_tolerance(self):
        """__str__ indicates when within tolerance."""
        comp = ResourceComparison(
            resource=ResourceType.cpu,
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
            ("annual_cost", "annual_cost"),
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
        assert out_of_tolerance[0].resource == ResourceType.cpu

    def test_returns_all_resource_types(self):
        """compare_plans returns comparisons for all 5 resource types."""
        result = compare_plans(_create_plan(), _create_plan())
        assert len(result.comparisons) == 5
        assert set(result.comparisons.keys()) == {
            ResourceType.annual_cost,
            ResourceType.cpu,
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

    def test_cpu_normalization_across_different_ipc(self):
        """CPU comparison normalizes for both GHz and IPC differences.

        Normalization formula:
            ref_cores = cores × (ghz × ipc) / (2.3 × 1.0)

        Where 2.3 GHz and IPC 1.0 is the default_reference_shape.

        Test cases:
        1. Different GHz, same IPC:
           - A: 100 cores @ 2.3 GHz, IPC 1.0 → 100 × 2.3/2.3 = 100 ref
           - B: 77 cores @ 3.0 GHz, IPC 1.0 → 77 × 3.0/2.3 = 100.4 ref
           Ratio ≈ 1.0 → equivalent

        2. Same GHz, different IPC (higher IPC = fewer cores needed):
           - A: 100 cores @ 2.3 GHz, IPC 1.0 → 100 ref
           - B: 50 cores @ 2.3 GHz, IPC 2.0 → 50 × 4.6/2.3 = 100 ref
           Ratio = 1.0 → equivalent

        3. Combined GHz + IPC difference:
           - A: 100 cores @ 2.3 GHz, IPC 1.0 → 100 ref
           - B: 40 cores @ 3.0 GHz, IPC 2.0 → 40 × 6.0/2.3 = 104.3 ref
           Ratio ≈ 1.04 → equivalent (within ±10%)

        4. IPC difference exceeds tolerance (failure case):
           - A: 100 cores @ 2.3 GHz, IPC 1.0 → 100 ref
           - B: 100 cores @ 2.3 GHz, IPC 0.5 → 100 × 1.15/2.3 = 50 ref
           Ratio = 0.5 → under-provisioned (outside ±10%)
        """

        def make_plan(cpu: int, cpu_ghz: float, cpu_ipc_scale: float) -> CapacityPlan:
            inst = _create_instance(
                cpu=cpu, cpu_ghz=cpu_ghz, cpu_ipc_scale=cpu_ipc_scale
            )
            return CapacityPlan(
                requirements=Requirements(
                    zonal=[
                        CapacityRequirement(
                            requirement_type="t", cpu_cores=certain_int(cpu)
                        )
                    ]
                ),
                candidate_clusters=Clusters(
                    annual_costs={"t": Decimal("1000")},
                    zonal=[
                        ZoneClusterCapacity(
                            cluster_type="t", count=1, instance=inst, annual_cost=1000.0
                        )
                    ],
                ),
            )

        # Case 1: Different GHz, same IPC
        baseline = make_plan(cpu=100, cpu_ghz=2.3, cpu_ipc_scale=1.0)
        comparison = make_plan(cpu=77, cpu_ghz=3.0, cpu_ipc_scale=1.0)
        result = compare_plans(baseline, comparison)
        assert result.cpu.is_equivalent
        assert 0.99 < result.cpu.ratio < 1.05  # 100.4/100 ≈ 1.004

        # Case 2: Same GHz, different IPC (2× IPC = half the cores)
        baseline = make_plan(cpu=100, cpu_ghz=2.3, cpu_ipc_scale=1.0)
        comparison = make_plan(cpu=50, cpu_ghz=2.3, cpu_ipc_scale=2.0)
        result = compare_plans(baseline, comparison)
        assert result.cpu.is_equivalent
        assert 0.99 < result.cpu.ratio < 1.01  # Exactly 1.0

        # Case 3: Combined GHz + IPC
        baseline = make_plan(cpu=100, cpu_ghz=2.3, cpu_ipc_scale=1.0)
        comparison = make_plan(cpu=40, cpu_ghz=3.0, cpu_ipc_scale=2.0)
        result = compare_plans(baseline, comparison)
        assert result.cpu.is_equivalent
        assert 1.0 < result.cpu.ratio < 1.1  # 104.3/100 ≈ 1.043

        # Case 4: IPC difference exceeds tolerance (NOT equivalent)
        #   - Baseline: 100 cores @ 2.3 GHz, IPC 1.0 → 100 ref
        #   - Comparison: 100 cores @ 2.3 GHz, IPC 0.5 → 100 × 1.15/2.3 = 50 ref
        #   Ratio = 0.5 → under-provisioned (outside ±10%)
        baseline = make_plan(cpu=100, cpu_ghz=2.3, cpu_ipc_scale=1.0)
        comparison = make_plan(cpu=100, cpu_ghz=2.3, cpu_ipc_scale=0.5)
        result = compare_plans(baseline, comparison)
        assert not result.cpu.is_equivalent
        assert result.cpu.exceeds_lower_bound  # ratio < 0.9 = under-provisioned
        assert 0.49 < result.cpu.ratio < 0.51  # Exactly 0.5
