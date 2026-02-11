"""Tests for plan comparison utility."""

import pytest
from decimal import Decimal

from service_capacity_modeling.interface import (
    CapacityPlan,
    CapacityRequirement,
    Clusters,
    default_reference_shape,
    Drive,
    Instance,
    Lifecycle,
    Platform,
    RegionClusterCapacity,
    Requirements,
    ZoneClusterCapacity,
    certain_float,
    certain_int,
)
from service_capacity_modeling.models.common import EFFECTIVE_DISK_PER_NODE_GIB
from service_capacity_modeling.models.common import normalize_cores_float
from service_capacity_modeling.models.plan_comparison import (
    compare_plans,
    ComparisonStrategy,
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


def _create_instance(
    *,
    name: str = "test.xlarge",
    cpu: int = 8,
    cpu_ghz: float = 2.3,  # Same as default_reference_shape for 1:1 normalization
    cpu_ipc_scale: float = 1.0,
    ram_gib: float = 32,
    net_mbps: float = 10000,
    drive: Drive | None = None,
) -> Instance:
    """Helper to create a test instance."""
    return Instance(
        name=name,
        cpu=cpu,
        cpu_ghz=cpu_ghz,
        cpu_ipc_scale=cpu_ipc_scale,
        ram_gib=ram_gib,
        net_mbps=net_mbps,
        drive=drive,
        lifecycle=Lifecycle.stable,
        platforms=[Platform.amd64],
    )


def _create_plan(
    *,
    cpu_cores: int = 100,
    mem_gib: float = 200.0,
    disk_gib: float = 1000.0,
    network_mbps: float = 5000.0,
    annual_cost: float = 10000.0,
    count: int = 1,
) -> CapacityPlan:
    """Helper to create test plans with specified resources.

    CPU is computed from candidate_clusters, so cpu_cores sets the instance CPU
    with count=1. The instance uses default_reference_shape IPC/GHz for 1:1
    normalization (cpu_cores directly equals reference-equivalent cores).

    Memory, disk, and network are set on the instance so that cluster-based
    aggregation produces the expected values. Requirements use decoy values
    (9999) so that reading from the wrong source produces obvious failures.
    """
    drive = Drive(name="test-drive", size_gib=int(disk_gib)) if disk_gib else None
    instance = _create_instance(
        cpu=cpu_cores,
        ram_gib=mem_gib,
        net_mbps=network_mbps,
        drive=drive,
    )
    # Decoy requirement values — if _aggregate_resources reads from requirements
    # instead of clusters, tests asserting baseline_value/comparison_value will fail.
    requirement = CapacityRequirement(
        requirement_type="test",
        cpu_cores=certain_int(9999),
        mem_gib=certain_float(9999.0),
        disk_gib=certain_float(9999.0),
        network_mbps=certain_float(9999.0),
    )
    cluster = ZoneClusterCapacity(
        cluster_type="test",
        count=count,
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
# normalize_cores_float tests
# -----------------------------------------------------------------------------


def _to_ref_cores(core_count, instance):
    """Test helper: normalize instance cores to reference-equivalent (float)."""
    return normalize_cores_float(
        core_count,
        target_shape=default_reference_shape,
        reference_shape=instance,
    )


class TestNormalizeCoresFloat:
    """Tests for normalize_cores_float() — float precision variant."""

    def test_same_as_reference_shape(self):
        """Instance matching reference shape returns cores unchanged.

        default_reference_shape: 2.3 GHz, IPC 1.0
        100 cores @ 2.3 GHz, IPC 1.0 → 100 × (2.3×1.0)/(2.3×1.0) = 100
        """
        instance = _create_instance(cpu=100, cpu_ghz=2.3, cpu_ipc_scale=1.0)
        assert _to_ref_cores(100, instance) == 100.0

    def test_faster_instance_gives_more_reference_cores(self):
        """Faster instance (higher GHz) yields more reference-equivalent cores.

        100 cores @ 3.0 GHz, IPC 1.0 → 100 × (3.0×1.0)/(2.3×1.0) = 130.4
        """
        instance = _create_instance(cpu=100, cpu_ghz=3.0, cpu_ipc_scale=1.0)
        result = _to_ref_cores(100, instance)
        assert 130.0 < result < 131.0  # 130.43

    def test_higher_ipc_gives_more_reference_cores(self):
        """Higher IPC yields more reference-equivalent cores.

        100 cores @ 2.3 GHz, IPC 2.0 → 100 × (2.3×2.0)/(2.3×1.0) = 200
        """
        instance = _create_instance(cpu=100, cpu_ghz=2.3, cpu_ipc_scale=2.0)
        assert _to_ref_cores(100, instance) == 200.0

    def test_slower_instance_gives_fewer_reference_cores(self):
        """Slower instance (lower GHz or IPC) yields fewer reference-equivalent cores.

        100 cores @ 2.0 GHz, IPC 0.8 → 100 × (2.0×0.8)/(2.3×1.0) = 69.6
        """
        instance = _create_instance(cpu=100, cpu_ghz=2.0, cpu_ipc_scale=0.8)
        result = _to_ref_cores(100, instance)
        assert 69.0 < result < 70.0  # 69.57

    def test_returns_float_not_int(self):
        """normalize_cores_float returns float for precise ratio calculations.

        32 cores @ 2.4 GHz, IPC 1.0 → 32 × (2.4×1.0)/(2.3×1.0) = 33.39
        normalize_cores would return ceil(33.39) = 34, but we need 33.39
        """
        instance = _create_instance(cpu=32, cpu_ghz=2.4, cpu_ipc_scale=1.0)
        result = _to_ref_cores(32, instance)
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

    @pytest.mark.parametrize("count", [1, 3])
    def test_identical_plans_are_equivalent(self, count):
        """Identical plans should be equivalent.

        Also verifies that resource values are aggregated from candidate_clusters
        (not requirements, which use decoy value 9999), and scaled by count.
        """
        plan = _create_plan(count=count)
        result = compare_plans(plan, plan)
        assert result.is_equivalent
        assert all(c.is_equivalent for c in result.comparisons.values())
        # Verify values come from clusters (scaled by count), not decoy reqs (9999)
        assert result.cpu.baseline_value == 100.0 * count
        assert result.memory.baseline_value == 200.0 * count
        assert result.disk.baseline_value == 1000.0 * count
        assert result.network.baseline_value == 5000.0 * count
        assert result.annual_cost.baseline_value == 10000.0

    def test_within_tolerance_is_equivalent(self):
        """Plans within default tolerance are equivalent."""
        baseline = _create_plan(cpu_cores=100)
        comparison = _create_plan(cpu_cores=105)  # ratio = 1.05 < 1.1
        result = compare_plans(baseline, comparison)
        assert result.is_equivalent

    @pytest.mark.parametrize("count", [1, 3])
    def test_exceeding_tolerance_is_not_equivalent(self, count):
        baseline = _create_plan(cpu_cores=100, count=count)
        comparison = _create_plan(cpu_cores=120, count=count)
        result = compare_plans(baseline, comparison)
        assert not result.is_equivalent
        assert result.cpu.exceeds_upper_bound
        assert result.cpu.baseline_value == 100.0 * count
        assert result.cpu.comparison_value == 120.0 * count

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
        baseline = make_plan(cpu=100, cpu_ghz=2.3, cpu_ipc_scale=1.0)
        comparison = make_plan(cpu=100, cpu_ghz=2.3, cpu_ipc_scale=0.5)
        result = compare_plans(baseline, comparison)
        assert not result.cpu.is_equivalent
        assert result.cpu.exceeds_lower_bound
        assert 0.49 < result.cpu.ratio < 0.51  # Exactly 0.5


# -----------------------------------------------------------------------------
# compare_plans requirements strategy tests
# -----------------------------------------------------------------------------


def _make_requirement_plan(
    *,
    cluster_type: str = "cassandra",
    req_cpu: int = 100,
    req_mem: float = 200.0,
    req_disk: float = 1000.0,
    req_net: float = 5000.0,
    req_reference_shape: Instance = default_reference_shape,
    cluster_annual_cost: float | None = None,
) -> CapacityPlan:
    """Build a plan with requirements (and optionally a matching cluster for cost)."""
    requirement = CapacityRequirement(
        requirement_type=cluster_type,
        reference_shape=req_reference_shape,
        cpu_cores=certain_int(req_cpu),
        mem_gib=certain_float(req_mem),
        disk_gib=certain_float(req_disk),
        network_mbps=certain_float(req_net),
    )
    zonal_clusters = []
    annual_costs = {}
    if cluster_annual_cost is not None:
        zonal_clusters.append(
            ZoneClusterCapacity(
                cluster_type=cluster_type,
                count=1,
                instance=_create_instance(),
                annual_cost_override=cluster_annual_cost,
            )
        )
        annual_costs[cluster_type] = Decimal(str(cluster_annual_cost))
    return CapacityPlan(
        requirements=Requirements(zonal=[requirement]),
        candidate_clusters=Clusters(
            annual_costs=annual_costs,
            zonal=zonal_clusters,
        ),
    )


def _wrap_cluster(cluster: ZoneClusterCapacity) -> CapacityPlan:
    """Wrap a single cluster in a minimal CapacityPlan."""
    return CapacityPlan(
        requirements=Requirements(),
        candidate_clusters=Clusters(
            annual_costs={cluster.cluster_type: Decimal(str(cluster.annual_cost))},
            zonal=[cluster],
        ),
    )


class TestCompareRequirementsStrategy:
    """Tests for compare_plans with strategy=requirements."""

    def test_exact_match(self):
        """Cluster matching plan requirement exactly → is_equivalent=True.

        Plan requires 100 ref-cores, 200 GiB mem, 1000 GiB disk, 5000 Mbps.
        Cluster provides exactly those values.
        """
        req_plan = _make_requirement_plan(
            req_cpu=100,
            req_mem=200.0,
            req_disk=1000.0,
            req_net=5000.0,
            cluster_annual_cost=15000.0,
        )
        cluster_plan = _wrap_cluster(
            ZoneClusterCapacity(
                cluster_type="cassandra",
                count=1,
                instance=_create_instance(
                    cpu=100,
                    ram_gib=200.0,
                    net_mbps=5000.0,
                    drive=Drive(name="d", size_gib=1000),
                ),
                annual_cost_override=15000.0,
            )
        )
        result = compare_plans(
            cluster_plan,
            req_plan,
            strategy=ComparisonStrategy.requirements,
        )
        assert result.is_equivalent
        assert result.cpu.ratio == pytest.approx(1.0)
        assert result.memory.ratio == pytest.approx(1.0)

    def test_under_provisioned(self):
        """Cluster with less CPU than requirement → exceeds_lower_bound.

        Plan requires 100 ref-cores. Cluster has 50 cores at same GHz/IPC.
        ratio = 50/100 = 0.5, outside default ±10% tolerance.
        """
        req_plan = _make_requirement_plan(req_cpu=100)
        cluster_plan = _wrap_cluster(
            ZoneClusterCapacity(
                cluster_type="cassandra",
                count=1,
                instance=_create_instance(
                    cpu=50,
                    ram_gib=200.0,
                    net_mbps=5000.0,
                    drive=Drive(name="d", size_gib=1000),
                ),
            )
        )
        result = compare_plans(
            cluster_plan,
            req_plan,
            strategy=ComparisonStrategy.requirements,
        )
        assert not result.is_equivalent
        assert result.cpu.exceeds_lower_bound
        assert result.cpu.ratio == pytest.approx(0.5)

    def test_cpu_normalization(self):
        """i3en cluster vs m6id-reference requirement, normalized to default.

        Requirement: 100 cores in default_reference_shape (2.3 GHz, IPC 1.0)
        Cluster: 77 cores @ 3.0 GHz, IPC 1.0 → 77 × 3.0/2.3 = 100.4 ref cores
        ratio = 100.4/100 ≈ 1.004 → equivalent
        """
        req_plan = _make_requirement_plan(req_cpu=100)
        cluster_plan = _wrap_cluster(
            ZoneClusterCapacity(
                cluster_type="cassandra",
                count=1,
                instance=_create_instance(
                    cpu=77,
                    cpu_ghz=3.0,
                    cpu_ipc_scale=1.0,
                    ram_gib=200.0,
                    net_mbps=5000.0,
                    drive=Drive(name="d", size_gib=1000),
                ),
            )
        )
        result = compare_plans(
            cluster_plan,
            req_plan,
            strategy=ComparisonStrategy.requirements,
        )
        assert result.cpu.is_equivalent
        assert 0.99 < result.cpu.ratio < 1.05

    def test_cost_comparison(self):
        """Cost compared against requirement plan's matching cluster.

        annual_cost on ClusterCapacity is a computed field.
        Use annual_cost_override to set specific cost values for testing.
        """
        req_plan = _make_requirement_plan(cluster_annual_cost=10000.0)
        cluster_plan = _wrap_cluster(
            ZoneClusterCapacity(
                cluster_type="cassandra",
                count=1,
                instance=_create_instance(
                    cpu=100,
                    ram_gib=200.0,
                    net_mbps=5000.0,
                    drive=Drive(name="d", size_gib=1000),
                ),
                annual_cost_override=10500.0,
            )
        )
        result = compare_plans(
            cluster_plan,
            req_plan,
            strategy=ComparisonStrategy.requirements,
        )
        assert ResourceType.annual_cost in result.comparisons
        assert result.annual_cost.baseline_value == 10000.0
        assert result.annual_cost.comparison_value == 10500.0
        assert result.annual_cost.ratio == pytest.approx(1.05)

    def test_cost_omitted_when_no_matching_cluster(self):
        """No matching cluster in requirement plan → no cost in result."""
        # Requirement plan has cassandra requirement but no cassandra cluster
        req_plan = _make_requirement_plan()  # no cluster_annual_cost
        cluster_plan = _wrap_cluster(
            ZoneClusterCapacity(
                cluster_type="cassandra",
                count=1,
                instance=_create_instance(
                    cpu=100,
                    ram_gib=200.0,
                    net_mbps=5000.0,
                    drive=Drive(name="d", size_gib=1000),
                ),
                annual_cost_override=12000.0,
            )
        )
        result = compare_plans(
            cluster_plan,
            req_plan,
            strategy=ComparisonStrategy.requirements,
        )
        assert ResourceType.annual_cost not in result.comparisons

    def test_matching_by_cluster_type(self):
        """Correct requirement selected when plan has multiple types."""
        cass_req = CapacityRequirement(
            requirement_type="cassandra",
            cpu_cores=certain_int(100),
            mem_gib=certain_float(200.0),
            disk_gib=certain_float(1000.0),
            network_mbps=certain_float(5000.0),
        )
        evcache_req = CapacityRequirement(
            requirement_type="evcache",
            cpu_cores=certain_int(50),
            mem_gib=certain_float(400.0),
            disk_gib=certain_float(0.0),
            network_mbps=certain_float(8000.0),
        )
        req_plan = CapacityPlan(
            requirements=Requirements(zonal=[cass_req, evcache_req]),
            candidate_clusters=Clusters(annual_costs={}, zonal=[]),
        )
        cluster_plan = _wrap_cluster(
            ZoneClusterCapacity(
                cluster_type="cassandra",
                count=1,
                instance=_create_instance(
                    cpu=105,
                    ram_gib=200.0,
                    net_mbps=5000.0,
                    drive=Drive(name="d", size_gib=1000),
                ),
            )
        )
        result = compare_plans(
            cluster_plan,
            req_plan,
            strategy=ComparisonStrategy.requirements,
        )
        # CPU baseline should be 100 (cassandra req), not 50 (evcache req)
        assert result.cpu.baseline_value == 100.0
        assert result.cpu.is_equivalent

    def test_cassandra_64_node_scenario(self):
        """Motivating case: 64-node i3en cluster vs plan with different shape.

        Plan requires 500 ref-cores total.
        Cluster: 64 × i3en.xlarge (4 vCPU @ 3.1 GHz, IPC 1.0)
          → 64 × 4 = 256 actual cores
          → 256 × 3.1/2.3 = 345.0 ref-cores
        ratio = 345/500 = 0.69 → under-provisioned
        """
        req_plan = _make_requirement_plan(
            req_cpu=500,
            req_mem=4096.0,
            req_disk=64000.0,
            req_net=640000.0,
        )
        i3en = _create_instance(
            name="i3en.xlarge",
            cpu=4,
            cpu_ghz=3.1,
            cpu_ipc_scale=1.0,
            ram_gib=32.0,
            net_mbps=25000,
            drive=Drive(name="nvme", size_gib=2500),
        )
        cluster_plan = _wrap_cluster(
            ZoneClusterCapacity(
                cluster_type="cassandra",
                count=64,
                instance=i3en,
            )
        )
        result = compare_plans(
            cluster_plan,
            req_plan,
            strategy=ComparisonStrategy.requirements,
        )
        assert not result.cpu.is_equivalent
        assert result.cpu.exceeds_lower_bound
        expected_ref_cores = 64 * 4 * (3.1 / 2.3)
        assert result.cpu.comparison_value == pytest.approx(expected_ref_cores)
        assert result.cpu.baseline_value == 500.0

    def test_over_provisioned(self):
        """Cluster with more CPU than requirement → exceeds_upper_bound.

        Plan requires 100 ref-cores. Cluster has 150 cores at same GHz/IPC.
        ratio = 150/100 = 1.5, outside default ±10% tolerance.
        """
        req_plan = _make_requirement_plan(req_cpu=100)
        cluster_plan = _wrap_cluster(
            ZoneClusterCapacity(
                cluster_type="cassandra",
                count=1,
                instance=_create_instance(
                    cpu=150,
                    ram_gib=200.0,
                    net_mbps=5000.0,
                    drive=Drive(name="d", size_gib=1000),
                ),
            )
        )
        result = compare_plans(
            cluster_plan,
            req_plan,
            strategy=ComparisonStrategy.requirements,
        )
        assert not result.is_equivalent
        assert result.cpu.exceeds_upper_bound
        assert result.cpu.ratio == pytest.approx(1.5)

    def test_empty_baseline_returns_empty_result(self):
        """Baseline plan with no clusters → empty PlanComparisonResult."""
        req_plan = _make_requirement_plan(req_cpu=100)
        empty_plan = CapacityPlan(
            requirements=Requirements(),
            candidate_clusters=Clusters(annual_costs={}, zonal=[], regional=[]),
        )
        result = compare_plans(
            empty_plan,
            req_plan,
            strategy=ComparisonStrategy.requirements,
        )
        assert result.is_equivalent  # No comparisons = vacuously true
        assert len(result.comparisons) == 0

    def test_effective_disk_from_cluster_params(self):
        """Cluster with EFFECTIVE_DISK_PER_NODE_GIB uses that for disk.

        When cluster_params contains effective_disk_per_node_gib, disk
        should be computed from that value × count, not from the drive.
        """
        req_plan = _make_requirement_plan(req_disk=500.0)
        cluster_plan = _wrap_cluster(
            ZoneClusterCapacity(
                cluster_type="cassandra",
                count=10,
                instance=_create_instance(
                    cpu=100,
                    ram_gib=200.0,
                    net_mbps=5000.0,
                    drive=Drive(name="nvme", size_gib=2000),  # 2000 per node
                ),
                cluster_params={EFFECTIVE_DISK_PER_NODE_GIB: 50.0},
            )
        )
        result = compare_plans(
            cluster_plan,
            req_plan,
            strategy=ComparisonStrategy.requirements,
        )
        # Disk should be 50.0 × 10 = 500, NOT 2000 × 10 = 20000
        assert result.disk.comparison_value == pytest.approx(500.0)
        assert result.disk.ratio == pytest.approx(1.0)

    def test_regional_cluster_matching(self):
        """Regional clusters are found and compared correctly.

        Tests that the chain(zonal, regional) search in the requirements
        strategy works end-to-end with regional clusters.
        """
        req_plan = CapacityPlan(
            requirements=Requirements(
                regional=[
                    CapacityRequirement(
                        requirement_type="read-only-kv",
                        cpu_cores=certain_int(100),
                        mem_gib=certain_float(200.0),
                        disk_gib=certain_float(1000.0),
                        network_mbps=certain_float(5000.0),
                    )
                ]
            ),
            candidate_clusters=Clusters(annual_costs={}, zonal=[], regional=[]),
        )
        cluster_plan = CapacityPlan(
            requirements=Requirements(),
            candidate_clusters=Clusters(
                annual_costs={"read-only-kv": Decimal("0")},
                zonal=[],
                regional=[
                    RegionClusterCapacity(
                        cluster_type="read-only-kv",
                        count=1,
                        instance=_create_instance(
                            cpu=100,
                            ram_gib=200.0,
                            net_mbps=5000.0,
                            drive=Drive(name="d", size_gib=1000),
                        ),
                    )
                ],
            ),
        )
        result = compare_plans(
            cluster_plan,
            req_plan,
            strategy=ComparisonStrategy.requirements,
        )
        assert result.is_equivalent
        assert result.cpu.ratio == pytest.approx(1.0)

    def test_no_matching_requirement_raises(self):
        """ValueError when no requirement matches the cluster type."""
        req_plan = _make_requirement_plan(cluster_type="evcache")
        cluster_plan = _wrap_cluster(
            ZoneClusterCapacity(
                cluster_type="cassandra",
                count=1,
                instance=_create_instance(),
            )
        )
        with pytest.raises(ValueError, match="No requirement.*cassandra"):
            compare_plans(
                cluster_plan,
                req_plan,
                strategy=ComparisonStrategy.requirements,
            )
