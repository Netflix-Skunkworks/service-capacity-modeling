"""Plan comparison utilities and types for capacity planning.

This module provides functionality to compare two capacity plans and determine
if they are roughly equivalent, with detailed explanations of any differences.

Example usage::

    from service_capacity_modeling.models.plan_comparison import (
        compare_plans,
        extract_baseline_plan,
        ignore_resource,
        lte,
        plus_or_minus,
        ResourceTolerances,
    )

    # Get recommendation from planner
    cap_plan = planner.plan_certain(
        model_name="org.netflix.cassandra",
        region="us-east-1",
        desires=desires,
    )
    recommendation = cap_plan.least_regret[0]

    # Get current deployment as baseline
    baseline = extract_baseline_plan(desires, region="us-east-1")

    # Compare with custom tolerances
    result = compare_plans(
        baseline,
        recommendation,
        tolerances=ResourceTolerances(
            cost=ignore_resource(),        # Don't care about cost
            cpu=lte(1.05),                 # CPU can be at most 5% over baseline
            disk=plus_or_minus(0.10),      # Storage within ±10%
        ),
    )

    if result.is_equivalent:
        print("Current capacity is sufficient")
    else:
        print("Capacity adjustments needed:")
        for diff in result.get_out_of_tolerance():
            print(f"  - {diff}")
"""

from decimal import Decimal
from functools import lru_cache
from itertools import chain

from pydantic import ConfigDict

from service_capacity_modeling.enum_utils import enum_docstrings
from service_capacity_modeling.enum_utils import StrEnum
from typing import TYPE_CHECKING
from typing import Any

from service_capacity_modeling.hardware import shapes
from service_capacity_modeling.interface import (
    CapacityDesires,
    CapacityPlan,
    CapacityRequirement,
    Clusters,
    CurrentClusterCapacity,
    default_reference_shape,
    ExcludeUnsetModel,
    Instance,
    RegionClusterCapacity,
    RegionContext,
    Requirements,
    ServiceCapacity,
    ZoneClusterCapacity,
    certain_float,
    certain_int,
)

if TYPE_CHECKING:
    from service_capacity_modeling.models import CapacityModel


@enum_docstrings
class ResourceType(StrEnum):
    """Resource types that can be compared between capacity plans."""

    cost = "cost"
    """Annual cost in dollars"""

    cpus = "cpus"
    """CPU cores required"""

    mem_gib = "mem_gib"
    """Memory in GiB"""

    disk_gib = "disk_gib"
    """Disk storage in GiB"""

    network_mbps = "network_mbps"
    """Network bandwidth in Mbps"""


# Tolerance for floating-point comparisons (e.g., exact_match)
_FLOAT_TOLERANCE = 1e-9


class Tolerance(ExcludeUnsetModel):
    """Tolerance bounds for comparison / baseline ratio.

    The bounds define the acceptable range for the ratio:
    - ratio > 1.0 = comparison exceeds baseline (need more)
    - ratio < 1.0 = baseline exceeds comparison (have extra capacity)
    - ratio = 1.0 = exact match

    Examples:
        Tolerance(lower=0.0, upper=1.1)            # lte(1.1) - ratio ≤ 1.1
        Tolerance(lower=0.9, upper=float('inf'))   # gte(0.9) - ratio ≥ 0.9
        Tolerance(lower=0.9, upper=1.1)            # tolerance(0.9, 1.1)
    """

    lower: float = 1.0
    """Lower bound for ratio (< 1.0 means can have extra capacity)"""

    upper: float = 1.0
    """Upper bound for ratio (> 1.0 means requirement can exceed baseline)"""

    model_config = ConfigDict(frozen=True)

    def __contains__(self, ratio: float) -> bool:
        """Check if a ratio is within tolerance bounds.

        Args:
            ratio: comparison / baseline

        Returns:
            True if within bounds (equivalent), False otherwise
        """
        return self.lower <= ratio <= self.upper


# -----------------------------------------------------------------------------
# Tolerance helper functions
# -----------------------------------------------------------------------------


@lru_cache(256)
def tolerance(lower: float, upper: float) -> Tolerance:
    """Create a tolerance with explicit ratio bounds.

    Args:
        lower: Lower bound for ratio (e.g., 0.9 means comparison ≥ 0.9× baseline)
        upper: Upper bound for ratio (e.g., 1.1 means comparison ≤ 1.1× baseline)

    Returns:
        Tolerance with specified bounds

    Example:
        >>> t = tolerance(0.9, 1.1)  # Allow 0.9× to 1.1× baseline
        >>> 0.95 in t  # 0.95× baseline: True
        >>> 1.05 in t  # 1.05× baseline: True
        >>> 1.15 in t  # 1.15× baseline: False
    """
    return Tolerance(lower=lower, upper=upper)


def lte(upper: float) -> Tolerance:
    """Ratio must be ≤ upper bound.

    Use this to limit how much the requirement can exceed the baseline.
    Any amount of extra capacity (ratio < 1.0) is acceptable.

    Args:
        upper: Upper bound as a ratio (e.g., 1.1 means comparison ≤ 1.1× baseline).

    Example:
        With baseline=100:

        >>> tol = lte(1.1)  # ratio must be ≤ 1.1

        comparison=110 → ratio = 1.1 → OK (at boundary)
        comparison=111 → ratio = 1.11 → NOT OK
        comparison=90  → ratio = 0.9 → OK (extra capacity)
    """
    return tolerance(0.0, upper)


def gte(lower: float) -> Tolerance:
    """Ratio must be ≥ lower bound.

    Use this to limit how much extra capacity is acceptable.
    Any amount of requirement exceeding baseline (ratio > 1.0) is acceptable.

    Args:
        lower: Lower bound as a ratio (e.g., 0.9 means comparison ≥ 0.9× baseline).

    Example:
        With baseline=100:

        >>> tol = gte(0.9)  # ratio must be ≥ 0.9

        comparison=90  → ratio = 0.9 → OK (at boundary)
        comparison=89  → ratio = 0.89 → NOT OK (too much extra)
        comparison=110 → ratio = 1.1 → OK (need more)
    """
    return tolerance(lower, float("inf"))


def plus_or_minus(percent: float) -> Tolerance:
    """Create a symmetric tolerance of ±percent around 1.0.

    Args:
        percent: The tolerance as a decimal (e.g., 0.10 for ±10%).

    Example:
        >>> tol = plus_or_minus(0.10)  # ratio must be 0.9 to 1.1

        comparison=110, baseline=100 → ratio = 1.1 → OK (at boundary)
        comparison=89, baseline=100 → ratio = 0.89 → NOT OK
    """
    return tolerance(1.0 - percent, 1.0 + percent)


def exact_match() -> Tolerance:
    """Create a tolerance requiring exact match (ratio = 1.0)."""
    return tolerance(1.0 - _FLOAT_TOLERANCE, 1.0 + _FLOAT_TOLERANCE)


def ignore_resource() -> Tolerance:
    """Create a tolerance that ignores a resource (any ratio acceptable)."""
    return tolerance(0.0, float("inf"))


class ResourceTolerances(ExcludeUnsetModel):
    """Per-resource tolerance configuration for plan comparison.

    Set specific tolerances to override for individual resources.
    Unset resources fall back to `default`.

    Example:
        >>> tolerances = ResourceTolerances(
        ...     default=lte(1.1),          # comparison ≤ 1.1× baseline
        ...     cpu=lte(1.05),             # Stricter for CPU
        ...     cost=ignore_resource(),    # Don't care about cost
        ... )
    """

    default: Tolerance = plus_or_minus(0.10)
    """Default tolerance: ±10% (ratio must be 0.9× to 1.1× baseline)"""

    cost: Tolerance | None = None
    """Tolerance for annual cost comparison"""

    cpu: Tolerance | None = None
    """Tolerance for CPU cores comparison"""

    memory: Tolerance | None = None
    """Tolerance for memory (GiB) comparison"""

    disk: Tolerance | None = None
    """Tolerance for disk (GiB) comparison"""

    network: Tolerance | None = None
    """Tolerance for network (Mbps) comparison"""

    def get_tolerance(self, resource: ResourceType) -> Tolerance:
        """Get tolerance for a resource type, falling back to default."""
        match resource:
            case ResourceType.cost:
                return self.cost or self.default
            case ResourceType.cpus:
                return self.cpu or self.default
            case ResourceType.mem_gib:
                return self.memory or self.default
            case ResourceType.disk_gib:
                return self.disk or self.default
            case ResourceType.network_mbps:
                return self.network or self.default
            case _:
                raise ValueError(f"Unknown resource type: {resource}")


class ResourceComparison(ExcludeUnsetModel):
    """Represents a comparison of a resource between baseline and comparison plans.

    Properties:
    - is_equivalent: ratio is within tolerance bounds
    - exceeds_upper_bound: ratio > upper bound (need more than tolerance allows)
    - exceeds_lower_bound: ratio < lower bound (have more extra than tolerance allows)
    """

    resource: ResourceType
    """Resource type being compared"""

    baseline_value: float
    """Value from the current deployment"""

    comparison_value: float
    """Value from the recommendation"""

    tolerance: Tolerance
    """The tolerance bounds that were applied for this comparison"""

    @property
    def ratio(self) -> float:
        """Ratio: comparison / baseline.

        - ratio > 1.0 = comparison exceeds baseline (need more)
        - ratio < 1.0 = baseline exceeds comparison (have extra capacity)
        - ratio = 1.0 = exact match

        Examples:
        - Baseline=100, Comparison=110 → ratio = 1.1 (need 10% more)
        - Baseline=100, Comparison=90 → ratio = 0.9 (have 10% extra)
        """
        if self.baseline_value == 0:
            if self.comparison_value == 0:
                return 1.0  # Both zero = exact match
            return float("inf")  # comparison > 0, baseline = 0
        return self.comparison_value / self.baseline_value

    @property
    def is_equivalent(self) -> bool:
        """True if the ratio is within tolerance bounds."""
        return self.ratio in self.tolerance

    @property
    def exceeds_lower_bound(self) -> bool:
        """True if ratio < lower bound (too much extra capacity)."""
        return self.ratio < self.tolerance.lower

    @property
    def exceeds_upper_bound(self) -> bool:
        """True if ratio > upper bound (requirement exceeds tolerance)."""
        return self.ratio > self.tolerance.upper

    def __str__(self) -> str:
        """Human-readable explanation of the comparison."""
        if self.tolerance.upper == float("inf"):
            bounds_str = f"≥ {self.tolerance.lower:.2f}×"
        elif self.tolerance.lower == 0.0:
            bounds_str = f"≤ {self.tolerance.upper:.2f}×"
        else:
            bounds_str = f"{self.tolerance.lower:.2f}× to {self.tolerance.upper:.2f}×"

        if self.is_equivalent:
            return (
                f"{self.resource.value}: {self.ratio:.2f}× "
                f"(within tolerance: {bounds_str})"
            )
        else:
            bound = "lower" if self.exceeds_lower_bound else "upper"
            return (
                f"{self.resource.value}: exceeds {bound} bound, "
                f"ratio={self.ratio:.2f}× (baseline={self.baseline_value:.2f}, "
                f"comparison={self.comparison_value:.2f}, tolerance: {bounds_str})"
            )


class PlanComparisonResult(ExcludeUnsetModel):
    """Result of comparing two capacity plans for equivalence."""

    is_equivalent: bool
    """True if plans are within tolerance, False if significant differences"""

    comparisons: dict[ResourceType, ResourceComparison] = {}
    """Resource comparisons keyed by resource type"""

    @property
    def cpu(self) -> ResourceComparison:
        """Get CPU comparison result."""
        return self.comparisons[ResourceType.cpus]

    @property
    def memory(self) -> ResourceComparison:
        """Get memory comparison result."""
        return self.comparisons[ResourceType.mem_gib]

    @property
    def disk(self) -> ResourceComparison:
        """Get disk comparison result."""
        return self.comparisons[ResourceType.disk_gib]

    @property
    def network(self) -> ResourceComparison:
        """Get network comparison result."""
        return self.comparisons[ResourceType.network_mbps]

    @property
    def cost(self) -> ResourceComparison:
        """Get cost comparison result."""
        return self.comparisons[ResourceType.cost]

    def get_out_of_tolerance(self) -> list[ResourceComparison]:
        """Get only comparisons that exceed tolerance bounds."""
        return [c for c in self.comparisons.values() if not c.is_equivalent]


def to_reference_cores(core_count: float, instance: Instance) -> float:
    """Convert instance cores to reference-equivalent cores.

    This is the inverse of normalize_cores() from models.common. While
    normalize_cores answers "how many target cores to match N reference cores",
    this answers "how many reference cores is N instance cores equivalent to".

    Mathematically equivalent to:
        normalize_cores(core_count, target=default_reference_shape, reference=instance)

    but returns float instead of ceiling int. We need float precision for
    accurate ratio comparisons - ceiling would distort ratios by up to ~3%.

    See normalize_cores() in models/common.py for the original implementation.

    Args:
        core_count: Number of cores on the instance
        instance: The instance shape (with cpu_ghz and cpu_ipc_scale)

    Returns:
        Equivalent cores on default_reference_shape (2.3 GHz, IPC 1.0)

    Example:
        # 32 cores on a 2.4 GHz instance = 33.4 reference cores
        to_reference_cores(32, instance_at_2_4_ghz)  # → 33.39
    """
    instance_speed = instance.cpu_ghz * instance.cpu_ipc_scale
    ref_speed = default_reference_shape.cpu_ghz * default_reference_shape.cpu_ipc_scale
    return core_count * (instance_speed / ref_speed)


def _aggregate_resources(plan: CapacityPlan) -> dict[ResourceType, float]:
    """Aggregate resource values from a plan, normalizing CPU to reference shape.

    CPU is computed from candidate_clusters and normalized to default_reference_shape
    using IPC and frequency factors. This ensures consistent comparison even when
    plans use different instance types with varying CPU performance characteristics.

    Memory, disk, and network are summed from requirements (no normalization needed).
    """
    totals: dict[ResourceType, float] = {
        ResourceType.cpus: 0.0,
        ResourceType.mem_gib: 0.0,
        ResourceType.disk_gib: 0.0,
        ResourceType.network_mbps: 0.0,
    }

    # CPU: compute from candidate_clusters, normalized to reference shape
    for cluster in chain(
        plan.candidate_clusters.zonal, plan.candidate_clusters.regional
    ):
        totals[ResourceType.cpus] += to_reference_cores(
            cluster.instance.cpu * cluster.count, cluster.instance
        )

    # Other resources: sum from requirements
    for req in chain(plan.requirements.zonal, plan.requirements.regional):
        totals[ResourceType.mem_gib] += req.mem_gib.mid
        totals[ResourceType.disk_gib] += req.disk_gib.mid
        totals[ResourceType.network_mbps] += req.network_mbps.mid

    return totals


def compare_plans(
    baseline: CapacityPlan,
    comparison: CapacityPlan,
    tolerances: ResourceTolerances | None = None,
) -> PlanComparisonResult:
    """Compare two capacity plans to determine if they are roughly equivalent.

    This function compares plans across multiple dimensions (cost, CPU, memory,
    disk, network) and determines if the differences are significant based on
    the provided tolerance bounds.

    Args:
        baseline: The reference plan (e.g., current production deployment)
        comparison: The plan to compare against baseline (e.g., new recommendation)
        tolerances: Per-resource tolerance configuration. If None, uses defaults.

    Returns:
        PlanComparisonResult containing:
        - is_equivalent: True if all resources within tolerance, False otherwise
        - differences: Dict of ResourceDifference for ALL resources (use
          get_out_of_tolerance() to filter to only problematic ones)

    Example:
        >>> result = compare_plans(baseline, recommended)
        >>> if result.cpu.exceeds_lower_bound:
        ...     print("Current has excess CPU capacity")
        >>> for diff in result.get_out_of_tolerance():
        ...     print(diff)  # Human-readable explanation
    """
    if tolerances is None:
        tolerances = ResourceTolerances()

    baseline_cost = float(baseline.candidate_clusters.total_annual_cost)
    comparison_cost = float(comparison.candidate_clusters.total_annual_cost)
    baseline_resources = _aggregate_resources(baseline)
    comparison_resources = _aggregate_resources(comparison)

    def make_comparison(
        resource: ResourceType, baseline_val: float, comparison_val: float
    ) -> ResourceComparison:
        return ResourceComparison(
            resource=resource,
            baseline_value=baseline_val,
            comparison_value=comparison_val,
            tolerance=tolerances.get_tolerance(resource),
        )

    comparisons = {
        ResourceType.cost: make_comparison(
            ResourceType.cost, baseline_cost, comparison_cost
        ),
        ResourceType.cpus: make_comparison(
            ResourceType.cpus,
            baseline_resources[ResourceType.cpus],
            comparison_resources[ResourceType.cpus],
        ),
        ResourceType.mem_gib: make_comparison(
            ResourceType.mem_gib,
            baseline_resources[ResourceType.mem_gib],
            comparison_resources[ResourceType.mem_gib],
        ),
        ResourceType.disk_gib: make_comparison(
            ResourceType.disk_gib,
            baseline_resources[ResourceType.disk_gib],
            comparison_resources[ResourceType.disk_gib],
        ),
        ResourceType.network_mbps: make_comparison(
            ResourceType.network_mbps,
            baseline_resources[ResourceType.network_mbps],
            comparison_resources[ResourceType.network_mbps],
        ),
    }

    return PlanComparisonResult(
        is_equivalent=all(c.is_equivalent for c in comparisons.values()),
        comparisons=comparisons,
    )


# -----------------------------------------------------------------------------
# Baseline extraction helper
# -----------------------------------------------------------------------------


def _get_disk_gib_from_cluster(cluster: CurrentClusterCapacity) -> float:
    """Get disk size in GiB from a current cluster capacity.

    Handles both attached storage (cluster_drive) and local storage
    (instance.drive). Returns 0 if no storage is configured.

    Args:
        cluster: The current cluster capacity to inspect

    Returns:
        Disk size in GiB per node, or 0 if no storage
    """
    # Check for attached storage first (e.g., EBS)
    if cluster.cluster_drive is not None:
        return cluster.cluster_drive.size_gib or 0

    # Fall back to local storage (e.g., NVMe SSD on instance)
    if (
        cluster.cluster_instance is not None
        and cluster.cluster_instance.drive is not None
    ):
        return cluster.cluster_instance.drive.size_gib or 0

    return 0


# pylint: disable-next=too-many-positional-arguments,too-many-locals
def _create_plan_from_current_cluster(
    cluster: CurrentClusterCapacity,
    is_zonal: bool,
    region: str,
    requirement_type: str,
    model: "CapacityModel | None" = None,
    context: RegionContext | None = None,
    desires: CapacityDesires | None = None,
    extra_model_arguments: dict[str, Any] | None = None,
) -> CapacityPlan:
    """Create a synthetic CapacityPlan from a CurrentClusterCapacity.

    This allows comparison between an existing deployment and a recommended
    plan using the compare_plans() function.

    Args:
        cluster: The current cluster capacity
        is_zonal: True if this is a zonal cluster, False for regional
        region: AWS region for looking up drive pricing and instance specs
        requirement_type: Label for the capacity requirement (e.g., "cassandra")
        model: Optional model class for calculating service costs (network, backup).
            When provided with context and desires, service_costs() is called.
        context: Optional RegionContext with service pricing. Required for
            service cost calculation.
        desires: Optional CapacityDesires for service cost calculation.
            Required for service cost calculation.
        extra_model_arguments: Optional model arguments (e.g., copies_per_region)

    Returns:
        A CapacityPlan representing the current deployment with cost calculated

    Raises:
        ValueError: If instance cannot be resolved from cluster_instance or
            cluster_instance_name
    """
    extra_model_arguments = extra_model_arguments or {}

    # Resolve instance: prefer cluster_instance, fall back to lookup by name
    if cluster.cluster_instance is not None:
        instance = cluster.cluster_instance
    else:
        regional_instances = shapes.region(region).instances
        instance_name = cluster.cluster_instance_name
        if instance_name not in regional_instances:
            raise ValueError(
                f"Cannot resolve instance '{instance_name}' in region '{region}'. "
                f"Either provide cluster_instance directly or use a valid "
                f"instance name from the hardware catalog."
            )
        instance = regional_instances[instance_name]

    # Validate that instance has pricing information
    if instance.annual_cost == 0:
        raise ValueError(
            f"Instance '{instance.name}' has annual_cost=0. "
            f"This would result in incorrect baseline cost calculations. "
            f"Either use an instance from the hardware catalog (which includes "
            f"pricing) or ensure cluster_instance.annual_cost is set."
        )

    count = int(cluster.cluster_instance_count.mid)
    disk_gib_per_node = _get_disk_gib_from_cluster(cluster)

    # Build requirements from instance specs * count
    # Note: CPU normalization for IPC/frequency is handled in _aggregate_resources()
    requirement = CapacityRequirement(
        requirement_type=requirement_type,
        reference_shape=instance,
        cpu_cores=certain_int(instance.cpu * count),
        mem_gib=certain_float(instance.ram_gib * count),
        network_mbps=certain_float(instance.net_mbps * count),
        disk_gib=certain_float(disk_gib_per_node * count),
    )

    if is_zonal:
        requirements = Requirements(zonal=[requirement])
    else:
        requirements = Requirements(regional=[requirement])

    # Calculate instance cost
    instance_cost = instance.annual_cost * count

    # Calculate drive cost using priced drive from hardware shapes
    drive_cost = 0.0
    attached_drives = []
    if cluster.cluster_drive is not None:
        # Get the priced version of the drive from hardware shapes
        regional_drives = shapes.region(region).drives
        drive_name = cluster.cluster_drive.name
        if drive_name not in regional_drives:
            raise ValueError(
                f"Cannot price drive '{drive_name}' in region '{region}'. "
                f"Available drives: {sorted(regional_drives.keys())}. "
                "Ensure the drive name matches a known drive type (e.g., 'gp3', 'io2')."
            )
        priced_drive = regional_drives[drive_name].model_copy()
        priced_drive.size_gib = cluster.cluster_drive.size_gib or 0
        priced_drive.read_io_per_s = cluster.cluster_drive.read_io_per_s
        priced_drive.write_io_per_s = cluster.cluster_drive.write_io_per_s
        drive_cost = priced_drive.annual_cost * count
        attached_drives = [priced_drive]

    # Calculate service costs (network, backup, etc.) if model provided
    services: list[ServiceCapacity] = []
    service_cost = 0.0
    if model is not None and context is not None and desires is not None:
        services = model.service_costs(
            service_type=requirement_type,
            context=context,
            desires=desires,
            requirement=requirement,
            extra_model_arguments=extra_model_arguments,
        )
        service_cost = sum(s.annual_cost for s in services)

    # Note: total_cost = instance_cost + drive_cost + service_cost
    # This is reflected in the sum of annual_costs dict values
    _ = service_cost  # Used indirectly via services list

    # Build annual_costs dict with breakdown
    annual_costs: dict[str, Decimal] = {}
    cluster_key = "current.zonal" if is_zonal else "current.regional"
    annual_costs[cluster_key] = Decimal(str(instance_cost + drive_cost))
    for service in services:
        annual_costs[service.service_type] = Decimal(str(service.annual_cost))

    if is_zonal:
        zonal_capacity = ZoneClusterCapacity(
            cluster_type="current",
            count=count,
            instance=instance,
            attached_drives=attached_drives,
            annual_cost=instance_cost + drive_cost,
        )
        clusters = Clusters(
            annual_costs=annual_costs,
            zonal=[zonal_capacity],
            services=services,
        )
    else:
        regional_capacity = RegionClusterCapacity(
            cluster_type="current",
            count=count,
            instance=instance,
            attached_drives=attached_drives,
            annual_cost=instance_cost + drive_cost,
        )
        clusters = Clusters(
            annual_costs=annual_costs,
            regional=[regional_capacity],
            services=services,
        )

    return CapacityPlan(
        requirements=requirements,
        candidate_clusters=clusters,
    )


def extract_baseline_plan(  # pylint: disable=too-many-positional-arguments
    desires: CapacityDesires,
    region: str,
    requirement_type: str = "baseline",
    model: "CapacityModel | None" = None,
    context: RegionContext | None = None,
    extra_model_arguments: dict[str, Any] | None = None,
) -> CapacityPlan:
    """Extract baseline plan from current clusters in desires.

    Args:
        desires: The capacity desires (must contain current_clusters)
        region: AWS region for looking up drive pricing from hardware shapes
        requirement_type: Label for the capacity requirement (default: "baseline")
        model: Optional model class for calculating service costs (network, backup).
            If provided along with context, service costs will be included.
        context: Optional RegionContext with service pricing. Required if model is
            provided for service cost calculation.
        extra_model_arguments: Optional model arguments (e.g., copies_per_region)

    Returns:
        CapacityPlan representing the current deployment

    Raises:
        ValueError: If desires.current_clusters is None or empty
        ValueError: If cluster_instance is not resolved in current_clusters
    """
    if desires.current_clusters is None:
        raise ValueError(
            "Cannot extract baseline: desires.current_clusters is None. "
            "This function requires an existing deployment to compare against."
        )

    zonal = desires.current_clusters.zonal
    regional = desires.current_clusters.regional

    if not zonal and not regional:
        raise ValueError(
            "Cannot extract baseline: desires.current_clusters has no zonal "
            "or regional clusters defined."
        )

    if zonal and regional:
        raise ValueError(
            "Cannot extract baseline: desires.current_clusters has both zonal "
            "and regional clusters. Only one type is supported for comparison."
        )

    # Determine which cluster list to use
    clusters = zonal if zonal else regional
    is_zonal = bool(zonal)

    # Validate homogeneity: all clusters must use the same instance type
    instance_names = {c.cluster_instance_name for c in clusters}
    if len(instance_names) > 1:
        raise ValueError(
            f"Cannot extract baseline: clusters are heterogeneous with different "
            f"instance types: {sorted(instance_names)}. "
            f"Baseline extraction requires all clusters to use the same instance type."
        )

    # Validate homogeneity: all clusters must have the same instance count
    instance_counts = {int(c.cluster_instance_count.mid) for c in clusters}
    if len(instance_counts) > 1:
        raise ValueError(
            f"Cannot extract baseline: clusters have different instance counts: "
            f"{sorted(instance_counts)}. "
            f"Baseline extraction requires all clusters to have the same count."
        )

    return _create_plan_from_current_cluster(
        clusters[0],
        is_zonal=is_zonal,
        region=region,
        requirement_type=requirement_type,
        model=model,
        context=context,
        desires=desires,
        extra_model_arguments=extra_model_arguments or {},
    )
