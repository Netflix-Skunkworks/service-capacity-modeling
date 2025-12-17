"""Plan comparison utilities and types for capacity planning.

This module provides functionality to compare two capacity plans and determine
if they are roughly equivalent, with detailed explanations of any differences.
"""

from decimal import Decimal
from functools import lru_cache
from itertools import chain
from typing import Any, Dict, List, Optional

from pydantic import ConfigDict

from service_capacity_modeling.enum_utils import enum_docstrings
from service_capacity_modeling.enum_utils import StrEnum
from service_capacity_modeling.hardware import shapes
from service_capacity_modeling.interface import (
    CapacityDesires,
    CapacityPlan,
    CapacityRequirement,
    Clusters,
    CurrentClusterCapacity,
    default_reference_shape,
    ExcludeUnsetModel,
    RegionClusterCapacity,
    Requirements,
    ZoneClusterCapacity,
    certain_float,
)
from service_capacity_modeling.models.common import normalize_cores


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


@enum_docstrings
class ComparisonResult(StrEnum):
    """Result of comparing current deployment to recommendation."""

    gt = "gt"
    """Current > recommendation (over-provisioned, excess capacity)"""

    lt = "lt"
    """Current < recommendation (under-provisioned, needs more)"""

    equivalent = "equivalent"
    """Current ≈ recommendation (within floating-point tolerance)"""


# Tolerance for floating-point comparisons when determining ordering
_FLOAT_TOLERANCE = 1e-9


class Tolerance(ExcludeUnsetModel):
    """Tolerance bounds as signed percentage values on a number line.

    The bounds represent the acceptable range for (comparison - baseline) / baseline.
    Negative values indicate under-provisioning, positive values indicate
    over-provisioning.

    Examples:
        Tolerance(lower=-0.10, upper=0.20)  # -10% to +20%
        Tolerance(lower=0.15, upper=0.30)   # +15% to +30% (must be over by 15-30%)
        Tolerance(lower=-0.05, upper=0.0)   # -5% to 0% (slightly under or exact)
    """

    lower: float = 0.0
    """Lower bound (negative = under-provisioned allowed)"""

    upper: float = 0.0
    """Upper bound (positive = over-provisioned allowed)"""

    model_config = ConfigDict(frozen=True)

    def __contains__(self, difference_percent: float) -> bool:
        """Check if a percentage difference is within tolerance bounds.

        Args:
            difference_percent: (comparison - baseline) / baseline

        Returns:
            True if within bounds (equivalent), False otherwise
        """
        return self.lower <= difference_percent <= self.upper


# -----------------------------------------------------------------------------
# Tolerance helper functions
# -----------------------------------------------------------------------------


@lru_cache(256)
def tolerance(lower: float, upper: float) -> Tolerance:
    """Create a tolerance with explicit lower and upper bounds.

    Args:
        lower: Lower bound (negative for under-provisioning tolerance)
        upper: Upper bound (positive for over-provisioning tolerance)

    Returns:
        Tolerance with specified bounds

    Example:
        >>> t = tolerance(-0.05, 0.20)  # Allow 5% under to 20% over
        >>> -0.03 in t  # 3% under: True
        >>> 0.15 in t   # 15% over: True
        >>> -0.10 in t  # 10% under: False
    """
    return Tolerance(lower=lower, upper=upper)


def symmetric_tolerance(percent: float) -> Tolerance:
    """Create a symmetric tolerance (+/- percent)."""
    return tolerance(-percent, percent)


def allow_over_provisioning(under_percent: float = 0.10) -> Tolerance:
    """Create a tolerance allowing unlimited over-provisioning.

    Over-provisioning means the baseline has MORE resources than the comparison
    (recommendation). This is safe - you just have extra capacity.

    Args:
        under_percent: Maximum allowed under-provisioning (comparison can be
            at most this much larger than baseline). Default 10%.

    Returns:
        Tolerance with lower=-inf (baseline can be arbitrarily larger) and
        upper=under_percent (comparison can only exceed baseline by this much).
    """
    return tolerance(float("-inf"), under_percent)


@lru_cache(maxsize=None)
def strict_tolerance() -> Tolerance:
    """Create a tolerance requiring exact match (no variance allowed)."""
    return tolerance(0.0, 0.0)


@lru_cache(maxsize=None)
def ignore_resource() -> Tolerance:
    """Create a tolerance that ignores a resource (any value acceptable)."""
    return tolerance(float("-inf"), float("inf"))


class ResourceTolerances(ExcludeUnsetModel):
    """Per-resource tolerance configuration for plan comparison.

    All tolerances default to None, meaning the default tolerance will be used.
    Set specific tolerances to override for individual resources.

    Example:
        >>> tolerances = ResourceTolerances(
        ...     cpu=tolerance(-0.05, 0.30),      # Strict on CPU under-provisioning
        ...     cost=symmetric_tolerance(0.15),  # +/- 15% for cost
        ... )
    """

    cost: Optional[Tolerance] = None
    """Tolerance for annual cost comparison"""

    cpu: Optional[Tolerance] = None
    """Tolerance for CPU cores comparison"""

    memory: Optional[Tolerance] = None
    """Tolerance for memory (GiB) comparison"""

    disk: Optional[Tolerance] = None
    """Tolerance for disk (GiB) comparison"""

    network: Optional[Tolerance] = None
    """Tolerance for network (Mbps) comparison"""

    def get_tolerance(self, resource: ResourceType, default: Tolerance) -> Tolerance:
        """Get tolerance for a resource type, falling back to default."""
        match resource:
            case ResourceType.cost:
                return self.cost or default
            case ResourceType.cpus:
                return self.cpu or default
            case ResourceType.mem_gib:
                return self.memory or default
            case ResourceType.disk_gib:
                return self.disk or default
            case ResourceType.network_mbps:
                return self.network or default
            case _:
                raise ValueError(f"Unknown resource type: {resource}")


class ResourceDifference(ExcludeUnsetModel):
    """Represents a difference in a specific resource dimension between two plans.

    Only stores the base values; all derived values (difference_percent,
    comparison_result, is_equivalent) are computed from these.
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
    def difference_percent(self) -> float:
        """Percentage difference: (comparison - baseline) / baseline."""
        if self.baseline_value == 0:
            if self.comparison_value == 0:
                return 0.0
            return float("inf") if self.comparison_value > 0 else float("-inf")
        return (self.comparison_value - self.baseline_value) / self.baseline_value

    @property
    def comparison_result(self) -> ComparisonResult:
        """Result: gt (over-provisioned), lt (under-provisioned), or equivalent."""
        diff = self.difference_percent
        if abs(diff) < _FLOAT_TOLERANCE:
            return ComparisonResult.equivalent
        return ComparisonResult.gt if diff > 0 else ComparisonResult.lt

    @property
    def is_equivalent(self) -> bool:
        """True if the difference is within tolerance bounds."""
        return self.difference_percent in self.tolerance

    @property
    def is_over_provisioned(self) -> bool:
        """True if current > recommendation (excess capacity)."""
        return self.comparison_result == ComparisonResult.gt

    @property
    def is_under_provisioned(self) -> bool:
        """True if current < recommendation (needs more)."""
        return self.comparison_result == ComparisonResult.lt

    def __str__(self) -> str:
        """Human-readable explanation of the difference."""
        if self.tolerance.upper == float("inf"):
            bounds_str = f">= {self.tolerance.lower * 100:.1f}%"
        elif self.tolerance.lower == float("-inf"):
            bounds_str = f"<= {self.tolerance.upper * 100:.1f}%"
        else:
            lower_pct = self.tolerance.lower * 100
            upper_pct = self.tolerance.upper * 100
            bounds_str = f"{lower_pct:.1f}% to {upper_pct:.1f}%"

        if self.is_equivalent:
            return (
                f"{self.resource.value}: {self.difference_percent * 100:+.1f}% "
                f"(within tolerance: {bounds_str})"
            )
        else:
            direction = "over" if self.is_over_provisioned else "under"
            return (
                f"{self.resource.value}: {direction}-provisioned by "
                f"{abs(self.difference_percent) * 100:.1f}% "
                f"(baseline={self.baseline_value:.2f}, "
                f"comparison={self.comparison_value:.2f}, tolerance: {bounds_str})"
            )


class PlanComparisonResult(ExcludeUnsetModel):
    """Result of comparing two capacity plans for equivalence."""

    is_equivalent: bool
    """True if plans are within tolerance, False if significant differences"""

    context: Dict[str, Any] = {}
    """Additional context about the comparison (default tolerance, costs, etc.)"""

    differences: List[ResourceDifference] = []
    """List of resource differences for ALL resources compared"""

    def _get_diff(self, resource: ResourceType) -> ResourceDifference:
        """Get the difference for a specific resource type.

        Args:
            resource: The resource type to look up

        Returns:
            ResourceDifference for the specified resource

        Raises:
            KeyError: If resource not in comparison results
        """
        for d in self.differences:
            if d.resource == resource:
                return d
        raise KeyError(f"Resource {resource} not in comparison results")

    @property
    def cpu(self) -> ResourceDifference:
        """Get CPU comparison result."""
        return self._get_diff(ResourceType.cpus)

    @property
    def memory(self) -> ResourceDifference:
        """Get memory comparison result."""
        return self._get_diff(ResourceType.mem_gib)

    @property
    def disk(self) -> ResourceDifference:
        """Get disk comparison result."""
        return self._get_diff(ResourceType.disk_gib)

    @property
    def network(self) -> ResourceDifference:
        """Get network comparison result."""
        return self._get_diff(ResourceType.network_mbps)

    @property
    def cost(self) -> ResourceDifference:
        """Get cost comparison result."""
        return self._get_diff(ResourceType.cost)

    def get_out_of_tolerance(self) -> List[ResourceDifference]:
        """Get only differences that exceed tolerance bounds.

        Returns:
            List of ResourceDifference where is_equivalent is False
        """
        return [d for d in self.differences if not d.is_equivalent]

    @property
    def out_of_tolerance_resources(self) -> List[ResourceType]:
        """Get list of resource types that are out of tolerance."""
        return [d.resource for d in self.get_out_of_tolerance()]


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------


def _aggregate_resources(plan: CapacityPlan) -> Dict[ResourceType, float]:
    """Aggregate resource requirements from all zonal and regional requirements."""
    totals: Dict[ResourceType, float] = {
        ResourceType.cpus: 0.0,
        ResourceType.mem_gib: 0.0,
        ResourceType.disk_gib: 0.0,
        ResourceType.network_mbps: 0.0,
    }
    for req in chain(plan.requirements.zonal, plan.requirements.regional):
        totals[ResourceType.cpus] += req.cpu_cores.mid
        totals[ResourceType.mem_gib] += req.mem_gib.mid
        totals[ResourceType.disk_gib] += req.disk_gib.mid
        totals[ResourceType.network_mbps] += req.network_mbps.mid
    return totals


def _compare_resource(
    resource: ResourceType,
    baseline_value: float,
    comparison_value: float,
    tolerance: Tolerance,
) -> ResourceDifference:
    """Compare a single resource dimension and return detailed difference.

    Args:
        resource: The resource type being compared
        baseline_value: Value from baseline plan
        comparison_value: Value from comparison plan
        tolerance: Tolerance bounds for this comparison

    Returns:
        ResourceDifference with computed properties for ordering and tolerance
    """
    return ResourceDifference(
        resource=resource,
        baseline_value=baseline_value,
        comparison_value=comparison_value,
        tolerance=tolerance,
    )


def compare_plans(
    baseline: CapacityPlan,
    comparison: CapacityPlan,
    tolerances: Optional[ResourceTolerances] = None,
    default_tolerance: Optional[Tolerance] = None,
) -> PlanComparisonResult:
    """Compare two capacity plans to determine if they are roughly equivalent.

    This function compares plans across multiple dimensions (cost, CPU, memory,
    disk, network) and determines if the differences are significant based on
    the provided tolerance bounds.

    Args:
        baseline: The reference plan (e.g., current production deployment)
        comparison: The plan to compare against baseline (e.g., new recommendation)
        tolerances: Per-resource tolerance configuration. If None, uses
            default_tolerance for all resources.
        default_tolerance: Default tolerance for resources not specified in
            tolerances. Defaults to allow_over_provisioning(0.10) if not specified.

    Returns:
        PlanComparisonResult containing:
        - is_equivalent: True if all resources within tolerance, False otherwise
        - context: Metadata about the comparison (default tolerance, costs)
        - differences: List of ResourceDifference for ALL resources (use
          get_out_of_tolerance() to filter to only problematic ones)

    Example:
        >>> # Simple: use defaults (10% under allowed, unlimited over)
        >>> result = compare_plans(baseline, recommended)
        >>> if result.is_equivalent:
        ...     print("Plans match")
        >>>
        >>> # Custom per-resource tolerances
        >>> tolerances = ResourceTolerances(
        ...     cpu=tolerance(-0.05, 0.30),      # 5% under, 30% over
        ...     cost=symmetric_tolerance(0.15),  # +/- 15%
        ... )
        >>> result = compare_plans(baseline, recommended, tolerances=tolerances)
        >>> for diff in result.get_out_of_tolerance():
        ...     print(f"{diff.resource}: {diff.ordering} by {diff.difference_percent}")
    """
    # Set up defaults
    if default_tolerance is None:
        default_tolerance = allow_over_provisioning(0.10)
    if tolerances is None:
        tolerances = ResourceTolerances()

    differences: List[ResourceDifference] = []

    # Compare cost
    baseline_cost = float(baseline.candidate_clusters.total_annual_cost)
    comparison_cost = float(comparison.candidate_clusters.total_annual_cost)

    cost_tolerance = tolerances.get_tolerance(ResourceType.cost, default_tolerance)
    cost_diff = _compare_resource(
        resource=ResourceType.cost,
        baseline_value=baseline_cost,
        comparison_value=comparison_cost,
        tolerance=cost_tolerance,
    )
    differences.append(cost_diff)

    # Compare resource dimensions
    baseline_resources = _aggregate_resources(baseline)
    comparison_resources = _aggregate_resources(comparison)

    for resource in [
        ResourceType.cpus,
        ResourceType.mem_gib,
        ResourceType.disk_gib,
        ResourceType.network_mbps,
    ]:
        resource_tolerance = tolerances.get_tolerance(resource, default_tolerance)
        resource_diff = _compare_resource(
            resource=resource,
            baseline_value=baseline_resources.get(resource, 0),
            comparison_value=comparison_resources.get(resource, 0),
            tolerance=resource_tolerance,
        )
        differences.append(resource_diff)

    # Determine overall result - equivalent if all resources are within tolerance
    is_equivalent = all(d.is_equivalent for d in differences)

    context: Dict[str, Any] = {
        "default_tolerance": {
            "lower": default_tolerance.lower,
            "upper": default_tolerance.upper,
        },
        "baseline_cost": baseline_cost,
        "comparison_cost": comparison_cost,
    }

    return PlanComparisonResult(
        is_equivalent=is_equivalent,
        context=context,
        differences=differences,
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


def _create_plan_from_current_cluster(
    cluster: CurrentClusterCapacity,
    is_zonal: bool,
    region: str,
) -> CapacityPlan:
    """Create a synthetic CapacityPlan from a CurrentClusterCapacity.

    This allows comparison between an existing deployment and a recommended
    plan using the compare_plans() function.

    Args:
        cluster: The current cluster capacity
        is_zonal: True if this is a zonal cluster, False for regional
        region: AWS region for looking up drive pricing

    Returns:
        A CapacityPlan representing the current deployment with cost calculated
    """
    if cluster.cluster_instance is None:
        raise ValueError(
            "Cannot create plan from current cluster: cluster_instance is None. "
            "Ensure the CurrentClusterCapacity has a resolved cluster_instance."
        )

    instance = cluster.cluster_instance
    count = int(cluster.cluster_instance_count.mid)
    disk_gib_per_node = _get_disk_gib_from_cluster(cluster)

    # Normalize CPU cores to default_reference_shape for consistent comparison
    # with recommended plans (which are already normalized)
    normalized_cpu = normalize_cores(
        instance.cpu * count,
        target_shape=default_reference_shape,
        reference_shape=instance,
    )

    # Build requirements from instance specs * count
    requirement = CapacityRequirement(
        requirement_type="current",
        reference_shape=default_reference_shape,
        cpu_cores=certain_float(normalized_cpu),
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

    total_cost = instance_cost + drive_cost

    if is_zonal:
        zonal_capacity = ZoneClusterCapacity(
            cluster_type="current",
            count=count,
            instance=instance,
            attached_drives=attached_drives,
            annual_cost=total_cost,
        )
        clusters = Clusters(
            annual_costs={"current.zonal": Decimal(str(total_cost))},
            zonal=[zonal_capacity],
        )
    else:
        regional_capacity = RegionClusterCapacity(
            cluster_type="current",
            count=count,
            instance=instance,
            attached_drives=attached_drives,
            annual_cost=total_cost,
        )
        clusters = Clusters(
            annual_costs={"current.regional": Decimal(str(total_cost))},
            regional=[regional_capacity],
        )

    return CapacityPlan(
        requirements=requirements,
        candidate_clusters=clusters,
    )


def extract_baseline_plan(desires: CapacityDesires, region: str) -> CapacityPlan:
    """Extract baseline plan from current clusters in desires.

    Args:
        desires: The capacity desires (must contain current_clusters)
        region: AWS region for looking up drive pricing from hardware shapes

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

    if zonal:
        return _create_plan_from_current_cluster(zonal[0], is_zonal=True, region=region)
    else:
        return _create_plan_from_current_cluster(
            regional[0], is_zonal=False, region=region
        )
