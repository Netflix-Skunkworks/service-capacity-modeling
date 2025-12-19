"""Plan comparison utilities and types for capacity planning.

This module provides functionality to compare two capacity plans and determine
if they are roughly equivalent, with detailed explanations of any differences.
"""

from decimal import Decimal
from functools import lru_cache
from itertools import chain

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


# Tolerance for floating-point comparisons (e.g., strict_tolerance)
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


def strict_tolerance() -> Tolerance:
    """Create a tolerance requiring near-exact match (float epsilon only)."""
    return tolerance(-_FLOAT_TOLERANCE, _FLOAT_TOLERANCE)


def ignore_resource() -> Tolerance:
    """Create a tolerance that ignores a resource (any value acceptable)."""
    return tolerance(float("-inf"), float("inf"))


class ResourceTolerances(ExcludeUnsetModel):
    """Per-resource tolerance configuration for plan comparison.

    Set specific tolerances to override for individual resources.
    Unset resources fall back to `default`.

    Example:
        >>> tolerances = ResourceTolerances(
        ...     cpu=tolerance(-0.05, 0.30),      # Strict on CPU under-provisioning
        ...     cost=symmetric_tolerance(0.15),  # +/- 15% for cost
        ... )
    """

    default: Tolerance = allow_over_provisioning(0.10)
    """Default tolerance for resources not explicitly configured"""

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
        """Percentage difference: (baseline - comparison) / comparison.

        Positive means baseline exceeds comparison (over-provisioned).
        Negative means baseline is below comparison (under-provisioned).
        """
        if self.comparison_value == 0:
            if self.baseline_value == 0:
                return 0.0
            return float("inf") if self.baseline_value > 0 else float("-inf")
        return (self.baseline_value - self.comparison_value) / self.comparison_value

    @property
    def is_equivalent(self) -> bool:
        """True if the difference is within tolerance bounds."""
        return self.difference_percent in self.tolerance

    @property
    def is_over_provisioned(self) -> bool:
        """True if over-provisioned beyond tolerance (diff exceeds upper bound)."""
        return self.difference_percent > self.tolerance.upper

    @property
    def is_under_provisioned(self) -> bool:
        """True if under-provisioned beyond tolerance (diff below lower bound)."""
        return self.difference_percent < self.tolerance.lower

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

    differences: dict[ResourceType, ResourceDifference] = {}
    """Resource differences keyed by resource type"""

    @property
    def cpu(self) -> ResourceDifference:
        """Get CPU comparison result."""
        return self.differences[ResourceType.cpus]

    @property
    def memory(self) -> ResourceDifference:
        """Get memory comparison result."""
        return self.differences[ResourceType.mem_gib]

    @property
    def disk(self) -> ResourceDifference:
        """Get disk comparison result."""
        return self.differences[ResourceType.disk_gib]

    @property
    def network(self) -> ResourceDifference:
        """Get network comparison result."""
        return self.differences[ResourceType.network_mbps]

    @property
    def cost(self) -> ResourceDifference:
        """Get cost comparison result."""
        return self.differences[ResourceType.cost]

    def get_out_of_tolerance(self) -> list[ResourceDifference]:
        """Get only differences that exceed tolerance bounds."""
        return [d for d in self.differences.values() if not d.is_equivalent]


def _aggregate_resources(plan: CapacityPlan) -> dict[ResourceType, float]:
    """Aggregate resource requirements from all zonal and regional requirements."""
    totals: dict[ResourceType, float] = {
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
        >>> if result.cpu.is_over_provisioned:
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

    def make_diff(
        resource: ResourceType, baseline_val: float, comparison_val: float
    ) -> ResourceDifference:
        return ResourceDifference(
            resource=resource,
            baseline_value=baseline_val,
            comparison_value=comparison_val,
            tolerance=tolerances.get_tolerance(resource),
        )

    differences = {
        ResourceType.cost: make_diff(ResourceType.cost, baseline_cost, comparison_cost),
        ResourceType.cpus: make_diff(
            ResourceType.cpus,
            baseline_resources[ResourceType.cpus],
            comparison_resources[ResourceType.cpus],
        ),
        ResourceType.mem_gib: make_diff(
            ResourceType.mem_gib,
            baseline_resources[ResourceType.mem_gib],
            comparison_resources[ResourceType.mem_gib],
        ),
        ResourceType.disk_gib: make_diff(
            ResourceType.disk_gib,
            baseline_resources[ResourceType.disk_gib],
            comparison_resources[ResourceType.disk_gib],
        ),
        ResourceType.network_mbps: make_diff(
            ResourceType.network_mbps,
            baseline_resources[ResourceType.network_mbps],
            comparison_resources[ResourceType.network_mbps],
        ),
    }

    return PlanComparisonResult(
        is_equivalent=all(d.is_equivalent for d in differences.values()),
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
    requirement_type: str,
) -> CapacityPlan:
    """Create a synthetic CapacityPlan from a CurrentClusterCapacity.

    This allows comparison between an existing deployment and a recommended
    plan using the compare_plans() function.

    Args:
        cluster: The current cluster capacity
        is_zonal: True if this is a zonal cluster, False for regional
        region: AWS region for looking up drive pricing
        requirement_type: Label for the capacity requirement (e.g., "cassandra")

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
        requirement_type=requirement_type,
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


def extract_baseline_plan(
    desires: CapacityDesires, region: str, requirement_type: str = "baseline"
) -> CapacityPlan:
    """Extract baseline plan from current clusters in desires.

    Args:
        desires: The capacity desires (must contain current_clusters)
        region: AWS region for looking up drive pricing from hardware shapes
        requirement_type: Label for the capacity requirement (default: "baseline")

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
        return _create_plan_from_current_cluster(
            zonal[0], is_zonal=True, region=region, requirement_type=requirement_type
        )
    else:
        return _create_plan_from_current_cluster(
            regional[0],
            is_zonal=False,
            region=region,
            requirement_type=requirement_type,
        )
