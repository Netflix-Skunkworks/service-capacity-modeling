"""Plan comparison utilities and types for capacity planning.

This module provides functionality to compare two capacity plans and determine
if they are roughly equivalent, with detailed explanations of any differences.

Example usage::

    from service_capacity_modeling.models.plan_comparison import (
        compare_plans,
        ComparisonStrategy,
        gte,
        ignore_resource,
        lte,
        plus_or_minus,
        ResourceTolerances,
    )

    # Get recommendation from planner
    recommendation = planner.plan_certain(
        model_name="org.netflix.cassandra",
        region="us-east-1",
        desires=desires,
    )[0]

    # Get current deployment as baseline
    baseline = planner.extract_baseline_plan(
        model_name="org.netflix.cassandra",
        region="us-east-1",
        desires=desires,  # must have current_clusters populated
        extra_model_arguments={"copies_per_region": 3},
    )

Two comparison strategies are available:

**Provisioned** (default) — compare aggregate cluster resources::

    # "Is the new recommendation similar to what we have deployed?"
    result = compare_plans(
        baseline,
        recommendation,
        tolerances=ResourceTolerances(
            annual_cost=ignore_resource(),  # Don't care about cost
            cpu=lte(1.05),                 # CPU can be at most 5% over baseline
            disk=plus_or_minus(0.10),      # Storage within ±10%
        ),
    )

**Requirements** — compare a plan's clusters against requirements::

    # "Does my current cluster satisfy the requirements without a
    #  significant change in cost?"
    result = compare_plans(
        baseline,                                   # provides clusters
        recommendation,                             # provides requirements
        strategy=ComparisonStrategy.requirements,
        tolerances=ResourceTolerances(
            default=gte(1.0),               # clusters must meet requirements
            annual_cost=plus_or_minus(0.10), # cost within ±10%
        ),
    )

Checking the result::

    if result.is_equivalent:
        print("Current capacity is sufficient")
    else:
        print("Capacity adjustments needed:")
        for diff in result.get_out_of_tolerance():
            print(f"  - {diff}")
"""

from functools import lru_cache
from itertools import chain

from pydantic import computed_field
from pydantic import ConfigDict

from service_capacity_modeling.enum_utils import enum_docstrings
from service_capacity_modeling.enum_utils import StrEnum

from service_capacity_modeling.interface import (
    CapacityPlan,
    CapacityRequirement,
    ClusterCapacity,
    default_reference_shape,
    ExcludeUnsetModel,
    Instance,
)
from service_capacity_modeling.models.common import EFFECTIVE_DISK_PER_NODE_GIB
from service_capacity_modeling.models.common import get_disk_size_gib


@enum_docstrings
class ResourceType(StrEnum):
    """Resource types that can be compared between capacity plans."""

    annual_cost = "annual_cost"
    """Annual cost in dollars"""

    cpu = "cpu"
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
        ...     annual_cost=ignore_resource(),  # Don't care about cost
        ... )
    """

    default: Tolerance = plus_or_minus(0.10)
    """Default tolerance: ±10% (ratio must be 0.9× to 1.1× baseline)"""

    annual_cost: Tolerance | None = None
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
            case ResourceType.annual_cost:
                return self.annual_cost or self.default
            case ResourceType.cpu:
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

    @computed_field(return_type=float)  # type: ignore
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

    @computed_field(return_type=bool)  # type: ignore
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

    comparisons: dict[ResourceType, ResourceComparison] = {}
    """Resource comparisons keyed by resource type"""

    @computed_field(return_type=bool)  # type: ignore
    @property
    def is_equivalent(self) -> bool:
        """True if plans are within tolerance, False if significant differences."""
        return all(c.is_equivalent for c in self.comparisons.values())

    @property
    def cpu(self) -> ResourceComparison:
        """Get CPU comparison result."""
        return self.comparisons[ResourceType.cpu]

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
    def annual_cost(self) -> ResourceComparison:
        """Get annual cost comparison result."""
        return self.comparisons[ResourceType.annual_cost]

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


class ComparisonStrategy(StrEnum):
    """Strategy for how baseline and comparison values are extracted."""

    requirements = "requirements"
    """Baseline's clusters vs comparison's requirements (matched by cluster_type)."""

    provisioned = "provisioned"
    """Aggregate cluster resources from both plans."""


def _find_matching_requirement(
    cluster_type: str, plan: CapacityPlan
) -> CapacityRequirement | None:
    """Find first requirement where requirement_type == cluster_type."""
    for req in chain(plan.requirements.zonal, plan.requirements.regional):
        if req.requirement_type == cluster_type:
            return req
    return None


def _find_matching_cluster(
    cluster_type: str, plan: CapacityPlan
) -> ClusterCapacity | None:
    """Find first plan cluster where cluster_type matches."""
    for cluster in chain(
        plan.candidate_clusters.zonal, plan.candidate_clusters.regional
    ):
        if cluster.cluster_type == cluster_type:
            return cluster
    return None


def _single_cluster_resources(cluster: ClusterCapacity) -> dict[ResourceType, float]:
    """Extract resource totals from a single cluster."""
    cluster_drive = cluster.attached_drives[0] if cluster.attached_drives else None
    effective_disk = cluster.cluster_params.get(EFFECTIVE_DISK_PER_NODE_GIB)
    if effective_disk is not None:
        disk = effective_disk * cluster.count
    else:
        disk = get_disk_size_gib(cluster_drive, cluster.instance) * cluster.count

    return {
        ResourceType.cpu: to_reference_cores(
            cluster.instance.cpu * cluster.count, cluster.instance
        ),
        ResourceType.mem_gib: cluster.instance.ram_gib * cluster.count,
        ResourceType.disk_gib: disk,
        ResourceType.network_mbps: cluster.instance.net_mbps * cluster.count,
    }


def _requirement_resources(
    requirement: CapacityRequirement,
) -> dict[ResourceType, float]:
    """Extract resource values from a CapacityRequirement.

    CPU cores are normalized from the requirement's reference_shape to
    default_reference_shape, giving the same basis as cluster CPU.
    """
    return {
        ResourceType.cpu: to_reference_cores(
            requirement.cpu_cores.mid, requirement.reference_shape
        ),
        ResourceType.mem_gib: requirement.mem_gib.mid,
        ResourceType.disk_gib: requirement.disk_gib.mid,
        ResourceType.network_mbps: requirement.network_mbps.mid,
    }


def _aggregate_resources(plan: CapacityPlan) -> dict[ResourceType, float]:
    """Aggregate resource values from a plan's candidate_clusters.

    All resources are computed from candidate_clusters (the actual provisioned
    instances), NOT from requirements (the demand). CPU is normalized to
    default_reference_shape using IPC and frequency factors for consistent
    comparison across different instance types.
    """
    totals: dict[ResourceType, float] = {
        ResourceType.cpu: 0.0,
        ResourceType.mem_gib: 0.0,
        ResourceType.disk_gib: 0.0,
        ResourceType.network_mbps: 0.0,
    }

    for cluster in chain(
        plan.candidate_clusters.zonal, plan.candidate_clusters.regional
    ):
        for resource_type, val in _single_cluster_resources(cluster).items():
            totals[resource_type] += val

    return totals


def compare_plans(
    baseline: CapacityPlan,
    comparison: CapacityPlan,
    tolerances: ResourceTolerances | None = None,
    strategy: ComparisonStrategy = ComparisonStrategy.provisioned,
) -> PlanComparisonResult:
    """Compare two capacity plans.

    Strategy controls what each plan contributes:

    - **provisioned** (default): Compare aggregate cluster resources from both
      plans. Answers: "how does the recommendation compare to what's deployed?"

    - **requirements**: Compare baseline's clusters against comparison's
      requirements, matched by cluster_type. Answers: "does the baseline's
      deployment satisfy the comparison's requirements?"

      Resource ratios are ``cluster / requirement`` — values ≥ 1.0 mean the
      cluster meets or exceeds the requirement. Cost ratios are
      ``baseline_cost / comparison_cost``.

    Args:
        baseline: The plan providing clusters (e.g., current deployment).
        comparison: The plan providing the reference to compare against.
            For provisioned: provides clusters.
            For requirements: provides requirements (and clusters for cost).
        tolerances: Per-resource tolerance configuration. If None, uses defaults.
        strategy: How to extract and match values between the plans.

    Returns:
        PlanComparisonResult with per-resource comparisons.

    Raises:
        ValueError: With requirements strategy, if no matching requirement
            is found for the baseline cluster's type.
    """
    if tolerances is None:
        tolerances = ResourceTolerances()

    # Extract resources based on strategy
    if strategy == ComparisonStrategy.requirements:
        clusters = list(
            chain(
                baseline.candidate_clusters.zonal,
                baseline.candidate_clusters.regional,
            )
        )
        if not clusters:
            return PlanComparisonResult()
        cluster = clusters[0]

        requirement = _find_matching_requirement(cluster.cluster_type, comparison)
        if requirement is None:
            raise ValueError(
                f"No requirement with type '{cluster.cluster_type}' in plan"
            )

        baseline_resources = _requirement_resources(requirement)
        comparison_resources = _single_cluster_resources(cluster)

        # Cost from comparison's matching cluster (if available)
        comp_cluster = _find_matching_cluster(cluster.cluster_type, comparison)
        baseline_cost = comp_cluster.annual_cost if comp_cluster else None
        comparison_cost = cluster.annual_cost if comp_cluster else None
    else:
        baseline_resources = _aggregate_resources(baseline)
        comparison_resources = _aggregate_resources(comparison)
        baseline_cost = float(baseline.candidate_clusters.total_annual_cost)
        comparison_cost = float(comparison.candidate_clusters.total_annual_cost)

    # Build comparisons from extracted resources
    comparisons = {
        resource_type: ResourceComparison(
            resource=resource_type,
            baseline_value=b_val,
            comparison_value=comparison_resources[resource_type],
            tolerance=tolerances.get_tolerance(resource_type),
        )
        for resource_type, b_val in baseline_resources.items()
    }
    if baseline_cost is not None:
        comparisons[ResourceType.annual_cost] = ResourceComparison(
            resource=ResourceType.annual_cost,
            baseline_value=baseline_cost,
            comparison_value=comparison_cost,
            tolerance=tolerances.get_tolerance(ResourceType.annual_cost),
        )

    return PlanComparisonResult(comparisons=comparisons)
