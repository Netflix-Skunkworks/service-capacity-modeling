"""
Shared utilities for property-based testing of capacity models.

This module provides reusable strategies, helper functions, and common
property test patterns for all Netflix capacity models.
"""

from collections import defaultdict
from itertools import chain
from typing import Optional

import humanize
from hypothesis import strategies as st
from pydantic import BaseModel

from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import CapacityPlan
from service_capacity_modeling.interface import certain_int
from service_capacity_modeling.interface import DataShape
from service_capacity_modeling.interface import QueryPattern

GiB = 1024 * 1024 * 1024


# ============================================================================
# Strategies for Generating Valid Test Data
# ============================================================================


@st.composite
def valid_qps(draw, min_qps=100, max_qps=1_000_000):
    """Generate valid queries-per-second values."""
    return draw(st.integers(min_value=min_qps, max_value=max_qps))


@st.composite
def valid_data_size_gib(draw, min_gib=1, max_gib=50_000):
    """Generate valid data sizes in GiB."""
    return draw(st.integers(min_value=min_gib, max_value=max_gib))


@st.composite
def valid_service_tier(draw):
    """Generate valid service tiers (0=critical, 1=prod, 2=non-critical, 3=dev)."""
    return draw(st.integers(min_value=0, max_value=3))


@st.composite
def valid_latency_ms(draw, min_ms=0.1, max_ms=1000.0):
    """Generate valid latency values in milliseconds."""
    return draw(st.floats(min_value=min_ms, max_value=max_ms))


@st.composite
def valid_item_count(draw, min_items=1000, max_items=1_000_000_000_000):
    """Generate valid item counts."""
    return draw(st.integers(min_value=min_items, max_value=max_items))


@st.composite
def capacity_desires_simple(  # pylint: disable=too-many-positional-arguments
    draw,
    # Query pattern (QPS and sizes)
    min_qps=100,
    max_qps=100_000,
    min_read_size_bytes=128,
    max_read_size_bytes=8192,
    min_write_size_bytes=128,
    max_write_size_bytes=8192,
    # Data shape
    min_data_gib=1,
    max_data_gib=10_000,
    # SLA tier
    min_tier=0,
    max_tier=2,
):
    """
    Generate simple CapacityDesires with certain (non-interval) values.

    Args:
        draw: Hypothesis draw function
        min_qps: Minimum QPS to generate
        max_qps: Maximum QPS to generate
        min_read_size_bytes: Minimum read size in bytes
        max_read_size_bytes: Maximum read size in bytes
        min_write_size_bytes: Minimum write size in bytes (0 for read-only)
        max_write_size_bytes: Maximum write size in bytes (0 for read-only)
        min_data_gib: Minimum data size in GiB
        max_data_gib: Maximum data size in GiB
        min_tier: Minimum service tier
        max_tier: Maximum service tier

    Returns:
        CapacityDesires with certain values for fast, deterministic testing
    """
    read_qps = draw(st.integers(min_value=min_qps, max_value=max_qps))
    write_qps = draw(st.integers(min_value=min_qps, max_value=max_qps))
    data_gib = draw(st.integers(min_value=min_data_gib, max_value=max_data_gib))
    tier = draw(st.integers(min_value=min_tier, max_value=max_tier))

    # Generate read/write sizes for network bandwidth calculations
    read_size = draw(
        st.integers(min_value=min_read_size_bytes, max_value=max_read_size_bytes)
    )
    write_size = draw(
        st.integers(min_value=min_write_size_bytes, max_value=max_write_size_bytes)
    )

    return CapacityDesires(
        service_tier=tier,
        query_pattern=QueryPattern(
            estimated_read_per_second=certain_int(read_qps),
            estimated_write_per_second=certain_int(write_qps),
            estimated_mean_read_size_bytes=certain_int(read_size),
            estimated_mean_write_size_bytes=certain_int(write_size),
        ),
        data_shape=DataShape(
            estimated_state_size_gib=certain_int(data_gib),
        ),
    )


@st.composite
def capacity_desires_with_items(
    draw,
    min_qps=100,
    max_qps=100_000,
    min_items=1000,
    max_items=1_000_000_000,
):
    """Generate CapacityDesires with item_count for models that need it."""
    read_qps = draw(st.integers(min_value=min_qps, max_value=max_qps))
    write_qps = draw(st.integers(min_value=min_qps, max_value=max_qps))
    item_count = draw(st.integers(min_value=min_items, max_value=max_items))
    tier = draw(st.integers(min_value=0, max_value=3))

    return CapacityDesires(
        service_tier=tier,
        query_pattern=QueryPattern(
            estimated_read_per_second=certain_int(read_qps),
            estimated_write_per_second=certain_int(write_qps),
        ),
        data_shape=DataShape(
            estimated_state_item_count=certain_int(item_count),
        ),
    )


def capacity_desires_for_model(model_name, **overrides):
    """
    Get a capacity_desires strategy appropriate for a specific model.

    This handles model-specific constraints like models that don't support tier 0
    and models with restrictive size limits. Configuration is loaded from the
    model's test file (via PROPERTY_TEST_CONFIG).

    Args:
        model_name: The model name (e.g., "org.netflix.postgres")
        **overrides: Override any strategy parameters

    Returns:
        A Hypothesis strategy for CapacityDesires
    """
    # Get model-specific configuration
    qps_range = _get_model_config(model_name, "qps_range")
    data_range_gib = _get_model_config(model_name, "data_range_gib")
    tier_range = _get_model_config(model_name, "tier_range")
    read_size_range = _get_model_config(model_name, "read_size_range")
    write_size_range = _get_model_config(model_name, "write_size_range")

    # Default parameters (order matches capacity_desires_simple signature)
    params = {
        # Query pattern
        "min_qps": overrides.get("min_qps", 1000),
        "max_qps": overrides.get("max_qps", 50_000),
        "min_read_size_bytes": overrides.get("min_read_size_bytes", 128),
        "max_read_size_bytes": overrides.get("max_read_size_bytes", 8192),
        "min_write_size_bytes": overrides.get("min_write_size_bytes", 128),
        "max_write_size_bytes": overrides.get("max_write_size_bytes", 8192),
        # Data shape
        "min_data_gib": overrides.get("min_data_gib", 1),
        "max_data_gib": overrides.get("max_data_gib", 10_000),
        # SLA tier
        "min_tier": overrides.get("min_tier", 0),
        "max_tier": overrides.get("max_tier", 2),
    }

    # Apply model-specific QPS range if configured
    if qps_range:
        params["min_qps"] = overrides.get("min_qps", qps_range[0])
        params["max_qps"] = overrides.get("max_qps", qps_range[1])

    # Apply model-specific data range if configured
    if data_range_gib:
        params["min_data_gib"] = overrides.get("min_data_gib", data_range_gib[0])
        params["max_data_gib"] = overrides.get("max_data_gib", data_range_gib[1])

    # Apply model-specific tier range if configured
    if tier_range:
        params["min_tier"] = overrides.get("min_tier", tier_range[0])
        params["max_tier"] = overrides.get("max_tier", tier_range[1])

    # Apply model-specific read/write size ranges if configured
    # Use (0, 0) for read-only models
    if read_size_range:
        params["min_read_size_bytes"] = overrides.get(
            "min_read_size_bytes", read_size_range[0]
        )
        params["max_read_size_bytes"] = overrides.get(
            "max_read_size_bytes", read_size_range[1]
        )
    if write_size_range:
        params["min_write_size_bytes"] = overrides.get(
            "min_write_size_bytes", write_size_range[0]
        )
        params["max_write_size_bytes"] = overrides.get(
            "max_write_size_bytes", write_size_range[1]
        )

    return capacity_desires_simple(**params)


# ============================================================================
# Helper Functions for Extracting Plan Metrics
# ============================================================================


def get_total_cpu(plan):
    """Get total CPU cores from all clusters in a capacity plan."""
    total = 0
    if plan.candidate_clusters.zonal:
        for cluster in plan.candidate_clusters.zonal:
            total += cluster.count * cluster.instance.cpu
    if plan.candidate_clusters.regional:
        for cluster in plan.candidate_clusters.regional:
            total += cluster.count * cluster.instance.cpu
    return total


def get_total_cost(plan):
    """Get total annual cost from a capacity plan."""
    return plan.candidate_clusters.total_annual_cost


def get_total_memory(plan):
    """Get total memory (RAM) in GiB from all clusters in a capacity plan."""
    total = 0.0
    if plan.candidate_clusters.zonal:
        for cluster in plan.candidate_clusters.zonal:
            total += cluster.count * cluster.instance.ram_gib
    if plan.candidate_clusters.regional:
        for cluster in plan.candidate_clusters.regional:
            total += cluster.count * cluster.instance.ram_gib
    return total


def get_total_storage(plan):
    """Get total storage in GiB from all clusters in a capacity plan."""
    from tests.util import get_drive_size_gib

    total = 0.0
    if plan.candidate_clusters.zonal:
        for cluster in plan.candidate_clusters.zonal:
            drive_size = get_drive_size_gib(cluster)
            if drive_size is not None:
                total += cluster.count * drive_size
    if plan.candidate_clusters.regional:
        for cluster in plan.candidate_clusters.regional:
            drive_size = get_drive_size_gib(cluster)
            if drive_size is not None:
                total += cluster.count * drive_size
    return total


def get_zonal_storage_gib(plan) -> Optional[int]:
    """Get total storage from all zonal clusters."""
    from tests.util import get_total_storage_gib

    total = None
    if plan.candidate_clusters.zonal:
        for cluster in plan.candidate_clusters.zonal:
            cluster_storage_gib = get_total_storage_gib(cluster)
            if cluster_storage_gib is not None:
                if total is None:
                    total = 0
                total += cluster_storage_gib
    return total


def get_instance_family(plan, cluster_type=None):
    """
    Get the instance family from a plan.

    Args:
        plan: CapacityPlan
        cluster_type: Optional cluster type to filter by

    Returns:
        Instance family (e.g., 'r5', 'c5', 'm6i')
    """
    clusters = []
    if plan.candidate_clusters.zonal:
        clusters.extend(plan.candidate_clusters.zonal)
    if plan.candidate_clusters.regional:
        clusters.extend(plan.candidate_clusters.regional)

    if cluster_type:
        clusters = [c for c in clusters if c.cluster_type == cluster_type]

    if not clusters:
        return None

    return clusters[0].instance.family


def get_cluster_count(plan, cluster_type=None):
    """Get the number of nodes in a cluster."""
    clusters = []
    if plan.candidate_clusters.zonal:
        clusters.extend(plan.candidate_clusters.zonal)
    if plan.candidate_clusters.regional:
        clusters.extend(plan.candidate_clusters.regional)

    if cluster_type:
        clusters = [c for c in clusters if c.cluster_type == cluster_type]

    if not clusters:
        return 0

    return clusters[0].count


def has_cluster_type(plan, cluster_type):
    """Check if plan includes a specific cluster type."""
    all_types = set()
    if plan.candidate_clusters.zonal:
        all_types.update(c.cluster_type for c in plan.candidate_clusters.zonal)
    if plan.candidate_clusters.regional:
        all_types.update(c.cluster_type for c in plan.candidate_clusters.regional)
    return cluster_type in all_types


# ============================================================================
# Planning Helper Functions
# ============================================================================


def _get_model_config(model_name: str, key: str, default=None):
    """
    Get a specific configuration value for a model from registered configs.

    This looks up the model's PROPERTY_TEST_CONFIG from its test file.
    """
    from tests.netflix.property_test_registry import get_property_test_config

    config = get_property_test_config(model_name)
    return config.get(key, default)


def plan_model(model_name, desires, region="us-east-1", **extra_args):
    """
    Helper to plan a specific model with certain desires.

    Args:
        model_name: Model to plan (e.g., "org.netflix.cassandra")
        desires: CapacityDesires object
        region: AWS region
        **extra_args: Extra model arguments

    Returns:
        First capacity plan or None if planning fails
    """
    # Check if model supports tier 0 (inferred from tier_range)
    tier_range = _get_model_config(model_name, "tier_range", default=(0, 2))
    supports_tier_0 = tier_range[0] == 0
    if not supports_tier_0 and desires.service_tier == 0:
        return None

    # Merge model-specific required args with user-provided args
    model_extra_args = _get_model_config(
        model_name, "extra_model_arguments", default={}
    )
    model_args = model_extra_args.copy()
    model_args.update(extra_args)

    try:
        plans = planner.plan_certain(
            model_name=model_name,
            region=region,
            desires=desires,
            extra_model_arguments=model_args,
        )
        return plans[0] if plans else None
    except Exception:  # pylint: disable=broad-exception-caught
        return None


# ============================================================================
# Common Property Test Patterns
# ============================================================================


def assert_monotonic_cpu(low_plan, high_plan, multiplier=2.0):
    """
    Assert that higher workload requires more CPU.

    Args:
        low_plan: Plan for lower workload
        high_plan: Plan for higher workload
        multiplier: Minimum CPU multiplier expected
    """
    low_cpu = get_total_cpu(low_plan)
    high_cpu = get_total_cpu(high_plan)

    assert high_cpu >= low_cpu * multiplier, (
        f"Higher workload should require at least {multiplier}x CPU: "
        f"Low = {low_cpu} cores, High = {high_cpu} cores"
    )


def assert_monotonic_cost(low_plan, high_plan):
    """Assert that higher workload costs more or equal."""
    low_cost = get_total_cost(low_plan)
    high_cost = get_total_cost(high_plan)

    assert high_cost >= low_cost, (
        f"Higher workload should cost at least as much: "
        f"Low = ${low_cost:,.0f}, High = ${high_cost:,.0f}"
    )


def assert_storage_sufficient(plan, min_data_gib, min_multiplier=1.0):
    """Assert that storage is sufficient for data."""
    storage = get_zonal_storage_gib(plan)
    assert storage >= min_data_gib * min_multiplier, (
        f"Storage ({storage} GiB) should be at least {min_multiplier}x "
        f"data size ({min_data_gib} GiB)"
    )


def assert_minimum_cluster_size(plan, min_count, tier=None, cluster_type=None):
    """Assert minimum cluster size for HA."""
    count = get_cluster_count(plan, cluster_type)
    tier_msg = f" for tier {tier}" if tier is not None else ""
    type_msg = f" {cluster_type}" if cluster_type else ""

    assert count >= min_count, (
        f"Cluster{type_msg}{tier_msg} should have at least {min_count} nodes, "
        f"got {count}"
    )


def assert_compute_optimized(plan, cluster_type=None):
    """Assert that instance family is compute-optimized."""
    family = get_instance_family(plan, cluster_type)
    assert family and family[0] in ("c", "m", "r", "i"), (
        f"Should use compute/balanced instance family, got {family}"
    )


def assert_deterministic_planning(model_name, desires, **extra_args):
    """Assert that planning is deterministic (same input = same output)."""
    plan1 = plan_model(model_name, desires, **extra_args)
    plan2 = plan_model(model_name, desires, **extra_args)

    assert plan1 is not None
    assert plan2 is not None
    assert plan1.model_dump_json() == plan2.model_dump_json(), (
        f"{model_name}: Planning should be deterministic\n"
        f"First plan:\n{plan_summary(plan1)}\n"
        f"Second plan:\n{plan_summary(plan2)}"
    )


# ============================================================================
# Model Categories (for organizing tests)
# ============================================================================


def get_all_model_names():
    """Get all registered Netflix model names."""
    from service_capacity_modeling.models.org import netflix

    return list(netflix.models().keys())


# Note: Model lists are now dynamically generated at test collection time
# by pytest_generate_tests() in conftest.py based on registered configurations


# ============================================================================
# Human-Readable Plan Summaries
# ============================================================================


class ClusterSummary(BaseModel):
    annual_cost: int = 0
    instance_type: str
    count: int = 0
    cpus: int = 0
    local_gib: int = 0
    attached_gib: int = 0

    def __str__(self):
        desc = f"{self.count}×{self.instance_type}"
        storage_parts = []
        if self.local_gib != 0:
            size = humanize.filesize.naturalsize(self.local_gib * GiB, True)
            storage_parts.append(f"{size} of local")
        if self.attached_gib != 0:
            size = humanize.filesize.naturalsize(self.attached_gib * GiB, True)
            storage_parts.append(f"{size} of attached")
        if storage_parts:
            desc += " with " + " ".join(storage_parts)
        return desc


class PlanSummary(BaseModel):
    annual_cost: int
    clusters: dict[str, ClusterSummary]

    def __str__(self):
        lines = [f"Annual cost: ${self.annual_cost:,}"]
        for cluster_type, cluster in self.clusters.items():
            lines.append(f"  {cluster_type}: {cluster}")
        return "\n".join(lines)


def plan_summary(plan: CapacityPlan) -> PlanSummary:
    """Generate a human-readable summary of a capacity plan."""
    total_cost = int(plan.candidate_clusters.total_annual_cost)

    clusters: dict = defaultdict(
        lambda: {
            "annual_cost": 0,
            "count": 0,
            "cpus": 0,
            "ram_gib": 0.0,
            "attached_gib": 0,
            "local_gib": 0,
        }
    )
    for cap in chain.from_iterable(
        [plan.candidate_clusters.zonal, plan.candidate_clusters.regional]
    ):
        csum = clusters[cap.cluster_type]
        csum["instance_type"] = cap.instance.name
        csum["annual_cost"] += int(cap.annual_cost)
        csum["count"] += cap.count
        csum["cpus"] += cap.instance.cpu * cap.count
        csum["ram_gib"] += cap.instance.ram_gib * cap.count
        if cap.instance.drive:
            csum["local_gib"] += cap.instance.drive.size_gib * cap.count
        for drive in cap.attached_drives:
            csum["attached_gib"] += drive.size_gib * cap.count

    summary = PlanSummary(
        annual_cost=total_cost,
        clusters=clusters,
    )
    return summary
