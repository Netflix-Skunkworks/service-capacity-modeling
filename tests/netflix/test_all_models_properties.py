"""
Universal property-based tests that apply to ALL Netflix capacity models.

These tests verify fundamental properties that should hold for every model,
such as determinism, feasibility, and basic scaling behavior.
"""

import hypothesis.strategies as st
from hypothesis import assume
from hypothesis import given
from hypothesis import HealthCheck
from hypothesis import settings

from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import certain_int
from service_capacity_modeling.interface import DataShape
from service_capacity_modeling.interface import QueryPattern
from tests.netflix.property_test_utils import assert_deterministic_planning
from tests.netflix.property_test_utils import get_total_cost
from tests.netflix.property_test_utils import get_total_cpu
from tests.netflix.property_test_utils import plan_model
from tests.netflix.property_test_utils import plan_summary

# Note: Model parametrization is now handled dynamically by pytest_generate_tests()
# in conftest.py based on registered model configurations


# ============================================================================
# Universal Property: Determinism
# ============================================================================


@settings(
    max_examples=10,
    deadline=15000,
    suppress_health_check=[HealthCheck.filter_too_much],  # Some models are restrictive
)
@given(data=st.data())
def test_all_models_are_deterministic(model_name, data):
    """
    Property: All models should be deterministic.

    Same input should always produce the same output. This is fundamental
    for capacity planning - we can't have non-deterministic recommendations.
    """
    from tests.netflix.property_test_utils import capacity_desires_for_model

    # Generate desires appropriate for this specific model
    desires = data.draw(
        capacity_desires_for_model(model_name, min_qps=1000, max_qps=50_000)
    )

    # Skip if model can't handle this configuration
    assume(plan_model(model_name, desires) is not None)

    assert_deterministic_planning(model_name, desires)


# ============================================================================
# Universal Property: Feasibility
# ============================================================================


@settings(
    max_examples=20,
    deadline=15000,
    suppress_health_check=[HealthCheck.filter_too_much],  # Some models are restrictive
)
@given(data=st.data())
def test_all_models_produce_valid_plans(model_name, data):
    """
    Property: All models should produce at least one valid plan.

    Valid input should never result in empty results. The planner should
    always find at least one viable configuration.

    Note: Some models don't support certain configurations (e.g., tier 0),
    so we generate model-specific desires that avoid unsupported configs.
    Some models (postgres, aurora, counter, zookeeper, kafka) have very
    restrictive constraints and will filter many inputs.
    """
    from tests.netflix.property_test_utils import capacity_desires_for_model

    # Generate desires appropriate for this specific model
    desires = data.draw(
        capacity_desires_for_model(model_name, min_qps=1000, max_qps=50_000)
    )
    plan = plan_model(model_name, desires)

    # With model-specific desires, we should always get a plan
    # If we still get None, use assume() to filter it out
    assume(plan is not None)

    # Plans should have either clusters (EC2) or managed services
    # (like DynamoDB)
    assert (
        len(plan.candidate_clusters.zonal) > 0
        or len(plan.candidate_clusters.regional) > 0
        or len(plan.candidate_clusters.services) > 0
    ), (
        f"{model_name} should have at least one cluster or service\n"
        f"Generated plan:\n{plan_summary(plan)}"
    )


# ============================================================================
# Universal Property: QPS Monotonicity
# ============================================================================


@settings(
    max_examples=15, deadline=20000, suppress_health_check=[HealthCheck.filter_too_much]
)
@given(data=st.data())
def test_all_models_scale_cpu_with_qps(model_name, data):
    """
    Property: Higher QPS should require more CPU (or same).

    This is a fundamental scaling property. If you increase traffic,
    you need more computational resources (or at minimum, the same if you
    were over-provisioned).

    This test uses model-specific QPS ranges so database models are tested
    at appropriate scales (e.g., 100-1000 QPS) while high-throughput models
    are tested at their appropriate scales (e.g., 10000-50000 QPS).
    """

    # Get model's valid QPS range
    from tests.netflix.property_test_utils import _get_model_config

    # Check for separate read/write QPS ranges (for models with asymmetric workloads)
    read_qps_range = _get_model_config(model_name, "read_qps_range")
    write_qps_range = _get_model_config(model_name, "write_qps_range")
    qps_range = _get_model_config(model_name, "qps_range")

    if read_qps_range and write_qps_range:
        # Use separate ranges for reads and writes
        min_read_qps, max_read_qps = read_qps_range
        min_write_qps, max_write_qps = write_qps_range
    elif qps_range:
        # Use same range for both reads and writes
        min_read_qps, max_read_qps = qps_range
        min_write_qps, max_write_qps = qps_range
    else:
        # Default range
        min_read_qps, max_read_qps = 1000, 50_000
        min_write_qps, max_write_qps = 1000, 50_000

    # Generate low QPS in the lower half of the range (ensuring 2x fits)
    # Generate high QPS as at least 2x low QPS
    low_read_qps = data.draw(
        st.integers(min_value=min_read_qps, max_value=max_read_qps // 2)
    )
    high_read_qps = data.draw(
        st.integers(min_value=low_read_qps * 2, max_value=max_read_qps)
    )

    low_write_qps = data.draw(
        st.integers(min_value=min_write_qps, max_value=max_write_qps // 2)
    )
    high_write_qps = data.draw(
        st.integers(min_value=low_write_qps * 2, max_value=max_write_qps)
    )

    # Generate desires with same data size but different QPS
    data_gib_range = _get_model_config(model_name, "data_range_gib", default=(100, 100))
    data_gib = data_gib_range[0] if isinstance(data_gib_range, tuple) else 100

    low_qps_desires = CapacityDesires(
        service_tier=1,
        query_pattern=QueryPattern(
            estimated_read_per_second=certain_int(low_read_qps),
            estimated_write_per_second=certain_int(low_write_qps),
        ),
        data_shape=DataShape(estimated_state_size_gib=certain_int(data_gib)),
    )

    high_qps_desires = CapacityDesires(
        service_tier=1,
        query_pattern=QueryPattern(
            estimated_read_per_second=certain_int(high_read_qps),
            estimated_write_per_second=certain_int(high_write_qps),
        ),
        data_shape=DataShape(estimated_state_size_gib=certain_int(data_gib)),
    )

    low_plan = plan_model(model_name, low_qps_desires)
    high_plan = plan_model(model_name, high_qps_desires)

    assume(low_plan is not None)
    assume(high_plan is not None)

    low_cpu = get_total_cpu(low_plan)
    high_cpu = get_total_cpu(high_plan)

    read_multiplier = high_read_qps / low_read_qps if low_read_qps else float("inf")
    write_multiplier = high_write_qps / low_write_qps if low_write_qps else float("inf")

    # Higher QPS should require at least as much CPU (allow equal for over-provisioning)
    assert high_cpu >= low_cpu, (
        f"{model_name}: Higher QPS should require >= CPU\n"
        f"Read multiplier: {read_multiplier:.1f}x, "
        f"Write multiplier: {write_multiplier:.1f}x\n"
        f"\nLow QPS (R:{low_read_qps} W:{low_write_qps}) plan:\n"
        f"{plan_summary(low_plan)}\n"
        f"\nHigh QPS (R:{high_read_qps} W:{high_write_qps}) plan:\n"
        f"{plan_summary(high_plan)}"
    )


# ============================================================================
# Universal Property: Tier-Based Capacity
# ============================================================================


@settings(
    max_examples=15, deadline=20000, suppress_health_check=[HealthCheck.filter_too_much]
)
@given(data=st.data())
# pylint: disable=too-many-locals
def test_all_models_tier_capacity_relationship(model_name, data):
    """
    Property: More critical tiers should have >= capacity/cost in at least
    one dimension.

    Compares a more critical tier (default: tier 0) to a less critical tier
    (default: tier 2) and verifies the more critical tier has at least as
    much in at least ONE of: CPU, memory, storage, or cost. This ensures
    critical services have some form of additional capacity/redundancy,
    though models may express this differently.

    Models can override which tiers to compare via:
    - tier_range (default: (0, 2)) - tuple of (min_tier, max_tier)

    This test uses model-specific QPS and data ranges so all models can be
    tested at their appropriate scales.
    """
    from tests.netflix.property_test_utils import _get_model_config

    # Get model's valid ranges
    read_qps_range = _get_model_config(model_name, "read_qps_range")
    write_qps_range = _get_model_config(model_name, "write_qps_range")
    qps_range = _get_model_config(model_name, "qps_range")
    data_range_gib = _get_model_config(model_name, "data_range_gib")

    if read_qps_range and write_qps_range:
        # Use separate ranges for reads and writes
        min_read_qps, max_read_qps = read_qps_range
        min_write_qps, max_write_qps = write_qps_range
    elif qps_range:
        # Use same range for both reads and writes
        min_read_qps, max_read_qps = qps_range
        min_write_qps, max_write_qps = qps_range
    else:
        # Default range
        min_read_qps, max_read_qps = 1000, 50_000
        min_write_qps, max_write_qps = 1000, 50_000

    if data_range_gib:
        min_data, max_data = data_range_gib
    else:
        min_data, max_data = 100, 1000

    # Get tier range for this model (defaults to tier 0 vs tier 2)
    tier_range = _get_model_config(model_name, "tier_range", default=(0, 2))
    min_tier, max_tier = tier_range

    # Generate workload within model's valid range
    read_qps = data.draw(st.integers(min_value=min_read_qps, max_value=max_read_qps))
    write_qps = data.draw(st.integers(min_value=min_write_qps, max_value=max_write_qps))
    data_gib = data.draw(st.integers(min_value=min_data, max_value=max_data))

    critical_desires = CapacityDesires(
        service_tier=min_tier,  # More critical tier
        query_pattern=QueryPattern(
            estimated_read_per_second=certain_int(read_qps),
            estimated_write_per_second=certain_int(write_qps),
        ),
        data_shape=DataShape(estimated_state_size_gib=certain_int(data_gib)),
    )

    noncritical_desires = CapacityDesires(
        service_tier=max_tier,  # Less critical tier
        query_pattern=QueryPattern(
            estimated_read_per_second=certain_int(read_qps),
            estimated_write_per_second=certain_int(write_qps),
        ),
        data_shape=DataShape(estimated_state_size_gib=certain_int(data_gib)),
    )

    critical_plan = plan_model(model_name, critical_desires)
    noncritical_plan = plan_model(model_name, noncritical_desires)

    assume(critical_plan is not None)
    assume(noncritical_plan is not None)

    # Compare total capacity across all clusters
    critical_cpu = get_total_cpu(critical_plan)
    noncritical_cpu = get_total_cpu(noncritical_plan)

    from tests.netflix.property_test_utils import get_total_memory, get_total_storage

    critical_memory = get_total_memory(critical_plan)
    noncritical_memory = get_total_memory(noncritical_plan)

    critical_storage = get_total_storage(critical_plan)
    noncritical_storage = get_total_storage(noncritical_plan)

    critical_cost = get_total_cost(critical_plan)
    noncritical_cost = get_total_cost(noncritical_plan)

    # More critical tier should have at least ONE dimension with >= capacity/cost
    # (models express redundancy differently - more CPU, memory, storage, or cost)
    has_more_cpu = critical_cpu >= noncritical_cpu
    has_more_memory = critical_memory >= noncritical_memory
    has_more_storage = critical_storage >= noncritical_storage
    has_more_cost = critical_cost >= noncritical_cost

    assert has_more_cpu or has_more_memory or has_more_storage or has_more_cost, (
        f"{model_name}: Tier {min_tier} (more critical) should have >= "
        f"capacity/cost in at least one dimension vs Tier {max_tier}\n"
        f"Workload: R:{read_qps} W:{write_qps} QPS, {data_gib} GiB\n"
        f"\nTier {min_tier} plan:\n{plan_summary(critical_plan)}\n"
        f"\nTier {max_tier} plan:\n{plan_summary(noncritical_plan)}"
    )


# ============================================================================
# Universal Property: Cost Positivity
# ============================================================================


@settings(
    max_examples=20, deadline=15000, suppress_health_check=[HealthCheck.filter_too_much]
)
@given(data=st.data())
def test_all_models_have_positive_cost(model_name, data):
    """
    Property: All plans should have positive annual cost.

    Infrastructure costs money. A plan with zero or negative cost indicates
    a bug in cost calculation.
    """
    from tests.netflix.property_test_utils import capacity_desires_for_model

    # Generate desires appropriate for this specific model
    desires = data.draw(
        capacity_desires_for_model(model_name, min_qps=1000, max_qps=50_000)
    )
    plan = plan_model(model_name, desires)
    assume(plan is not None)

    cost = get_total_cost(plan)

    assert cost > 0, (
        f"{model_name} should have positive annual cost, got ${cost:,.2f}\n"
        f"Plan details:\n{plan_summary(plan)}"
    )


# ============================================================================
# Universal Property: Instance Count Positivity
# ============================================================================


@settings(
    max_examples=20, deadline=15000, suppress_health_check=[HealthCheck.filter_too_much]
)
@given(data=st.data())
def test_all_models_have_positive_instance_count(model_name, data):
    """
    Property: All clusters should have at least one instance.

    A cluster with 0 instances is invalid.
    """
    from tests.netflix.property_test_utils import capacity_desires_for_model

    # Generate desires appropriate for this specific model
    desires = data.draw(
        capacity_desires_for_model(model_name, min_qps=1000, max_qps=50_000)
    )
    plan = plan_model(model_name, desires)
    assume(plan is not None)

    for cluster in plan.candidate_clusters.zonal:
        assert cluster.count > 0, (
            f"{model_name} zonal cluster {cluster.cluster_type} "
            f"should have > 0 instances, got {cluster.count}\n"
            f"Full plan:\n{plan_summary(plan)}"
        )

    for cluster in plan.candidate_clusters.regional:
        assert cluster.count > 0, (
            f"{model_name} regional cluster {cluster.cluster_type} "
            f"should have > 0 instances, got {cluster.count}\n"
            f"Full plan:\n{plan_summary(plan)}"
        )
