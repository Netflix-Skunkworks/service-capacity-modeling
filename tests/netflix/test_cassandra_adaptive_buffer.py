"""Tests for adaptive storage buffer: data-size-dependent buffer ratio.

The fixed 4x storage buffer over-provisions large clusters because they already
have enormous absolute headroom in GiB. The adaptive buffer uses a logistic
decay from max_ratio (4.0) to min_ratio (2.0) as zonal data grows, allowing
cheaper instance types that physically fit the data.
"""

import pytest

from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.interface import (
    CapacityDesires,
    DataShape,
    Interval,
    QueryPattern,
)
from service_capacity_modeling.models.org.netflix.cassandra import (
    NflxCassandraArguments,
)
from service_capacity_modeling.models.org.netflix.cassandra import (
    _adaptive_storage_buffer_ratio,
)


class TestAdaptiveBufferFormula:
    """Unit tests for _adaptive_storage_buffer_ratio()."""

    def test_monotonically_decreasing(self):
        """Buffer ratio should decrease as data grows."""
        sizes = [50, 400, 5_000, 10_000, 35_000, 200_000]
        ratios = [_adaptive_storage_buffer_ratio(s) for s in sizes]
        for i in range(len(ratios) - 1):
            assert ratios[i] > ratios[i + 1], (
                f"ratio({sizes[i]})={ratios[i]:.3f} should be > "
                f"ratio({sizes[i + 1]})={ratios[i + 1]:.3f}"
            )

    def test_small_cluster_gets_near_max_buffer(self):
        """50 GiB cluster should get close to max_ratio (4.0)."""
        ratio = _adaptive_storage_buffer_ratio(50)
        assert ratio > 3.9

    def test_large_cluster_gets_reduced_buffer(self):
        """35 TiB cluster should get a meaningfully reduced buffer."""
        ratio = _adaptive_storage_buffer_ratio(35_000)
        assert 2.0 < ratio < 3.0

    def test_huge_cluster_approaches_min_buffer(self):
        """200 TiB cluster should be near min_ratio (2.0)."""
        ratio = _adaptive_storage_buffer_ratio(200_000)
        assert ratio < 2.2

    def test_zero_data_returns_max(self):
        assert _adaptive_storage_buffer_ratio(0) == 4.0

    def test_negative_data_returns_max(self):
        assert _adaptive_storage_buffer_ratio(-100) == 4.0

    def test_custom_bounds(self):
        ratio = _adaptive_storage_buffer_ratio(10_000, max_ratio=6.0, min_ratio=3.0)
        # At midpoint, should be roughly halfway between max and min
        assert 4.0 < ratio < 5.0

    def test_stays_within_bounds(self):
        """Ratio should never exceed max or go below min."""
        for size in [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000]:
            ratio = _adaptive_storage_buffer_ratio(size)
            assert 2.0 <= ratio <= 4.0

    def test_rejects_inverted_bounds(self):
        """min > max should raise ValueError."""
        with pytest.raises(ValueError, match="min_storage_buffer_ratio"):
            NflxCassandraArguments(
                min_storage_buffer_ratio=5.0,
                max_storage_buffer_ratio=4.0,
            )


# Large cluster: 35 TiB state, high read/write traffic (the problem scenario)
LARGE_CLUSTER_DESIRES = CapacityDesires(
    service_tier=1,
    query_pattern=QueryPattern(
        estimated_read_per_second=Interval(
            low=4000, mid=400_000, high=400_000, confidence=0.98
        ),
        estimated_write_per_second=Interval(
            low=4000, mid=200_000, high=200_000, confidence=0.98
        ),
    ),
    data_shape=DataShape(
        estimated_state_size_gib=Interval(
            low=20_000, mid=35_000, high=50_000, confidence=0.98
        ),
    ),
)

# Small cluster: 400 GiB state
SMALL_CLUSTER_DESIRES = CapacityDesires(
    service_tier=1,
    query_pattern=QueryPattern(
        estimated_read_per_second=Interval(
            low=1000, mid=50_000, high=50_000, confidence=0.98
        ),
        estimated_write_per_second=Interval(
            low=1000, mid=25_000, high=25_000, confidence=0.98
        ),
    ),
    data_shape=DataShape(
        estimated_state_size_gib=Interval(low=200, mid=400, high=600, confidence=0.98),
    ),
)


def _plan_instance_names(desires, extra_model_arguments, families=None):
    """Return list of instance names from plan_certain results."""
    kwargs = {
        "model_name": "org.netflix.cassandra",
        "region": "us-east-1",
        "desires": desires,
        "extra_model_arguments": extra_model_arguments,
        "num_results": 20,
        "max_results_per_family": 10,
    }
    if families:
        kwargs["instance_families"] = families
    plans = planner.plan_certain(**kwargs)
    seen = set()
    names = []
    for p in plans:
        z = p.candidate_clusters.zonal[0]
        if z.instance.name not in seen:
            seen.add(z.instance.name)
            names.append(z.instance.name)
    return names


class TestLargeClusterGetsSmallInstances:
    """35 TiB cluster should accept m6id.8xlarge with adaptive buffer."""

    def test_8xlarge_appears_for_large_cluster(self):
        """With adaptive buffer (default), 8xlarge should be a valid option."""
        names = _plan_instance_names(
            LARGE_CLUSTER_DESIRES,
            extra_model_arguments={},
            families=["m6id"],
        )
        has_8xl = any("8xlarge" in n for n in names)
        assert has_8xl, (
            f"Expected m6id.8xlarge to appear for 35 TiB cluster "
            f"with adaptive buffer, got: {names}"
        )


class TestOptOut:
    """adaptive_storage_buffer=False restores fixed 4x behavior."""

    def test_fixed_buffer_when_disabled(self):
        """Disabling adaptive buffer should use fixed max_storage_buffer_ratio."""
        names_adaptive = _plan_instance_names(
            LARGE_CLUSTER_DESIRES,
            extra_model_arguments={},
            families=["m6id"],
        )
        names_fixed = _plan_instance_names(
            LARGE_CLUSTER_DESIRES,
            extra_model_arguments={"adaptive_storage_buffer": False},
            families=["m6id"],
        )
        # Fixed 4x buffer should accept same or fewer instance types
        assert len(names_fixed) <= len(names_adaptive), (
            f"Fixed buffer should accept same or fewer instance types. "
            f"Adaptive: {names_adaptive}, Fixed: {names_fixed}"
        )
        # Verify the buffer actually changed — adaptive should rank cheaper
        # instances higher (8xlarge before 16xlarge)
        if any("8xlarge" in n for n in names_adaptive) and any(
            "16xlarge" in n for n in names_adaptive
        ):
            idx_8_a = next(i for i, n in enumerate(names_adaptive) if "8xlarge" in n)
            idx_16_a = next(i for i, n in enumerate(names_adaptive) if "16xlarge" in n)
            assert idx_8_a < idx_16_a, (
                f"Adaptive should rank 8xlarge above 16xlarge, got: {names_adaptive}"
            )


class TestSmallClusterUnchanged:
    """Small clusters should see negligible buffer change."""

    def test_small_cluster_gets_similar_results(self):
        """400 GiB cluster should get similar plans with or without adaptive."""
        names_adaptive = _plan_instance_names(
            SMALL_CLUSTER_DESIRES,
            extra_model_arguments={},
            families=["m6id"],
        )
        names_fixed = _plan_instance_names(
            SMALL_CLUSTER_DESIRES,
            extra_model_arguments={"adaptive_storage_buffer": False},
            families=["m6id"],
        )
        # Same instance types should be available for small clusters
        assert set(names_adaptive) == set(names_fixed), (
            f"Small cluster instance options should be the same. "
            f"Adaptive: {names_adaptive}, Fixed: {names_fixed}"
        )
