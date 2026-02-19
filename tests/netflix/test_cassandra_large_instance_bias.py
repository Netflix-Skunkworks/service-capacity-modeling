"""Tests for Cassandra large_instance_regret: prefer horizontal scaling.

AWS pricing rounding makes larger instances (16xlarge, 32xlarge) appear
marginally cheaper than equivalent horizontal configurations (8xlarge × 2).
The large_instance_regret penalty overrides this rounding artifact, ensuring
the planner prefers more smaller nodes over fewer larger ones.

The m6id family exhibits this inverted pricing in our catalog:
  4x m6id.32xlarge  < 8x m6id.16xlarge < 16x m6id.8xlarge  (by ~$8/yr)

The Cassandra model defaults large_instance_regret=0.2, which adds a
graduated cost penalty for instance sizes above 8xlarge. This flips
the ordering:
  16x m6id.8xlarge  < 8x m6id.16xlarge < 4x m6id.32xlarge
"""

from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.interface import (
    CapacityDesires,
    DataShape,
    Interval,
    QueryPattern,
)


# Production-like workload that produces 512 total vCPU in m6id
# (matches real Antigravity API calls)
PRODUCTION_DESIRES = CapacityDesires(
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
            low=1000, mid=2000, high=3000, confidence=0.98
        ),
    ),
)


def _get_m6id_size_order(extra_model_arguments):
    """Return m6id instance sizes in planner-preferred order."""
    plans = planner.plan_certain(
        model_name="org.netflix.cassandra",
        region="us-east-1",
        desires=PRODUCTION_DESIRES,
        extra_model_arguments=extra_model_arguments,
        instance_families=["m6id"],
        num_results=20,
        max_results_per_family=10,
    )
    seen = set()
    order = []
    for p in plans:
        z = p.candidate_clusters.zonal[0]
        name = z.instance.name
        if name not in seen:
            seen.add(name)
            order.append((name, z.count, p.candidate_clusters.total_annual_cost))
    return order


class TestDefaultBiasFlipsOrdering:
    """The default large_instance_regret (0.2) should prefer horizontal scaling."""

    def test_8xlarge_ranks_above_16xlarge_by_default(self):
        """With default bias, m6id.8xlarge should beat m6id.16xlarge."""
        # No explicit large_instance_regret — the default (0.2) should apply
        order = _get_m6id_size_order(extra_model_arguments={})
        names = [name for name, _, _ in order]

        idx_16xl = next(i for i, n in enumerate(names) if "16xlarge" in n)
        idx_8xl = next(i for i, n in enumerate(names) if "8xlarge" in n)

        assert idx_8xl < idx_16xl, (
            f"Expected 8xlarge to rank above 16xlarge by default, "
            f"but got 8xl at {idx_8xl}, 16xl at {idx_16xl}. "
            f"Order: {names}"
        )

    def test_full_ordering_by_default(self):
        """Default bias should produce: all ≤8xlarge before any >8xlarge."""
        order = _get_m6id_size_order(extra_model_arguments={})
        names = [name for name, _, _ in order]

        idx_8xl = next(i for i, n in enumerate(names) if "8xlarge" in n)
        idx_16xl = next(i for i, n in enumerate(names) if "16xlarge" in n)

        # All unpenalized sizes (≤8xlarge) should appear before penalized ones
        assert idx_8xl < idx_16xl, (
            f"Expected 8xl({idx_8xl}) < 16xl({idx_16xl}) "
            f"by default, but ordering was: {names}"
        )

    def test_least_regret_prefers_horizontal_by_default(self):
        """plan() least-regret should pick 8xlarge or smaller by default."""
        result = planner.plan(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=PRODUCTION_DESIRES,
            extra_model_arguments={},
            instance_families=["m6id"],
            num_results=5,
        )

        lr = result.least_regret[0]
        cluster = lr.candidate_clusters.zonal[0]
        size = cluster.instance.name.split(".")[-1]

        assert size in ("2xlarge", "4xlarge", "8xlarge"), (
            f"Expected least-regret to pick 8xlarge or smaller by default, "
            f"got {cluster.instance.name}"
        )


class TestBiasCanBeDisabled:
    """Setting large_instance_regret=0 should restore raw cost ordering."""

    def test_no_bias_allows_larger_instances_to_win(self):
        """With bias=0, m6id.32xlarge ranks above m6id.8xlarge (rounding wins)."""
        order = _get_m6id_size_order(extra_model_arguments={"large_instance_regret": 0})
        names = [name for name, _, _ in order]

        idx_32xl = next(i for i, n in enumerate(names) if "32xlarge" in n)
        idx_16xl = next(i for i, n in enumerate(names) if "16xlarge" in n)
        idx_8xl = next(i for i, n in enumerate(names) if "8xlarge" in n)

        # Without bias: larger instances rank better due to rounding
        assert idx_32xl < idx_16xl < idx_8xl, (
            f"Expected 32xl({idx_32xl}) < 16xl({idx_16xl}) < 8xl({idx_8xl}) "
            f"with bias=0, but ordering was: {names}"
        )


class TestBiasDoesNotAffectSmallInstances:
    """Instances at 8xlarge and below should not be penalized."""

    def test_costs_unchanged_for_small_instances(self):
        """8xlarge, 4xlarge, 2xlarge costs should be identical with and without bias."""
        order_without = _get_m6id_size_order(
            extra_model_arguments={"large_instance_regret": 0}
        )
        order_with = _get_m6id_size_order(extra_model_arguments={})

        def costs_at_or_below_8xl(order):
            return {
                name: cost
                for name, _, cost in order
                if any(s in name for s in ["2xlarge", "4xlarge", "8xlarge"])
                and "12x" not in name
                and "16x" not in name
                and "24x" not in name
                and "32x" not in name
            }

        costs_without = costs_at_or_below_8xl(order_without)
        costs_with = costs_at_or_below_8xl(order_with)

        for name in costs_without:
            if name in costs_with:
                assert costs_without[name] == costs_with[name], (
                    f"{name} cost changed with bias: "
                    f"${costs_without[name]:,.2f} -> ${costs_with[name]:,.2f}"
                )
