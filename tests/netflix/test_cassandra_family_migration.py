"""Tests for Cassandra same_family_bias: prefer current instance family.

When a cluster currently runs on m6id, migrating to c6id requires going
through on-demand pricing (reserved instances are family-specific). The
same_family_bias (default 5%) adds a cost-proportional penalty to cross-
family plans, so you need >5% savings to justify switching.

Since c6id is ~13% cheaper than m6id for this workload, the default 5%
bias is NOT enough to prevent the switch — this is intentional. The bias
only prevents switching when cost differences are marginal.
"""

from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.interface import (
    CapacityDesires,
    CurrentClusters,
    CurrentZoneClusterCapacity,
    DataShape,
    Interval,
    QueryPattern,
)


def _desires_with_current_family(instance_name: str) -> CapacityDesires:
    """Build desires for a cluster currently running on a specific instance."""
    return CapacityDesires(
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
        current_clusters=CurrentClusters(
            zonal=[
                CurrentZoneClusterCapacity(
                    cluster_instance_name=instance_name,
                    cluster_instance_count=Interval(
                        low=16, mid=16, high=16, confidence=1.0
                    ),
                    cpu_utilization=Interval(low=10, mid=40, high=60, confidence=0.98),
                )
            ]
        ),
    )


DESIRES_NO_CURRENT = CapacityDesires(
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


def _get_family_order(desires, extra_model_arguments, families=None):
    """Return instance families in planner-preferred order with costs."""
    if families is None:
        families = ["m6id", "c6id"]
    plans = planner.plan_certain(
        model_name="org.netflix.cassandra",
        region="us-east-1",
        desires=desires,
        extra_model_arguments=extra_model_arguments,
        instance_families=families,
        num_results=20,
        max_results_per_family=10,
    )
    seen = set()
    order = []
    for p in plans:
        z = p.candidate_clusters.zonal[0]
        family = z.instance.family
        if family not in seen:
            seen.add(family)
            order.append(
                (family, z.instance.name, p.candidate_clusters.total_annual_cost)
            )
    return order


class TestSameFamilyBiasPrefersCurrent:
    """With a high enough bias, current family should be preferred."""

    def test_high_bias_prefers_current_family(self):
        """A 15% bias should overcome the ~13% m6id→c6id cost difference."""
        desires = _desires_with_current_family("m6id.8xlarge")
        order = _get_family_order(
            desires,
            extra_model_arguments={"same_family_bias": 0.15},
        )
        families = [fam for fam, _, _ in order]

        assert "m6id" in families, f"m6id not found in results: {families}"
        assert "c6id" in families, f"c6id not found in results: {families}"

        idx_m6id = families.index("m6id")
        idx_c6id = families.index("c6id")

        assert idx_m6id < idx_c6id, (
            f"Expected m6id to rank above c6id with 15% bias "
            f"(>13% cost diff), but got m6id at {idx_m6id}, c6id at "
            f"{idx_c6id}. Order: {order}"
        )

    def test_default_bias_allows_significant_savings(self):
        """Default 5% bias should NOT prevent switching when savings are ~13%."""
        desires = _desires_with_current_family("m6id.8xlarge")
        order = _get_family_order(desires, extra_model_arguments={})
        families = [fam for fam, _, _ in order]

        assert "c6id" in families, f"c6id not found in results: {families}"
        assert "m6id" in families, f"m6id not found in results: {families}"

        idx_c6id = families.index("c6id")
        idx_m6id = families.index("m6id")

        assert idx_c6id < idx_m6id, (
            f"Expected c6id to rank above m6id with default bias "
            f"(13% savings > 5% bias), but got c6id at {idx_c6id}, "
            f"m6id at {idx_m6id}. Order: {order}"
        )


class TestSameFamilyBiasCanBeDisabled:
    """Setting same_family_bias=0 lets the cheaper family win."""

    def test_no_bias_pure_cost_ranking(self):
        """With same_family_bias=0, c6id should rank by pure cost."""
        desires = _desires_with_current_family("m6id.8xlarge")
        order = _get_family_order(
            desires,
            extra_model_arguments={
                "same_family_bias": 0,
                "large_instance_regret": 0,
            },
        )
        families = [fam for fam, _, _ in order]

        assert "c6id" in families, f"c6id not found in results: {families}"
        assert "m6id" in families, f"m6id not found in results: {families}"

        idx_c6id = families.index("c6id")
        idx_m6id = families.index("m6id")

        assert idx_c6id < idx_m6id, (
            f"Expected c6id to rank above m6id with bias disabled, "
            f"but got c6id at {idx_c6id}, m6id at {idx_m6id}. "
            f"Order: {order}"
        )


class TestSameFamilyBiasNoCurrentCluster:
    """Without current_clusters, no family penalty should apply."""

    def test_no_penalty_for_new_provisioning(self):
        """New clusters (no current_clusters) should not have family bias."""
        order = _get_family_order(DESIRES_NO_CURRENT, extra_model_arguments={})
        families = [fam for fam, _, _ in order]

        assert "c6id" in families, f"c6id not found in results: {families}"
        assert "m6id" in families, f"m6id not found in results: {families}"

        # For new provisioning, c6id should win on pure cost (no migration penalty)
        idx_c6id = families.index("c6id")
        idx_m6id = families.index("m6id")

        assert idx_c6id < idx_m6id, (
            f"Expected c6id to rank above m6id for new provisioning "
            f"(no family migration penalty), "
            f"but got c6id at {idx_c6id}, m6id at {idx_m6id}. "
            f"Order: {order}"
        )
