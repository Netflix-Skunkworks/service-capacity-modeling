"""Tests for Cassandra different_family_regret: prefer current instance family.

When a cluster currently runs on m6id, migrating to c6id requires going
through on-demand pricing (reserved instances are family-specific). The
different_family_regret (default 10%) adds a cost-proportional penalty to
cross-family plans, so you need >10% savings to justify switching.

Since c6id is ~13% cheaper than m6id for this workload, the default 10%
regret is NOT enough to prevent the switch -- this is intentional. The penalty
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
    """Return instance families in planner-preferred order."""
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
        family = p.candidate_clusters.zonal[0].instance.family
        if family not in seen:
            seen.add(family)
            order.append(family)
    return order


class TestSameFamilyRegretPrefersCurrent:
    """With a high enough regret, current family should be preferred."""

    def test_high_regret_prefers_current_family(self):
        """A 15% regret should overcome the ~13% m6id->c6id cost difference."""
        desires = _desires_with_current_family("m6id.8xlarge")
        families = _get_family_order(
            desires,
            extra_model_arguments={"different_family_regret": 0.15},
        )

        assert "m6id" in families, f"m6id not found in results: {families}"
        assert "c6id" in families, f"c6id not found in results: {families}"

        idx_m6id = families.index("m6id")
        idx_c6id = families.index("c6id")

        assert idx_m6id < idx_c6id, (
            f"Expected m6id to rank above c6id with 15% regret "
            f"(>13% cost diff), but got m6id at {idx_m6id}, c6id at "
            f"{idx_c6id}. Order: {families}"
        )

    def test_default_regret_allows_significant_savings(self):
        """Default 10% regret should NOT prevent switching when savings are ~13%."""
        desires = _desires_with_current_family("m6id.8xlarge")
        families = _get_family_order(desires, extra_model_arguments={})

        assert "c6id" in families, f"c6id not found in results: {families}"
        assert "m6id" in families, f"m6id not found in results: {families}"

        idx_c6id = families.index("c6id")
        idx_m6id = families.index("m6id")

        assert idx_c6id < idx_m6id, (
            f"Expected c6id to rank above m6id with default regret "
            f"(13% savings > 10% regret), but got c6id at {idx_c6id}, "
            f"m6id at {idx_m6id}. Order: {families}"
        )


class TestSameFamilyRegretCanBeDisabled:
    """Setting different_family_regret=0 lets the cheaper family win."""

    def test_no_regret_pure_cost_ranking(self):
        """With different_family_regret=0, c6id should rank by pure cost."""
        desires = _desires_with_current_family("m6id.8xlarge")
        families = _get_family_order(
            desires,
            extra_model_arguments={"different_family_regret": 0},
        )

        assert "c6id" in families, f"c6id not found in results: {families}"
        assert "m6id" in families, f"m6id not found in results: {families}"

        idx_c6id = families.index("c6id")
        idx_m6id = families.index("m6id")

        assert idx_c6id < idx_m6id, (
            f"Expected c6id to rank above m6id with regret disabled, "
            f"but got c6id at {idx_c6id}, m6id at {idx_m6id}. "
            f"Order: {families}"
        )


class TestSameFamilyRegretNoCurrentCluster:
    """Without current_clusters, no family regret should apply."""

    def test_no_regret_for_new_provisioning(self):
        """New clusters (no current_clusters) should not have family regret."""
        families = _get_family_order(DESIRES_NO_CURRENT, extra_model_arguments={})

        assert "c6id" in families, f"c6id not found in results: {families}"
        assert "m6id" in families, f"m6id not found in results: {families}"

        # For new provisioning, c6id should win on pure cost (no migration regret)
        idx_c6id = families.index("c6id")
        idx_m6id = families.index("m6id")

        assert idx_c6id < idx_m6id, (
            f"Expected c6id to rank above m6id for new provisioning "
            f"(no family migration regret), "
            f"but got c6id at {idx_c6id}, m6id at {idx_m6id}. "
            f"Order: {families}"
        )
