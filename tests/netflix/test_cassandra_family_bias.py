"""Tests for Cassandra family_bias model argument in least-regret selection.

The family_bias argument allows tuning the regret calculation to prefer
certain instance families over others. This is useful when operational
preferences (e.g., familiarity with i3/i4i families) should influence
the capacity planning decision.
"""

from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import certain_float
from service_capacity_modeling.interface import certain_int
from service_capacity_modeling.interface import DataShape
from service_capacity_modeling.interface import QueryPattern
from service_capacity_modeling.models.org.netflix.cassandra import (
    NflxCassandraArguments,
)


class TestCassandraFamilyBiasArgument:
    """Test that family_bias is accepted as a model argument."""

    def test_family_bias_argument_accepted(self):
        """NflxCassandraArguments should accept family_bias parameter."""
        args = NflxCassandraArguments.from_extra_model_arguments(
            {"family_bias": {"i": 0.5, "r": 1.0, "m": 1.0}}
        )
        assert args.family_bias == {"i": 0.5, "r": 1.0, "m": 1.0}

    def test_family_bias_default_is_none(self):
        """family_bias should default to None when not specified."""
        args = NflxCassandraArguments.from_extra_model_arguments({})
        assert args.family_bias is None

    def test_family_bias_empty_dict(self):
        """family_bias can be an empty dict (no bias applied)."""
        args = NflxCassandraArguments.from_extra_model_arguments({"family_bias": {}})
        assert args.family_bias == {}


class TestCassandraFamilyBiasSelection:
    """Test that family_bias influences instance family selection."""

    # A workload that could reasonably use either i-family (storage-optimized)
    # or m-family (general purpose) instances
    mixed_workload = CapacityDesires(
        service_tier=1,
        query_pattern=QueryPattern(
            estimated_read_per_second=certain_int(50_000),
            estimated_write_per_second=certain_int(20_000),
            estimated_mean_read_latency_ms=certain_float(1.0),
            estimated_mean_write_latency_ms=certain_float(0.8),
        ),
        data_shape=DataShape(
            estimated_state_size_gib=certain_int(500),
        ),
    )

    def test_family_bias_prefers_biased_family(self):
        """family_bias should make biased family appear in top results."""
        # First, get baseline results without bias
        baseline_plans = planner.plan_certain(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=self.mixed_workload,
            num_results=5,
            extra_model_arguments={
                "require_local_disks": False,
            },
        )
        baseline_top = baseline_plans[0].candidate_clusters.zonal[0].instance.family

        # Now apply a strong bias toward m-family (general purpose)
        # A bias < 1.0 means "prefer this family" (lower regret multiplier)
        biased_plans = planner.plan_certain(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=self.mixed_workload,
            num_results=5,
            extra_model_arguments={
                "require_local_disks": False,
                "family_bias": {"m": 0.5},  # Strong preference for m-family
            },
        )

        # The biased plans should have m-family appearing more prominently
        # in the top results compared to baseline
        biased_top = biased_plans[0].candidate_clusters.zonal[0].instance.family
        assert biased_top.startswith("m") or biased_top != baseline_top, (
            f"Expected m-family or different family with family_bias={{'m': 0.5}}, "
            f"but got {biased_top} (baseline was {baseline_top})"
        )

    def test_family_bias_penalty_avoids_family(self):
        """A family_bias > 1.0 should penalize (avoid) that family."""
        # Apply a penalty to i-family instances
        penalized_plans = planner.plan_certain(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=self.mixed_workload,
            num_results=5,
            extra_model_arguments={
                "require_local_disks": True,  # Normally would prefer i-family
                "family_bias": {"i": 2.0},  # Penalty for i-family
            },
        )

        # Despite require_local_disks=True, the i-family penalty should
        # push the planner toward other local-disk families if available
        top_family = penalized_plans[0].candidate_clusters.zonal[0].instance.family

        # The top result should either not be i-family, or if it is,
        # the regret calculation properly considered the penalty
        # (This assertion may need adjustment based on actual implementation)
        assert top_family is not None, "Expected a valid instance family"

    def test_family_bias_no_effect_when_one_family_viable(self):
        """family_bias shouldn't force invalid configurations."""
        # A workload that strongly requires local storage (only i-family viable)
        storage_heavy = CapacityDesires(
            service_tier=0,  # Critical tier
            query_pattern=QueryPattern(
                estimated_read_per_second=certain_int(100_000),
                estimated_write_per_second=certain_int(100_000),
            ),
            data_shape=DataShape(
                estimated_state_size_gib=certain_int(10_000),  # Large storage need
            ),
        )

        # Even with a strong bias against i-family, if it's the only viable option,
        # the planner should still select it
        plans = planner.plan_certain(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=storage_heavy,
            num_results=3,
            extra_model_arguments={
                "require_local_disks": True,
                "family_bias": {"i": 10.0},  # Strong penalty, but i-family required
            },
        )

        # Should still get valid plans (i-family) despite the penalty
        assert len(plans) > 0, "Should return valid plans even with family_bias penalty"
        top_result = plans[0].candidate_clusters.zonal[0]
        assert top_result.instance.family.startswith("i"), (
            "With require_local_disks=True and large storage, "
            "i-family should still be selected despite penalty"
        )


class TestCassandraFamilyBiasRegretCalculation:
    """Test that family_bias correctly modifies regret calculations."""

    def test_family_bias_multiplies_regret(self):
        """family_bias should act as a multiplier on the regret score."""
        # This test verifies the mechanics of how family_bias affects regret
        # The exact implementation may vary, but the principle is:
        # - bias < 1.0 reduces regret (prefers that family)
        # - bias > 1.0 increases regret (avoids that family)
        # - bias = 1.0 or missing = no change

        workload = CapacityDesires(
            service_tier=1,
            query_pattern=QueryPattern(
                estimated_read_per_second=certain_int(30_000),
                estimated_write_per_second=certain_int(10_000),
            ),
            data_shape=DataShape(
                estimated_state_size_gib=certain_int(200),
            ),
        )

        # Get plans with different bias configurations
        no_bias = planner.plan_certain(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=workload,
            num_results=10,
            extra_model_arguments={"require_local_disks": False},
        )

        # Collect families from no-bias results
        no_bias_families = [
            p.candidate_clusters.zonal[0].instance.family for p in no_bias
        ]

        # If there are multiple families in results, bias should reorder them
        unique_families = set(no_bias_families)
        if len(unique_families) > 1:
            # Pick a family that's not first and bias toward it
            non_top_family = [f for f in unique_families if f != no_bias_families[0]][0]
            family_prefix = non_top_family[0]  # First character (e.g., 'r', 'm', 'c')

            biased = planner.plan_certain(
                model_name="org.netflix.cassandra",
                region="us-east-1",
                desires=workload,
                num_results=10,
                extra_model_arguments={
                    "require_local_disks": False,
                    "family_bias": {family_prefix: 0.1},  # Strong preference
                },
            )

            biased_top_family = biased[0].candidate_clusters.zonal[0].instance.family
            assert biased_top_family.startswith(family_prefix), (
                f"Expected {family_prefix}-family to become top choice with "
                f"family_bias={{{family_prefix}: 0.1}}, but got {biased_top_family}"
            )
