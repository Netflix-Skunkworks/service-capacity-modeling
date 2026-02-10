"""Tests for Cassandra large_instance_bias model argument.

The large_instance_bias argument adds a rank penalty when the proposed plan
uses the largest instance in its family. This discourages selecting max-size
instances, reserving them for emergency scale-ups (e.g., avoid 16xlarge so
it's available if we need to scale up quickly).
"""

from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import certain_int
from service_capacity_modeling.interface import DataShape
from service_capacity_modeling.interface import QueryPattern
from service_capacity_modeling.models.org.netflix.cassandra import (
    NflxCassandraArguments,
)


class TestCassandraLargeInstanceBiasArgument:
    """Test that large_instance_bias is accepted as a model argument."""

    def test_large_instance_bias_argument_accepted(self):
        """NflxCassandraArguments should accept large_instance_bias parameter."""
        args = NflxCassandraArguments.from_extra_model_arguments(
            {"large_instance_bias": 2.0}
        )
        assert args.large_instance_bias == 2.0

    def test_large_instance_bias_default_is_none(self):
        """large_instance_bias should default to None when not specified."""
        args = NflxCassandraArguments.from_extra_model_arguments({})
        assert args.large_instance_bias is None


class TestCassandraLargeInstanceBiasSelection:
    """Test that large_instance_bias influences instance selection."""

    def test_large_instance_bias_penalizes_largest(self):
        """With large_instance_bias, largest instances get rank penalty."""
        # Large workload that will include some plans using largest instances
        # (e.g., i3.4xlarge is the largest i3 in the hardware catalog)
        desires = CapacityDesires(
            service_tier=1,
            query_pattern=QueryPattern(
                estimated_read_per_second=certain_int(500_000),
                estimated_write_per_second=certain_int(200_000),
            ),
            data_shape=DataShape(
                estimated_state_size_gib=certain_int(10_000),
            ),
        )

        # Get plans with large_instance_bias
        plans = planner.plan_certain(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=desires,
            num_results=30,
            extra_model_arguments={
                "require_local_disks": False,
                "large_instance_bias": 5.0,
            },
        )

        # Find plans that are marked as largest in their family
        largest_instance_plans = [
            (i, p)
            for i, p in enumerate(plans)
            if p.candidate_clusters.zonal[0].cluster_params.get("is_largest_in_family")
        ]
        non_largest_plans = [
            (i, p)
            for i, p in enumerate(plans)
            if not p.candidate_clusters.zonal[0].cluster_params.get(
                "is_largest_in_family"
            )
        ]

        # We should have some of each (otherwise test isn't useful)
        assert len(largest_instance_plans) > 0, (
            "Expected some plans using largest instances"
        )
        assert len(non_largest_plans) > 0, (
            "Expected some plans not using largest instances"
        )

        # With a strong bias, largest instance plans should tend to rank lower
        # Check that the average position of largest-instance plans is higher
        # (i.e., they appear later in the sorted list)
        avg_largest_pos = sum(i for i, _ in largest_instance_plans) / len(
            largest_instance_plans
        )
        avg_non_largest_pos = sum(i for i, _ in non_largest_plans) / len(
            non_largest_plans
        )

        # With bias=5.0, largest instances should average later positions.
        # Not guaranteed for every workload, but with strong bias it should hold
        assert avg_largest_pos > avg_non_largest_pos, (
            f"Expected largest instances to rank lower on average. "
            f"Avg position of largest: {avg_largest_pos:.1f}, "
            f"Avg position of non-largest: {avg_non_largest_pos:.1f}"
        )

    def test_large_instance_bias_metadata_stored(self):
        """Plans should have large_instance_bias metadata in cluster_params."""
        desires = CapacityDesires(
            service_tier=1,
            query_pattern=QueryPattern(
                estimated_read_per_second=certain_int(50_000),
                estimated_write_per_second=certain_int(20_000),
            ),
            data_shape=DataShape(
                estimated_state_size_gib=certain_int(500),
            ),
        )

        plans = planner.plan_certain(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=desires,
            num_results=5,
            extra_model_arguments={
                "require_local_disks": False,
                "large_instance_bias": 2.0,
            },
        )

        # All plans should have the metadata
        for plan in plans:
            cluster = plan.candidate_clusters.zonal[0]
            assert "large_instance_bias" in cluster.cluster_params
            assert "is_largest_in_family" in cluster.cluster_params
            assert cluster.cluster_params["large_instance_bias"] == 2.0

    def test_large_instance_bias_no_effect_without_bias(self):
        """Without large_instance_bias, metadata should not be present."""
        desires = CapacityDesires(
            service_tier=1,
            query_pattern=QueryPattern(
                estimated_read_per_second=certain_int(50_000),
                estimated_write_per_second=certain_int(20_000),
            ),
            data_shape=DataShape(
                estimated_state_size_gib=certain_int(500),
            ),
        )

        plans = planner.plan_certain(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=desires,
            num_results=5,
            extra_model_arguments={
                "require_local_disks": False,
            },
        )

        # Plans should not have the large_instance metadata
        for plan in plans:
            cluster = plan.candidate_clusters.zonal[0]
            assert "large_instance_bias" not in cluster.cluster_params
            assert "is_largest_in_family" not in cluster.cluster_params
