"""Tests for Cassandra same_family_bias model argument.

The same_family_bias argument adds a cost penalty when the proposed plan
uses a different instance family than the current cluster. This encourages
staying on the same family during capacity changes, reducing operational
risk from family switches.
"""

from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import certain_float
from service_capacity_modeling.interface import certain_int
from service_capacity_modeling.interface import CurrentClusters
from service_capacity_modeling.interface import CurrentZoneClusterCapacity
from service_capacity_modeling.interface import DataShape
from service_capacity_modeling.interface import Interval
from service_capacity_modeling.interface import QueryPattern
from service_capacity_modeling.models.org.netflix.cassandra import (
    NflxCassandraArguments,
)


class TestCassandraSameFamilyBiasArgument:
    """Test that same_family_bias is accepted as a model argument."""

    def test_same_family_bias_argument_accepted(self):
        """NflxCassandraArguments should accept same_family_bias parameter."""
        args = NflxCassandraArguments.from_extra_model_arguments(
            {"same_family_bias": 2.0}
        )
        assert args.same_family_bias == 2.0

    def test_same_family_bias_default_is_none(self):
        """same_family_bias should default to None when not specified."""
        args = NflxCassandraArguments.from_extra_model_arguments({})
        assert args.same_family_bias is None


class TestCassandraSameFamilyBiasSelection:
    """Test that same_family_bias influences instance family selection."""

    def test_same_family_bias_prefers_current_family(self):
        """With same_family_bias, plans should prefer current family."""
        # Current cluster is on i4i family
        current_cluster = CurrentZoneClusterCapacity(
            cluster_instance_name="i4i.2xlarge",
            cluster_instance_count=Interval(low=4, mid=4, high=4, confidence=1),
            cpu_utilization=certain_float(50.0),
            memory_utilization_gib=certain_float(32.0),
            disk_utilization_gib=certain_float(500.0),
            network_utilization_mbps=certain_float(100.0),
        )

        desires_with_current = CapacityDesires(
            service_tier=1,
            current_clusters=CurrentClusters(zonal=[current_cluster]),
            query_pattern=QueryPattern(
                estimated_read_per_second=certain_int(50_000),
                estimated_write_per_second=certain_int(20_000),
            ),
            data_shape=DataShape(
                estimated_state_size_gib=certain_int(500),
            ),
        )

        # With same_family_bias - should prefer staying on i-family
        biased_plans = planner.plan_certain(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=desires_with_current,
            num_results=5,
            extra_model_arguments={
                "require_local_disks": False,
                "same_family_bias": 2.0,  # 2x cost penalty for switching families
            },
        )

        # The biased plans should have i-family at the top (same as current)
        biased_top = biased_plans[0].candidate_clusters.zonal[0].instance.family
        current_family = "i4i"

        # With bias, same family should be preferred
        assert biased_top.startswith("i"), (
            f"Expected i-family (same as current {current_family}) with "
            f"same_family_bias=2.0, but got {biased_top}"
        )

    def test_same_family_bias_no_effect_without_current(self):
        """same_family_bias should have no effect when no current cluster."""
        desires_no_current = CapacityDesires(
            service_tier=1,
            query_pattern=QueryPattern(
                estimated_read_per_second=certain_int(50_000),
                estimated_write_per_second=certain_int(20_000),
            ),
            data_shape=DataShape(
                estimated_state_size_gib=certain_int(500),
            ),
        )

        # Without same_family_bias
        baseline_plans = planner.plan_certain(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=desires_no_current,
            num_results=5,
            extra_model_arguments={
                "require_local_disks": False,
            },
        )

        # With same_family_bias (should have no effect - no current cluster)
        biased_plans = planner.plan_certain(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=desires_no_current,
            num_results=5,
            extra_model_arguments={
                "require_local_disks": False,
                "same_family_bias": 2.0,
            },
        )

        # Results should be the same since there's no current cluster
        baseline_top = baseline_plans[0].candidate_clusters.zonal[0].instance.family
        biased_top = biased_plans[0].candidate_clusters.zonal[0].instance.family

        assert baseline_top == biased_top, (
            f"Expected same results without current cluster, but baseline got "
            f"{baseline_top} and biased got {biased_top}"
        )

    def test_same_family_bias_higher_value_stronger_preference(self):
        """Higher same_family_bias values should more strongly prefer same family."""
        # Current cluster is on m6i family
        current_cluster = CurrentZoneClusterCapacity(
            cluster_instance_name="m6i.4xlarge",
            cluster_instance_count=Interval(low=6, mid=6, high=6, confidence=1),
            cpu_utilization=certain_float(40.0),
            memory_utilization_gib=certain_float(48.0),
            disk_utilization_gib=certain_float(200.0),
            network_utilization_mbps=certain_float(150.0),
        )

        desires = CapacityDesires(
            service_tier=1,
            current_clusters=CurrentClusters(zonal=[current_cluster]),
            query_pattern=QueryPattern(
                estimated_read_per_second=certain_int(30_000),
                estimated_write_per_second=certain_int(10_000),
            ),
            data_shape=DataShape(
                estimated_state_size_gib=certain_int(200),
            ),
        )

        # With very high same_family_bias - m-family should definitely be preferred
        strongly_biased_plans = planner.plan_certain(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=desires,
            num_results=5,
            extra_model_arguments={
                "require_local_disks": False,
                "same_family_bias": 10.0,  # Very strong preference for same family
            },
        )

        top_family = (
            strongly_biased_plans[0].candidate_clusters.zonal[0].instance.family
        )
        assert top_family.startswith("m"), (
            f"Expected m-family (same as current m6i) with same_family_bias=10.0, "
            f"but got {top_family}"
        )
