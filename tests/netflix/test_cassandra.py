import pytest

from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.hardware import shapes
from service_capacity_modeling.interface import AccessConsistency
from service_capacity_modeling.interface import AccessPattern
from service_capacity_modeling.interface import Buffer
from service_capacity_modeling.interface import BufferComponent
from service_capacity_modeling.interface import BufferIntent
from service_capacity_modeling.interface import Buffers
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import certain_float
from service_capacity_modeling.interface import certain_int
from service_capacity_modeling.interface import Consistency
from service_capacity_modeling.interface import CurrentClusters
from service_capacity_modeling.interface import CurrentZoneClusterCapacity
from service_capacity_modeling.interface import DataShape
from service_capacity_modeling.interface import fixed_float
from service_capacity_modeling.interface import FixedInterval
from service_capacity_modeling.interface import GlobalConsistency
from service_capacity_modeling.interface import Interval
from service_capacity_modeling.interface import QueryPattern
from service_capacity_modeling.models.org.netflix.cassandra import (
    NflxCassandraCapacityModel,
)
from tests.util import assert_similar_compute

small_but_high_qps = CapacityDesires(
    service_tier=1,
    query_pattern=QueryPattern(
        estimated_read_per_second=certain_int(100_000),
        estimated_write_per_second=certain_int(100_000),
        estimated_mean_read_latency_ms=certain_float(0.5),
        estimated_mean_write_latency_ms=certain_float(0.4),
    ),
    data_shape=DataShape(
        estimated_state_size_gib=certain_int(10),
    ),
)

high_writes = CapacityDesires(
    service_tier=1,
    query_pattern=QueryPattern(
        estimated_read_per_second=certain_int(10_000),
        estimated_write_per_second=certain_int(100_000),
    ),
    data_shape=DataShape(
        estimated_state_size_gib=certain_int(300),
    ),
)

large_footprint = CapacityDesires(
    service_tier=1,
    query_pattern=QueryPattern(
        estimated_read_per_second=certain_int(60000),
        estimated_write_per_second=certain_int(60000),
        estimated_mean_read_latency_ms=certain_float(0.8),
        estimated_mean_write_latency_ms=certain_float(0.5),
    ),
    data_shape=DataShape(
        estimated_state_size_gib=certain_int(4000),
    ),
)


class TestCassandraCapacityPlanning:
    """Test basic capacity planning scenarios."""

    def test_capacity_small_fast(self):
        for require_local_disks in (True, False):
            cap_plan = planner.plan_certain(
                model_name="org.netflix.cassandra",
                region="us-east-1",
                desires=small_but_high_qps,
                extra_model_arguments={"require_local_disks": require_local_disks},
            )[0]
            small_result = cap_plan.candidate_clusters.zonal[0]
            # We really should just pay for CPU here
            assert small_result.instance.name.startswith("c")

            cores = small_result.count * small_result.instance.cpu
            assert 30 <= cores <= 80
            # Even though it's a small dataset we need IOs so should end up
            # with lots of ebs_gp2 to handle the read IOs
            if small_result.attached_drives:
                assert (
                    small_result.count
                    * sum(d.size_gib for d in small_result.attached_drives)
                    > 1000
                )

            assert small_result.cluster_params["cassandra.heap.write.percent"] == 0.25
            assert small_result.cluster_params["cassandra.heap.table.percent"] == 0.11

    def test_capacity_high_writes(self):
        cap_plan = planner.plan_certain(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=high_writes,
            extra_model_arguments={"copies_per_region": 2},
        )[0]
        high_writes_result = cap_plan.candidate_clusters.zonal[0]
        assert high_writes_result.instance.family.startswith("c")
        assert high_writes_result.count > 4

        num_cpus = high_writes_result.instance.cpu * high_writes_result.count
        assert 30 <= num_cpus <= 128
        if high_writes_result.attached_drives:
            assert (
                high_writes_result.count
                * high_writes_result.attached_drives[0].size_gib
                >= 400
            )
        elif high_writes_result.instance.drive is not None:
            assert (
                high_writes_result.count * high_writes_result.instance.drive.size_gib
                >= 400
            )
        else:
            raise AssertionError("Should have drives")
        assert (
            cap_plan.candidate_clusters.annual_costs["cassandra.zonal-clusters"]
            < 40_000
        )

    def test_capacity_large_footprint(self):
        cap_plan = planner.plan_certain(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=large_footprint,
            extra_model_arguments={
                "require_local_disks": True,
                "required_cluster_size": 16,
            },
        )[0]

        large_footprint_result = cap_plan.candidate_clusters.zonal[0]
        assert large_footprint_result.instance.name.startswith("i")
        assert large_footprint_result.count == 16

        # Should have been able to use default heap settings
        assert (
            large_footprint_result.cluster_params["cassandra.heap.write.percent"]
            == 0.25
        )
        assert (
            large_footprint_result.cluster_params["cassandra.heap.table.percent"]
            == 0.11
        )
        assert (
            large_footprint_result.cluster_params["cassandra.compaction.min_threshold"]
            == 4
        )

    def test_capacity_non_power_of_two(self):
        cap_plan = planner.plan_certain(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=large_footprint,
            extra_model_arguments={
                "require_local_disks": True,
                "required_cluster_size": 12,
            },
        )[0]

        result = cap_plan.candidate_clusters.zonal[0]
        assert result.count == 12
        assert result.instance.name.startswith("i")


class TestCassandraStorage:
    """Test storage-related scenarios."""

    def test_ebs_high_reads(self):
        cap_plan = planner.plan_certain(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=CapacityDesires(
                service_tier=1,
                query_pattern=QueryPattern(
                    estimated_read_per_second=certain_int(100_000),
                    estimated_write_per_second=certain_int(1_000),
                ),
                data_shape=DataShape(
                    estimated_state_size_gib=certain_int(1_000),
                ),
            ),
            extra_model_arguments={
                "require_attached_disks": True,
                "require_local_disks": False,
            },
        )[0]
        result = cap_plan.candidate_clusters.zonal[0]

        cores = result.count * result.instance.cpu
        assert 64 <= cores <= 128
        # Should get gp3
        assert result.attached_drives[0].name == "gp3"
        # 1TiB / ~32 nodes
        assert result.attached_drives[0].read_io_per_s is not None
        ios = result.attached_drives[0].read_io_per_s * result.count
        # Each zone is handling ~33k reads per second, so total disk ios should be < 3x
        # that 3 from each level
        assert 100_000 < ios < 400_000

    def test_ebs_high_writes(self):
        cap_plan = planner.plan_certain(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=CapacityDesires(
                service_tier=1,
                query_pattern=QueryPattern(
                    estimated_read_per_second=certain_int(10_000),
                    estimated_write_per_second=certain_int(100_000),
                    estimated_mean_write_size_bytes=certain_int(1024 * 8),
                ),
                data_shape=DataShape(
                    estimated_state_size_gib=certain_int(10_000),
                ),
            ),
            extra_model_arguments={
                "require_attached_disks": True,
                "require_local_disks": False,
            },
        )[0]
        result = cap_plan.candidate_clusters.zonal[0]

        cores = result.count * result.instance.cpu
        assert 128 <= cores <= 512
        # Should get gp3
        assert result.attached_drives[0].name == "gp3"
        # 1TiB / ~32 nodes
        assert result.attached_drives[0].read_io_per_s is not None
        assert result.attached_drives[0].write_io_per_s is not None

        read_ios = result.attached_drives[0].read_io_per_s * result.count
        write_ios = result.attached_drives[0].write_io_per_s * result.count

        # 10TiB ~= 4 IO/read -> 3.3k r/zone/s -> 12k /s
        assert 20_000 < read_ios < 60_000
        # 33k wps * 8KiB  / 256KiB write IO size = 16.5k / s * 4 for compaction = 6.4k
        assert 4_000 < write_ios < 7_000


def test_capacity_high_writes():
    cap_plan = planner.plan_certain(
        model_name="org.netflix.cassandra",
        region="us-east-1",
        desires=high_writes,
        extra_model_arguments={"copies_per_region": 2},
    )[0]
    high_writes_result = cap_plan.candidate_clusters.zonal[0]
    assert high_writes_result.instance.family.startswith("c")
    assert high_writes_result.count > 4

    num_cpus = high_writes_result.instance.cpu * high_writes_result.count
    assert 30 <= num_cpus <= 128
    if high_writes_result.attached_drives:
        assert (
            high_writes_result.count * high_writes_result.attached_drives[0].size_gib
            >= 400
        )
    elif high_writes_result.instance.drive is not None:
        assert (
            high_writes_result.count * high_writes_result.instance.drive.size_gib >= 400
        )
    else:
        raise AssertionError("Should have drives")
    assert cap_plan.candidate_clusters.annual_costs["cassandra.zonal-clusters"] < 40_000


class TestCassandraThroughput:
    """Test high throughput scenarios."""

    def test_high_write_throughput(self):
        desires = CapacityDesires(
            service_tier=1,
            query_pattern=QueryPattern(
                estimated_read_per_second=certain_int(1000),
                estimated_write_per_second=certain_int(1_000_000),
                # Really large writes
                estimated_mean_write_size_bytes=certain_int(4096),
            ),
            data_shape=DataShape(
                estimated_state_size_gib=certain_int(100_000),
            ),
        )

        cap_plan = planner.plan_certain(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=desires,
            extra_model_arguments={"max_regional_size": 96 * 2},
        )[0]
        high_writes_result = cap_plan.candidate_clusters.zonal[0]
        assert high_writes_result.instance.family not in ("m5", "r5")
        assert high_writes_result.count > 16

        cluster_cost = cap_plan.candidate_clusters.annual_costs[
            "cassandra.zonal-clusters"
        ]
        assert 125_000 < cluster_cost < 900_000

        # We should require more than 4 tiering in order to meet this requirement
        assert (
            high_writes_result.cluster_params["cassandra.compaction.min_threshold"] > 4
        )

    def test_high_write_throughput_ebs(self):
        desires = CapacityDesires(
            service_tier=1,
            query_pattern=QueryPattern(
                estimated_read_per_second=certain_int(1000),
                estimated_write_per_second=certain_int(1_000_000),
                # Really large writes
                estimated_mean_write_size_bytes=certain_int(4096),
            ),
            data_shape=DataShape(
                estimated_state_size_gib=certain_int(100_000),
            ),
        )

        cap_plan = planner.plan_certain(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=desires,
            extra_model_arguments={
                "max_regional_size": 96 * 2,
                "require_local_disks": False,
                "require_attached_disks": True,
            },
        )[0]
        high_writes_result = cap_plan.candidate_clusters.zonal[0]

        assert high_writes_result.instance.family[0] in ("m", "r")
        assert high_writes_result.count > 32

        assert high_writes_result.attached_drives[0].size_gib >= 400
        assert (
            300_000
            > high_writes_result.count * high_writes_result.attached_drives[0].size_gib
            >= 100_000
        )

        cluster_cost = cap_plan.candidate_clusters.annual_costs[
            "cassandra.zonal-clusters"
        ]
        assert 125_000 < cluster_cost < 900_000

        # We should require more than 4 tiering in order to meet this requirement
        assert (
            high_writes_result.cluster_params["cassandra.compaction.min_threshold"] > 4
        )


class TestCassandraDurability:
    """Test durability and consistency scenarios."""

    def test_reduced_durability(self):
        expensive = CapacityDesires(
            service_tier=1,
            query_pattern=QueryPattern(
                estimated_read_per_second=certain_int(1000),
                estimated_write_per_second=certain_int(1_000_000),
            ),
            data_shape=DataShape(
                estimated_state_size_gib=certain_int(100_000),
            ),
        )

        cheaper = CapacityDesires(
            service_tier=1,
            query_pattern=QueryPattern(
                estimated_read_per_second=certain_int(1000),
                estimated_write_per_second=certain_int(1_000_000),
                access_consistency=GlobalConsistency(
                    same_region=Consistency(
                        target_consistency=AccessConsistency.eventual
                    )
                ),
            ),
            data_shape=DataShape(
                estimated_state_size_gib=certain_int(100_000),
                durability_slo_order=FixedInterval(low=10, mid=100, high=100000),
            ),
        )

        expensive_plan = planner.plan_certain(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=expensive,
        )[0]

        cheap_plan = planner.plan_certain(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=cheaper,
        )[0]

        assert cheap_plan.candidate_clusters.total_annual_cost < (
            0.7 * float(expensive_plan.candidate_clusters.total_annual_cost)
        )
        # The reduced durability and consistency requirement let's us
        # use less compute
        assert expensive_plan.requirements.zonal[0].context["replication_factor"] == 3
        assert cheap_plan.requirements.zonal[0].context["replication_factor"] == 2

        # Due to high writes both should have high heap write buffering
        for plan in (expensive_plan, cheap_plan):
            assert (
                plan.candidate_clusters.zonal[0].cluster_params[
                    "cassandra.heap.write.percent"
                ]
                == 0.5
            )
            assert (
                plan.candidate_clusters.zonal[0].cluster_params[
                    "cassandra.heap.table.percent"
                ]
                == 0.2
            )
            assert (
                plan.candidate_clusters.zonal[0].cluster_params[
                    "cassandra.compaction.min_threshold"
                ]
                == 8
            )

        assert (
            cheap_plan.candidate_clusters.zonal[0].cluster_params[
                "cassandra.keyspace.rf"
            ]
            == 2
        )


class TestCassandraCurrentCapacity:
    """Test scenarios with current capacity information."""

    def test_plan_certain(self):
        """
        Use cpu utilization to determine instance types directly as supposed to
        extrapolating it from the Data Shape
        """
        cluster_capacity = CurrentZoneClusterCapacity(
            cluster_instance_name="i4i.8xlarge",
            cluster_instance_count=Interval(low=8, mid=8, high=8, confidence=1),
            cpu_utilization=Interval(
                low=10.12, mid=13.1, high=14.194801291058118, confidence=1
            ),
            memory_utilization_gib=certain_float(32.0),
            network_utilization_mbps=certain_float(128.0),
        )

        worn_desire = CapacityDesires(
            service_tier=1,
            current_clusters=CurrentClusters(zonal=[cluster_capacity]),
            query_pattern=QueryPattern(
                access_pattern=AccessPattern(AccessPattern.latency),
                estimated_read_per_second=Interval(
                    low=234248, mid=351854, high=485906, confidence=0.98
                ),
                estimated_write_per_second=Interval(
                    low=19841, mid=31198, high=37307, confidence=0.98
                ),
            ),
            # We think we're going to have around 200 TiB of data
            data_shape=DataShape(
                estimated_state_size_gib=Interval(
                    low=2006.083, mid=2252.5, high=2480.41, confidence=0.98
                ),
                estimated_compression_ratio=Interval(
                    low=1, mid=1, high=1, confidence=1
                ),
            ),
        )
        cap_plan = planner.plan_certain(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            num_results=3,
            num_regions=4,
            desires=worn_desire,
            extra_model_arguments={
                "required_cluster_size": 8,
            },
        )

        # Use a similar number of CPU cores but allocate less disk
        lr_clusters = cap_plan[0].candidate_clusters.zonal[0]
        assert_similar_compute(
            shapes.instance("i4i.4xlarge"), lr_clusters.instance, 8, lr_clusters.count
        )

    def test_preserve_memory(self):
        cluster_capacity = CurrentZoneClusterCapacity(
            cluster_instance_name="r5d.4xlarge",
            cluster_instance_count=Interval(low=2, mid=2, high=2, confidence=1),
            cpu_utilization=Interval(
                low=10.12, mid=13.2, high=14.194801291058118, confidence=1
            ),
            memory_utilization_gib=certain_float(32.0),
            disk_utilization_gib=certain_float(100),
            network_utilization_mbps=certain_float(128.0),
        )

        derived_buffer = Buffers(
            derived={
                "memory": Buffer(
                    intent=BufferIntent.preserve,
                    components=[BufferComponent.memory],
                )
            }
        )

        worn_desire = CapacityDesires(
            service_tier=1,
            current_clusters=CurrentClusters(zonal=[cluster_capacity]),
            query_pattern=QueryPattern(
                estimated_read_per_second=certain_int(10_000),
                estimated_write_per_second=certain_int(100_000),
            ),
            data_shape=DataShape(
                estimated_state_size_gib=certain_int(300),
            ),
            buffers=derived_buffer,
        )
        cap_plan = planner.plan_certain(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            num_results=3,
            num_regions=4,
            desires=worn_desire,
            extra_model_arguments={
                "required_cluster_size": 2,
            },
        )

        lr_clusters = cap_plan[0].candidate_clusters.zonal[0]
        assert lr_clusters.instance.ram_gib == 128

    def test_capacity_non_power_of_two(self):
        cluster_capacity = CurrentZoneClusterCapacity(
            cluster_instance_name="r5d.4xlarge",
            cluster_instance_count=fixed_float(3),
            cpu_utilization=certain_float(80),
            memory_utilization_gib=certain_float(32.0),
            disk_utilization_gib=certain_float(2048),
            network_utilization_mbps=certain_float(128.0),
        )
        desires = CapacityDesires(
            service_tier=1,
            current_clusters=CurrentClusters(zonal=[cluster_capacity]),
        )
        cap_plan = planner.plan_certain(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=desires,
            extra_model_arguments={
                "require_local_disks": True,
            },
        )[0]
        result = cap_plan.candidate_clusters.zonal[0]
        # Doubles a 3 node cluster to 6
        assert result.count == 6

    def test_capacity_non_power_of_two_with_required_size(self):
        cluster_capacity = CurrentZoneClusterCapacity(
            cluster_instance_name="r5d.4xlarge",
            cluster_instance_count=fixed_float(3),
            cpu_utilization=Interval(
                low=10.12, mid=30, high=14.194801291058118, confidence=1
            ),
            memory_utilization_gib=certain_float(32.0),
            disk_utilization_gib=certain_float(1024),
            network_utilization_mbps=certain_float(128.0),
        )
        desires = CapacityDesires(
            service_tier=1,
            current_clusters=CurrentClusters(zonal=[cluster_capacity]),
        )
        cap_plan = planner.plan_certain(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=desires,
            extra_model_arguments={
                "require_local_disks": True,
                "required_cluster_size": 24,
            },
        )[0]

        result = cap_plan.candidate_clusters.zonal[0]
        assert result.count == 24


class TestCassandraExtraModelArguments:
    """Test model argument validation."""

    @pytest.mark.parametrize(
        "tier, extra_model_arguments, expected_result",
        [
            # Non-critical tier, no required_cluster_size
            (2, {}, None),
            # Non-critical tier, required_cluster_size provided
            (2, {"required_cluster_size": 5}, 5),
            # Critical tier, required_cluster_size >= CRITICAL_TIER_MIN_CLUSTER_SIZE
            (0, {"required_cluster_size": 3}, 3),
            (0, {"required_cluster_size": 2}, 2),
            # Critical tier, no required_cluster_size
            (0, {}, None),
        ],
    )
    def test_get_required_cluster_size_valid(
        self, tier, extra_model_arguments, expected_result
    ):
        result = NflxCassandraCapacityModel.get_required_cluster_size(
            tier, extra_model_arguments
        )
        assert result == expected_result

    @pytest.mark.parametrize(
        "tier, extra_model_arguments",
        [
            # Critical tier(s), required_cluster_size < CRITICAL_TIER_MIN_CLUSTER_SIZE
            (
                1,
                {"required_cluster_size": 1},
            ),
            (
                0,
                {"required_cluster_size": 1},
            ),
        ],
    )
    def test_get_required_cluster_size_exceptions(self, tier, extra_model_arguments):
        with pytest.raises(ValueError):
            NflxCassandraCapacityModel.get_required_cluster_size(
                tier, extra_model_arguments
            )
