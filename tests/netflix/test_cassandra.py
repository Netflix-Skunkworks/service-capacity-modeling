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
    _default_cluster_size_mode,
    _get_cluster_size_lambda,
    _get_min_count,
    CassandraClusterSizeMode,
    NflxCassandraCapacityModel,
)
from tests.util import assert_minimum_storage_gib
from tests.util import assert_similar_compute
from tests.util import get_total_storage_gib
from tests.util import has_local_storage
from tests.util import simple_drive

# TODO(homatthew): This is a workaround since EBS is disabled broadly for new
# provisionings (require_local_disks=True by default), but we still want to test
# with both local and attached disks in unit tests.
EXTRA_MODEL_ARGS = {"require_local_disks": False}

# Property test configuration for Cassandra model.
# See tests/netflix/PROPERTY_TESTING.md for configuration options and examples.
PROPERTY_TEST_CONFIG = {
    "org.netflix.cassandra": {
        # Cassandra critical tiers share the same default cluster-size policy.
        # Non-critical tiers can pick a different local-disk shape with more raw
        # capacity for the same workload, so the universal tier property should
        # compare the critical tier boundary directly.
        "tier_range": (0, 1),
    },
}

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
        estimated_write_per_second=certain_int(500_000),
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
            # CPU-bound workload: should pick compute or general-purpose family,
            # not memory-heavy r/i families (~7.6 GiB/vCPU)
            assert small_result.instance.ram_gib / small_result.instance.cpu <= 4.5

            cores = small_result.count * small_result.instance.cpu
            assert 30 <= cores <= 80
            # Even though it's a small dataset we need IOs so should end up
            # with lots of storage to handle the read IOs
            assert get_total_storage_gib(small_result) >= 1000

            assert small_result.cluster_params["cassandra.heap.write.percent"] == 0.25
            assert small_result.cluster_params["cassandra.heap.table.percent"] == 0.11

    def test_capacity_high_writes(self):
        cap_plan = planner.plan_certain(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=high_writes,
            extra_model_arguments={**EXTRA_MODEL_ARGS, "copies_per_region": 2},
        )[0]
        high_writes_result = cap_plan.candidate_clusters.zonal[0]
        assert high_writes_result.instance.family.startswith("c")

        # Storage should be sufficient for the data (300 GiB with buffer)
        assert_minimum_storage_gib(high_writes_result, 400)
        assert_similar_compute(
            shapes.instance("c7a.4xlarge"),
            high_writes_result.instance,
            expected_count=8,
            actual_count=high_writes_result.count,
            expected_attached_disk=simple_drive(
                size_gib=100, read_io_per_s=3400, write_io_per_s=200
            ),
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
        # With require_local_disks=True, should get local storage instances
        assert has_local_storage(result), (
            "Expected local storage with require_local_disks=True"
        )
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
        # Should get attached storage since we explicitly requested it
        assert result.attached_drives, (
            "Expected attached drives with require_attached_disks=True"
        )
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
        assert 64 <= cores <= 512
        # Should get attached storage since we explicitly requested it
        assert result.attached_drives, (
            "Expected attached drives with require_attached_disks=True"
        )
        assert result.attached_drives[0].name == "gp3"
        # 1TiB / ~32 nodes
        assert result.attached_drives[0].read_io_per_s is not None
        assert result.attached_drives[0].write_io_per_s is not None

        read_ios = result.attached_drives[0].read_io_per_s * result.count
        write_ios = result.attached_drives[0].write_io_per_s * result.count

        # 10TiB ~= 4 IO/read -> 3.3k r/zone/s -> 12k /s
        assert 20_000 < read_ios < 60_000
        # 33k wps * 8KiB / 256KiB write IO = ~6.4k base; page-cache cap
        # constrains memory denominator → more nodes → higher total IOs
        assert 4_000 < write_ios < 16_000

    def test_existing_ebs_volume_shrink_requires_extra_argument(self):
        current_drive_size_gib = 6000
        current_cluster = CurrentZoneClusterCapacity(
            cluster_instance=shapes.instance("m6a.4xlarge"),
            cluster_instance_name="m6a.4xlarge",
            cluster_drive=simple_drive(size_gib=current_drive_size_gib),
            cluster_instance_count=certain_int(8),
            cluster_type="cassandra",
            cpu_utilization=certain_float(1),
            memory_utilization_gib=certain_float(8),
            disk_utilization_gib=certain_float(4000),
            network_utilization_mbps=certain_float(1),
        )
        desires = CapacityDesires(
            service_tier=1,
            current_clusters=CurrentClusters(zonal=[current_cluster]),
            query_pattern=QueryPattern(
                access_pattern=AccessPattern.latency,
                estimated_read_per_second=certain_int(100),
                estimated_write_per_second=certain_int(100),
            ),
            data_shape=DataShape(estimated_state_size_gib=certain_int(100)),
            buffers=Buffers(
                derived={
                    "storage": Buffer(
                        ratio=0.5,
                        intent=BufferIntent.scale_down,
                        components=[BufferComponent.storage],
                    )
                }
            ),
        )

        def plan(allow_ebs_volume_shrink: bool):
            return planner.plan_certain(
                model_name="org.netflix.cassandra",
                region="us-east-1",
                desires=desires,
                extra_model_arguments={
                    "require_attached_disks": True,
                    "require_local_disks": False,
                    "cluster_size_mode": "unrestricted",
                    "allow_ebs_volume_shrink": allow_ebs_volume_shrink,
                },
                instance_families=["m6a"],
                num_results=1,
            )[0].candidate_clusters.zonal[0]

        default_result = plan(False)
        shrink_result = plan(True)

        assert default_result.attached_drives[0].size_gib >= current_drive_size_gib
        assert shrink_result.attached_drives[0].size_gib < current_drive_size_gib


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
            extra_model_arguments={**EXTRA_MODEL_ARGS, "max_regional_size": 96 * 2},
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

        # With attached disks requested, stay in stateful datastore families.
        assert high_writes_result.instance.family in {
            "c6a",
            "c7a",
            "m6a",
            "m7a",
            "r6a",
            "r7a",
        }
        assert high_writes_result.count >= 32

        # Should have attached storage since we explicitly requested it
        assert high_writes_result.attached_drives, (
            "Expected attached drives with require_attached_disks=True"
        )
        assert high_writes_result.attached_drives[0].size_gib >= 400
        total_storage = get_total_storage_gib(high_writes_result)
        # EBS applies a hotter disk buffer on top of the adaptive storage buffer.
        assert 30_000 <= total_storage < 100_000

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
            extra_model_arguments=EXTRA_MODEL_ARGS,
        )[0]

        cheap_plan = planner.plan_certain(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=cheaper,
            extra_model_arguments=EXTRA_MODEL_ARGS,
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
        # A CPU threshold larger than this will cause CPU to remain the same.
        # This is a magic number based on the current logic and does not hold any
        # particular significance. Modify this value slightly if necessary as
        # new logic is introduced and behaviors change
        cpu_threshold = 13.1
        cluster_capacity = CurrentZoneClusterCapacity(
            cluster_instance_name="i4i.8xlarge",
            cluster_instance_count=Interval(low=8, mid=8, high=8, confidence=1),
            cpu_utilization=Interval(low=10, mid=cpu_threshold, high=14, confidence=1),
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
                **EXTRA_MODEL_ARGS,
                "required_cluster_size": 8,
            },
        )

        lr_clusters = cap_plan[0].candidate_clusters.zonal[0]
        assert lr_clusters.instance.ram_gib < 256, (
            f"Cap should prevent 256 GiB RAM instances, got {lr_clusters.instance.name}"
        )

    def test_preserve_memory(self):
        """Memory preserve buffer keeps current cluster's page cache."""
        cluster = CurrentZoneClusterCapacity(
            cluster_instance_name="r5d.4xlarge",
            cluster_instance=shapes.instance("r5d.4xlarge"),
            cluster_instance_count=certain_int(2),
            cpu_utilization=certain_float(13.0),
            disk_utilization_gib=certain_float(100),
            network_utilization_mbps=certain_float(128.0),
        )
        desire = CapacityDesires(
            service_tier=1,
            current_clusters=CurrentClusters(zonal=[cluster]),
            query_pattern=QueryPattern(
                estimated_read_per_second=certain_int(10_000),
                estimated_write_per_second=certain_int(100_000),
            ),
            data_shape=DataShape(estimated_state_size_gib=certain_int(300)),
            buffers=Buffers(
                derived={
                    "memory": Buffer(
                        intent=BufferIntent.preserve,
                        components=[BufferComponent.memory],
                    )
                }
            ),
        )
        cap_plan = planner.plan_certain(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=desire,
            extra_model_arguments={
                **EXTRA_MODEL_ARGS,
                "required_cluster_size": 8,
            },
            instance_families=["r6id"],
        )
        assert cap_plan, "Expected at least one plan for preserve memory"

        # Preserve is applied at hard-memory node sizing, after the raw
        # Cassandra memory requirement has been calculated.
        cluster_params = cap_plan[0].candidate_clusters.zonal[0].cluster_params
        assert cluster_params["required_nodes_by_type"]["memory"] == 8

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
        assert result.count == 6

    def test_capacity_non_power_of_two_with_doubling_mode(self):
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
                "cluster_size_mode": "doubling",
            },
        )[0]
        result = cap_plan.candidate_clusters.zonal[0]
        counts = result.cluster_params["required_nodes_by_type"]
        assert counts["min_count"] == 6

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
                "cluster_size_mode": "unrestricted",
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

    @pytest.mark.parametrize("tier", [2, 3, 4])
    def test_non_critical_tiers_do_not_round_cluster_size(self, tier):
        cluster_size = _get_cluster_size_lambda(
            cluster_size_mode=_default_cluster_size_mode(tier),
        )

        assert (
            _get_min_count(
                tier=tier,
                required_cluster_size=None,
                needed_disk_gib=3,
                disk_per_node_gib=1,
                cluster_size_lambda=cluster_size,
            )
            == 3
        )

    def test_cluster_size_lambda_defaults_to_doubling_mode(self):
        cluster_size = _get_cluster_size_lambda()

        assert cluster_size(3) == 4

    @pytest.mark.parametrize(
        "tier, expected_mode",
        [
            (0, CassandraClusterSizeMode.doubling),
            (1, CassandraClusterSizeMode.doubling),
            (2, CassandraClusterSizeMode.unrestricted),
            (3, CassandraClusterSizeMode.unrestricted),
            (4, CassandraClusterSizeMode.unrestricted),
        ],
    )
    def test_default_cluster_size_mode_is_tier_based(self, tier, expected_mode):
        assert _default_cluster_size_mode(tier) == expected_mode

    @pytest.mark.parametrize("tier", [2, 3, 4])
    def test_non_critical_tiers_do_not_round_above_required_cluster_size(self, tier):
        cluster_size = _get_cluster_size_lambda(
            cluster_size_mode=_default_cluster_size_mode(tier),
        )

        assert (
            _get_min_count(
                tier=tier,
                required_cluster_size=5,
                needed_disk_gib=6,
                disk_per_node_gib=1,
                cluster_size_lambda=cluster_size,
            )
            == 6
        )

    @pytest.mark.parametrize("tier", [0, 1])
    def test_critical_tiers_keep_doubling_cluster_size(self, tier):
        cluster_size = _get_cluster_size_lambda(
            cluster_size_mode=_default_cluster_size_mode(tier),
        )

        assert (
            _get_min_count(
                tier=tier,
                required_cluster_size=None,
                needed_disk_gib=3,
                disk_per_node_gib=1,
                cluster_size_lambda=cluster_size,
            )
            == 4
        )

    @pytest.mark.parametrize("tier", [2, 3, 4])
    def test_cluster_size_mode_can_force_doubling_for_non_critical_tiers(self, tier):
        cluster_size = _get_cluster_size_lambda(
            cluster_size_mode=CassandraClusterSizeMode.doubling,
        )

        assert (
            _get_min_count(
                tier=tier,
                required_cluster_size=None,
                needed_disk_gib=3,
                disk_per_node_gib=1,
                cluster_size_lambda=cluster_size,
            )
            == 4
        )

    def test_cluster_size_mode_does_not_double_from_required_size(self):
        cluster_size = _get_cluster_size_lambda(
            cluster_size_mode=CassandraClusterSizeMode.doubling,
            required_cluster_size=5,
        )

        assert (
            _get_min_count(
                tier=2,
                required_cluster_size=5,
                needed_disk_gib=6,
                disk_per_node_gib=1,
                cluster_size_lambda=cluster_size,
            )
            == 6
        )

    def test_required_cluster_size_remains_the_min_count_floor(self):
        cluster_size = _get_cluster_size_lambda(
            cluster_size_mode=CassandraClusterSizeMode.doubling,
            required_cluster_size=5,
        )

        assert (
            _get_min_count(
                tier=2,
                required_cluster_size=5,
                needed_disk_gib=4,
                disk_per_node_gib=1,
                cluster_size_lambda=cluster_size,
            )
            == 5
        )

    def test_cluster_size_mode_doubles_from_current_non_power_of_two_size(self):
        cluster_size = _get_cluster_size_lambda(
            cluster_size_mode=CassandraClusterSizeMode.doubling,
            current_cluster_size=6,
        )

        assert (
            _get_min_count(
                tier=2,
                required_cluster_size=None,
                needed_disk_gib=7,
                disk_per_node_gib=1,
                cluster_size_lambda=cluster_size,
            )
            == 12
        )

    @pytest.mark.parametrize("tier", [0, 1])
    def test_cluster_size_mode_can_force_unrestricted_for_critical_tiers(self, tier):
        cluster_size = _get_cluster_size_lambda(
            cluster_size_mode=CassandraClusterSizeMode.unrestricted,
        )

        assert (
            _get_min_count(
                tier=tier,
                required_cluster_size=None,
                needed_disk_gib=3,
                disk_per_node_gib=1,
                cluster_size_lambda=cluster_size,
            )
            == 3
        )

    def test_page_cache_cap_default(self):
        from service_capacity_modeling.models.org.netflix.cassandra import (
            NflxCassandraArguments,
        )

        args = NflxCassandraArguments.from_extra_model_arguments({})
        assert args.max_page_cache_gib == 28.0

    def test_cluster_size_mode_extra_argument(self):
        from service_capacity_modeling.models.org.netflix.cassandra import (
            NflxCassandraArguments,
        )

        assert (
            NflxCassandraArguments.from_extra_model_arguments({}).cluster_size_mode
            is None
        )
        assert (
            NflxCassandraArguments.from_extra_model_arguments(
                {"cluster_size_mode": "doubling"}
            ).cluster_size_mode
            == CassandraClusterSizeMode.doubling
        )
        assert (
            NflxCassandraArguments.from_extra_model_arguments(
                {"cluster_size_mode": "unrestricted"}
            ).cluster_size_mode
            == CassandraClusterSizeMode.unrestricted
        )

    def test_allow_ebs_volume_shrink_extra_argument(self):
        from service_capacity_modeling.models.org.netflix.cassandra import (
            NflxCassandraArguments,
        )

        assert (
            NflxCassandraArguments.from_extra_model_arguments(
                {}
            ).allow_ebs_volume_shrink
            is False
        )
        assert (
            NflxCassandraArguments.from_extra_model_arguments(
                {"allow_ebs_volume_shrink": True}
            ).allow_ebs_volume_shrink
            is True
        )

    def test_cluster_size_mode_schema_exposes_enum_docstrings(self):
        from service_capacity_modeling.models.org.netflix.cassandra import (
            NflxCassandraArguments,
        )

        schema = NflxCassandraArguments.model_json_schema()
        cluster_size_mode = schema["$defs"]["CassandraClusterSizeMode"]

        assert cluster_size_mode["oneOf"] == [
            {
                "const": CassandraClusterSizeMode.doubling.value,
                "title": CassandraClusterSizeMode.doubling.name,
                "description": CassandraClusterSizeMode.doubling.__doc__,
            },
            {
                "const": CassandraClusterSizeMode.unrestricted.value,
                "title": CassandraClusterSizeMode.unrestricted.name,
                "description": CassandraClusterSizeMode.unrestricted.__doc__,
            },
        ]
