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
    # "org.netflix.cassandra": {
    #     "extra_model_arguments": {},
    # },
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
            # We really should just pay for CPU here
            assert small_result.instance.name.startswith("c")

            cores = small_result.count * small_result.instance.cpu
            assert 30 <= cores <= 80
            # Even though it's a small dataset we need IOs so should end up
            # with lots of storage to handle the read IOs
            assert 1000 <= get_total_storage_gib(small_result) <= 2000

            # Data per node is a finicky assertion because it can vary heavily
            # baesd on generational improvements. If this breaks from a
            # generational improvement, remove or change this assertion
            node_density = get_total_storage_gib(small_result) / small_result.count
            assert 300 < node_density < 400

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
        assert 128 <= cores <= 512
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
        # 33k wps * 8KiB  / 256KiB write IO size = 16.5k / s * 4 for compaction = 6.4k
        assert 4_000 < write_ios < 7_000


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

        # With attached disks requested, should get general-purpose instances
        assert high_writes_result.instance.family[0] in ("m", "r")
        assert high_writes_result.count > 32

        # Should have attached storage since we explicitly requested it
        assert high_writes_result.attached_drives, (
            "Expected attached drives with require_attached_disks=True"
        )
        assert high_writes_result.attached_drives[0].size_gib >= 400
        total_storage = get_total_storage_gib(high_writes_result)
        assert 100_000 <= total_storage < 300_000

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


def _make_memory_test_desire(
    disk_utilization_gib: float = 100,
    state_size_gib: int = 300,
    reads: int = 10_000,
    writes: int = 100_000,
) -> CapacityDesires:
    """Build a CapacityDesires for memory model tests with common defaults."""
    cluster = CurrentZoneClusterCapacity(
        cluster_instance_name="r5d.4xlarge",
        cluster_instance=shapes.instance("r5d.4xlarge"),
        cluster_instance_count=certain_int(2),
        cpu_utilization=certain_float(13.0),
        disk_utilization_gib=certain_float(disk_utilization_gib),
        network_utilization_mbps=certain_float(128.0),
    )
    return CapacityDesires(
        service_tier=1,
        current_clusters=CurrentClusters(zonal=[cluster]),
        query_pattern=QueryPattern(
            estimated_read_per_second=certain_int(reads),
            estimated_write_per_second=certain_int(writes),
        ),
        data_shape=DataShape(estimated_state_size_gib=certain_int(state_size_gib)),
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
                "experimental_memory_model": True,
            },
        )

        # Page cache cap (32 GiB default) limits working set on i4i.8xlarge
        # (256 GiB RAM → raw page_cache≈224, capped to 32).  This prevents
        # the planner from picking 256+ GiB RAM instances to satisfy inflated
        # memory requirements.
        lr_clusters = cap_plan[0].candidate_clusters.zonal[0]
        assert lr_clusters.instance.ram_gib < 256, (
            f"Cap should prevent 256 GiB RAM instances, got {lr_clusters.instance.name}"
        )

    def test_page_cache_cap_limits_working_set(self):
        """Page cache cap (default 32 GiB) limits effective working set."""
        # r5d.4xlarge: 128 GiB RAM, heap=30, base≈2 → raw page_cache≈96
        # capped to 32, disk_per_node=100 → ws=0.32
        desire = _make_memory_test_desire()
        cap_plan = planner.plan_certain(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=desire,
            extra_model_arguments={
                **EXTRA_MODEL_ARGS,
                "required_cluster_size": 2,
                "experimental_memory_model": True,
            },
        )

        ctx = cap_plan[0].requirements.zonal[0].context
        assert ctx["working_set"] < 0.35
        assert ctx["write_buffer_gib"] > 0

    def test_custom_page_cache_cap(self):
        """Custom max_page_cache_gib overrides default cap."""
        desire = _make_memory_test_desire()
        # With cap=64, ws = min(1.0, 64/100) = 0.64 → more RAM needed
        cap_plan = planner.plan_certain(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=desire,
            extra_model_arguments={
                **EXTRA_MODEL_ARGS,
                "required_cluster_size": 2,
                "experimental_memory_model": True,
                "max_page_cache_gib": 64.0,
            },
        )

        ctx = cap_plan[0].requirements.zonal[0].context
        assert ctx["working_set"] > 0.5

    def test_legacy_path_ignores_cap(self):
        """Without experimental_memory_model, legacy theoretical path is used."""
        desire = _make_memory_test_desire()
        cap_plan = planner.plan_certain(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=desire,
            extra_model_arguments={**EXTRA_MODEL_ARGS, "required_cluster_size": 2},
        )

        ctx = cap_plan[0].requirements.zonal[0].context
        assert ctx["working_set"] < 0.5

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
                "required_cluster_size": 2,
                "experimental_memory_model": True,
            },
        )

        # Preserve → write_buffer zeroed, RAM preserved at current level
        ctx = cap_plan[0].requirements.zonal[0].context
        assert ctx["write_buffer_gib"] == 0

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

    def test_experimental_memory_model_defaults_to_false(self):
        """Verify experimental_memory_model defaults to False."""
        from service_capacity_modeling.models.org.netflix.cassandra import (
            NflxCassandraArguments,
        )

        args = NflxCassandraArguments.from_extra_model_arguments({})
        assert args.experimental_memory_model is False


class TestCassandraPageCacheCap:
    """Test page cache cap in the experimental memory model."""

    def test_cap_with_high_disk_per_node(self):
        """High disk per node with cap → low working set."""
        desire = _make_memory_test_desire(
            disk_utilization_gib=500,
            state_size_gib=300,
            reads=10_000,
            writes=50_000,
        )
        cap_plan = planner.plan_certain(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=desire,
            extra_model_arguments={
                **EXTRA_MODEL_ARGS,
                "required_cluster_size": 2,
                "experimental_memory_model": True,
            },
        )

        # r5d.4xlarge: 128 GiB RAM, page_cache capped to 32, disk=500
        # ws = 32/500 = 0.064
        ctx = cap_plan[0].requirements.zonal[0].context
        assert ctx["working_set"] < 0.1

    def test_large_state_cluster(self):
        """Large-state cluster benefits from page cache cap."""
        cluster = CurrentZoneClusterCapacity(
            cluster_instance_name="r6a.4xlarge",
            cluster_instance_count=certain_int(16),
            cpu_utilization=certain_float(15.0),
            disk_utilization_gib=certain_float(3000),
            network_utilization_mbps=certain_float(50),
        )
        desire = CapacityDesires(
            service_tier=1,
            current_clusters=CurrentClusters(zonal=[cluster] * 3),
            query_pattern=QueryPattern(
                estimated_read_per_second=certain_int(60_000),
                estimated_write_per_second=certain_int(170_000),
            ),
            data_shape=DataShape(
                estimated_state_size_gib=certain_int(10_000),
                estimated_compression_ratio=certain_float(1.0),
            ),
        )
        cap_plan = planner.plan_certain(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=desire,
            extra_model_arguments={
                **EXTRA_MODEL_ARGS,
                "experimental_memory_model": True,
            },
        )

        assert cap_plan, "Page cache cap should produce a plan"
        ctx = cap_plan[0].requirements.zonal[0].context
        # r6a.4xlarge: 122 GiB RAM → raw page_cache≈90, capped to 32
        # disk_per_node=3000 → ws ≈ 32/3000 ≈ 0.01
        assert ctx["working_set"] < 0.02


class TestExperimentalMemoryModel:
    """Test EBS soft memory and cache skew features."""

    def _plan_ebs(self, extra_args=None):
        """Helper: plan an EBS cluster with experimental memory model."""
        desires = CapacityDesires(
            service_tier=1,
            query_pattern=QueryPattern(
                estimated_read_per_second=certain_int(200_000),
                estimated_write_per_second=certain_int(10_000),
                estimated_mean_read_latency_ms=certain_float(1.0),
                estimated_mean_write_latency_ms=certain_float(0.5),
            ),
            data_shape=DataShape(
                estimated_state_size_gib=certain_int(2_000),
                estimated_compression_ratio=certain_float(1.0),
            ),
        )
        args = {
            "require_local_disks": False,
            "require_attached_disks": True,
            "experimental_memory_model": True,
        }
        if extra_args:
            args.update(extra_args)
        return planner.plan_certain(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=desires,
            extra_model_arguments=args,
        )

    def test_ebs_soft_memory_produces_page_cache_params(self):
        """EBS + experimental model → page_cache params in cluster_params."""
        plans = self._plan_ebs()
        assert plans, "Should produce at least one plan"
        result = plans[0].candidate_clusters.zonal[0]
        assert "cassandra.page_cache.coverage_pct" in result.cluster_params
        assert "cassandra.page_cache.cpu_factor" in result.cluster_params
        assert result.cluster_params["cassandra.page_cache.cpu_factor"] >= 1.0

    def test_local_disk_no_soft_memory(self):
        """Local disk instances should NOT get page_cache params."""
        desires = CapacityDesires(
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
        plans = planner.plan_certain(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=desires,
            extra_model_arguments={
                "require_local_disks": True,
                "experimental_memory_model": True,
            },
        )
        assert plans
        result = plans[0].candidate_clusters.zonal[0]
        assert "cassandra.page_cache.coverage_pct" not in result.cluster_params
        assert "cassandra.page_cache.cpu_factor" not in result.cluster_params

    def test_cache_skew_in_cluster_params(self):
        """cache_skew_factor > 1.0 → recorded in cluster_params."""
        plans = self._plan_ebs(extra_args={"cache_skew_factor": 2.0})
        assert plans
        result = plans[0].candidate_clusters.zonal[0]
        assert result.cluster_params.get("cassandra.cache_skew_factor") == 2.0

    def test_no_experimental_flag_no_soft_memory(self):
        """Without experimental_memory_model, no page_cache params."""
        desires = CapacityDesires(
            service_tier=1,
            query_pattern=QueryPattern(
                estimated_read_per_second=certain_int(200_000),
                estimated_write_per_second=certain_int(10_000),
                estimated_mean_read_latency_ms=certain_float(1.0),
                estimated_mean_write_latency_ms=certain_float(0.5),
            ),
            data_shape=DataShape(
                estimated_state_size_gib=certain_int(2_000),
                estimated_compression_ratio=certain_float(1.0),
            ),
        )
        plans = planner.plan_certain(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=desires,
            extra_model_arguments={
                "require_local_disks": False,
                "require_attached_disks": True,
                # experimental_memory_model defaults to False
            },
        )
        assert plans
        result = plans[0].candidate_clusters.zonal[0]
        assert "cassandra.page_cache.coverage_pct" not in result.cluster_params
