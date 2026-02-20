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
        # With soft page cache memory, EBS write-heavy workloads are sized
        # by CPU/IO rather than memory, so fewer cores are needed
        assert 32 <= cores <= 512
        # Should get attached storage since we explicitly requested it
        assert result.attached_drives, (
            "Expected attached drives with require_attached_disks=True"
        )
        assert result.attached_drives[0].name == "gp3"
        assert result.attached_drives[0].read_io_per_s is not None
        assert result.attached_drives[0].write_io_per_s is not None

        read_ios = result.attached_drives[0].read_io_per_s * result.count
        write_ios = result.attached_drives[0].write_io_per_s * result.count

        # IO assertions need wider bounds since node count may differ
        assert 5_000 < read_ios < 60_000
        assert 1_000 < write_ios < 10_000


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

        # With soft page cache memory, EBS write-heavy workloads may use
        # compute-optimized instances since the page cache memory requirement
        # no longer drives instance selection
        assert high_writes_result.instance.family[0] in ("c", "m", "r")
        assert high_writes_result.count > 16

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
            extra_model_arguments={**EXTRA_MODEL_ARGS, "required_cluster_size": 8},
        )

        # Use a similar number of CPU cores but allocate less disk
        lr_clusters = cap_plan[0].candidate_clusters.zonal[0]
        assert_similar_compute(
            shapes.instance("m6id.8xlarge"), lr_clusters.instance, 8, lr_clusters.count
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
            extra_model_arguments={**EXTRA_MODEL_ARGS, "required_cluster_size": 2},
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


class TestCassandraPageCacheSoftMemory:
    """Test that EBS page cache memory is treated as a soft cost (CPU overhead)
    rather than a hard rejection.

    When the working set cannot fit in RAM, the model should still produce a
    plan but communicate the page cache coverage and CPU overhead factor.
    """

    def test_large_ebs_dataset_produces_plan(self):
        """A 10 TiB dataset on EBS at 16 nodes should produce a plan instead
        of returning None. Previously this was rejected because 30.6% of
        10 TiB = 3.3 TiB RAM/zone, which was impossible at 16 nodes."""
        desires = CapacityDesires(
            service_tier=1,
            query_pattern=QueryPattern(
                estimated_read_per_second=certain_int(50_000),
                estimated_write_per_second=certain_int(50_000),
            ),
            data_shape=DataShape(
                estimated_state_size_gib=certain_int(10_000),
            ),
        )

        cap_plan = planner.plan_certain(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=desires,
            extra_model_arguments={
                "require_attached_disks": True,
                "require_local_disks": False,
                "required_cluster_size": 16,
            },
        )

        # Should produce at least one plan (was previously None/empty)
        assert len(cap_plan) > 0, (
            "Expected at least one plan for 10 TiB EBS with 16 nodes"
        )
        result = cap_plan[0].candidate_clusters.zonal[0]
        assert result.count == 16

        # Should have page cache params
        assert "cassandra.page_cache.coverage_pct" in result.cluster_params
        assert "cassandra.page_cache.cpu_factor" in result.cluster_params

        # Coverage should be less than 100% (can't fit full working set)
        coverage = result.cluster_params["cassandra.page_cache.coverage_pct"]
        assert coverage < 100.0, (
            f"Expected partial page cache coverage, got {coverage}%"
        )

        # CPU factor should be > 1.0 since we have cache misses with reads
        cpu_factor = result.cluster_params["cassandra.page_cache.cpu_factor"]
        assert cpu_factor > 1.0, (
            f"Expected CPU factor > 1.0 due to cache misses, got {cpu_factor}"
        )

    def test_write_heavy_workload_low_cpu_factor(self):
        """Write-heavy workloads should have a low CPU factor (~1.0) because
        writes go to commitlog/memtable and don't incur page cache misses."""
        desires = CapacityDesires(
            service_tier=1,
            query_pattern=QueryPattern(
                estimated_read_per_second=certain_int(1_000),
                estimated_write_per_second=certain_int(100_000),
            ),
            data_shape=DataShape(
                estimated_state_size_gib=certain_int(5_000),
            ),
        )

        cap_plan = planner.plan_certain(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=desires,
            extra_model_arguments={
                "require_attached_disks": True,
                "require_local_disks": False,
                "required_cluster_size": 16,
            },
        )

        assert len(cap_plan) > 0
        result = cap_plan[0].candidate_clusters.zonal[0]

        cpu_factor = result.cluster_params["cassandra.page_cache.cpu_factor"]
        # With ~1% read fraction, CPU factor should be very close to 1.0
        assert cpu_factor < 1.1, (
            f"Expected CPU factor < 1.1 for write-heavy workload, got {cpu_factor}"
        )

    def test_read_heavy_workload_higher_cpu_factor(self):
        """Read-heavy workloads with insufficient page cache should have a
        higher CPU factor because reads hit EBS instead of RAM."""
        desires = CapacityDesires(
            service_tier=1,
            query_pattern=QueryPattern(
                estimated_read_per_second=certain_int(100_000),
                estimated_write_per_second=certain_int(1_000),
            ),
            data_shape=DataShape(
                estimated_state_size_gib=certain_int(10_000),
            ),
        )

        # Don't constrain cluster size -- let the planner pick the natural
        # topology. The IO requirements for 10 TiB read-heavy need > 16 nodes.
        cap_plan = planner.plan_certain(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=desires,
            extra_model_arguments={
                "require_attached_disks": True,
                "require_local_disks": False,
            },
        )

        assert len(cap_plan) > 0
        result = cap_plan[0].candidate_clusters.zonal[0]

        cpu_factor = result.cluster_params["cassandra.page_cache.cpu_factor"]
        # With ~99% read fraction and low page cache coverage,
        # CPU factor should be notably > 1.0
        assert cpu_factor > 1.2, (
            f"Expected CPU factor > 1.2 for read-heavy workload, got {cpu_factor}"
        )

    def test_local_disk_instances_unchanged(self):
        """Local disk instances should still have the original behavior
        (page cache params present but coverage at 100% or near it for
        appropriately sized clusters)."""
        desires = CapacityDesires(
            service_tier=1,
            query_pattern=QueryPattern(
                estimated_read_per_second=certain_int(60_000),
                estimated_write_per_second=certain_int(60_000),
            ),
            data_shape=DataShape(
                estimated_state_size_gib=certain_int(4_000),
            ),
        )

        cap_plan = planner.plan_certain(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=desires,
            extra_model_arguments={
                "require_local_disks": True,
                "required_cluster_size": 16,
            },
        )

        assert len(cap_plan) > 0
        result = cap_plan[0].candidate_clusters.zonal[0]

        # Local disk instances should have page cache params set
        assert "cassandra.page_cache.coverage_pct" in result.cluster_params
        assert "cassandra.page_cache.cpu_factor" in result.cluster_params

        # For local disk, page cache is still a hard requirement (the old
        # behavior), so coverage should be 100% and cpu_factor 1.0
        assert result.cluster_params["cassandra.page_cache.coverage_pct"] == 100.0
        assert result.cluster_params["cassandra.page_cache.cpu_factor"] == 1.0
