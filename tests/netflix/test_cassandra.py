import pytest

from service_capacity_modeling.capacity_planner import planner
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
from service_capacity_modeling.interface import CurrentClusterCapacity
from service_capacity_modeling.interface import CurrentClusters
from service_capacity_modeling.interface import CurrentZoneClusterCapacity
from service_capacity_modeling.interface import DataShape
from service_capacity_modeling.interface import FixedInterval
from service_capacity_modeling.interface import GlobalConsistency
from service_capacity_modeling.interface import Interval
from service_capacity_modeling.interface import QueryPattern
from service_capacity_modeling.models.org.netflix.cassandra import (
    NflxCassandraCapacityModel,
)

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


class TestCassandraCapacityPlanningFromDesires:
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
                low=10.12, mid=13.2, high=14.194801291058118, confidence=1
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

        lr_clusters = cap_plan[0].candidate_clusters.zonal[0]
        assert lr_clusters.count == 8
        assert lr_clusters.instance.cpu == 12

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


class TestCassandraBufferScenarios:
    """
    Test buffer scenarios for Cassandra capacity planning.

    Preserve Buffer:
    * We want to preserve the *default* buffer for a given component. This is an
      explicit scale down and also is equivalent to a

    In other words, if the cluster is below the default buffer percent, we can
    choose instance types that are smaller than the current cluster.
    But if the cluster is above the default buffer percent, we must scale up the
    cluster to meet the default buffer percent.

    E2E Buffer tests:
    * A cassandra cluster has a ~ 1.5x compute buffer and a 4x disk buffer. And
      there are a few permutations to think about:

    1. Scenario 1: Current Cluster CPU is high (~50%) and the disk is low. We
       want to scale up the cluster to meet the CPU scale but the disk buffer
       needs to be "preserved". For example:
        * (Case A) Derived = [Scale([compute], 1.5), Preserve([storage])
          Translates to (scale compute by 1.5x and use the default (or derived
          buffer). But preserve the default storage buffer).
          In the case of an i4i.4xlarge, this might mean we have a
          m6id.8xlarge cluster (equivalent storage but not much more compute)

        * (Case B) Derived = [Scale([compute], 1.5)) (default storage)]
          We also scale up the storage buffer using the default buffer with the
          possibility of scaling down. (Preserve is same as default)

    2. Current Cluster CPU is low (~10%) and the disk is high. We do not scale
       down the cluster to meet the CPU scale but preserve the disk buffer which
       means we scale up the cluster to meet the 4x disk scale (default buffer).
        * Case C: Derived = [Scale([compute], 1.5), Preserve([storage])
          The CPU remains as-is because scale does not allow us to right-size.
          But the preserve storage will scale up the storage buffer to meet the
          4x disk scale.
        * Case D: Derived = [Scale([compute])] (default storage)]
          Same behavior as case C

    TODO: When EBS is supported, then this test should also test for EBS output
    """

    # i4i.4xlarge cluster specifications
    I4I_4XLARGE_VCPU = 16
    I4I_4XLARGE_RAM_GIB = 128
    I4I_4XLARGE_DISK_GIB = 3750  # 3.75 TB
    CLUSTER_SIZE = 8

    # Calculated totals for i4i.4xlarge cluster
    I4I_4XLARGE_TOTAL_VCPU = CLUSTER_SIZE * I4I_4XLARGE_VCPU  # 128 vCPU
    I4I_4XLARGE_TOTAL_STORAGE_GIB = CLUSTER_SIZE * I4I_4XLARGE_DISK_GIB  # 30TB

    # Test scenarios
    HIGH_CPU_LOW_DISK_CLUSTER = CurrentZoneClusterCapacity(
        cluster_instance_name="i4i.4xlarge",
        cluster_instance_count=certain_int(CLUSTER_SIZE),
        cpu_utilization=Interval(low=45, mid=50, high=55, confidence=0.9),  # High CPU
        memory_utilization_gib=certain_float(64.0),  # ~50% memory
        disk_utilization_gib=certain_float(200),  # Low disk usage
        network_utilization_mbps=certain_float(128.0),
    )

    LOW_CPU_HIGH_DISK_CLUSTER = CurrentZoneClusterCapacity(
        cluster_instance_name="i4i.4xlarge",
        cluster_instance_count=certain_int(CLUSTER_SIZE),
        cpu_utilization=Interval(low=8, mid=10, high=12, confidence=0.9),  # Low CPU
        memory_utilization_gib=certain_float(16.0),  # ~12% memory
        disk_utilization_gib=certain_float(1200),  # High disk usage
        network_utilization_mbps=certain_float(128.0),
    )

    SCALE_FACTOR = 1.5

    SCALE_COMPUTE_PRESERVE_STORAGE = Buffers(
        derived={
            "compute": Buffer(
                intent=BufferIntent.scale,
                components=[BufferComponent.cpu],
                ratio=1.5,
            ),
            "storage": Buffer(
                intent=BufferIntent.preserve,
                components=[BufferComponent.disk],
            ),
        }
    )

    SCALE_COMPUTE = Buffers(
        derived={
            "compute": Buffer(
                intent=BufferIntent.scale,
                components=[BufferComponent.cpu],
                ratio=SCALE_FACTOR,
            )
        }
    )

    @staticmethod
    def _cur_state_size(cluster: CurrentClusterCapacity):
        return cluster.cluster_instance_count.mid * cluster.disk_utilization_gib.mid

    @pytest.mark.parametrize(
        "buffers",
        [SCALE_COMPUTE_PRESERVE_STORAGE, SCALE_COMPUTE],
    )
    def test_scenario_1(self, buffers):
        """
        Scenario 1 Case A: High CPU (~50%), low disk, scale compute by 1.5x,
        preserve storage buffer (or absent preserve buffer).

        Expected: Scale up compute but preserve storage buffer (equivalent
        storage, more compute).
        """
        cluster = self.HIGH_CPU_LOW_DISK_CLUSTER
        desires = CapacityDesires(
            service_tier=1,
            current_clusters=CurrentClusters(zonal=[cluster]),
            data_shape=DataShape(
                estimated_state_size_gib=certain_int(500),
            ),
            buffers=buffers,
        )

        cap_plan = planner.plan_certain(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=desires,
        )[0]

        result = cap_plan.candidate_clusters.zonal[0]

        # Expected: More compute (scaled by 1.5x) but equivalent storage
        # Current: 8 * 16 = 128 vCPU, should scale to ~192 vCPU
        expected_min_cpu = self.I4I_4XLARGE_TOTAL_VCPU * self.SCALE_FACTOR
        actual_cpu = result.count * result.instance.cpu
        # This math is hacky because it ignores IPC per cycle differences
        assert expected_min_cpu <= actual_cpu

        # Storage should be preserved (not scaled up significantly)
        state_size = TestCassandraBufferScenarios._cur_state_size(cluster)
        actual_storage = result.count * result.instance.drive.size_gib

        # Should be above the buffer amount, but we should not expect
        # more disk (if this turns out to be flaky, it can be within a
        # percentage of total_storage GiB e.g. 20%)
        expected_default_overhead = 4
        assert state_size * expected_default_overhead < actual_storage
        assert actual_storage <= self.I4I_4XLARGE_TOTAL_STORAGE_GIB

    @pytest.mark.parametrize(
        "buffers",
        [SCALE_COMPUTE_PRESERVE_STORAGE, SCALE_COMPUTE],
    )
    def test_scenario_2(self, buffers):
        """
        Scenario 2 Case C: Low CPU (~10%), high disk, scale compute by 1.5x,
        preserve storage buffer.

        Expected: CPU remains as-is (scale doesn't allow right-sizing), but
        storage scales up to meet 4x disk buffer.
        """
        cluster = self.LOW_CPU_HIGH_DISK_CLUSTER

        desires = CapacityDesires(
            service_tier=1,
            current_clusters=CurrentClusters(zonal=[cluster]),
            data_shape=DataShape(
                estimated_state_size_gib=certain_int(500),
            ),
            buffers=buffers,
        )

        cap_plan = planner.plan_certain(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=desires,
        )[0]

        result = cap_plan.candidate_clusters.zonal[0]

        # Expected: CPU might not scale down much due to scale intent, but
        # storage should scale up to meet preserve buffer
        # Current: 8 * 16 = 128 vCPU
        actual_cpu = result.count * result.instance.cpu

        # CPU is significantly scaled down from current due to low CPU
        assert actual_cpu <= self.I4I_4XLARGE_TOTAL_VCPU * 0.8

        # Storage should be scaled up significantly to meet preserve buffer
        # (4x default)
        actual_storage = result.count * result.instance.drive.size_gib
        # Should be significantly more storage due to preserve buffer (4x)
        expected_storage_buffer = 4
        current_usage = self._cur_state_size(self.LOW_CPU_HIGH_DISK_CLUSTER)
        assert actual_storage >= expected_storage_buffer * current_usage
