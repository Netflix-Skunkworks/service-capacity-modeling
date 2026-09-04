# pylint: disable=too-many-lines

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
from service_capacity_modeling.interface import Drive
from service_capacity_modeling.interface import DriveType
from service_capacity_modeling.interface import Excuse
from service_capacity_modeling.interface import fixed_float
from service_capacity_modeling.interface import FixedInterval
from service_capacity_modeling.interface import GlobalConsistency
from service_capacity_modeling.interface import Interval
from service_capacity_modeling.interface import QueryPattern
from service_capacity_modeling.interface import RegionContext
from service_capacity_modeling.models.org.netflix.cassandra import (
    _get_cluster_size_lambda,
    _get_min_count,
    CASSANDRA_MAX_DISK_UTILIZATION,
    CassandraClusterSizeMode,
    CassandraKeyspacePlacement,
    CassandraKeyspaceTopology,
    NflxCassandraArguments,
    NflxCassandraCapacityModel,
)
from tests.util import assert_minimum_storage_gib
from tests.util import assert_similar_compute
from tests.util import get_total_storage_gib
from tests.util import has_local_storage
from tests.util import simple_drive

# Explicitly allow both storage types in tests that exercise their shared path.
EXTRA_MODEL_ARGS = {"require_local_disks": False}

# Property test configuration for Cassandra model.
# See tests/netflix/PROPERTY_TESTING.md for configuration options and examples.
PROPERTY_TEST_CONFIG = {
    "org.netflix.cassandra": {
        # Tiers can pick different local-disk shapes with different raw capacity
        # for the same workload, so compare the critical tier boundary directly.
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
        inst = high_writes_result.instance
        assert inst.ram_gib / inst.cpu <= 4.5

        # Storage should be sufficient for the data (300 GiB with buffer)
        assert_minimum_storage_gib(high_writes_result, 400)
        assert_similar_compute(
            shapes.instance("c8a.4xlarge"),
            high_writes_result.instance,
            expected_count=5,
            actual_count=high_writes_result.count,
            expected_attached_disk=simple_drive(
                size_gib=100, read_io_per_s=5400, write_io_per_s=200
            ),
            actual_attached_disk=high_writes_result.attached_drives[0],
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


class TestCassandraStorage:  # pylint: disable=too-many-public-methods
    """Test storage-related scenarios."""

    @staticmethod
    def _existing_ebs_desires(
        disk_utilization_gib: float = 900,
    ) -> CapacityDesires:
        current = CurrentZoneClusterCapacity(
            cluster_instance_name="r7a.4xlarge",
            cluster_instance_count=certain_int(64),
            cluster_drive=simple_drive(
                size_gib=max(1200, disk_utilization_gib),
            ),
            cpu_utilization=certain_float(20),
            disk_utilization_gib=certain_float(disk_utilization_gib),
            network_utilization_mbps=certain_float(100),
        )
        return CapacityDesires(
            service_tier=1,
            query_pattern=QueryPattern(
                estimated_read_per_second=certain_int(300_000),
                estimated_write_per_second=certain_int(300_000),
            ),
            data_shape=DataShape(estimated_state_size_gib=certain_int(70_000)),
            current_clusters=CurrentClusters(
                zonal=[current.model_copy(deep=True) for _ in range(3)]
            ),
        )

    @staticmethod
    def _ebs_plan(desires: CapacityDesires, **extra_model_arguments):
        plan = planner.plan_certain(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=desires,
            extra_model_arguments={
                "require_attached_disks": True,
                "require_local_disks": False,
                "cluster_size_mode": "unrestricted",
                "max_regional_size": 600,
                **extra_model_arguments,
            },
            instance_families=["r7a"],
            num_results=1,
        )[0]
        return plan.candidate_clusters.zonal[0]

    @staticmethod
    def _ebs_iops_evidence(
        *,
        peak_iops_per_node: float = 12_000,
        configured_iops_per_node: int = 16_000,
        regional_read_per_second: float = 300_000,
        regional_write_per_second: float = 300_000,
        mean_read_size_bytes: float = 1024,
        mean_write_size_bytes: float = 256,
    ):
        return {
            "ebs_iops_evidence": {
                "peak_iops_per_node": peak_iops_per_node,
                "configured_iops_per_node": configured_iops_per_node,
                "observed_regional_workload": {
                    "read_per_second": regional_read_per_second,
                    "write_per_second": regional_write_per_second,
                    "mean_read_size_bytes": mean_read_size_bytes,
                    "mean_write_size_bytes": mean_write_size_bytes,
                },
            }
        }

    def _ebs_explained(
        self,
        desires,
        *,
        peak_iops_per_node=12_000,
        configured_iops_per_node=16_000,
        regional_read_per_second=300_000,
        regional_write_per_second=300_000,
        mean_read_size_bytes=1024,
        mean_write_size_bytes=256,
        num_results=1,
        **extra_model_arguments,
    ):
        return planner.plan_certain_explained(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=desires,
            extra_model_arguments={
                "require_attached_disks": True,
                "require_local_disks": False,
                "cluster_size_mode": "unrestricted",
                "max_regional_size": 600,
                **self._ebs_iops_evidence(
                    peak_iops_per_node=peak_iops_per_node,
                    configured_iops_per_node=configured_iops_per_node,
                    regional_read_per_second=regional_read_per_second,
                    regional_write_per_second=regional_write_per_second,
                    mean_read_size_bytes=mean_read_size_bytes,
                    mean_write_size_bytes=mean_write_size_bytes,
                ),
                **extra_model_arguments,
            },
            instance_families=["r7a"],
            num_results=num_results,
        )

    def test_ebs_evidence_calibrates_total_iops_demand(self):
        cluster = (
            self._ebs_explained(
                self._existing_ebs_desires(),
                peak_iops_per_node=8_000,
                required_cluster_size=64,
            )
            .plans[0]
            .candidate_clusters.zonal[0]
        )

        calibration = cluster.cluster_params["cassandra.ebs_io_calibration"]
        headroom = cluster.cluster_params["cassandra.disk_iops_headroom"]
        assert calibration["iops_calibration_factor"] == pytest.approx(
            8_000 / calibration["modeled_current_iops_per_node"]
        )
        assert calibration["iops_calibration_factor"] < 1
        assert headroom["expected_peak_iops_per_node"] == pytest.approx(
            headroom["modeled_candidate_iops_per_node"]
            * calibration["iops_calibration_factor"],
            abs=0.02,
        )

    def test_ebs_evidence_uses_planning_sizes_when_observation_omits_them(self):
        desires = self._existing_ebs_desires()
        desires.query_pattern.estimated_mean_read_size_bytes = certain_int(2048)
        desires.query_pattern.estimated_mean_write_size_bytes = certain_int(4096)
        evidence = self._ebs_iops_evidence()["ebs_iops_evidence"]
        evidence["observed_regional_workload"].pop("mean_read_size_bytes")
        evidence["observed_regional_workload"].pop("mean_write_size_bytes")

        cluster = (
            self._ebs_explained(
                desires,
                required_cluster_size=64,
                ebs_iops_evidence=evidence,
            )
            .plans[0]
            .candidate_clusters.zonal[0]
        )

        workload = cluster.cluster_params["cassandra.ebs_io_calibration"][
            "calibration_workload"
        ]
        assert workload["mean_read_size_bytes"] == (
            desires.query_pattern.estimated_mean_read_size_bytes.mid
        )
        assert workload["mean_write_size_bytes"] == (
            desires.query_pattern.estimated_mean_write_size_bytes.mid
        )
        assert workload["mean_read_size_source"] == "planning_query_pattern"
        assert workload["mean_write_size_source"] == "planning_query_pattern"

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("peak_iops_per_node", 0),
            ("configured_iops_per_node", 200),
            ("read_per_second", float("nan")),
            ("write_per_second", float("inf")),
            ("mean_read_size_bytes", 0),
            ("mean_write_size_bytes", float("inf")),
        ],
    )
    def test_ebs_iops_evidence_validates_measurements(self, field, value):
        evidence = self._ebs_iops_evidence()["ebs_iops_evidence"]
        if field in evidence:
            evidence[field] = value
        else:
            evidence["observed_regional_workload"][field] = value

        with pytest.raises(ValueError):
            NflxCassandraArguments.from_extra_model_arguments(
                {"ebs_iops_evidence": evidence}
            )

    def test_ebs_iops_evidence_requires_observed_traffic(self):
        with pytest.raises(ValueError, match="requires read or write traffic"):
            NflxCassandraArguments.from_extra_model_arguments(
                self._ebs_iops_evidence(
                    regional_read_per_second=0,
                    regional_write_per_second=0,
                )
            )

    def test_ebs_iops_evidence_rejects_peak_above_configured_limit(self):
        with pytest.raises(
            ValueError,
            match="peak_iops_per_node must be less than or equal to",
        ):
            NflxCassandraArguments.from_extra_model_arguments(
                self._ebs_iops_evidence(
                    peak_iops_per_node=16_001,
                    configured_iops_per_node=16_000,
                )
            )

    def test_comfortable_evidence_uses_calibrated_model_with_disk_iops_buffer(self):
        cluster = (
            self._ebs_explained(
                self._existing_ebs_desires(),
                peak_iops_per_node=8_000,
                required_cluster_size=64,
            )
            .plans[0]
            .candidate_clusters.zonal[0]
        )

        calibration = cluster.cluster_params["cassandra.ebs_io_calibration"]
        headroom = cluster.cluster_params["cassandra.disk_iops_headroom"]
        assert calibration["current_topology_iops_governor"] == "deployed_topology"
        assert calibration["observation_at_configured_limit"] is False
        assert headroom["demand_source"] == "calibrated_model"
        assert headroom["target_utilization"] == 0.9
        assert headroom["planned_utilization"] <= 0.9

    def test_caller_disk_iops_buffer_composes_with_model_policy(self):
        desires = self._existing_ebs_desires()
        desires.buffers = Buffers(
            desired={
                "caller-disk-iops": Buffer(
                    ratio=1.25,
                    components=[BufferComponent.disk_iops],
                )
            }
        )

        cluster = (
            self._ebs_explained(
                desires,
                peak_iops_per_node=8_000,
                required_cluster_size=64,
            )
            .plans[0]
            .candidate_clusters.zonal[0]
        )

        assert cluster.cluster_params["cassandra.disk_iops_buffer_ratio"] == 1.39

    def test_at_limit_evidence_is_a_lower_bound_and_can_add_nodes(self):
        cluster = (
            self._ebs_explained(
                self._existing_ebs_desires(),
                peak_iops_per_node=16_000,
                configured_iops_per_node=16_000,
                regional_read_per_second=600_000,
                regional_write_per_second=600_000,
                num_results=20,
                max_regional_size=600,
            )
            .plans[0]
            .candidate_clusters.zonal[0]
        )

        calibration = cluster.cluster_params["cassandra.ebs_io_calibration"]
        headroom = cluster.cluster_params["cassandra.disk_iops_headroom"]
        assert calibration["current_topology_iops_governor"] == (
            "candidate_model_observation_at_configured_limit"
        )
        assert calibration["raw_iops_calibration_factor"] < 1
        assert calibration["iops_calibration_factor"] == 1
        assert cluster.count > 64
        assert headroom["demand_source"] == "deployed_peak_lower_bound"
        assert headroom["planned_utilization"] <= 0.9

    def test_evidence_topology_is_derived_from_current_clusters(self):
        desires = self._existing_ebs_desires()
        desires.current_clusters.zonal[1].cluster_drive = Drive(
            name="gp2",
            drive_type=DriveType.attached_ssd,
            size_gib=1200,
        )

        explained = self._ebs_explained(desires, num_results=20)
        calibrations = [
            plan.candidate_clusters.zonal[0].cluster_params[
                "cassandra.ebs_io_calibration"
            ]
            for plan in explained.plans
        ] + [
            excuse.context["ebs_io_calibration"]
            for excuse in explained.excuses_by_model["org.netflix.cassandra"]
            if "ebs_io_calibration" in excuse.context
        ]
        assert calibrations
        assert all(
            calibration["current_topology_iops_governor"]
            == "deployed_evidence_topology_mismatch"
            for calibration in calibrations
        )
        assert all(
            calibration["iops_calibration_factor"] is None
            for calibration in calibrations
        )

        baseline = self._ebs_plan(desires)
        rejected = self._ebs_plan(desires, **self._ebs_iops_evidence())
        assert rejected.count == baseline.count
        assert rejected.attached_drives == baseline.attached_drives

    def test_ebs_iops_evidence_requires_current_cluster(self):
        with pytest.raises(
            ValueError,
            match="requires a current deployed cluster",
        ):
            planner.plan_certain(
                model_name="org.netflix.cassandra",
                region="us-east-1",
                desires=large_footprint,
                extra_model_arguments={
                    "require_attached_disks": True,
                    "require_local_disks": False,
                    **self._ebs_iops_evidence(),
                },
            )

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
        assert 32 <= cores <= 512
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

    def test_existing_ebs_volume_floor_uses_hottest_reported_zone(self):
        desires = self._existing_ebs_desires(disk_utilization_gib=100)
        desires.current_clusters.zonal[1].disk_utilization_gib = certain_float(5000)

        result = self._ebs_plan(desires)

        assert result.attached_drives[0].size_gib >= int(
            5000 / CASSANDRA_MAX_DISK_UTILIZATION
        )

    def test_existing_ebs_volume_floor_uses_largest_reported_zone_volume(self):
        desires = self._existing_ebs_desires(disk_utilization_gib=100)
        desires.current_clusters.zonal[1].cluster_drive.size_gib = 5000

        result = self._ebs_plan(desires)

        assert result.attached_drives[0].size_gib >= 5000


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
                "ephemeral_maintenance_regret": 0,
            },
        )[0]
        result = cap_plan.candidate_clusters.zonal[0]
        counts = result.cluster_params["required_nodes_by_type"]
        assert result.count == counts["cluster_size"] == counts["cpu"] == 8

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
                "ephemeral_maintenance_regret": 0,
            },
        )[0]
        result = cap_plan.candidate_clusters.zonal[0]
        counts = result.cluster_params["required_nodes_by_type"]
        assert result.count == 6
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

    @pytest.mark.parametrize("tier", [0, 1, 2, 3, 4])
    def test_all_tiers_do_not_round_cluster_size_by_default(self, tier):
        cluster_size = _get_cluster_size_lambda()

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

    @pytest.mark.parametrize("tier", [2, 3, 4])
    def test_non_critical_tiers_do_not_round_above_required_cluster_size(self, tier):
        cluster_size = _get_cluster_size_lambda()

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

    @pytest.mark.parametrize("tier", [0, 1, 2, 3, 4])
    def test_cluster_size_mode_can_force_doubling_for_all_tiers(self, tier):
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

    def test_page_cache_cap_default(self):
        args = NflxCassandraArguments.from_extra_model_arguments({})
        assert args.max_page_cache_gib == 28.0

    def test_min_instance_ram_gib_exclusive_default(self):
        args = NflxCassandraArguments.from_extra_model_arguments({})
        assert args.min_instance_ram_gib_exclusive == 16.0

    def test_default_min_instance_ram_rejects_m6id_xlarge(self):
        hardware = shapes.region("us-east-1")
        result = NflxCassandraCapacityModel.capacity_plan(
            instance=hardware.instances["m6id.xlarge"],
            drive=hardware.drives["gp3"],
            context=RegionContext(
                zones_in_region=hardware.zones_in_region,
                services=hardware.services,
            ),
            desires=small_but_high_qps,
            extra_model_arguments={},
        )

        assert isinstance(result, Excuse)
        assert result.context["ram_gib"] == 15.26
        assert result.context["min_ram_gib_exclusive"] == 16.0
        assert "requires > 16 GiB" in result.reason

    def test_min_instance_ram_override_allows_m6id_xlarge(self):
        hardware = shapes.region("us-east-1")
        result = NflxCassandraCapacityModel.capacity_plan(
            instance=hardware.instances["m6id.xlarge"],
            drive=hardware.drives["gp3"],
            context=RegionContext(
                zones_in_region=hardware.zones_in_region,
                services=hardware.services,
            ),
            desires=small_but_high_qps,
            extra_model_arguments={"min_instance_ram_gib_exclusive": 15.0},
        )

        assert not isinstance(result, Excuse)
        assert result is not None
        assert result.candidate_clusters.zonal[0].instance.name == "m6id.xlarge"

    def test_cluster_size_mode_extra_argument(self):
        assert (
            NflxCassandraArguments.from_extra_model_arguments({}).cluster_size_mode
            == CassandraClusterSizeMode.unrestricted
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


class TestCassandraServiceCosts:
    @staticmethod
    def _services(
        *,
        num_regions=4,
        backup_retention_days=None,
        keyspace_topology=None,
        state_size_gib=300,
        writes_per_second=100,
        copies_per_region=None,
    ):
        hardware = shapes.region("us-east-1")
        desires = CapacityDesires(
            query_pattern=QueryPattern(
                estimated_write_per_second=certain_float(writes_per_second),
                estimated_mean_write_size_bytes=certain_int(512),
            ),
            data_shape=DataShape(
                estimated_state_size_gib=certain_float(state_size_gib)
            ),
        )
        return NflxCassandraCapacityModel.service_costs(
            "cassandra",
            RegionContext(
                zones_in_region=hardware.zones_in_region,
                num_regions=num_regions,
                services=hardware.services,
            ),
            desires,
            {
                "backup_retention_days": backup_retention_days,
                "keyspace_topology": keyspace_topology,
                "copies_per_region": copies_per_region,
            },
        )

    def test_service_costs_return_one_regional_share(self):
        one_region = {
            service.service_type: service for service in self._services(num_regions=1)
        }
        four_regions = {
            service.service_type: service for service in self._services(num_regions=4)
        }

        assert four_regions["cassandra.net.intra.region"].annual_cost == pytest.approx(
            one_region["cassandra.net.intra.region"].annual_cost / 4
        )
        assert (
            four_regions["cassandra.backup.s3-standard"].service_params["snapshot_gib"]
            == one_region["cassandra.backup.s3-standard"].service_params["snapshot_gib"]
        )
        assert four_regions["cassandra.backup.s3-standard"].service_params[
            "daily_write_gib"
        ] == pytest.approx(
            one_region["cassandra.backup.s3-standard"].service_params["daily_write_gib"]
            / 4,
            abs=0.1,
        )

    def test_backup_retention_controls_backup_service_cost(self):
        enabled = {service.service_type: service for service in self._services()}
        seven_days = {
            service.service_type: service
            for service in self._services(backup_retention_days=7)
        }
        disabled = {
            service.service_type: service
            for service in self._services(backup_retention_days=0)
        }

        assert "cassandra.backup.s3-standard" in enabled
        assert (
            seven_days["cassandra.backup.s3-standard"].service_params["retention_days"]
            == 7
        )
        assert "cassandra.backup.s3-standard" not in disabled
        assert "cassandra.net.inter.region" in disabled
        assert "cassandra.net.intra.region" in disabled

    def test_keyspace_placements_equal_independently_modeled_service_costs(self):
        placements = [
            CassandraKeyspacePlacement(
                keyspaces=["rf1"],
                copies_per_region=1,
                regions=["us-east-1", "us-west-2"],
                logical_state_size_gib=90,
                write_per_second=40,
            ),
            CassandraKeyspacePlacement(
                keyspaces=["rf3"],
                copies_per_region=3,
                regions=["us-east-1", "eu-west-1"],
                logical_state_size_gib=210,
                write_per_second=60,
            ),
        ]
        topology = CassandraKeyspaceTopology(
            planning_region="us-east-1",
            placements=placements,
            physical_state_size_gib_by_region={
                "us-east-1": 900,
                "us-west-2": 90,
                "eu-west-1": 630,
            },
        )
        combined = {
            service.service_type: service
            for service in self._services(num_regions=3, keyspace_topology=topology)
        }
        independent = [
            {
                service.service_type: service
                for service in self._services(
                    num_regions=placement.replicated_region_count,
                    backup_retention_days=0,
                    state_size_gib=placement.logical_state_size_gib,
                    writes_per_second=placement.write_per_second,
                    copies_per_region=placement.copies_per_region,
                )
            }
            for placement in placements
        ]

        for service_type in (
            "cassandra.net.inter.region",
            "cassandra.net.intra.region",
        ):
            assert combined[service_type].annual_cost == pytest.approx(
                sum(services[service_type].annual_cost for services in independent)
            )
            assert [
                (
                    placement["keyspaces"],
                    placement["copies_per_region"],
                    placement["replicated_region_count"],
                    placement["logical_state_size_gib"],
                    placement["write_per_second"],
                    placement.get("num_regions"),
                )
                for placement in combined[service_type].service_params["placements"]
            ] == [
                (["rf1"], 1, 2, 90, 40, None),
                (["rf3"], 3, 2, 210, 60, None),
            ]
            assert combined[service_type].service_params["placements"][0][
                "regions"
            ] == ["us-east-1", "us-west-2"]

        backup = combined["cassandra.backup.s3-standard"]
        assert backup.service_params["snapshot_gib"] == 300

    def test_region_without_keyspace_placements_has_an_explicit_receipt(self):
        topology = CassandraKeyspaceTopology(
            planning_region="eu-west-1",
            placements=[
                CassandraKeyspacePlacement(
                    keyspaces=["events"],
                    copies_per_region=3,
                    regions=["us-east-1"],
                    logical_state_size_gib=10,
                    write_per_second=20,
                )
            ],
            physical_state_size_gib_by_region={"us-east-1": 30, "eu-west-1": 0},
        )
        services = self._services(keyspace_topology=topology)

        assert len(services) == 1
        assert services[0].service_type == "cassandra.cost.placements"
        assert services[0].annual_cost == 0
        assert services[0].service_params == {
            "contract_version": 1,
            "placement_count": 0,
        }

    def test_keyspace_topology_requires_disjoint_keyspaces(self):
        placement = CassandraKeyspacePlacement(
            keyspaces=["events"],
            copies_per_region=1,
            regions=["us-east-1"],
            logical_state_size_gib=1,
            write_per_second=1,
        )

        with pytest.raises(ValueError, match="must not repeat keyspaces"):
            CassandraKeyspaceTopology(
                planning_region="us-east-1",
                placements=[placement, placement],
                physical_state_size_gib_by_region={"us-east-1": 1},
            )

    @pytest.mark.parametrize(
        "placement",
        [
            {},
            {"regions": [""]},
            {"regions": ["us-east-1", "us-east-1"]},
        ],
    )
    def test_keyspace_placement_region_scope_is_unambiguous(self, placement):
        with pytest.raises(ValueError):
            CassandraKeyspacePlacement(
                keyspaces=["events"],
                copies_per_region=3,
                logical_state_size_gib=1,
                write_per_second=1,
                **placement,
            )

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), -1])
    def test_topology_physical_state_must_be_finite_and_nonnegative(self, value):
        with pytest.raises(ValueError):
            CassandraKeyspaceTopology(
                planning_region="us-east-1",
                placements=[
                    CassandraKeyspacePlacement(
                        keyspaces=["events"],
                        copies_per_region=3,
                        regions=["us-east-1"],
                        logical_state_size_gib=1,
                        write_per_second=1,
                    )
                ],
                physical_state_size_gib_by_region={"us-east-1": value},
            )

    @pytest.mark.parametrize(
        ("planning_region", "placement_region", "physical_state"),
        [
            ("us-west-2", "us-east-1", {"us-east-1": 1}),
            ("us-east-1", "us-west-2", {"us-east-1": 1}),
            (
                "us-east-1",
                "us-east-1",
                {"us-east-1": 1, "us-west-2": 1},
            ),
        ],
    )
    def test_keyspace_topology_rejects_incomplete_region_mapping(
        self, planning_region, placement_region, physical_state
    ):
        with pytest.raises(ValueError):
            CassandraKeyspaceTopology(
                planning_region=planning_region,
                placements=[
                    CassandraKeyspacePlacement(
                        keyspaces=["events"],
                        copies_per_region=3,
                        regions=[placement_region],
                        logical_state_size_gib=1,
                        write_per_second=1,
                    )
                ],
                physical_state_size_gib_by_region=physical_state,
            )

    def test_capacity_plan_sizes_every_region_from_largest_physical_state(self):
        common_arguments = {
            "require_attached_disks": True,
            "require_local_disks": False,
            "cluster_size_mode": "unrestricted",
            "max_regional_size": 600,
            "keyspace_topology": {
                "planning_region": "us-east-1",
                "physical_state_size_gib_by_region": {
                    "us-east-1": 210_000,
                    "us-west-2": 30_000,
                },
                "placements": [
                    CassandraKeyspacePlacement(
                        keyspaces=["regional"],
                        copies_per_region=3,
                        regions=["us-east-1", "us-west-2"],
                        logical_state_size_gib=10,
                        write_per_second=100,
                    ).model_dump(mode="json")
                ],
            },
        }
        plans = {}
        for planning_region, regional_disk_gib in (
            ("us-east-1", 900),
            ("us-west-2", 150),
        ):
            current = CurrentZoneClusterCapacity(
                cluster_instance_name="r7a.4xlarge",
                cluster_instance_count=certain_int(64),
                cluster_drive=simple_drive(size_gib=1200),
                cpu_utilization=certain_float(20),
                disk_utilization_gib=certain_float(regional_disk_gib),
                network_utilization_mbps=certain_float(100),
            )
            desires = CapacityDesires(
                service_tier=1,
                query_pattern=QueryPattern(
                    estimated_read_per_second=certain_int(300_000),
                    estimated_write_per_second=certain_int(300_000),
                ),
                data_shape=DataShape(estimated_state_size_gib=certain_int(10_000)),
                current_clusters=CurrentClusters(zonal=[current] * 3),
            )
            arguments = {**common_arguments}
            arguments["keyspace_topology"] = {
                **common_arguments["keyspace_topology"],
                "planning_region": planning_region,
            }
            plans[planning_region] = planner.plan_certain(
                model_name="org.netflix.cassandra",
                region="us-east-1",
                desires=desires,
                extra_model_arguments=arguments,
                instance_families=["r7a"],
                num_results=1,
            )[0]

        east = plans["us-east-1"]
        west = plans["us-west-2"]
        assert east.requirements == west.requirements
        assert east.requirements.zonal[0].disk_gib.mid > 210_000 / 3
        services = {
            service.service_type: service
            for service in east.candidate_clusters.services
        }
        inter_region = services["cassandra.net.inter.region"]
        assert inter_region.annual_cost > 0
        assert inter_region.service_params["contract_version"] == 1
        assert inter_region.service_params["placements"][0]["keyspaces"] == ["regional"]
