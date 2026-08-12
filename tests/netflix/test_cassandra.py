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
    _cass_io_per_read,
    _default_cluster_size_mode,
    _deployed_topology_saturation_min_count,
    _get_cluster_size_lambda,
    _get_min_count,
    CassandraClusterSizeMode,
    NflxCassandraArguments,
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
        inst = high_writes_result.instance
        assert inst.ram_gib / inst.cpu <= 4.5

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
                read_io_per_s=8_000,
                write_io_per_s=8_000,
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
            current_clusters=CurrentClusters(zonal=[current] * 3),
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
    def _deployed_topology_evidence(observed_iops=12_000):
        return {
            "observed_ebs_max_total_iops_per_node": observed_iops,
            "observed_ebs_node_count_at_peak": 768,
            "ebs_planning_baseline_read_per_second": 300_000,
            "ebs_planning_baseline_write_per_second": 300_000,
            "ebs_planning_baseline_mean_read_size_bytes": 1024,
            "ebs_planning_baseline_mean_write_size_bytes": 256,
            "ebs_planning_baseline_copies_per_region": 3,
        }

    def _ebs_explained(
        self,
        desires,
        *,
        observed_iops=12_000,
        instance_family="r7a",
        num_results=1,
        same_data_as_deployed=True,
        **extra_model_arguments,
    ):
        planning_arguments = {
            **self._deployed_topology_evidence(observed_iops),
            **extra_model_arguments,
        }
        if same_data_as_deployed:
            current = desires.current_clusters.zonal[0]
            merged_compression = NflxCassandraCapacityModel.default_desires(
                desires, planning_arguments
            ).data_shape.estimated_compression_ratio.mid
            desires = desires.model_copy(
                update={
                    "data_shape": desires.data_shape.model_copy(
                        update={
                            "estimated_state_size_gib": certain_float(
                                current.disk_utilization_gib.mid
                                * current.cluster_instance_count.mid
                                * merged_compression
                            ),
                            "estimated_compression_ratio": certain_float(
                                merged_compression
                            ),
                        }
                    )
                }
            )
        return planner.plan_certain_explained(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=desires,
            extra_model_arguments={
                "require_attached_disks": True,
                "require_local_disks": False,
                "cluster_size_mode": "unrestricted",
                **planning_arguments,
            },
            num_regions=4,
            instance_families=[instance_family],
            num_results=num_results,
            max_results_per_family=num_results,
        )

    @staticmethod
    def _compute_buffer(kind, ratio):
        intent = BufferIntent.scale_up if kind == "derived" else BufferIntent.desired
        return Buffers(
            **{
                kind: {
                    "scale_compute": Buffer(
                        ratio=ratio,
                        intent=intent,
                        components=[BufferComponent.compute],
                    )
                }
            }
        )

    def test_ebs_io_per_request_calibrates_candidate_iops(self):
        desires = self._existing_ebs_desires()
        baseline = self._ebs_plan(desires)
        calibrated = self._ebs_plan(
            desires,
            observed_ebs_read_io_per_read=0.0002,
            observed_ebs_write_io_per_write=1.0,
        )

        calibration = calibrated.cluster_params["cassandra.ebs_io_calibration"]
        assert "cassandra.ebs_io_calibration" not in baseline.cluster_params
        assert (
            baseline.attached_drives[0].read_io_per_s,
            baseline.attached_drives[0].write_io_per_s,
        ) == (11_400, 200)
        assert calibration["observed_read_io_per_read"] == 0.0002
        modeled_read_io_per_read = _cass_io_per_read(900) * 1_563 * 64 / 100_000
        assert calibration["modeled_read_io_per_read"] == pytest.approx(
            modeled_read_io_per_read
        )
        assert calibration["read_io_calibration_factor"] == pytest.approx(
            0.0002 / modeled_read_io_per_read
        )
        assert calibration["observed_write_io_per_write"] == 1.0
        assert (
            calibrated.attached_drives[0].read_io_per_s
            < baseline.attached_drives[0].read_io_per_s
        )
        assert (
            calibrated.attached_drives[0].write_io_per_s
            > baseline.attached_drives[0].write_io_per_s
        )

    @pytest.mark.parametrize(
        "field",
        ["observed_ebs_read_io_per_read", "observed_ebs_write_io_per_write"],
    )
    @pytest.mark.parametrize("value", [0, float("nan"), float("inf")])
    def test_ebs_io_per_request_must_be_positive_and_finite(self, field, value):
        with pytest.raises(ValueError):
            self._ebs_plan(self._existing_ebs_desires(), **{field: value})

    def test_ebs_read_calibration_requires_current_disk_utilization(self):
        desires = self._existing_ebs_desires(disk_utilization_gib=0)
        baseline = self._ebs_plan(desires)
        calibrated = self._ebs_plan(
            desires,
            observed_ebs_read_io_per_read=2.0,
            observed_ebs_write_io_per_write=1.0,
        )

        calibration = calibrated.cluster_params["cassandra.ebs_io_calibration"]
        assert set(calibration) == {
            "observed_write_io_per_write",
            "modeled_write_io_per_write",
            "write_io_calibration_factor",
        }
        baseline_drive = baseline.attached_drives[0]
        calibrated_drive = calibrated.attached_drives[0]
        assert calibrated_drive.read_io_per_s == baseline_drive.read_io_per_s
        assert calibrated_drive.write_io_per_s > baseline_drive.write_io_per_s

    def test_ebs_read_calibration_uses_current_data_not_future_storage_scale(self):
        def candidate(storage_scale: float):
            desires = self._existing_ebs_desires(disk_utilization_gib=1500)
            desires.buffers = Buffers(
                derived={
                    BufferIntent.scale: Buffer(
                        ratio=storage_scale,
                        intent=BufferIntent.scale,
                        components=[BufferComponent.storage],
                    )
                }
            )
            return self._ebs_plan(
                desires,
                max_disk_utilization=1.0,
                max_regional_size=3000,
                observed_ebs_read_io_per_read=2.0,
            )

        current_load = candidate(1.0)
        future_growth = candidate(2.0)
        current_calibration = current_load.cluster_params[
            "cassandra.ebs_io_calibration"
        ]
        future_calibration = future_growth.cluster_params[
            "cassandra.ebs_io_calibration"
        ]

        assert (
            current_calibration["read_io_calibration_factor"]
            == future_calibration["read_io_calibration_factor"]
        )
        assert [
            (
                candidate.count,
                candidate.attached_drives[0].read_io_per_s,
                _cass_io_per_read(1500 * 64 * scale / candidate.count),
            )
            for candidate, scale in ((current_load, 1), (future_growth, 2))
        ] == [(89, 2400, 10), (122, 2000, 12)]

    def test_aligned_observed_iops_preserve_the_exact_deployed_topology(self):
        observed_iops = 15_800
        desires = self._existing_ebs_desires()
        explained = self._ebs_explained(
            desires,
            observed_iops=observed_iops,
            required_cluster_size=64,
            observed_ebs_read_io_per_read=20.0,
            observed_ebs_write_io_per_write=1.0,
        )

        cluster = explained.plans[0].candidate_clusters.zonal[0]
        drive = cluster.attached_drives[0]
        calibration = cluster.cluster_params["cassandra.ebs_io_calibration"]
        assert (cluster.count, cluster.instance.name) == (64, "r7a.4xlarge")
        assert (drive.name, drive.drive_type) == (
            "gp3",
            desires.current_clusters.zonal[0].cluster_drive.drive_type,
        )
        assert (
            calibration["observed_max_total_iops_per_node"],
            calibration["observed_node_count_at_peak"],
            calibration["current_topology_iops_governor"],
            drive.read_io_per_s,
            drive.write_io_per_s,
        ) == (observed_iops, 768, "deployed_topology", 15_000, 800)

    def test_deployed_topology_matches_minimal_current_drive_shape(self):
        desires = self._existing_ebs_desires()
        current = desires.current_clusters.zonal[0]
        current.cluster_drive = Drive(
            name="gp3", drive_type=DriveType.attached_ssd, size_gib=1200
        )

        explained = self._ebs_explained(
            desires,
            observed_iops=12_000,
            required_cluster_size=64,
        )

        candidate_excuse = next(
            excuse
            for excuse in explained.excuses_by_model["org.netflix.cassandra"]
            if excuse.instance == "r7a.4xlarge"
        )
        calibration = candidate_excuse.context["ebs_io_calibration"]
        assert calibration["current_topology_iops_governor"] == (
            "candidate_model_missing_configured_iops_limit"
        )
        assert calibration["configured_iops_limit_missing"] is True
        assert calibration["projected_max_total_iops_per_node"] is None

    @pytest.mark.parametrize("observed_iops", [15_801, 15_999])
    def test_deployed_topology_requires_one_iops_quantum_of_headroom(
        self, observed_iops
    ):
        desires = self._existing_ebs_desires()
        desires.current_clusters.zonal[0].cluster_drive = simple_drive(
            size_gib=1200,
            read_io_per_s=8_000,
            write_io_per_s=8_000,
        )

        explained = self._ebs_explained(
            desires,
            observed_iops=observed_iops,
            required_cluster_size=64,
        )

        candidate_excuse = next(
            excuse
            for excuse in explained.excuses_by_model["org.netflix.cassandra"]
            if excuse.instance == "r7a.4xlarge"
        )
        calibration = candidate_excuse.context["ebs_io_calibration"]
        assert calibration["current_topology_iops_governor"] == (
            "candidate_model_observation_at_configured_limit"
        )
        assert calibration["projected_max_total_iops_per_node"] is None

    def test_deployed_topology_evidence_at_configured_drive_limit_is_not_demand(self):
        desires = self._existing_ebs_desires()
        current = desires.current_clusters.zonal[0]
        current.cluster_drive = simple_drive(
            size_gib=1200,
            read_io_per_s=8_000,
            write_io_per_s=8_000,
        )

        explained = self._ebs_explained(
            desires,
            observed_iops=16_000,
            required_cluster_size=64,
        )

        candidate_excuse = next(
            excuse
            for excuse in explained.excuses_by_model["org.netflix.cassandra"]
            if excuse.instance == "r7a.4xlarge"
        )
        calibration = candidate_excuse.context["ebs_io_calibration"]
        assert calibration["current_topology_iops_governor"] == (
            "candidate_model_observation_at_configured_limit"
        )
        assert calibration["deployed_configured_iops_limit"] == 16_000
        assert calibration["observation_at_configured_limit"] is True
        assert calibration["projected_max_total_iops_per_node"] is None
        assert "deployed_topology_saturation_min_count" not in calibration

    @pytest.mark.parametrize(
        ("compression_ratio", "logical_data_multiplier"),
        [(3.0, 3.0), (0.5, 1.0)],
    )
    def test_deployed_topology_compares_data_in_on_disk_units(
        self, compression_ratio, logical_data_multiplier
    ):
        desires = self._existing_ebs_desires()
        current = desires.current_clusters.zonal[0]
        physical_zonal_gib = (
            current.disk_utilization_gib.mid * current.cluster_instance_count.mid
        )
        desires.data_shape = DataShape(
            estimated_state_size_gib=certain_float(
                physical_zonal_gib * logical_data_multiplier
            ),
            estimated_compression_ratio=certain_float(compression_ratio),
        )

        explained = self._ebs_explained(
            desires,
            observed_iops=12_000,
            required_cluster_size=64,
            same_data_as_deployed=False,
        )

        cluster = explained.plans[0].candidate_clusters.zonal[0]
        calibration = cluster.cluster_params["cassandra.ebs_io_calibration"]
        assert (cluster.count, cluster.instance.name) == (64, "r7a.4xlarge")
        assert calibration["current_topology_iops_governor"] == "deployed_topology"

    def test_deployed_topology_compares_zonal_data_at_non_default_rf(self):
        desires = self._existing_ebs_desires()
        current = desires.current_clusters.zonal[0]
        physical_zonal_gib = (
            current.disk_utilization_gib.mid * current.cluster_instance_count.mid
        )
        desires.data_shape = DataShape(
            estimated_state_size_gib=certain_float(
                physical_zonal_gib
                * 1.5
                * desires.data_shape.estimated_compression_ratio.mid
            ),
            estimated_compression_ratio=(
                desires.data_shape.estimated_compression_ratio
            ),
        )

        explained = self._ebs_explained(
            desires,
            observed_iops=12_000,
            required_cluster_size=64,
            same_data_as_deployed=False,
            copies_per_region=2,
            ebs_planning_baseline_copies_per_region=2,
        )

        cluster = explained.plans[0].candidate_clusters.zonal[0]
        calibration = cluster.cluster_params["cassandra.ebs_io_calibration"]
        assert (cluster.count, cluster.instance.name) == (64, "r7a.4xlarge")
        assert calibration["current_topology_iops_governor"] == "deployed_topology"

    def test_deployed_topology_evidence_does_not_cross_replication_factor(self):
        explained = self._ebs_explained(
            self._existing_ebs_desires(),
            required_cluster_size=64,
            copies_per_region=2,
        )

        candidate_excuse = next(
            excuse
            for excuse in explained.excuses_by_model["org.netflix.cassandra"]
            if excuse.instance == "r7a.4xlarge"
        )
        calibration = candidate_excuse.context["ebs_io_calibration"]
        assert calibration["same_planning_demand"] is False
        assert calibration["planning_baseline_copies_per_region"] == 3
        assert (
            calibration["current_topology_iops_governor"]
            == "candidate_model_unprojectable_demand"
        )

    def test_deployed_topology_evidence_requires_all_fields(self):
        evidence = self._deployed_topology_evidence(15_999)
        for missing_field in evidence:
            incomplete = evidence | {missing_field: None}
            with pytest.raises(ValueError, match=f"missing: {missing_field}"):
                NflxCassandraArguments.from_extra_model_arguments(incomplete)

    def test_aligned_iops_project_compute_growth_for_deployed_topology(self):
        desires = self._existing_ebs_desires()
        desires = desires.model_copy(
            update={"buffers": self._compute_buffer("derived", 1.12)}
        )
        explained = self._ebs_explained(
            desires,
            observed_iops=8_000,
            required_cluster_size=64,
            observed_ebs_read_io_per_read=20.0,
        )

        cluster = explained.plans[0].candidate_clusters.zonal[0]
        calibration = cluster.cluster_params["cassandra.ebs_io_calibration"]
        assert (cluster.count, cluster.instance.name) == (64, "r7a.4xlarge")
        assert (
            calibration["projected_iops_scale"],
            calibration["projected_max_total_iops_per_node"],
            calibration["current_topology_iops_governor"],
            cluster.attached_drives[0].read_io_per_s
            + cluster.attached_drives[0].write_io_per_s,
        ) == (1.12, pytest.approx(8_960), "deployed_topology_projected", 9_000)

    def test_deployed_topology_evidence_does_not_treat_desired_headroom_as_load(self):
        desires = self._existing_ebs_desires().model_copy(
            update={"buffers": self._compute_buffer("desired", 3.6)}
        )

        explained = self._ebs_explained(
            desires,
            observed_iops=8_000,
            num_results=20,
        )

        candidate_excuse = next(
            excuse
            for excuse in explained.excuses_by_model["org.netflix.cassandra"]
            if excuse.instance == "r7a.4xlarge"
        )
        calibration = candidate_excuse.context["ebs_io_calibration"]
        assert calibration["same_desired_buffer_policy"] is False
        assert calibration["projected_max_total_iops_per_node"] is None
        assert (
            calibration["current_topology_iops_governor"]
            == "candidate_model_unprojectable_demand"
        )

    def test_deployed_topology_evidence_does_not_justify_fewer_nodes(self):
        explained = self._ebs_explained(
            self._existing_ebs_desires(),
            observed_iops=2_000,
            num_results=20,
        )

        current_cluster = next(
            plan.candidate_clusters.zonal[0]
            for plan in explained.plans
            if plan.candidate_clusters.zonal[0].instance.name == "r7a.4xlarge"
        )
        assert current_cluster.count == 64
        calibration = current_cluster.cluster_params["cassandra.ebs_io_calibration"]
        assert calibration["current_topology_iops_governor"] == "deployed_topology"
        assert (
            sum(
                (
                    current_cluster.attached_drives[0].read_io_per_s,
                    current_cluster.attached_drives[0].write_io_per_s,
                )
            )
            == 2_000
        )

    def test_deployed_topology_evidence_projects_across_larger_node_count(self):
        explained = self._ebs_explained(
            self._existing_ebs_desires(),
            observed_iops=15_000,
            required_cluster_size=96,
            max_regional_size=600,
            num_results=20,
        )

        cluster = next(
            plan.candidate_clusters.zonal[0]
            for plan in explained.plans
            if plan.candidate_clusters.zonal[0].instance.name == "r7a.4xlarge"
        )
        drive = cluster.attached_drives[0]
        assert cluster.count == 96
        assert drive.read_io_per_s + drive.write_io_per_s == 10_000

    def test_deployed_topology_evidence_requires_directional_demand_fingerprint(self):
        desires = self._existing_ebs_desires()
        desires.query_pattern = desires.query_pattern.model_copy(
            update={
                "estimated_read_per_second": certain_int(100_000),
                "estimated_write_per_second": certain_int(500_000),
            }
        )
        explained = self._ebs_explained(
            desires,
            observed_iops=2_000,
            required_cluster_size=64,
            ebs_planning_baseline_read_per_second=500_000,
            ebs_planning_baseline_write_per_second=100_000,
            num_results=20,
        )

        cluster = next(
            plan.candidate_clusters.zonal[0]
            for plan in explained.plans
            if plan.candidate_clusters.zonal[0].instance.name == "r7a.4xlarge"
        )
        calibration = cluster.cluster_params["cassandra.ebs_io_calibration"]
        assert calibration["same_planning_demand"] is False
        assert (
            calibration["current_topology_iops_governor"]
            == "candidate_model_unprojectable_demand"
        )

    def test_deployed_topology_evidence_requires_matching_fleet_node_count(self):
        explained = self._ebs_explained(
            self._existing_ebs_desires(),
            observed_iops=2_000,
            required_cluster_size=64,
            observed_ebs_node_count_at_peak=1,
        )

        current_shape_excuse = next(
            excuse
            for excuse in explained.excuses_by_model["org.netflix.cassandra"]
            if excuse.instance == "r7a.4xlarge"
        )
        calibration = current_shape_excuse.context["ebs_io_calibration"]
        assert calibration["expected_deployed_node_count"] == 768
        assert calibration["same_observed_node_count"] is False
        assert calibration["same_deployed_topology"] is False
        assert (
            calibration["current_topology_iops_governor"]
            == "candidate_model_different_topology"
        )

    def test_deployed_topology_evidence_does_not_cross_request_size_boundary(self):
        desires = self._existing_ebs_desires()
        desires.query_pattern = desires.query_pattern.model_copy(
            update={"estimated_mean_read_size_bytes": certain_int(64 * 1024)}
        )
        explained = self._ebs_explained(
            desires,
            observed_iops=2_000,
            num_results=20,
            max_regional_size=768,
        )

        current_family_cluster = next(
            plan.candidate_clusters.zonal[0]
            for plan in explained.plans
            if plan.candidate_clusters.zonal[0].instance.name == "r7a.4xlarge"
        )
        calibration = current_family_cluster.cluster_params[
            "cassandra.ebs_io_calibration"
        ]
        assert current_family_cluster.count > 64
        assert (
            calibration["current_topology_iops_governor"]
            == "candidate_model_unprojectable_demand"
        )
        assert calibration["same_planning_demand"] is False

    @pytest.mark.parametrize(
        "boundary",
        ["different_topology", "different_drive", "storage_growth", "data_growth"],
    )
    def test_aligned_iops_do_not_cross_topology_or_storage_boundaries(self, boundary):
        desires = self._existing_ebs_desires()
        instance_family = "r7a"
        if boundary == "different_topology":
            instance_family = "m7a"
        elif boundary == "different_drive":
            desires.current_clusters.zonal[0].cluster_drive = Drive(
                name="gp2", size_gib=1200
            )
        elif boundary == "storage_growth":
            desires = desires.model_copy(
                update={
                    "buffers": Buffers(
                        derived={
                            "scale_storage": Buffer(
                                ratio=2.0,
                                intent=BufferIntent.scale_up,
                                components=[BufferComponent.storage],
                            )
                        }
                    )
                }
            )
        elif boundary == "data_growth":
            current = desires.current_clusters.zonal[0]
            desires.data_shape = DataShape(
                estimated_state_size_gib=certain_float(
                    current.disk_utilization_gib.mid
                    * current.cluster_instance_count.mid
                    * desires.data_shape.estimated_compression_ratio.mid
                    * 2
                ),
                estimated_compression_ratio=(
                    desires.data_shape.estimated_compression_ratio
                ),
            )
        explained = self._ebs_explained(
            desires,
            observed_iops=8_000,
            instance_family=instance_family,
            required_cluster_size=64,
            observed_ebs_read_io_per_read=20.0,
            same_data_as_deployed=boundary != "data_growth",
        )

        candidate_excuse = next(
            excuse
            for excuse in explained.excuses_by_model["org.netflix.cassandra"]
            if excuse.instance == f"{instance_family}.4xlarge"
        )
        calibration = candidate_excuse.context["ebs_io_calibration"]
        assert not explained.plans
        expected_governor = (
            "candidate_model_different_topology"
            if boundary in ("different_topology", "different_drive")
            else "candidate_model_unprojectable_demand"
        )
        assert calibration["current_topology_iops_governor"] == expected_governor
        assert "deployed_topology_saturation_min_count" not in calibration

    def test_projected_deployed_iops_at_drive_limit_is_saturated(self):
        desires = self._existing_ebs_desires()
        desires = desires.model_copy(
            update={"buffers": self._compute_buffer("derived", 2.0)}
        )
        explained = self._ebs_explained(
            desires,
            observed_iops=8_000,
            required_cluster_size=64,
            observed_ebs_read_io_per_read=20.0,
        )

        candidate_excuse = next(
            excuse
            for excuse in explained.excuses_by_model["org.netflix.cassandra"]
            if excuse.instance == "r7a.4xlarge"
        )
        calibration = candidate_excuse.context["ebs_io_calibration"]
        assert not explained.plans
        assert (
            calibration["projected_max_total_iops_per_node"],
            calibration["current_topology_iops_governor"],
        ) == (16_000, "deployed_topology_saturation")

    def test_saturation_floor_handles_drive_limit_at_iops_quantum(self):
        assert _deployed_topology_saturation_min_count(64, 200, 200) == 65

    def test_deployed_topology_evidence_falls_back_without_comparable_data(self):
        desires = self._existing_ebs_desires(disk_utilization_gib=0)
        explained = self._ebs_explained(
            desires,
            required_cluster_size=64,
            observed_ebs_read_io_per_read=20.0,
            same_data_as_deployed=False,
        )

        current_shape = next(
            plan.candidate_clusters.zonal[0]
            for plan in explained.plans
            if plan.candidate_clusters.zonal[0].instance.name == "r7a.4xlarge"
        )
        calibration = current_shape.cluster_params["cassandra.ebs_io_calibration"]
        assert "read_io_calibration_factor" not in calibration
        assert (
            calibration["current_topology_iops_governor"]
            == "candidate_model_unprojectable_demand"
        )
        assert "deployed_topology_saturation_min_count" not in calibration

    @pytest.mark.parametrize(
        "read_per_second,write_per_second,derived_scale,cluster_size_mode,expected",
        [
            (300_000, 300_000, None, "unrestricted", (None, 12_000)),
            (300_000, 300_000, 1.325, "unrestricted", (65, 15_800)),
            (300_000, 300_000, 4 / 3, "unrestricted", (65, 15_800)),
            (600_000, 600_000, None, "unrestricted", (None, None)),
            (300_000, 300_000, 2.0, "unrestricted", (98, 15_800)),
            (150_000, 150_000, None, "unrestricted", (None, None)),
            (600_000, 150_000, None, "unrestricted", (None, None)),
            (300_000, 300_000, 2.0, "doubling", (128, 12_000)),
        ],
    )
    def test_deployed_topology_saturation_floors_count_and_models_larger_candidates(
        self,
        read_per_second,
        write_per_second,
        derived_scale,
        cluster_size_mode,
        expected,
    ):
        expected_floor, expected_drive_iops = expected
        desires = self._existing_ebs_desires()
        desires.query_pattern = desires.query_pattern.model_copy(
            update={
                "estimated_read_per_second": certain_int(read_per_second),
                "estimated_write_per_second": certain_int(write_per_second),
            }
        )
        if derived_scale is not None:
            desires = desires.model_copy(
                update={"buffers": self._compute_buffer("derived", derived_scale)}
            )
        explained = self._ebs_explained(
            desires,
            num_results=20,
            max_regional_size=768,
            cluster_size_mode=cluster_size_mode,
        )

        assert explained.plans
        clusters = [plan.candidate_clusters.zonal[0] for plan in explained.plans]
        current_cluster = next(
            cluster for cluster in clusters if cluster.instance.name == "r7a.4xlarge"
        )
        current_calibration = current_cluster.cluster_params[
            "cassandra.ebs_io_calibration"
        ]
        assert (
            current_calibration.get("deployed_topology_saturation_min_count")
            == expected_floor
        )
        assert expected_floor is None or current_cluster.count >= expected_floor
        if expected_drive_iops is not None:
            drive = current_cluster.attached_drives[0]
            assert drive.read_io_per_s is not None
            assert drive.write_io_per_s is not None
            assert drive.read_io_per_s + drive.write_io_per_s == expected_drive_iops
        assert all(
            "deployed_topology_saturation_min_count"
            not in cluster.cluster_params["cassandra.ebs_io_calibration"]
            for cluster in clusters
            if cluster.instance.name != "r7a.4xlarge"
        )
        if read_per_second == write_per_second == 150_000:
            assert current_cluster.count < 64
        if (read_per_second, write_per_second) != (300_000, 300_000):
            assert (
                current_calibration["current_topology_iops_governor"]
                == "candidate_model_unprojectable_demand"
            )
        if cluster_size_mode == "doubling":
            assert current_cluster.count == 128

    @pytest.mark.parametrize(
        "constraint",
        [{"required_cluster_size": 64}, {"max_regional_size": 3}],
    )
    def test_ebs_calibration_is_in_no_plan_excuse_context(self, constraint):
        explained = planner.plan_certain_explained(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=self._existing_ebs_desires(),
            extra_model_arguments={
                "require_attached_disks": True,
                "require_local_disks": False,
                "cluster_size_mode": "unrestricted",
                "max_regional_size": 600,
                "observed_ebs_read_io_per_read": 20.0,
                **constraint,
            },
            instance_families=["r7a"],
            num_results=1,
        )
        calibrated_excuses = [
            excuse
            for excuses in explained.excuses_by_model.values()
            for excuse in excuses
            if "ebs_io_calibration" in excuse.context
        ]

        assert not explained.plans
        assert calibrated_excuses
        calibration = calibrated_excuses[0].context["ebs_io_calibration"]
        assert calibration["observed_read_io_per_read"] == 20.0
        assert calibration["read_io_calibration_factor"] > 1

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
    ):
        hardware = shapes.region("us-east-1")
        desires = CapacityDesires(
            query_pattern=QueryPattern(
                estimated_write_per_second=certain_float(100),
                estimated_mean_write_size_bytes=certain_int(512),
            ),
            data_shape=DataShape(estimated_state_size_gib=certain_float(300)),
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
