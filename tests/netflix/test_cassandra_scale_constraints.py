import pytest

from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.interface import Buffer
from service_capacity_modeling.interface import BufferComponent
from service_capacity_modeling.interface import BufferIntent
from service_capacity_modeling.interface import Buffers
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import certain_float
from service_capacity_modeling.interface import certain_int
from service_capacity_modeling.interface import CurrentClusters
from service_capacity_modeling.interface import CurrentZoneClusterCapacity
from service_capacity_modeling.interface import DataShape
from service_capacity_modeling.interface import Interval


class TestCassandraScaleWithConstraints:
    """
    Test scaling scenarios with different storage constraints.

    Scenario 1:
    * CPU is running cool, but storage is running hot

    Case A: scale_up / scale_floor
        cluster = i4i.4xlarge (8 cores, 32GB RAM, 3.75TB storage)
        state_size = 3TB
        buffers = {
            'compute_scale': Buffer(
                intent=BufferIntent.scale,
                components=[BufferComponent.compute],
                ratio=1.5,
            ),
            # Storage is running hot and keep it that way
            'storage_scale': Buffer(
                intent=BufferIntent.scale,
                components=[BufferComponent.disk],
                ratio=1.0,
            ),
            'storage_floor': Buffer(
                intent=BufferIntent.floor,
                components=[BufferComponent.disk],
                ratio=1.0,
            ),
        }
        Expected:
        * may scale up the compute to meet desired buffer (1.5x)
        * scales up the storage to meet the desired buffer (4x)

    Case B: storage scale_down buffer / ceiling
        cluster = i4i.4xlarge (8 cores, 32GB RAM, 3.75TB storage)
        state_size = 3TB
        buffers = {
            'compute_scale': Buffer(
                intent=BufferIntent.scale,
                components=[BufferComponent.compute],
                ratio=1.5,
            ),
            'storage_scale_up': Buffer(
                intent=BufferIntent.scale,
                components=[BufferComponent.disk],
                ratio=1.0,
            ),
            'storage_ceiling': Buffer(
                intent=BufferIntent.ceiling,
                components=[BufferComponent.disk],
                ratio=1.0,
            )
        }

        Expected:
        * new cluster does not scale up storage. There should be at least 3.75TB of
        storage because the ceiling constraints prevents scaling up storage.

    Scenario 2:
    * CPU is running hot, but storage is running cool

    Case A: scale_down
        cluster = i4i.4xlarge (8 cores, 32GB RAM, 3.75TB storage)
        state_size = 100GB
        buffers = {
            'compute_scale': Buffer(
                intent=BufferIntent.scale,
                components=[BufferComponent.compute],
                ratio=1.5,
            ),
            'storage_scale_down': Buffer(
                intent=BufferIntent.scale_down,
                components=[BufferComponent.disk],
                ratio=1.0,
            )
        }
        Expected:
            * Cluster has a very high existing storage buffer. new cluster should
            just meet the 4x storage buffer, which lowers the required
            storage to ~4 * 100GB = 400GB.
    Case B: storage scale_up buffer / floor
        cluster = i4i.4xlarge (8 cores, 32GB RAM, 3.75TB storage)
        state_size = 100GB
        buffers = {
            'compute_scale': Buffer(
                intent=BufferIntent.scale,
                components=[BufferComponent.compute],
                ratio=1.5,
            ),
            'storage_scale_up': Buffer(
                intent=BufferIntent.scale_up,
                components=[BufferComponent.disk],
                ratio=1.0,
            )
        }

        Expected:
            * The new cluster must have at least 3.75TB of storage because
            the scale_up intent allows storage to scale up to meet the
            desired buffer and prevents us from scaling down


    """

    # i4i.4xlarge cluster specifications
    I4I_4XLARGE_VCPU = 16
    I4I_4XLARGE_RAM_GIB = 128
    I4I_4XLARGE_DISK_GIB = 3750  # 3.75 TB
    CLUSTER_SIZE = 4

    # Calculated totals for i4i.4xlarge cluster
    I4I_4XLARGE_TOTAL_VCPU = CLUSTER_SIZE * I4I_4XLARGE_VCPU  # 64 vCPU
    I4I_4XLARGE_TOTAL_STORAGE_GIB = CLUSTER_SIZE * I4I_4XLARGE_DISK_GIB  # 15TB

    # Test scenarios
    LOW_CPU_HIGH_STORAGE_CLUSTER = CurrentZoneClusterCapacity(
        cluster_instance_name="i4i.4xlarge",
        cluster_instance_count=certain_int(CLUSTER_SIZE),
        cpu_utilization=Interval(low=8, mid=10, high=12, confidence=0.9),  # Low CPU
        memory_utilization_gib=certain_float(16.0),  # ~12% memory
        disk_utilization_gib=certain_float(1200),  # High disk usage
        network_utilization_mbps=certain_float(128.0),
    )

    SCALE_FACTOR = 1.5

    # Top-level buffer configurations for different scaling scenarios
    SCALE_COMPUTE = Buffers(
        derived={
            "compute": Buffer(
                intent=BufferIntent.scale,
                components=[BufferComponent.cpu],
                ratio=SCALE_FACTOR,
            )
        }
    )

    SCALE_UP_STORAGE = Buffers(
        derived={
            "compute": Buffer(
                intent=BufferIntent.scale,
                components=[BufferComponent.cpu],
                ratio=SCALE_FACTOR,
            ),
            "storage": Buffer(
                intent=BufferIntent.scale_up,
                components=[BufferComponent.disk],
                ratio=1.0,
            ),
        }
    )

    SCALE_DOWN_STORAGE = Buffers(
        derived={
            "compute": Buffer(
                intent=BufferIntent.scale,
                components=[BufferComponent.cpu],
                ratio=SCALE_FACTOR,
            ),
            "storage": Buffer(
                intent=BufferIntent.scale_down,
                components=[BufferComponent.disk],
                ratio=1.0,
            ),
        }
    )

    SCALE_FLOOR_STORAGE = Buffers(
        derived={
            "compute": Buffer(
                intent=BufferIntent.scale,
                components=[BufferComponent.cpu],
                ratio=SCALE_FACTOR,
            ),
            "storage": Buffer(
                intent=BufferIntent.scale,
                components=[BufferComponent.disk],
                ratio=1.0,
            ),
            "storage_floor": Buffer(
                intent=BufferIntent.floor,
                components=[BufferComponent.disk],
                ratio=1.0,
            ),
        }
    )

    SCALE_CEILING_STORAGE = Buffers(
        derived={
            "compute": Buffer(
                intent=BufferIntent.scale,
                components=[BufferComponent.cpu],
                ratio=SCALE_FACTOR,
            ),
            "storage": Buffer(
                intent=BufferIntent.scale,
                components=[BufferComponent.disk],
                ratio=1.0,
            ),
            "storage_ceiling": Buffer(
                intent=BufferIntent.ceiling,
                components=[BufferComponent.disk],
                ratio=1.0,
            ),
        }
    )

    @staticmethod
    def _cur_state_size(cluster: CurrentZoneClusterCapacity):
        return cluster.cluster_instance_count.mid * cluster.disk_utilization_gib.mid

    @pytest.mark.parametrize(
        "buffers",
        [
            SCALE_FLOOR_STORAGE,
            SCALE_UP_STORAGE,
        ],
    )
    def test_scenario_1_case_a_scale_up_scale_floor(self, buffers):
        """
        Scenario 1: Storage is running hot (3TB state on 3.75TB storage)
        Case A: scale + floor constraint

        EXPECTATION: Cluster scales up storage to meet the desired buffer (4x)
        WHY: Storage is running hot (3TB state on 3.75TB storage), so we need to
             scale up storage. The floor constraint prevents scaling down below current
             levels, ensuring we maintain
             at least the current storage capacity while scaling up to meet demand.
        """
        # Create scaling desire with high storage demand
        cluster = self.LOW_CPU_HIGH_STORAGE_CLUSTER
        desires = CapacityDesires(
            service_tier=1,
            current_clusters=CurrentClusters(zonal=[cluster]),
            data_shape=DataShape(
                estimated_state_size_gib=certain_int(3000),  # 3TB state
            ),
            buffers=buffers,
        )

        # Run capacity planner
        cap_plan = planner.plan_certain(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=desires,
        )[0]
        result = cap_plan.candidate_clusters.zonal[0]

        # Extract current and result values
        current_cores = self.I4I_4XLARGE_TOTAL_VCPU
        current_storage = self.I4I_4XLARGE_TOTAL_STORAGE_GIB
        result_cores = result.count * result.instance.cpu
        result_storage = result.count * result.instance.drive.size_gib

        # Assertions for CPU cores
        # CPU should scale up by 1.5x: 64 cores → 96 cores
        expected_cpu_cores = current_cores * 1.5
        assert result_cores >= expected_cpu_cores, (
            f"CPU should scale up to at least {expected_cpu_cores} cores, "
            f"got {result_cores} cores"
        )

        # Assertions for storage
        # Storage should scale up to meet 4x buffer requirement
        expected_storage = 3000 * 4  # 4x buffer on 3TB state
        assert result_storage >= expected_storage, (
            f"Storage should scale up to at least {expected_storage} GiB, "
            f"got {result_storage} GiB"
        )

        # Storage should not go below floor (current storage)
        assert result_storage >= current_storage, (
            f"Storage should not go below floor of {current_storage} GiB, "
            f"got {result_storage} GiB"
        )

    @pytest.mark.parametrize(
        "buffers",
        [
            SCALE_CEILING_STORAGE,
            SCALE_DOWN_STORAGE,
        ],
    )
    def test_scenario_1_case_b_storage_scale_down_buffer_ceiling(self, buffers):
        """
        Scenario 1: Storage is over-provisioned (100GB state on 3.75TB storage)
        Case B: scale + ceiling constraint

        EXPECTATION: New cluster does not scale up storage, but cores increase (1.5x)
        WHY: State size is only 100GB on 3.75TB storage, so storage is over-provisioned.
             The ceiling constraint prevents scaling up storage beyond current levels.
             However, compute still scales up to handle increased traffic (1.5x ratio).
        """
        # Create scaling desire with low storage demand
        cluster = self.LOW_CPU_HIGH_STORAGE_CLUSTER
        desires = CapacityDesires(
            service_tier=1,
            current_clusters=CurrentClusters(zonal=[cluster]),
            data_shape=DataShape(
                estimated_state_size_gib=certain_int(
                    100
                ),  # 100GB state on 3.75TB storage
            ),
            buffers=buffers,
        )

        # Run capacity planner
        cap_plan = planner.plan_certain(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=desires,
        )[0]
        result = cap_plan.candidate_clusters.zonal[0]

        # Extract current and result values
        current_cores = self.I4I_4XLARGE_TOTAL_VCPU
        current_storage = self.I4I_4XLARGE_TOTAL_STORAGE_GIB
        result_cores = result.count * result.instance.cpu
        result_storage = result.count * result.instance.drive.size_gib

        # Assertions for CPU cores
        # CPU should scale up by 1.5x: 64 cores → 96 cores
        expected_cpu_cores = current_cores * 1.5
        assert result_cores >= expected_cpu_cores, (
            f"CPU should scale up to at least {expected_cpu_cores} cores, "
            f"got {result_cores} cores"
        )

        # Assertions for storage
        # Storage should not exceed ceiling (current storage)
        assert result_storage <= current_storage, (
            f"Storage should not exceed ceiling of {current_storage} GiB, "
            f"got {result_storage} GiB"
        )

        # Storage should still meet the 4x buffer requirement
        expected_storage = 100 * 4  # 4x buffer on 100GB state
        assert result_storage >= expected_storage, (
            f"Storage should meet 4x buffer requirement of {expected_storage} GiB, "
            f"got {result_storage} GiB"
        )

    @pytest.mark.parametrize(
        "buffers",
        [
            SCALE_CEILING_STORAGE,
            SCALE_DOWN_STORAGE,
        ],
    )
    def test_scenario_2_case_a_scale_down(self, buffers):
        """
        Scenario 2: Storage is massively over-provisioned (100GB state on 3.75TB
        storage) Case A: scale_down intent

        EXPECTATION: New cluster has a very high existing storage buffer. New cluster
        should just meet the 4x storage buffer.
        WHY: State size is only 100GB on 3.75TB storage, so storage is massively
             over-provisioned. The scale_down intent allows storage to scale down to
             meet the actual demand,
             but we still maintain a 4x buffer for safety and performance.
        """
        # Create scaling desire with low storage demand
        cluster = self.LOW_CPU_HIGH_STORAGE_CLUSTER
        desires = CapacityDesires(
            service_tier=1,
            current_clusters=CurrentClusters(zonal=[cluster]),
            data_shape=DataShape(
                estimated_state_size_gib=certain_int(
                    100
                ),  # 100GB state on 3.75TB storage
            ),
            buffers=buffers,
        )

        # Run capacity planner
        cap_plan = planner.plan_certain(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=desires,
        )[0]
        result = cap_plan.candidate_clusters.zonal[0]

        # Extract current and result values
        current_cores = self.I4I_4XLARGE_TOTAL_VCPU
        current_storage = self.I4I_4XLARGE_TOTAL_STORAGE_GIB
        result_cores = result.count * result.instance.cpu
        result_storage = result.count * result.instance.drive.size_gib

        # Assertions for CPU cores
        # CPU should scale up by 1.5x: 64 cores → 96 cores
        expected_cpu_cores = current_cores * 1.5
        assert result_cores >= expected_cpu_cores, (
            f"CPU should scale up to at least {expected_cpu_cores} cores, "
            f"got {result_cores} cores"
        )

        # Assertions for storage
        # Storage should scale down to meet 4x buffer requirement
        expected_storage = 100 * 4  # 4x buffer on 100GB state
        assert result_storage >= expected_storage, (
            f"Storage should meet 4x buffer requirement of {expected_storage} GiB, "
            f"got {result_storage} GiB"
        )

        # Storage should scale down from current over-provisioned level
        assert result_storage <= current_storage, (
            f"Storage should scale down from {current_storage} GiB, "
            f"got {result_storage} GiB"
        )

    @pytest.mark.parametrize(
        "buffers",
        [
            SCALE_FLOOR_STORAGE,
            SCALE_UP_STORAGE,
        ],
    )
    def test_scenario_2_case_b_storage_scale_up_buffer_floor(self, buffers):
        """
        Scenario 2: Storage is massively over-provisioned (100GB state on 3.75TB
        storage) Case B: scale_up intent (implicit floor)

        EXPECTATION: New cluster has a very high existing storage buffer. New cluster
        should not change the storage on the cluster (we should have at least 3.75TB
        of storage).
        WHY: State size is only 100GB on 3.75TB storage, so storage is massively
             over-provisioned. The scale_up intent with floor constraint prevents
             scaling down storage below current levels, maintaining the existing
             3.75TB storage capacity even though it's over-provisioned.
        """
        # Create scaling desire with low storage demand
        cluster = self.LOW_CPU_HIGH_STORAGE_CLUSTER
        desires = CapacityDesires(
            service_tier=1,
            current_clusters=CurrentClusters(zonal=[cluster]),
            data_shape=DataShape(
                estimated_state_size_gib=certain_int(
                    100
                ),  # 100GB state on 3.75TB storage
            ),
            buffers=buffers,
        )

        # Run capacity planner
        cap_plan = planner.plan_certain(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=desires,
        )[0]
        result = cap_plan.candidate_clusters.zonal[0]

        # Extract current and result values
        current_cores = self.I4I_4XLARGE_TOTAL_VCPU
        current_storage = self.I4I_4XLARGE_TOTAL_STORAGE_GIB
        result_cores = result.count * result.instance.cpu
        result_storage = result.count * result.instance.drive.size_gib

        # Assertions for CPU cores
        # CPU should scale up by 1.5x: 64 cores → 96 cores
        expected_cpu_cores = current_cores * 1.5
        assert result_cores >= expected_cpu_cores, (
            f"CPU should scale up to at least {expected_cpu_cores} cores, "
            f"got {result_cores} cores"
        )

        # Assertions for storage
        # Storage should maintain current level (floor constraint)
        assert result_storage >= current_storage, (
            f"Storage should maintain at least {current_storage} GiB due to floor "
            f"constraint, got {result_storage} GiB"
        )

        # Storage should still meet the 4x buffer requirement
        expected_storage = 100 * 4  # 4x buffer on 100GB state
        assert result_storage >= expected_storage, (
            f"Storage should meet 4x buffer requirement of {expected_storage} GiB, "
            f"got {result_storage} GiB"
        )
