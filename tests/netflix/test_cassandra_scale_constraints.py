from typing import Optional

from _pytest.python_api import approx

from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.hardware import shapes
from service_capacity_modeling.interface import Buffer
from service_capacity_modeling.interface import BufferComponent
from service_capacity_modeling.interface import BufferIntent
from service_capacity_modeling.interface import Buffers
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import certain_float
from service_capacity_modeling.interface import certain_int
from service_capacity_modeling.interface import ClusterCapacity
from service_capacity_modeling.interface import CurrentClusters
from service_capacity_modeling.interface import CurrentZoneClusterCapacity
from service_capacity_modeling.interface import Instance
from service_capacity_modeling.models.common import cpu_headroom_target
from tests.util import assert_similar_compute


def get_ideal_cpu_for_instance(instance_name: str) -> float:
    """
    Get the CPU headroom target for a given instance type without any
    scale buffers and using the default model buffers

    Args:
        instance_name: The name of the instance (e.g., 'i4i.4xlarge')

    Returns:
        The CPU headroom target as a percentage (e.g., 0.3 for 30%)
    """
    instance = shapes.instance(instance_name)
    return ((1 - cpu_headroom_target(instance)) * 100) / 3  # 2 * 1.5


def total_ipc_per_second(instance: Instance, count: int) -> float:
    return instance.cpu * instance.cpu_ghz * instance.cpu_ipc_scale * count


def get_drive_size_gib(cluster: ClusterCapacity) -> int:
    """Safely get drive size in GiB from attached or instance drives."""
    # Check attached_drives first (for EBS)
    if cluster.attached_drives:
        return cluster.attached_drives[0].size_gib

    # Fall back to checking the instance's built-in drive
    if cluster.instance.drive is not None:
        return cluster.instance.drive.size_gib

    raise ValueError(
        f"Instance {cluster.instance.name} has no drive or attached drives"
    )


class ResourceTargets:
    """Hot, cold, and ideal utilization targets for a resource."""

    def __init__(self, ideal: float):
        self.ideal = ideal

    @property
    def hot(self) -> float:
        """Hot utilization target (above ideal)."""
        return self.ideal * 2

    @property
    def cold(self) -> float:
        """Cold utilization target (below ideal)."""
        return self.ideal // 2


class I4i4xlarge:
    """i4i.4xlarge instance specifications with computed utilization targets."""

    def __init__(self, cluster_size: int = 4):
        self.cluster_size = cluster_size
        self.instance_name = "i4i.4xlarge"
        self.instance = shapes.instance(self.instance_name)

        # Per-instance specifications
        self.vcpu_per_instance = self.instance.cpu
        assert self.vcpu_per_instance == 16

        self.ram_gib_per_instance = self.instance.ram_gib
        assert self.ram_gib_per_instance == 122.07

        # i4i instances have built-in NVMe storage
        assert self.instance.drive is not None, (
            f"Instance {self.instance_name} has no drive"
        )
        self.disk_gib_per_instance = self.instance.drive.size_gib
        assert self.disk_gib_per_instance == 3492

    @property
    def total_vcpu(self) -> int:
        """Total vCPU across the cluster."""
        return self.cluster_size * self.vcpu_per_instance

    @property
    def total_ram_gib(self) -> float:
        """Total RAM across the cluster."""
        return self.cluster_size * self.ram_gib_per_instance

    @property
    def total_disk_gib(self) -> int:
        """Total disk across the cluster."""
        return self.cluster_size * self.disk_gib_per_instance

    @property
    def cpu(self) -> ResourceTargets:
        """CPU utilization targets derived from headroom target."""
        ideal_cpu = get_ideal_cpu_for_instance(self.instance_name)
        return ResourceTargets(ideal=ideal_cpu)

    @property
    def memory(self) -> ResourceTargets:
        """Memory utilization targets based on typical Cassandra usage."""
        ideal_ram = int(self.ram_gib_per_instance / 4)
        return ResourceTargets(ideal=ideal_ram)

    @property
    def disk(self) -> ResourceTargets:
        """Storage utilization targets based on 4x buffer requirement."""
        ideal_storage = int(self.disk_gib_per_instance / 4)  # 4x buffer
        return ResourceTargets(ideal=ideal_storage)

    @property
    def network(self) -> ResourceTargets:
        """
        Network utilization targets derived from actual instance
        bandwidth with default compute buffer.
        """
        inst = shapes.instance(self.instance_name)
        # Apply default compute buffer ratio 1.5x + 2x background
        ideal_network = int(inst.net_mbps / 3)
        return ResourceTargets(ideal=ideal_network)

    def get(
        self,
        cpu: Optional[float] = None,
        memory_gib: Optional[float] = None,
        disk_gib: Optional[float] = None,
        network: Optional[float] = None,
    ) -> CurrentZoneClusterCapacity:
        return CurrentZoneClusterCapacity(
            cluster_instance_name=self.instance_name,
            cluster_instance=shapes.instance(self.instance_name),
            cluster_instance_count=certain_int(self.cluster_size),
            cpu_utilization=certain_float(cpu if cpu is not None else self.cpu.cold),
            memory_utilization_gib=certain_float(
                memory_gib if memory_gib is not None else self.memory.cold
            ),
            disk_utilization_gib=certain_float(
                disk_gib if disk_gib is not None else self.disk.cold
            ),
            network_utilization_mbps=certain_float(
                network if network is not None else self.network.cold
            ),
        )

    def get_desires(self, cluster, buffers: Buffers) -> CapacityDesires:
        return CapacityDesires(
            service_tier=1,
            current_clusters=CurrentClusters(zonal=[cluster]),
            # data_shape=DataShape(
            #     estimated_state_size_gib=certain_int(state_size_gib),
            # ),
            buffers=buffers,
        )

    def __post_init__(self):
        """Validate the ideal CPU matches expected value after initialization."""
        ideal_cpu = self.cpu.ideal
        assert ideal_cpu == 21, f"Expected ideal CPU to be 21, got {ideal_cpu}"


# Create instance for use in tests
CLUSTER = I4i4xlarge()
CLUSTER_SIZE = CLUSTER.cluster_size


class TestStorageScalingConstraints:
    """
    Unit tests for storage scaling constraints in isolation.
    Tests scale_up and scale_down intents for storage only.
    """

    # Test scenarios - disk-focused only
    DISK_HOT_CAPACITY = CLUSTER.get(disk_gib=CLUSTER.disk.hot)

    DISK_COLD_CAPACITY = CLUSTER.get(disk_gib=CLUSTER.disk.cold)

    # Storage-only buffer configurations
    STORAGE_SCALE_UP = Buffers(
        derived={
            "storage": Buffer(
                intent=BufferIntent.scale_up,
                components=[BufferComponent.disk],
                ratio=1.0,
            ),
        }
    )

    STORAGE_SCALE_DOWN = Buffers(
        derived={
            "storage": Buffer(
                intent=BufferIntent.scale_down,
                components=[BufferComponent.disk],
                ratio=1.0,
            ),
        }
    )

    def test_storage_hot_scale_up(self):
        """
        Test: Storage is running hot (above ideal threshold) with scale_up intent

        EXPECTATION: Storage should scale up to meet 4x buffer requirement
        WHY: Current storage is insufficient for the 4x buffer,
             so scale_up should expand it
        """
        cluster_capacity = self.DISK_HOT_CAPACITY
        desires = CLUSTER.get_desires(
            cluster=cluster_capacity, buffers=self.STORAGE_SCALE_UP
        )

        cap_plan = planner.plan_certain(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            num_results=20,
            desires=desires,
        )
        result = cap_plan[0].candidate_clusters.zonal[0]
        result_storage = result.count * get_drive_size_gib(result)

        # Storage should scale up to meet 4x buffer requirement
        expected_storage = (
            CLUSTER.disk.hot * 4 * CLUSTER_SIZE
        )  # 4x buffer on hot state size based on default buffer

        # Assert the magic number for readability / clarity
        assert expected_storage == 27936
        assert expected_storage <= result_storage <= expected_storage * 2.5

    def test_storage_hot_scale_down(self):
        """
        Test: Storage utilization is high with scale_down intent

        EXPECTATION: Storage should not scale down below current levels because
        WHY: Current storage is already under-provisioned, so scale_down should
            not reduce it
        """
        cluster_capacity = self.DISK_HOT_CAPACITY
        desires = CLUSTER.get_desires(cluster_capacity, self.STORAGE_SCALE_DOWN)

        cap_plan = planner.plan_certain(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=desires,
        )
        result = cap_plan[0].candidate_clusters.zonal[0]

        result_storage = result.count * get_drive_size_gib(result)

        # Storage should not scale down from current over-provisioned level
        # It's acceptable to be within a 5% margin of current storage because
        # the capacity treats this as a lower bound
        current_storage = CLUSTER.total_disk_gib
        assert result_storage == approx(current_storage, 0.05 * current_storage)

    def test_storage_cold_scale_up(self):
        """
        Test: Storage is cold (below ideal) with scale_up intent

        EXPECTATION: scale_up should prevent scaling down below current levels
        WHY: scale_up intent means "only scale up, never scale down" - this
        tests the constraint
        """
        cluster_capacity = self.DISK_COLD_CAPACITY
        desires = CLUSTER.get_desires(cluster_capacity, self.STORAGE_SCALE_UP)

        cap_plan = planner.plan_certain(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=desires,
        )[0]
        result = cap_plan.candidate_clusters.zonal[0]

        result_storage = result.count * get_drive_size_gib(result)

        # scale_up should prevent scaling down below current levels
        # Even if demand is low, we should maintain at least current storage capacity
        assert result_storage >= CLUSTER.total_disk_gib, (
            f"scale_up should prevent scaling down below current "
            f"{CLUSTER.total_disk_gib} GiB, got {result_storage} GiB"
        )

        expected_storage = CLUSTER.disk.cold * 4 * CLUSTER_SIZE
        assert result_storage >= expected_storage

    def test_storage_cold_scale_down(self):
        """
        Test: Storage is cold with scale_down intent

        EXPECTATION: scale_down should reduce the provisioned storage
        WHY: The storage is already over-provisioned, so scale down will right-size
        """
        cluster_capacity = self.DISK_COLD_CAPACITY
        desires = CLUSTER.get_desires(
            cluster=cluster_capacity, buffers=self.STORAGE_SCALE_DOWN
        )

        cap_plan = planner.plan_certain(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            num_results=20,
            desires=desires,
        )
        result = cap_plan[0].candidate_clusters.zonal[0]

        # The current cost of the cluster should be smaller
        # This annual cost is a naive calculation that doesn't include
        # S3 / network costs but currently the output doesn't include those
        # either
        assert cluster_capacity.cluster_instance is not None
        current_cost = (
            cluster_capacity.cluster_instance_count.mid
            * cluster_capacity.cluster_instance.annual_cost
        )
        # Assert that the result's annual cost is just the instance count * cost
        assert result.annual_cost == result.count * result.instance.annual_cost, (
            f"Expected annual cost to be "
            f"{result.count * result.instance.annual_cost}. Seems like the "
            f"way annual cost is calculated has changed breaking later "
            f"assertions in this test "
        )

        result_storage = result.count * get_drive_size_gib(result)
        # scale_down should scale down below current levels
        assert result.annual_cost <= current_cost
        assert result_storage <= CLUSTER.total_disk_gib


class TestCPUScalingConstraints:
    """
    Unit tests for CPU scaling constraints in isolation.
    Tests scale_up and scale_down intents for CPU only.
    """

    # Class-level constants
    CLUSTER_SIZE = 4

    # Test scenarios - CPU-focused only
    CPU_COLD_CAPACITY = CLUSTER.get(cpu=CLUSTER.cpu.cold)

    CPU_HOT_CAPACITY = CLUSTER.get(cpu=CLUSTER.cpu.hot)

    # CPU-only buffer configurations
    CPU_SCALE_UP = Buffers(
        derived={
            "compute": Buffer(
                intent=BufferIntent.scale_up,
                components=[BufferComponent.cpu],
                ratio=1.0,
            ),
        }
    )

    CPU_SCALE_DOWN = Buffers(
        derived={
            "compute": Buffer(
                intent=BufferIntent.scale_down,
                components=[BufferComponent.cpu],
                ratio=1.0,
            ),
        }
    )

    def test_cpu_running_cool_scale_up(self):
        """
        Test: CPU is running cool (8-12 cores on 64 cores) with scale_up intent

        EXPECTATION: CPU remain as-is
        WHY: Current CPU is under-utilized, so scale_up should keep the existing
        capacity
        """
        cluster_capacity = self.CPU_COLD_CAPACITY
        desires = CLUSTER.get_desires(cluster_capacity, self.CPU_SCALE_UP)

        cap_plan = planner.plan_certain(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=desires,
        )
        result = cap_plan[0].candidate_clusters.zonal[0]

        if cluster_capacity.cluster_instance is None:
            raise ValueError("cluster_instance cannot be None")

        # It's likely to suggest to a large EBS if we are compute bound
        assert_similar_compute(
            shapes.instance("c6a.8xlarge"),
            result.instance,
            int(cluster_capacity.cluster_instance_count.mid) // 2,
            result.count,
        )

    def test_cpu_running_cool_scale_down(self):
        """
        Test: CPU is running cool (8-12 cores on 64 cores) with scale_down intent

        EXPECTATION: CPU should scale down to meet 1.5x buffer requirement
        WHY: Current CPU is over-provisioned, so scale_down should reduce it
        """
        cluster_capacity = self.CPU_COLD_CAPACITY
        desires = CLUSTER.get_desires(cluster_capacity, self.CPU_SCALE_DOWN)

        cap_plan = planner.plan_certain(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=desires,
        )
        result = cap_plan[0].candidate_clusters.zonal[0]
        # CPU should scale down from current over-provisioned level
        assert_similar_compute(
            shapes.instance("i4i.4xlarge"),
            result.instance,
            CLUSTER_SIZE // 2,
            result.count,
        )

    def test_cpu_running_hot_scale_up(self):
        """
        Test: CPU is running hot (56-64 cores on 64 cores) with scale_up intent

        EXPECTATION: CPU should scale up to meet 1.5x buffer requirement
        WHY: Current CPU is at capacity, so scale_up should expand it
        """
        cluster_capacity = self.CPU_HOT_CAPACITY
        desires = CLUSTER.get_desires(cluster_capacity, self.CPU_SCALE_UP)

        cap_plan = planner.plan_certain(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=desires,
        )[0]
        result = cap_plan.candidate_clusters.zonal[0]

        result_cores = result.count * result.instance.cpu

        # CPU should scale up by 1.5x: 64 cores → 96 cores
        expected_cpu_cores = CLUSTER.total_vcpu * 1.5
        assert result_cores >= expected_cpu_cores, (
            f"CPU should scale up to at least {expected_cpu_cores} cores, "
            f"got {result_cores} cores"
        )
        assert_similar_compute(
            shapes.instance("m6id.4xlarge"),
            result.instance,
            CLUSTER_SIZE * 2,
            result.count,
        )

    def test_cpu_running_hot_scale_down(self):
        """
        Test: CPU is running hot (56-64 cores on 64 cores) with scale_down intent

        EXPECTATION: CPU should not scale down
        WHY: Even though CPU is hot, scale_down will not allow a scale up
        """
        cluster_capacity = self.CPU_HOT_CAPACITY
        desires = CLUSTER.get_desires(cluster_capacity, self.CPU_SCALE_DOWN)

        cap_plan = planner.plan_certain(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=desires,
        )[0]
        result = cap_plan.candidate_clusters.zonal[0]
        result_cores = result.count * result.instance.cpu

        # CPU should scale down to meet 1.5x buffer requirement
        assert_similar_compute(
            shapes.instance("c6a.4xlarge"), result.instance, CLUSTER_SIZE, result.count
        )
        assert result_cores <= CLUSTER.total_vcpu


class TestStorageAndCPUIntegration:
    """
    E2E integration tests for storage and CPU scaling constraints together.
    Tests realistic scenarios where both components are constrained.
    """

    # Test scenarios - both storage and CPU are constrained
    BOTH_CONSTRAINED_CAPACITY = CLUSTER.get(
        cpu=CLUSTER.cpu.hot,  # High CPU - above ideal
        disk_gib=CLUSTER.disk.hot,  # High disk usage - above ideal
    )

    # Buffer configurations for both storage and CPU constraints
    BOTH_SCALE_UP = Buffers(
        derived={
            "compute": Buffer(
                intent=BufferIntent.scale_up,
                components=[BufferComponent.cpu],
                ratio=2,
            ),
            "storage": Buffer(
                intent=BufferIntent.scale_up,
                components=[BufferComponent.disk],
                ratio=2,
            ),
        }
    )

    BOTH_SCALE_DOWN = Buffers(
        derived={
            "compute": Buffer(
                intent=BufferIntent.scale_down,
                components=[BufferComponent.cpu],
                ratio=2,
            ),
            "storage": Buffer(
                intent=BufferIntent.scale_down,
                components=[BufferComponent.disk],
                ratio=2,
            ),
        }
    )
    # Additional buffer combinations for mixed CPU/DISK scaling
    CPU_UP_DISK_DOWN = Buffers(
        derived={
            "compute": Buffer(
                intent=BufferIntent.scale_up,
                components=[BufferComponent.cpu],
                ratio=2,
            ),
            "storage": Buffer(
                intent=BufferIntent.scale_down,
                components=[BufferComponent.disk],
                ratio=2,
            ),
        }
    )
    CPU_DOWN_DISK_UP = Buffers(
        derived={
            "compute": Buffer(
                intent=BufferIntent.scale_down,
                components=[BufferComponent.cpu],
                ratio=2,
            ),
            "storage": Buffer(
                intent=BufferIntent.scale_up,
                components=[BufferComponent.disk],
                ratio=2,
            ),
        }
    )

    def test_cpu_scale_up_disk_scale_up(self):
        """
        Test: Both CPU and storage are constrained, both use scale_up intent

        EXPECTATION: Both CPU and storage should scale up to meet their requirements
        WHY: This tests the realistic scenario where both components need expansion
        """
        cluster_capacity = self.BOTH_CONSTRAINED_CAPACITY
        desires = CLUSTER.get_desires(cluster_capacity, self.BOTH_SCALE_UP)

        cap_plan = planner.plan_certain(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=desires,
            num_results=20,
        )
        result = cap_plan[0].candidate_clusters.zonal[0]

        result_cores = result.count * result.instance.cpu
        result_storage = result.count * get_drive_size_gib(result)

        # With EBS, planner selects compute-optimized instances with attached storage
        assert_similar_compute(
            shapes.instance("m7i.4xlarge"),
            result.instance,
            CLUSTER_SIZE * 4,
            result.count,
        )

        # CPU should scale up by 1.5x: 64 cores → 96 cores
        expected_cpu_cores = CLUSTER.total_vcpu * 1.5
        assert result_cores >= expected_cpu_cores, (
            f"CPU should scale up to at least {expected_cpu_cores} cores, "
            f"got {result_cores} cores"
        )

        # Storage should scale up to meet 4x buffer requirement
        expected_storage = CLUSTER.disk.hot * 4 * 2  # 4x buffer on hot disk utilization
        assert result_storage >= expected_storage, (
            f"Storage should scale up to at least {expected_storage} GiB, "
            f"got {result_storage} GiB"
        )

    def test_cpu_scale_down_disk_scale_down(self):
        """
        Test: Both CPU and storage are constrained, both use scale_down intent

        EXPECTATION: Both CPU and storage should remain
        WHY: scale_down prevents a scale up even with high utilization
        """
        cluster_capacity = self.BOTH_CONSTRAINED_CAPACITY
        desires = CLUSTER.get_desires(cluster_capacity, self.BOTH_SCALE_DOWN)

        cap_plan = planner.plan_certain(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=desires,
        )[0]
        result = cap_plan.candidate_clusters.zonal[0]

        if cluster_capacity.cluster_instance is None:
            raise ValueError("cluster_instance cannot be None")
        assert_similar_compute(
            cluster_capacity.cluster_instance,
            result.instance,
            CLUSTER_SIZE,
            result.count,
        )

        result_storage = result.count * get_drive_size_gib(result)
        assert result_storage == approx(CLUSTER.total_disk_gib, 0.05)

    def test_cpu_scale_up_disk_scale_down(self):
        """
        Test: CPU scale_up and Disk scale_down
        EXPECTATION: CPU should scale up, Disk should not exceed current capacity
        """
        cluster_capacity = self.BOTH_CONSTRAINED_CAPACITY
        desires = CLUSTER.get_desires(cluster_capacity, self.CPU_UP_DISK_DOWN)
        cap_plan = planner.plan_certain(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=desires,
            num_results=20,
        )
        result = cap_plan[0].candidate_clusters.zonal[0]
        result_cores = result.count * result.instance.cpu
        result_storage = result.count * get_drive_size_gib(result)
        assert_similar_compute(
            shapes.instance("m6id.8xlarge"), result.instance, 8, result.count
        )
        assert result_cores >= CLUSTER.total_vcpu * 2

        # Storage should not exceed current capacity
        assert result_storage == approx(CLUSTER.total_disk_gib, 0.05)
        assert result_storage >= CLUSTER.total_disk_gib

    def test_cpu_scale_down_disk_scale_up(self):
        """
        Test: CPU scale_down and Disk scale_up
        EXPECTATION: CPU should not exceed current capacity, Disk should scale up
        """
        cluster_capacity = self.BOTH_CONSTRAINED_CAPACITY
        desires = CLUSTER.get_desires(cluster_capacity, self.CPU_DOWN_DISK_UP)
        cap_plan = planner.plan_certain(
            model_name="org.netflix.cassandra",
            region="us-east-1",
            desires=desires,
        )[0]
        result = cap_plan.candidate_clusters.zonal[0]
        result_cores = result.count * result.instance.cpu
        result_storage = result.count * get_drive_size_gib(result)
        # CPU should not exceed current capacity * scale factor (2)
        assert result_cores <= CLUSTER.total_vcpu * 2, (
            f"CPU should not exceed current "
            f"{CLUSTER.total_vcpu} cores, got {result_cores} cores"
        )

        expected_storage = CLUSTER.total_disk_gib * 2
        assert result_storage >= expected_storage, (
            f"Storage should scale up to at least {expected_storage} GiB, "
            f"got {result_storage} GiB"
        )
