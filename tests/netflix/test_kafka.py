# pylint: disable=too-many-lines
import logging
from typing import Any

from pydantic import TypeAdapter

from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.interface import AccessPattern
from service_capacity_modeling.interface import Buffer
from service_capacity_modeling.interface import BufferComponent
from service_capacity_modeling.interface import BufferIntent
from service_capacity_modeling.interface import Buffers
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import certain_float
from service_capacity_modeling.interface import CurrentClusters
from service_capacity_modeling.interface import CurrentZoneClusterCapacity
from service_capacity_modeling.interface import DataShape
from service_capacity_modeling.interface import Drive
from service_capacity_modeling.interface import DriveType
from service_capacity_modeling.interface import ExcludeUnsetModel
from service_capacity_modeling.interface import FixedInterval
from service_capacity_modeling.interface import Interval
from service_capacity_modeling.interface import QueryPattern
from service_capacity_modeling.models.common import normalize_cores
from service_capacity_modeling.models.org.netflix.kafka import ClusterType

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.WARNING,  # Set to DEBUG for detailed capacity planner reasoning
    format="%(name)s - %(levelname)s - %(message)s",
    force=True,
)

logger.setLevel(logging.DEBUG)


def test_kafka_basic():
    # 100 MiB / second
    throughput = 100 * 1024 * 1024
    desires = CapacityDesires(
        service_tier=0,
        query_pattern=QueryPattern(
            # 2 consumers
            estimated_read_per_second=Interval(low=1, mid=2, high=2, confidence=0.98),
            # 1 producer
            estimated_write_per_second=Interval(low=1, mid=1, high=1, confidence=0.98),
            # Write throughput of 100 MiB/s
            estimated_mean_write_size_bytes=Interval(
                low=throughput, mid=throughput, high=throughput * 2, confidence=0.98
            ),
        ),
    )

    plan = planner.plan(
        model_name="org.netflix.kafka",
        region="us-east-1",
        desires=desires,
        extra_model_arguments={
            "cluster_type": ClusterType.strong,
            "retention": "PT4H",
        },
    )

    lr = plan.least_regret[0]
    lr_cluster = lr.candidate_clusters.zonal

    assert 8_000 < lr.candidate_clusters.total_annual_cost < 20_000
    assert lr_cluster[0].instance.family in ("i3en", "i4i", "r5", "r7a")


def test_kafka_large_scale():
    # 1 GiB / second
    throughput = 1024 * 1024 * 1024
    desires = CapacityDesires(
        service_tier=0,
        query_pattern=QueryPattern(
            # 2 consumers
            estimated_read_per_second=Interval(low=1, mid=2, high=2, confidence=0.98),
            # 1 producer
            estimated_write_per_second=Interval(low=1, mid=1, high=1, confidence=0.98),
            # Write throughput of 100 MiB/s
            estimated_mean_write_size_bytes=Interval(
                low=throughput, mid=throughput, high=throughput * 2, confidence=0.98
            ),
        ),
    )

    plan = planner.plan(
        model_name="org.netflix.kafka",
        region="us-east-1",
        desires=desires,
        extra_model_arguments={
            "cluster_type": ClusterType.strong,
            "retention": "PT4H",
        },
    )

    lr = plan.least_regret[0]
    lr_cluster = lr.candidate_clusters.zonal

    assert lr.candidate_clusters.total_annual_cost < 200_000
    assert lr_cluster[0].instance.family in ("i3en", "i4i", "r5")


def test_kafka_high_throughput():
    # 2.8 GiB / second
    throughput = 2.8 * 1024 * 1024 * 1024
    desires = CapacityDesires(
        service_tier=1,
        query_pattern=QueryPattern(
            # 2 consumers
            estimated_read_per_second=Interval(low=1, mid=2, high=2, confidence=0.98),
            # 1 producer
            estimated_write_per_second=Interval(low=1, mid=1, high=1, confidence=0.98),
            # Write throughput of 100 MiB/s
            estimated_mean_write_size_bytes=Interval(
                low=throughput, mid=throughput, high=throughput * 2, confidence=0.98
            ),
        ),
    )

    plan = planner.plan(
        model_name="org.netflix.kafka",
        region="us-east-1",
        desires=desires,
        extra_model_arguments={
            "cluster_type": ClusterType.ha,
            "retention": "PT3H",
        },
        num_results=3,
    )

    lr = plan.least_regret[0]
    lr_zone_requirements = lr.requirements.zonal[0]
    # This should be doable with ~136 cpu cores @3.1 = 182 cores,
    # ~1TiB RAM, and 25Gbps networking
    # Note that network reserves 40% headroom for streaming.
    expected_cpu = (120, 200)
    expected_ram = (400, 1200)
    expected_net = (20_000 * 1.4, 28_000 * 1.4)
    expected_disk = (30_000, 75_000)

    assert expected_cpu[0] < lr_zone_requirements.cpu_cores.mid < expected_cpu[1]
    assert expected_ram[0] < lr_zone_requirements.mem_gib.mid < expected_ram[1]
    assert expected_net[0] < lr_zone_requirements.network_mbps.mid < expected_net[1]
    assert expected_disk[0] < lr_zone_requirements.disk_gib.mid < expected_disk[1]

    lr_zone_cluster = lr.candidate_clusters.zonal[0]
    # 17 r5.2xlarge is this much memory
    expected_memory = 17 * 62
    # subtract out the 8G heaps + system
    expected_cache = 17 * 52

    assert expected_cache * 0.5 < lr.requirements.zonal[0].mem_gib.mid < expected_cache
    assert (
        expected_memory * 0.5
        < lr_zone_cluster.instance.ram_gib * lr_zone_cluster.count
        < expected_memory * 2
    )

    for lr in plan.least_regret:
        logger.debug(lr.candidate_clusters.zonal[0])
        # 37 i3en.xlarge 166k
        # 18 i3en.2xlarge 162k
        assert 50_000 < lr.candidate_clusters.total_annual_cost < 205_000
        clstr = lr.candidate_clusters.zonal[0]
        if clstr.instance.drive is None:
            assert clstr.attached_drives[0].name == "gp3"
            disk = clstr.attached_drives[0].size_gib * clstr.count
            assert expected_disk[0] < disk < expected_disk[1] * 2.5
        else:
            disk = clstr.instance.drive.size_gib * clstr.count
            assert expected_disk[0] < disk < expected_disk[1] * 5


def test_kafka_high_throughput_ebs():
    # 2.8 GiB / second
    throughput = 2.8 * 1024 * 1024 * 1024
    desires = CapacityDesires(
        service_tier=1,
        query_pattern=QueryPattern(
            # 2 consumers
            estimated_read_per_second=Interval(low=1, mid=2, high=2, confidence=0.98),
            # 1 producer
            estimated_write_per_second=Interval(low=1, mid=1, high=1, confidence=0.98),
            # Write throughput of 100 MiB/s
            estimated_mean_write_size_bytes=Interval(
                low=throughput, mid=throughput, high=throughput * 2, confidence=0.98
            ),
        ),
    )

    plan = planner.plan(
        model_name="org.netflix.kafka",
        region="us-east-1",
        desires=desires,
        extra_model_arguments={
            "cluster_type": ClusterType.ha,
            "retention": "PT3H",
            # Force to attached drives
            "max_local_disk_gib": 125,
        },
        num_results=3,
    )

    lr = plan.least_regret[0]
    lr_zone_cluster = lr.candidate_clusters.zonal[0]
    lr_zone_requirements = lr.requirements.zonal[0]
    # This should be doable with ~136 cpu cores @3.1 = 182 cores,
    # ~1TiB RAM, and 25Gbps networking
    # Note that network reserves 40% headroom for streaming.
    expected_cpu = (80, 140)
    expected_ram = (400, 1200)
    expected_net = (20_000 * 1.4, 28_000 * 1.4)
    expected_disk = (40000, 63500)
    req = lr_zone_requirements

    assert (
        expected_cpu[0]
        < normalize_cores(
            req.cpu_cores.mid,
            target_shape=desires.reference_shape,
            reference_shape=lr_zone_cluster.instance,
        )
        < expected_cpu[1]
    )
    assert expected_ram[0] < lr_zone_requirements.mem_gib.mid < expected_ram[1]
    assert expected_net[0] < lr_zone_requirements.network_mbps.mid < expected_net[1]
    assert expected_disk[0] < lr_zone_requirements.disk_gib.mid < expected_disk[1]

    # 17 r5.2xlarge is this much memory
    expected_memory = 17 * 62
    # subtract out the 8G heaps + system
    expected_cache = 17 * 52

    assert expected_cache * 0.5 < lr.requirements.zonal[0].mem_gib.mid < expected_cache
    assert (
        expected_memory * 0.5
        < lr_zone_cluster.instance.ram_gib * lr_zone_cluster.count
        < expected_memory * 2
    )
    assert lr.candidate_clusters.total_annual_cost < 250_000

    for lr in plan.least_regret:
        assert lr.candidate_clusters.total_annual_cost < 250_000
        clstr = lr.candidate_clusters.zonal[0]
        if clstr.instance.drive is None:
            assert clstr.instance.family[0] in ("r", "m")
            assert clstr.attached_drives[0].name == "gp3"
            disk = clstr.attached_drives[0].size_gib * clstr.count
            assert expected_disk[0] < disk < expected_disk[1] * 2.5


def test_kafka_model_constraints():
    # 2.8 GiB / second
    throughput = 2.8 * 1024 * 1024 * 1024
    desires = CapacityDesires(
        service_tier=1,
        query_pattern=QueryPattern(
            # 2 consumers
            estimated_read_per_second=Interval(low=1, mid=2, high=2, confidence=0.98),
            # 1 producer
            estimated_write_per_second=Interval(low=1, mid=1, high=1, confidence=0.98),
            # Write throughput of 100 MiB/s
            estimated_mean_write_size_bytes=Interval(
                low=throughput, mid=throughput, high=throughput * 2, confidence=0.98
            ),
        ),
    )

    required_zone_size = 10
    min_instance_cpu = 16
    # Force to attached drives
    require_attached_disks = True
    plan = planner.plan(
        model_name="org.netflix.kafka",
        region="us-east-1",
        desires=desires,
        extra_model_arguments={
            "cluster_type": ClusterType.ha,
            "retention": "PT3H",
            "require_attached_disks": require_attached_disks,
            "min_instance_cpu": min_instance_cpu,
            "required_zone_size": required_zone_size,
        },
        num_results=3,
    )
    expected_min_zonal_cpu = required_zone_size * min_instance_cpu

    for lr in plan.least_regret:
        for z in range(3):
            clstr = lr.candidate_clusters.zonal[z]
            assert clstr.instance.drive is None
            assert (clstr.instance.cpu * clstr.count) >= expected_min_zonal_cpu

    # force to local disks
    plan = planner.plan(
        model_name="org.netflix.kafka",
        region="us-east-1",
        desires=desires,
        extra_model_arguments={
            "cluster_type": ClusterType.ha,
            "retention": "PT3H",
            "require_local_disks": True,
            "min_instance_cpu": min_instance_cpu,
            "required_zone_size": required_zone_size,
        },
        num_results=3,
    )
    expected_min_zonal_cpu = required_zone_size * min_instance_cpu

    for lr in plan.least_regret:
        for z in range(3):
            clstr = lr.candidate_clusters.zonal[z]
            assert clstr.instance.drive is not None
            assert (clstr.instance.cpu * clstr.count) >= expected_min_zonal_cpu


def test_plan_certain():
    """
    Use current clusters cpu utilization to determine instance types directly as
    supposed to extrapolating it from the Data Shape
    """
    cluster_capacity = CurrentZoneClusterCapacity(
        cluster_instance_name="r5.2xlarge",
        cluster_drive=Drive(
            name="gp3",
            drive_type=DriveType.attached_ssd,
            size_gib=5000,
            block_size_kib=16,
        ),
        cluster_instance_count=Interval(low=27, mid=27, high=27, confidence=1),
        cpu_utilization=Interval(low=11.6, mid=19.29, high=27.57, confidence=1),
        memory_utilization_gib=certain_float(32.0),
        network_utilization_mbps=certain_float(128.0),
        disk_utilization_gib=Interval(low=1000, mid=1500, high=2000, confidence=0.98),
    )

    throughput = 2 * 1024 * 1024 * 1024
    desires = CapacityDesires(
        service_tier=1,
        current_clusters=CurrentClusters(zonal=[cluster_capacity]),
        query_pattern=QueryPattern(
            # 2 consumers
            estimated_read_per_second=Interval(low=1, mid=2, high=2, confidence=0.98),
            # 1 producer
            estimated_write_per_second=Interval(low=1, mid=1, high=1, confidence=0.98),
            estimated_mean_write_size_bytes=Interval(
                low=throughput, mid=throughput, high=throughput * 2, confidence=0.98
            ),
        ),
    )

    cap_plan = planner.plan_certain(
        model_name="org.netflix.kafka",
        region="us-east-1",
        num_results=3,
        num_regions=4,
        desires=desires,
        extra_model_arguments={
            "cluster_type": ClusterType.ha,
            "retention": "PT8H",
            "require_attached_disks": True,
            "required_zone_size": cluster_capacity.cluster_instance_count.mid,
        },
    )

    assert len(cap_plan) >= 1
    lr_clusters = cap_plan[0].candidate_clusters.zonal
    assert len(lr_clusters) >= 1
    logger.debug(lr_clusters[0].instance.name)
    assert lr_clusters[0].count == cluster_capacity.cluster_instance_count.high


def test_plan_certain_data_shape():
    """
    Use current clusters cpu utilization to determine instance types directly as
    supposed to extrapolating it from the Data Shape
    """
    cluster_capacity = CurrentZoneClusterCapacity(
        cluster_instance_name="r7a.4xlarge",
        cluster_drive=Drive(
            name="gp3",
            drive_type=DriveType.attached_ssd,
            size_gib=5000,
            block_size_kib=16,
        ),
        cluster_instance_count=Interval(low=15, mid=15, high=15, confidence=1),
        cpu_utilization=Interval(
            low=5.441147804260254,
            mid=13.548842955300195,
            high=25.11203956604004,
            confidence=1,
        ),
        memory_utilization_gib=Interval(low=0, mid=0, high=0, confidence=1),
        network_utilization_mbps=Interval(
            low=217.18696,
            mid=590.5934259505216,
            high=1220.205184,
            confidence=1,
        ),
        disk_utilization_gib=Interval(
            low=1000,
            mid=1500,
            high=2000,
            confidence=1,
        ),
    )

    desires = CapacityDesires(
        service_tier=1,
        current_clusters=CurrentClusters(zonal=[cluster_capacity]),
        query_pattern=QueryPattern(
            access_pattern=AccessPattern(AccessPattern.latency),
            # 2 consumers
            estimated_read_per_second=Interval(low=2, mid=2, high=4, confidence=1),
            # 1 producer
            estimated_write_per_second=Interval(low=1, mid=1, high=1, confidence=0.98),
            estimated_mean_read_latency_ms=Interval(low=1, mid=1, high=1, confidence=1),
            estimated_mean_write_latency_ms=Interval(
                low=1, mid=1, high=1, confidence=1
            ),
            estimated_mean_read_size_bytes=Interval(
                low=1024, mid=1024, high=1024, confidence=1
            ),
            estimated_mean_write_size_bytes=Interval(
                low=125000000, mid=579000000, high=1351000000, confidence=0.98
            ),
            estimated_read_parallelism=Interval(low=1, mid=1, high=1, confidence=1),
            estimated_write_parallelism=Interval(low=1, mid=1, high=1, confidence=1),
            read_latency_slo_ms=FixedInterval(low=0.4, mid=4, high=10, confidence=0.98),
            write_latency_slo_ms=FixedInterval(
                low=0.4, mid=4, high=10, confidence=0.98
            ),
        ),
        data_shape=DataShape(
            estimated_state_size_gib=Interval(
                low=44000, mid=86000, high=91000, confidence=1
            ),
        ),
    )

    cap_plan = planner.plan_certain(
        model_name="org.netflix.kafka",
        region="us-east-1",
        num_results=3,
        num_regions=4,
        desires=desires,
        extra_model_arguments={
            "cluster_type": ClusterType.ha,
            "retention": "PT8H",
            "require_attached_disks": True,
            "required_zone_size": cluster_capacity.cluster_instance_count.mid,
            "require_same_instance_family": False,
        },
    )

    assert len(cap_plan) >= 1
    lr_clusters = cap_plan[0].candidate_clusters.zonal
    assert len(lr_clusters) >= 1
    logger.debug(lr_clusters[0].instance.name)
    assert lr_clusters[0].count == cluster_capacity.cluster_instance_count.high
    for lr in cap_plan:
        logger.debug(lr.candidate_clusters.zonal[0])
    families = set(
        map(
            lambda curr_plan: curr_plan.candidate_clusters.zonal[0].instance.family,
            cap_plan,
        )
    )
    # check that we did not restrict the instance family to only r7a
    assert families != {"r7a"}

    # check that we have the same instance count as the disk pressure
    # should not be too high
    assert lr_clusters[0].count == cluster_capacity.cluster_instance_count.high


def test_plan_certain_data_shape_same_instance_type():
    """
    Use current clusters cpu utilization to determine instance types directly as
    supposed to extrapolating it from the Data Shape
    """
    cluster_capacity = CurrentZoneClusterCapacity(
        cluster_instance_name="r7a.4xlarge",
        cluster_drive=Drive(
            name="gp3",
            drive_type=DriveType.attached_ssd,
            size_gib=5000,
            block_size_kib=16,
        ),
        cluster_instance_count=Interval(low=15, mid=15, high=15, confidence=1),
        cpu_utilization=Interval(
            low=5.441147804260254,
            mid=13.548842955300195,
            high=25.11203956604004,
            confidence=1,
        ),
        memory_utilization_gib=Interval(low=0, mid=0, high=0, confidence=1),
        network_utilization_mbps=Interval(
            low=217.18696,
            mid=590.5934259505216,
            high=1220.205184,
            confidence=1,
        ),
        disk_utilization_gib=Interval(
            low=1000,
            mid=1500,
            high=2000,
            confidence=1,
        ),
    )

    desires = CapacityDesires(
        service_tier=1,
        current_clusters=CurrentClusters(zonal=[cluster_capacity]),
        query_pattern=QueryPattern(
            access_pattern=AccessPattern(AccessPattern.latency),
            # 2 consumers
            estimated_read_per_second=Interval(low=2, mid=2, high=4, confidence=1),
            # 1 producer
            estimated_write_per_second=Interval(low=1, mid=1, high=1, confidence=0.98),
            estimated_mean_read_latency_ms=Interval(low=1, mid=1, high=1, confidence=1),
            estimated_mean_write_latency_ms=Interval(
                low=1, mid=1, high=1, confidence=1
            ),
            estimated_mean_read_size_bytes=Interval(
                low=1024, mid=1024, high=1024, confidence=1
            ),
            estimated_mean_write_size_bytes=Interval(
                low=125000000, mid=579000000, high=1351000000, confidence=0.98
            ),
            estimated_read_parallelism=Interval(low=1, mid=1, high=1, confidence=1),
            estimated_write_parallelism=Interval(low=1, mid=1, high=1, confidence=1),
            read_latency_slo_ms=FixedInterval(low=0.4, mid=4, high=10, confidence=0.98),
            write_latency_slo_ms=FixedInterval(
                low=0.4, mid=4, high=10, confidence=0.98
            ),
        ),
        data_shape=DataShape(
            estimated_state_size_gib=Interval(
                low=44000, mid=86000, high=91000, confidence=1
            ),
        ),
    )

    cap_plan = planner.plan_certain(
        model_name="org.netflix.kafka",
        region="us-east-1",
        num_results=3,
        num_regions=4,
        desires=desires,
        extra_model_arguments={
            "cluster_type": ClusterType.ha,
            "retention": "PT8H",
            "require_attached_disks": True,
            "required_zone_size": cluster_capacity.cluster_instance_count.mid,
            "require_same_instance_family": True,
        },
    )

    assert len(cap_plan) >= 1
    lr_clusters = cap_plan[0].candidate_clusters.zonal
    assert len(lr_clusters) >= 1
    logger.debug(lr_clusters[0].instance.name)
    assert lr_clusters[0].count == cluster_capacity.cluster_instance_count.high

    families = set(
        map(
            lambda curr_plan: curr_plan.candidate_clusters.zonal[0].instance.family,
            cap_plan,
        )
    )
    # check that we restricted the instance family to only r7a
    assert families == {"r7a"}

    # check that we have the same instance count as the disk pressure
    # should not be too high
    assert lr_clusters[0].count == cluster_capacity.cluster_instance_count.high

    for lr in cap_plan:
        logger.debug(lr.candidate_clusters.zonal[0])


def test_scale_up_using_buffers():
    """
    Use current clusters cpu utilization to determine instance types directly as
    supposed to extrapolating it from the Data Shape
    """
    cluster_capacity = CurrentZoneClusterCapacity(
        cluster_instance_name="r7a.4xlarge",
        cluster_drive=Drive(
            name="gp3",
            drive_type=DriveType.attached_ssd,
            size_gib=5000,
            block_size_kib=16,
        ),
        cluster_instance_count=Interval(low=15, mid=15, high=15, confidence=1),
        cpu_utilization=Interval(
            low=5.441147804260254,
            mid=13.548842955300195,
            high=25.11203956604004,
            confidence=1,
        ),
        memory_utilization_gib=Interval(low=0, mid=0, high=0, confidence=1),
        network_utilization_mbps=Interval(
            low=217.18696,
            mid=590.5934259505216,
            high=1220.205184,
            confidence=1,
        ),
        # use lower numbers here for testing since cap planner only
        # allows 5TB max disk per node
        disk_utilization_gib=Interval(
            low=100,
            mid=500,
            high=1000,
            confidence=1,
        ),
    )

    scale_ratio = 1.70
    buffer_ratio = 2.5
    buffers = Buffers(
        default=Buffer(ratio=1.5),
        desired={
            # Amount of compute buffer that we need to reserve in addition to
            # cpu_headroom_target that is reserved on a per instance basis
            "compute": Buffer(ratio=buffer_ratio, components=[BufferComponent.compute]),
            # This makes sure we use only 40% of the available storage
            "storage": Buffer(ratio=buffer_ratio, components=[BufferComponent.storage]),
        },
        derived={
            "compute": Buffer(
                ratio=scale_ratio,
                intent=BufferIntent.scale,
                components=[BufferComponent.compute],
            ),
            "storage": Buffer(
                ratio=scale_ratio,
                intent=BufferIntent.scale,
                components=[BufferComponent.storage],
            ),
        },
    )

    desires = CapacityDesires(
        service_tier=1,
        buffers=buffers,
        current_clusters=CurrentClusters(zonal=[cluster_capacity]),
        query_pattern=QueryPattern(
            access_pattern=AccessPattern(AccessPattern.latency),
            # 2 consumers
            estimated_read_per_second=Interval(low=2, mid=2, high=4, confidence=1),
            # 1 producer
            estimated_write_per_second=Interval(low=1, mid=1, high=1, confidence=0.98),
            estimated_mean_read_latency_ms=Interval(low=1, mid=1, high=1, confidence=1),
            estimated_mean_write_latency_ms=Interval(
                low=1, mid=1, high=1, confidence=1
            ),
            estimated_mean_read_size_bytes=Interval(
                low=1024, mid=1024, high=1024, confidence=1
            ),
            estimated_mean_write_size_bytes=Interval(
                low=125000000, mid=579000000, high=1351000000, confidence=0.98
            ),
            estimated_read_parallelism=Interval(low=1, mid=1, high=1, confidence=1),
            estimated_write_parallelism=Interval(low=1, mid=1, high=1, confidence=1),
            read_latency_slo_ms=FixedInterval(low=0.4, mid=4, high=10, confidence=0.98),
            write_latency_slo_ms=FixedInterval(
                low=0.4, mid=4, high=10, confidence=0.98
            ),
        ),
        data_shape=DataShape(
            estimated_state_size_gib=Interval(
                low=44000, mid=86000, high=91000, confidence=1
            ),
        ),
    )

    cap_plan = planner.plan_certain(
        model_name="org.netflix.kafka",
        region="us-east-1",
        num_results=3,
        num_regions=4,
        desires=desires,
        extra_model_arguments={
            "cluster_type": ClusterType.ha,
            "retention": "PT8H",
            "require_attached_disks": True,
            "required_zone_size": cluster_capacity.cluster_instance_count.mid,
            "require_same_instance_family": True,
        },
    )

    assert len(cap_plan) >= 1
    lr_clusters = cap_plan[0].candidate_clusters.zonal
    assert len(lr_clusters) >= 1
    logger.debug(lr_clusters[0].instance.name)
    assert lr_clusters[0].count == cluster_capacity.cluster_instance_count.high

    families = set(
        map(
            lambda curr_plan: curr_plan.candidate_clusters.zonal[0].instance.family,
            cap_plan,
        )
    )
    # check that we restricted the instance family to only r7a
    assert families == {"r7a"}

    # check that we scaled to something higher than r7a.4xlarge
    # get the starting integer of the size of the instance type i.e. 8 for r7a.8xlarge
    assert (
        int(cap_plan[0].candidate_clusters.zonal[0].instance.name.split(".")[1][0]) > 4
    )

    # check that we have at least as many instances as the current cluster
    assert lr_clusters[0].count >= cluster_capacity.cluster_instance_count.high

    # Check that we provisioned enough storage
    minimum_provisioned_disk = (
        cluster_capacity.disk_utilization_gib.high
        * cluster_capacity.cluster_instance_count.mid
        * buffer_ratio
        * scale_ratio
    )
    assert (
        lr_clusters[0].attached_drives[0].size_gib * lr_clusters[0].count
        >= minimum_provisioned_disk
    )

    for lr in cap_plan:
        logger.debug(lr.candidate_clusters.zonal[0])


def test_scale_up_using_buffers_high_disk_change_instance_count():
    """
    Use current clusters cpu utilization to determine instance types directly as
    supposed to extrapolating it from the Data Shape
    """
    cluster_capacity = CurrentZoneClusterCapacity(
        cluster_instance_name="r7a.4xlarge",
        cluster_drive=Drive(
            name="gp3",
            drive_type=DriveType.attached_ssd,
            size_gib=5000,
            block_size_kib=16,
        ),
        cluster_instance_count=Interval(low=15, mid=15, high=15, confidence=1),
        cpu_utilization=Interval(
            low=5.441147804260254,
            mid=13.548842955300195,
            high=25.11203956604004,
            confidence=1,
        ),
        memory_utilization_gib=Interval(low=0, mid=0, high=0, confidence=1),
        network_utilization_mbps=Interval(
            low=217.18696,
            mid=590.5934259505216,
            high=1220.205184,
            confidence=1,
        ),
        disk_utilization_gib=Interval(
            low=1341.579345703125,
            mid=1940.8741284013684,
            high=2437.607421875,
            confidence=1,
        ),
    )

    scale_ratio = 1.70
    buffer_ratio = 2.5
    buffers = Buffers(
        default=Buffer(ratio=1.5),
        desired={
            # Amount of compute buffer that we need to reserve in addition to
            # cpu_headroom_target that is reserved on a per instance basis
            "compute": Buffer(ratio=buffer_ratio, components=[BufferComponent.compute]),
            # This makes sure we use only 40% of the available storage
            "storage": Buffer(ratio=buffer_ratio, components=[BufferComponent.storage]),
        },
        derived={
            "compute": Buffer(
                ratio=scale_ratio,
                intent=BufferIntent.scale,
                components=[BufferComponent.compute],
            ),
            "storage": Buffer(
                ratio=scale_ratio,
                intent=BufferIntent.scale,
                components=[BufferComponent.storage],
            ),
        },
    )

    desires = CapacityDesires(
        service_tier=1,
        buffers=buffers,
        current_clusters=CurrentClusters(zonal=[cluster_capacity]),
        query_pattern=QueryPattern(
            access_pattern=AccessPattern(AccessPattern.latency),
            # 2 consumers
            estimated_read_per_second=Interval(low=2, mid=2, high=4, confidence=1),
            # 1 producer
            estimated_write_per_second=Interval(low=1, mid=1, high=1, confidence=0.98),
            estimated_mean_read_latency_ms=Interval(low=1, mid=1, high=1, confidence=1),
            estimated_mean_write_latency_ms=Interval(
                low=1, mid=1, high=1, confidence=1
            ),
            estimated_mean_read_size_bytes=Interval(
                low=1024, mid=1024, high=1024, confidence=1
            ),
            estimated_mean_write_size_bytes=Interval(
                low=125000000, mid=579000000, high=1351000000, confidence=0.98
            ),
            estimated_read_parallelism=Interval(low=1, mid=1, high=1, confidence=1),
            estimated_write_parallelism=Interval(low=1, mid=1, high=1, confidence=1),
            read_latency_slo_ms=FixedInterval(low=0.4, mid=4, high=10, confidence=0.98),
            write_latency_slo_ms=FixedInterval(
                low=0.4, mid=4, high=10, confidence=0.98
            ),
        ),
        data_shape=DataShape(
            estimated_state_size_gib=Interval(
                low=44000, mid=86000, high=91000, confidence=1
            ),
        ),
    )

    cap_plan = planner.plan_certain(
        model_name="org.netflix.kafka",
        region="us-east-1",
        num_results=3,
        num_regions=4,
        desires=desires,
        extra_model_arguments={
            "cluster_type": ClusterType.ha,
            "retention": "PT8H",
            "require_attached_disks": True,
            "required_zone_size": cluster_capacity.cluster_instance_count.mid,
            "require_same_instance_family": True,
        },
    )

    assert len(cap_plan) >= 1
    lr_clusters = cap_plan[0].candidate_clusters.zonal
    assert len(lr_clusters) >= 1
    logger.debug(lr_clusters[0].instance.name)

    families = set(
        map(
            lambda curr_plan: curr_plan.candidate_clusters.zonal[0].instance.family,
            cap_plan,
        )
    )
    # check that we restricted the instance family to only r7a
    assert families == {"r7a"}

    # check that we have at least as many instances as the current cluster
    assert lr_clusters[0].count >= cluster_capacity.cluster_instance_count.high

    # Since the disk required per instance is > 5TB allowed by cap planner, we
    # allow higher instance count. This means we may not have vertically scaled
    # the instance type up since a lower instance type may be ok with the higher count

    # Check that we provisioned enough storage
    minimum_provisioned_disk = (
        cluster_capacity.disk_utilization_gib.high
        * cluster_capacity.cluster_instance_count.mid
        * buffer_ratio
        * scale_ratio
    )
    assert (
        lr_clusters[0].attached_drives[0].size_gib * lr_clusters[0].count
        >= minimum_provisioned_disk
    )

    for lr in cap_plan:
        logger.debug(lr.candidate_clusters.zonal[0])


def test_non_ebs():
    cluster_capacity = CurrentZoneClusterCapacity(
        cluster_instance_name="i3en.2xlarge",
        cluster_instance=None,
        cluster_drive=None,
        cluster_instance_count=Interval(low=44.0, mid=44.0, high=44.0, confidence=1.0),
        cpu_utilization=Interval(
            low=3.0162736316287893,
            mid=17.47713213503852,
            high=28.850521087646484,
            confidence=1.0,
        ),
        memory_utilization_gib=Interval(low=0.0, mid=0.0, high=0.0, confidence=1.0),
        network_utilization_mbps=Interval(
            low=18.110018933333333,
            mid=908.3751514114257,
            high=2515.735296,
            confidence=1.0,
        ),
        disk_utilization_gib=Interval(
            low=500.24108505249023,
            mid=750.8721423142003,
            high=1300.127197265625,
            confidence=1.0,
        ),
    )

    scale_ratio = 1.47
    buffer_ratio = 2.5
    buffers = Buffers(
        default=Buffer(ratio=1.5, intent=BufferIntent.desired, components=["compute"]),
        desired={
            "compute": Buffer(
                ratio=buffer_ratio, intent=BufferIntent.desired, components=["compute"]
            ),
            "storage": Buffer(
                ratio=buffer_ratio, intent=BufferIntent.desired, components=["storage"]
            ),
        },
        derived={
            "compute": Buffer(
                ratio=scale_ratio, intent=BufferIntent.scale, components=["compute"]
            ),
            "storage": Buffer(
                ratio=scale_ratio, intent=BufferIntent.scale, components=["storage"]
            ),
        },
    )

    desires = CapacityDesires(
        service_tier=1,
        buffers=buffers,
        current_clusters=CurrentClusters(zonal=[cluster_capacity]),
        query_pattern=QueryPattern(
            access_pattern=AccessPattern.latency,
            estimated_read_per_second=Interval(
                low=4.0, mid=4.0, high=4.0, confidence=1.0
            ),
            estimated_write_per_second=Interval(
                low=1.0, mid=1.0, high=1.0, confidence=0.98
            ),
            estimated_mean_read_latency_ms=Interval(
                low=1.0, mid=1.0, high=1.0, confidence=1.0
            ),
            estimated_mean_write_latency_ms=Interval(
                low=1.0, mid=1.0, high=1.0, confidence=1.0
            ),
            estimated_mean_read_size_bytes=Interval(
                low=1024.0, mid=1024.0, high=1024.0, confidence=1.0
            ),
            estimated_mean_write_size_bytes=Interval(
                low=45934839.712777786,
                mid=2832737185.7793183,
                high=5291668222.459846,
                confidence=0.98,
            ),
            estimated_read_parallelism=Interval(
                low=1.0, mid=1.0, high=1.0, confidence=1.0
            ),
            estimated_write_parallelism=Interval(
                low=1.0, mid=1.0, high=1.0, confidence=1.0
            ),
            read_latency_slo_ms=FixedInterval(
                low=0.4, mid=4.0, high=10.0, confidence=0.98
            ),
            write_latency_slo_ms=FixedInterval(
                low=0.4, mid=4.0, high=10.0, confidence=0.98
            ),
        ),
        data_shape=DataShape(
            estimated_state_size_gib=Interval(
                low=35428.05778477986,
                mid=209861.32096354378,
                high=294807.85949808755,
                confidence=1.0,
            ),
            reserved_instance_app_mem_gib=2,
            reserved_instance_system_mem_gib=1,
            estimated_compression_ratio=Interval(
                low=1.0, mid=1.0, high=1.0, confidence=1.0
            ),
        ),
    )

    cap_plan = planner.plan_certain(
        model_name="org.netflix.kafka",
        region="us-east-1",
        num_results=3,
        num_regions=1,
        desires=desires,
        extra_model_arguments={
            "cluster_type": ClusterType.ha,
            "retention": "PT8H",
            "context": "fleet analysis from AG.kafka",
            "context-memo": "fleet analysis from AG.kafka",
            "nflx-sensitivedata": False,
            "required_zone_size": 44.0,
            "require_local_disks": True,
        },
    )

    assert len(cap_plan) >= 1
    lr_clusters = cap_plan[0].candidate_clusters.zonal
    assert len(lr_clusters) >= 1
    logger.debug(lr_clusters[0].instance.name)

    families = set(
        map(
            lambda curr_plan: curr_plan.candidate_clusters.zonal[0].instance.family,
            cap_plan,
        )
    )
    # check that we restricted the instance family to only i3en
    assert families == {"i3en"}

    # check that we have at least as many instances as the current cluster
    assert lr_clusters[0].count >= cluster_capacity.cluster_instance_count.high

    # Check that we provisioned enough storage
    minimum_provisioned_disk = (
        cluster_capacity.disk_utilization_gib.high
        * cluster_capacity.cluster_instance_count.mid
        * buffer_ratio
        * scale_ratio
    )
    assert (
        lr_clusters[0].instance.drive is not None
        and lr_clusters[0].instance.drive.size_gib * lr_clusters[0].count
        >= minimum_provisioned_disk
    )

    for lr in cap_plan:
        logger.debug(lr.candidate_clusters.zonal[0])


def test_non_ebs_force_horizontal():
    cluster_capacity = CurrentZoneClusterCapacity(
        cluster_instance_name="i3en.2xlarge",
        cluster_instance=None,
        cluster_drive=None,
        cluster_instance_count=Interval(low=34.0, mid=34.0, high=34.0, confidence=1.0),
        cpu_utilization=Interval(
            low=7.976402722859067,
            mid=23.6481902437047,
            high=32.95874639028172,
            confidence=1.0,
        ),
        memory_utilization_gib=Interval(low=0.0, mid=0.0, high=0.0, confidence=1.0),
        network_utilization_mbps=Interval(
            low=261.2259746737884,
            mid=1400.0759607382988,
            high=2102.2521095975226,
            confidence=1.0,
        ),
        disk_utilization_gib=Interval(
            low=568.1681300331992,
            mid=1265.9443437846992,
            high=1631.3559736741358,
            confidence=1.0,
        ),
    )

    scale_ratio = 1.47
    buffer_ratio = 2.5
    buffers = Buffers(
        default=Buffer(ratio=1.5, intent=BufferIntent.desired, components=["compute"]),
        desired={
            "compute": Buffer(
                ratio=buffer_ratio, intent=BufferIntent.desired, components=["compute"]
            ),
            "storage": Buffer(
                ratio=buffer_ratio, intent=BufferIntent.desired, components=["storage"]
            ),
        },
        derived={
            "compute": Buffer(
                ratio=scale_ratio, intent=BufferIntent.scale, components=["compute"]
            ),
            "storage": Buffer(
                ratio=scale_ratio, intent=BufferIntent.scale, components=["storage"]
            ),
        },
    )

    desires = CapacityDesires(
        service_tier=1,
        buffers=buffers,
        current_clusters=CurrentClusters(zonal=[cluster_capacity]),
        query_pattern=QueryPattern(
            access_pattern=AccessPattern.latency,
            estimated_read_per_second=Interval(
                low=4.0, mid=4.0, high=4.0, confidence=1.0
            ),
            estimated_write_per_second=Interval(
                low=1.0, mid=1.0, high=1.0, confidence=0.98
            ),
            estimated_mean_read_latency_ms=Interval(
                low=1.0, mid=1.0, high=1.0, confidence=1.0
            ),
            estimated_mean_write_latency_ms=Interval(
                low=1.0, mid=1.0, high=1.0, confidence=1.0
            ),
            estimated_mean_read_size_bytes=Interval(
                low=1024.0, mid=1024.0, high=1024.0, confidence=1.0
            ),
            estimated_mean_write_size_bytes=Interval(
                low=176860051.61423847,
                mid=2124963552.9652083,
                high=3525502574.7887697,
                confidence=0.98,
            ),
            estimated_read_parallelism=Interval(
                low=1.0, mid=1.0, high=1.0, confidence=1.0
            ),
            estimated_write_parallelism=Interval(
                low=1.0, mid=1.0, high=1.0, confidence=1.0
            ),
            read_latency_slo_ms=FixedInterval(
                low=0.4, mid=4.0, high=10.0, confidence=0.98
            ),
            write_latency_slo_ms=FixedInterval(
                low=0.4, mid=4.0, high=10.0, confidence=0.98
            ),
        ),
        data_shape=DataShape(
            estimated_state_size_gib=Interval(
                low=30065.005002339683,
                mid=147923.7993778993,
                high=166897.69089864095,
                confidence=1.0,
            ),
            reserved_instance_app_mem_gib=2,
            reserved_instance_system_mem_gib=1,
            estimated_compression_ratio=Interval(
                low=1.0, mid=1.0, high=1.0, confidence=1.0
            ),
        ),
    )

    cap_plan = planner.plan_certain(
        model_name="org.netflix.kafka",
        region="us-east-1",
        num_results=3,
        num_regions=1,
        desires=desires,
        extra_model_arguments={
            "cluster_type": ClusterType.ha,
            "retention": "PT8H",
            "context": "fleet analysis from AG.kafka",
            "context-memo": "fleet analysis from AG.kafka",
            "nflx-sensitivedata": False,
            "required_zone_size": cluster_capacity.cluster_instance_count.mid,
            "require_local_disks": True,
        },
    )

    assert len(cap_plan) >= 1
    lr_clusters = cap_plan[0].candidate_clusters.zonal
    assert len(lr_clusters) >= 1
    print(lr_clusters[0].instance.name)

    families = set(
        map(
            lambda curr_plan: curr_plan.candidate_clusters.zonal[0].instance.family,
            cap_plan,
        )
    )
    # check that we restricted the instance family to only i3en
    assert families == {"i3en"}

    # check that we have at least as many instances as the current cluster
    assert lr_clusters[0].count >= cluster_capacity.cluster_instance_count.high

    # Check that we provisioned enough storage
    minimum_provisioned_disk = (
        cluster_capacity.disk_utilization_gib.high
        * cluster_capacity.cluster_instance_count.mid
        * buffer_ratio
        * scale_ratio
    )
    assert (
        lr_clusters[0].instance.drive is not None
        and lr_clusters[0].instance.drive.size_gib * lr_clusters[0].count
        >= minimum_provisioned_disk
    )

    for lr in cap_plan:
        print(lr.candidate_clusters.zonal[0])


class PlanningInput(ExcludeUnsetModel):
    regions: list[str]
    desires: CapacityDesires
    extra_model_arguments: dict[str, Any]
    # Unused field for devs to understand how the request was constructed
    context: dict[str, Any] = None

    def plan(self):
        return planner.plan_certain(
            model_name="org.netflix.kafka",
            region=self.regions[0],
            num_results=3,
            num_regions=4,
            desires=self.desires,
            extra_model_arguments=self.extra_model_arguments,
        )


def test_my_node_density():
    test_blob = """
    {
      "regions": [
        "us-east-1"
      ],
      "desires": {
        "service_tier": 1,
        "query_pattern": {
          "access_pattern": "latency",
          "access_consistency": {
            "same_region": {
              "target_consistency": null,
              "staleness_slo_sec": {
                "low": 0,
                "mid": 0.1,
                "high": 1,
                "confidence": 1,
                "model_with": "beta",
                "allow_simulate": false,
                "minimum_value": null,
                "maximum_value": null
              }
            },
            "cross_region": {
              "target_consistency": null,
              "staleness_slo_sec": {
                "low": 10,
                "mid": 60,
                "high": 600,
                "confidence": 1,
                "model_with": "beta",
                "allow_simulate": false,
                "minimum_value": null,
                "maximum_value": null
              }
            }
          },
          "estimated_read_per_second": {
            "low": 3.424604762980999,
            "mid": 3.9231941764982463,
            "high": 4,
            "confidence": 1,
            "model_with": "beta",
            "allow_simulate": true,
            "minimum_value": null,
            "maximum_value": null
          },
          "estimated_write_per_second": {
            "low": 1,
            "mid": 1,
            "high": 1,
            "confidence": 0.98,
            "model_with": "beta",
            "allow_simulate": true,
            "minimum_value": null,
            "maximum_value": null
          },
          "estimated_mean_read_latency_ms": {
            "low": 1,
            "mid": 1,
            "high": 1,
            "confidence": 1,
            "model_with": "beta",
            "allow_simulate": true,
            "minimum_value": null,
            "maximum_value": null
          },
          "estimated_mean_write_latency_ms": {
            "low": 1,
            "mid": 1,
            "high": 1,
            "confidence": 1,
            "model_with": "beta",
            "allow_simulate": true,
            "minimum_value": null,
            "maximum_value": null
          },
          "estimated_mean_read_size_bytes": {
            "low": 1024,
            "mid": 1024,
            "high": 1024,
            "confidence": 1,
            "model_with": "beta",
            "allow_simulate": true,
            "minimum_value": null,
            "maximum_value": null
          },
          "estimated_mean_write_size_bytes": {
            "low": 18500944.360506184,
            "mid": 994809143.9717082,
            "high": 1716676785.3169494,
            "confidence": 0.98,
            "model_with": "beta",
            "allow_simulate": true,
            "minimum_value": null,
            "maximum_value": null
          },
          "estimated_read_parallelism": {
            "low": 1,
            "mid": 1,
            "high": 1,
            "confidence": 1,
            "model_with": "beta",
            "allow_simulate": true,
            "minimum_value": null,
            "maximum_value": null
          },
          "estimated_write_parallelism": {
            "low": 1,
            "mid": 1,
            "high": 1,
            "confidence": 1,
            "model_with": "beta",
            "allow_simulate": true,
            "minimum_value": null,
            "maximum_value": null
          },
          "read_latency_slo_ms": {
            "low": 0.4,
            "mid": 4,
            "high": 10,
            "confidence": 0.98,
            "model_with": "beta",
            "allow_simulate": false,
            "minimum_value": null,
            "maximum_value": null
          },
          "write_latency_slo_ms": {
            "low": 0.4,
            "mid": 4,
            "high": 10,
            "confidence": 0.98,
            "model_with": "beta",
            "allow_simulate": false,
            "minimum_value": null,
            "maximum_value": null
          }
        },
        "data_shape": {
          "estimated_state_size_gib": {
            "low": 19648.113148752847,
            "mid": 48897.53595507853,
            "high": 59691.70344924928,
            "confidence": 1,
            "model_with": "beta",
            "allow_simulate": false,
            "minimum_value": null,
            "maximum_value": null
          },
          "estimated_state_item_count": null,
          "estimated_working_set_percent": null,
          "estimated_compression_ratio": {
            "low": 1,
            "mid": 1,
            "high": 1,
            "confidence": 1,
            "model_with": "beta",
            "allow_simulate": true,
            "minimum_value": null,
            "maximum_value": null
          },
          "reserved_instance_app_mem_gib": 2,
          "reserved_instance_system_mem_gib": 1,
          "durability_slo_order": {
            "low": 1000,
            "mid": 10000,
            "high": 100000,
            "confidence": 0.98,
            "model_with": "beta",
            "allow_simulate": false,
            "minimum_value": null,
            "maximum_value": null
          }
        },
        "current_clusters": {
          "zonal": [
            {
              "cluster_instance_name": "r7a.2xlarge",
              "cluster_instance": null,
              "cluster_drive": {
                "name": "gp3",
                "drive_type": "attached-ssd",
                "size_gib": 4350,
                "read_io_per_s": null,
                "write_io_per_s": null,
                "throughput": null,
                "single_tenant": true,
                "max_scale_size_gib": 0,
                "max_scale_io_per_s": 0,
                "block_size_kib": 4,
                "group_size_kib": 4,
                "lifecycle": "stable",
                "compatible_families": [],
                "annual_cost_per_gib": 0,
                "annual_cost_per_read_io": [],
                "annual_cost_per_write_io": [],
                "read_io_latency_ms": {
                  "low": 0.8,
                  "mid": 1,
                  "high": 2,
                  "confidence": 0.9,
                  "model_with": "beta",
                  "allow_simulate": false,
                  "minimum_value": null,
                  "maximum_value": null
                },
                "write_io_latency_ms": {
                  "low": 0.6,
                  "mid": 2,
                  "high": 3,
                  "confidence": 0.9,
                  "model_with": "beta",
                  "allow_simulate": false,
                  "minimum_value": null,
                  "maximum_value": null
                },
                "annual_cost": 0
              },
              "cluster_instance_count": {
                "low": 23,
                "mid": 23,
                "high": 23,
                "confidence": 1,
                "model_with": "beta",
                "allow_simulate": true,
                "minimum_value": null,
                "maximum_value": null
              },
              "cpu_utilization": {
                "low": 5.075449556721986,
                "mid": 9.885813697428473,
                "high": 14.976325733265925,
                "confidence": 1,
                "model_with": "beta",
                "allow_simulate": true,
                "minimum_value": null,
                "maximum_value": null
              },
              "memory_utilization_gib": {
                "low": 0,
                "mid": 0,
                "high": 0,
                "confidence": 1,
                "model_with": "beta",
                "allow_simulate": true,
                "minimum_value": null,
                "maximum_value": null
              },
              "network_utilization_mbps": {
                "low": 302.65452242254446,
                "mid": 687.4712410197809,
                "high": 1119.570789256165,
                "confidence": 1,
                "model_with": "beta",
                "allow_simulate": true,
                "minimum_value": null,
                "maximum_value": null
              },
              "disk_utilization_gib": {
                "low": 389.99999661951597,
                "mid": 626.6531370364696,
                "high": 880.1773931084945,
                "confidence": 1,
                "model_with": "beta",
                "allow_simulate": true,
                "minimum_value": null,
                "maximum_value": null
              }
            }
          ],
          "regional": [],
          "services": []
        },
        "buffers": {
          "default": {
            "ratio": 1.5,
            "intent": "desired",
            "components": [
              "compute"
            ],
            "sources": {}
          },
          "desired": {
            "compute": {
              "ratio": 2.5,
              "intent": "desired",
              "components": [
                "compute"
              ],
              "sources": {}
            },
            "storage": {
              "ratio": 2,
              "intent": "desired",
              "components": [
                "storage"
              ],
              "sources": {}
            }
          },
          "derived": {}
        }
      },
      "extra_model_arguments": {
        "cluster_type": "high-availability",
        "retention": "PT8H",
        "context": "fleet analysis from AG.kafka",
        "context-memo": "fleet analysis from AG.kafka",
        "nflx-sensitivedata": false,
        "required_zone_size": 23,
        "require_attached_disks": true
      },
      "allowed_lifecycles": [
        "beta",
        "stable"
      ]
    }
    """
    type_adapter = TypeAdapter(PlanningInput)
    test_input = type_adapter.validate_json(test_blob)
    scale = 1.35
    test_input.desires.buffers.derived = {
        "scale_compute": Buffer(
            ratio=scale, intent=BufferIntent.scale, components=[BufferComponent.compute]
        ),
        "scale_storage": Buffer(
            ratio=scale, intent=BufferIntent.scale, components=[BufferComponent.storage]
        ),
    }

    assert test_input is not None
    plan = test_input.plan()
    assert plan
    result = plan[0].candidate_clusters.zonal[0]
    assert result.count == 23
    assert result.instance.name == "r7a.4xlarge"
