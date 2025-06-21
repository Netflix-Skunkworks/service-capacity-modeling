# pylint: disable=too-many-lines
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
from service_capacity_modeling.interface import FixedInterval
from service_capacity_modeling.interface import Interval
from service_capacity_modeling.interface import QueryPattern
from service_capacity_modeling.models.common import normalize_cores
from service_capacity_modeling.models.org.netflix.kafka import ClusterType


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
        print(lr.candidate_clusters.zonal[0])
        assert 50_000 < lr.candidate_clusters.total_annual_cost < 200_000
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
            "max_local_disk_gib": 500,
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
    print(lr_clusters[0].instance.name)
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
    print(lr_clusters[0].instance.name)
    assert lr_clusters[0].count == cluster_capacity.cluster_instance_count.high
    for lr in cap_plan:
        print(lr.candidate_clusters.zonal[0])
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
    print(lr_clusters[0].instance.name)
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
        print(lr.candidate_clusters.zonal[0])


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
    print(lr_clusters[0].instance.name)
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
        print(lr.candidate_clusters.zonal[0])


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
    print(lr_clusters[0].instance.name)

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
        print(lr.candidate_clusters.zonal[0])


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
            low=311.24108505249023,
            mid=1413.8721423142003,
            high=2350.127197265625,
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
        lr_clusters[0].instance.drive.size_gib * lr_clusters[0].count
        >= minimum_provisioned_disk
    )

    for lr in cap_plan:
        print(lr.candidate_clusters.zonal[0])
