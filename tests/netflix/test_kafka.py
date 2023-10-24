from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import Interval
from service_capacity_modeling.interface import QueryPattern


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
            "cluster_type": "strong",
            "retention": "PT4H",
        },
    )

    lr = plan.least_regret[0]
    lr_cluster = lr.candidate_clusters.zonal

    assert lr.candidate_clusters.total_annual_cost < 30_000
    assert lr_cluster[0].instance.family in ("i3en", "i4i", "r5")


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
            "cluster_type": "strong",
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
            "cluster_type": "ha",
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
    expected_disk = (22000, 25400)

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
        assert lr.candidate_clusters.total_annual_cost < 200_000
        clstr = lr.candidate_clusters.zonal[0]
        if clstr.instance.drive is None:
            assert clstr.instance.family in ("r5", "m6i")
            assert clstr.attached_drives[0].name == "gp3"
            disk = clstr.attached_drives[0].size_gib * clstr.count
            assert expected_disk[0] < disk < expected_disk[1] * 2.5
        else:
            assert clstr.instance.family in ("i3en", "i4i")
            disk = clstr.instance.drive.size_gib * clstr.count
            assert expected_disk[0] < disk


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
            "cluster_type": "ha",
            "retention": "PT3H",
            # Force to attached drives
            "max_local_disk_gib": 500,
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
    expected_disk = (22000, 25400)
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
    assert lr.candidate_clusters.total_annual_cost < 250_000

    for lr in plan.least_regret:
        assert lr.candidate_clusters.total_annual_cost < 250_000
        clstr = lr.candidate_clusters.zonal[0]
        if clstr.instance.drive is None:
            assert clstr.instance.family in ("r5", "r5n", "m5", "m6i")
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
            "cluster_type": "ha",
            "retention": "PT3H",
            "require_attached_disks": require_attached_disks,
            "min_instance_cpu": min_instance_cpu,
            "required_zone_size": required_zone_size
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
            "cluster_type": "ha",
            "retention": "PT3H",
            "require_local_disks": True,
            "min_instance_cpu": min_instance_cpu,
            "required_zone_size": required_zone_size
        },
        num_results=3,
    )
    expected_min_zonal_cpu = required_zone_size * min_instance_cpu

    for lr in plan.least_regret:
        for z in range(3):
            clstr = lr.candidate_clusters.zonal[z]
            assert clstr.instance.drive is not None
            assert (clstr.instance.cpu * clstr.count) >= expected_min_zonal_cpu
