from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import DataShape
from service_capacity_modeling.interface import Interval
from service_capacity_modeling.interface import QueryPattern
from service_capacity_modeling.models.org.netflix.evcache import (
    calculate_read_cpu_time_evcache_ms,
)


def test_evcache_read_latency():
    # 256 bits = 32 bytes 10
    small = calculate_read_cpu_time_evcache_ms(32)
    # 1600 bits = 200 bytes 41
    medium = calculate_read_cpu_time_evcache_ms(200)
    # 8192 bits = 1024 bytes 66
    large = calculate_read_cpu_time_evcache_ms(1024)
    # 24   KiB  = 133
    very_large = calculate_read_cpu_time_evcache_ms(24 * 1024)
    # 40   KiB  = 158
    extra_large = calculate_read_cpu_time_evcache_ms(40 * 1024)

    assert calculate_read_cpu_time_evcache_ms(1) > 0
    assert 0.008 < small < 0.015
    assert 0.030 < medium < 0.050
    assert 0.060 < large < 0.080
    assert 0.120 < very_large < 0.140
    assert 0.140 < extra_large < 0.160


def test_evcache_inmemory_low_latency_reads_cpu():
    inmemory_cluster_low_latency_reads_qps = CapacityDesires(
        service_tier=1,
        query_pattern=QueryPattern(
            estimated_read_per_second=Interval(
                low=18300000, mid=34200000, high=34200000 * 1.2, confidence=1.0
            ),
            estimated_write_per_second=Interval(
                low=228000, mid=536000, high=536000 * 1.2, confidence=1.0
            ),
            estimated_mean_write_size_bytes=Interval(
                low=3778, mid=3778, high=3778 * 1.2, confidence=1.0
            ),
            estimated_mean_read_size_bytes=Interval(
                low=35, mid=35, high=35 * 1.2, confidence=1.0
            ),
        ),
        data_shape=DataShape(
            estimated_state_size_gib=Interval(low=36, mid=36, high=36, confidence=1.0),
            estimated_state_item_count=Interval(
                low=416000000, mid=804000000, high=804000000 * 1.2, confidence=1.0
            ),
        ),
    )

    plan = planner.plan_certain(
        model_name="org.netflix.evcache",
        region="us-east-1",
        desires=inmemory_cluster_low_latency_reads_qps,
    )

    for candidate in plan:
        total_cpu_power = (
            candidate.candidate_clusters.zonal[0].count
            * candidate.candidate_clusters.zonal[0].instance.cpu
            * candidate.candidate_clusters.zonal[0].instance.cpu_ghz
        )

        assert total_cpu_power > 1100


def test_evcache_inmemory_medium_latency_reads_cpu():
    inmemory_cluster_medium_latency_reads_qps = CapacityDesires(
        service_tier=0,
        query_pattern=QueryPattern(
            estimated_read_per_second=Interval(
                low=470000, mid=1800000, high=1800000 * 1.2, confidence=1.0
            ),
            estimated_mean_write_per_second=Interval(
                low=505000, mid=861000, high=861000 * 1.2, confidence=1.0
            ),
            estimated_mean_write_size_bytes=Interval(
                low=365, mid=365, high=365 * 1.2, confidence=1.0
            ),
            estimated_mean_read_size_bytes=Interval(
                low=193, mid=193, high=193 * 1.2, confidence=1.0
            ),
        ),
        data_shape=DataShape(
            estimated_state_size_gib=Interval(low=61, mid=61, high=61, confidence=1.0),
            estimated_state_item_count=Interval(
                low=125000000, mid=202000000, high=202000000 * 1.2, confidence=1.0
            ),
        ),
    )

    plan = planner.plan_certain(
        model_name="org.netflix.evcache",
        region="us-east-1",
        desires=inmemory_cluster_medium_latency_reads_qps,
    )

    for candidate in plan:
        total_cpu_power = (
            candidate.candidate_clusters.zonal[0].count
            * candidate.candidate_clusters.zonal[0].instance.cpu
            * candidate.candidate_clusters.zonal[0].instance.cpu_ghz
        )

        assert total_cpu_power > 400


def test_evcache_inmemory_high_latency_reads_cpu():
    inmemory_cluster_high_latency_reads_qps = CapacityDesires(
        service_tier=0,
        query_pattern=QueryPattern(
            estimated_read_per_second=Interval(
                low=113000, mid=441000, high=441000 * 1.2, confidence=1.0
            ),
            estimated_write_per_second=Interval(
                low=19000, mid=35000, high=35000 * 1.2, confidence=1.0
            ),
            estimated_mean_write_size_bytes=Interval(
                low=7250, mid=7250, high=7250 * 1.2, confidence=1.0
            ),
            estimated_mean_read_size_bytes=Interval(
                low=5100, mid=5100, high=5100 * 1.2, confidence=1.0
            ),
        ),
        data_shape=DataShape(
            estimated_state_size_gib=Interval(
                low=1662, mid=1662, high=1662, confidence=1.0
            ),
            estimated_state_item_count=Interval(
                low=750000000, mid=750000000, high=750000000 * 1.2, confidence=1.0
            ),
        ),
    )

    plan = planner.plan_certain(
        model_name="org.netflix.evcache",
        region="us-east-1",
        desires=inmemory_cluster_high_latency_reads_qps,
    )

    for candidate in plan:
        total_cpu_power = (
            candidate.candidate_clusters.zonal[0].count
            * candidate.candidate_clusters.zonal[0].instance.cpu
            * candidate.candidate_clusters.zonal[0].instance.cpu_ghz
        )

        assert total_cpu_power > 100


def test_evcache_ondisk_low_latency_reads_cpu():
    ondisk_cluster_low_latency_reads_qps = CapacityDesires(
        service_tier=0,
        query_pattern=QueryPattern(
            estimated_read_per_second=Interval(
                low=284, mid=7110000, high=7110000 * 1.2, confidence=1.0
            ),
            estimated_write_per_second=Interval(
                low=0, mid=2620000, high=2620000 * 1.2, confidence=1.0
            ),
            estimated_mean_write_size_bytes=Interval(
                low=12000, mid=12000, high=12000 * 1.2, confidence=1.0
            ),
            estimated_mean_read_size_bytes=Interval(
                low=16000, mid=16000, high=16000 * 1.2, confidence=1.0
            ),
        ),
        data_shape=DataShape(
            estimated_state_size_gib=Interval(
                low=2306867, mid=2306867, high=2306867, confidence=1.0
            ),
            estimated_state_item_count=Interval(
                low=132000000000,
                mid=132000000000,
                high=132000000000 * 1.2,
                confidence=1.0,
            ),
        ),
    )

    plan = planner.plan_certain(
        model_name="org.netflix.evcache",
        region="us-east-1",
        desires=ondisk_cluster_low_latency_reads_qps,
    )

    for candidate in plan:
        total_cpu_power = (
            candidate.candidate_clusters.zonal[0].count
            * candidate.candidate_clusters.zonal[0].instance.cpu
            * candidate.candidate_clusters.zonal[0].instance.cpu_ghz
        )

        assert total_cpu_power > 8000


def test_evcache_ondisk_high_latency_reads_cpu():
    ondisk_cluster_high_latency_reads_qps = CapacityDesires(
        service_tier=0,
        query_pattern=QueryPattern(
            estimated_read_per_second=Interval(
                low=312000, mid=853000, high=853000 * 1.2, confidence=1.0
            ),
            estimated_write_per_second=Interval(
                low=0, mid=310000, high=310000 * 1.2, confidence=1.0
            ),
            estimated_write_size_bytes=Interval(
                low=34500, mid=34500, high=34500 * 1.2, confidence=1.0
            ),
            estimated_mean_read_size_bytes=Interval(
                low=41000, mid=41000, high=41000 * 1.2, confidence=1.0
            ),
        ),
        data_shape=DataShape(
            estimated_state_size_gib=Interval(
                low=281000, mid=281000, high=281000, confidence=1.0
            ),
            estimated_state_item_count=Interval(
                low=8518318523, mid=8518318523, high=8518318523 * 1.2, confidence=1.0
            ),
        ),
    )

    plan = planner.plan_certain(
        model_name="org.netflix.evcache",
        region="us-east-1",
        desires=ondisk_cluster_high_latency_reads_qps,
    )

    for candidate in plan:
        total_cpu_power = (
            candidate.candidate_clusters.zonal[0].count
            * candidate.candidate_clusters.zonal[0].instance.cpu
            * candidate.candidate_clusters.zonal[0].instance.cpu_ghz
            * candidate.candidate_clusters.zonal[0].instance.cpu_ipc_scale
        )

        assert total_cpu_power > 800


def test_evcache_inmemory_ram_usage():
    inmemory_qps = CapacityDesires(
        service_tier=1,
        query_pattern=QueryPattern(
            estimated_read_per_second=Interval(
                low=18300000, mid=34200000, high=34200000 * 1.2, confidence=1.0
            ),
            estimated_write_per_second=Interval(
                low=228000, mid=536000, high=536000 * 1.2, confidence=1.0
            ),
            estimated_mean_write_size_bytes=Interval(
                low=3778, mid=3778, high=3778 * 1.2, confidence=1.0
            ),
            estimated_mean_read_size_bytes=Interval(
                low=35, mid=35, high=35 * 1.2, confidence=1.0
            ),
        ),
        data_shape=DataShape(
            estimated_state_size_gib=Interval(low=36, mid=36, high=36, confidence=1.0),
            estimated_state_item_count=Interval(
                low=416000000, mid=804000000, high=804000000 * 1.2, confidence=1.0
            ),
        ),
    )

    plan = planner.plan_certain(
        model_name="org.netflix.evcache",
        region="us-east-1",
        desires=inmemory_qps,
    )

    for candidate in plan:
        total_ram = (
            candidate.candidate_clusters.zonal[0].instance.ram_gib
            * candidate.candidate_clusters.zonal[0].count
        )

        assert total_ram > inmemory_qps.data_shape.estimated_state_size_gib.mid


def test_evcache_ondisk_disk_usage():
    inmemory_qps = CapacityDesires(
        service_tier=1,
        query_pattern=QueryPattern(
            estimated_read_per_second=Interval(
                low=18300000, mid=34200000, high=34200000 * 1.2, confidence=1.0
            ),
            estimated_write_per_second=Interval(
                low=228000, mid=536000, high=536000 * 1.2, confidence=1.0
            ),
            estimated_mean_write_size_bytes=Interval(
                low=3778, mid=3778, high=3778 * 1.2, confidence=1.0
            ),
            estimated_mean_read_size_bytes=Interval(
                low=35, mid=35, high=35 * 1.2, confidence=1.0
            ),
        ),
        data_shape=DataShape(
            estimated_state_size_gib=Interval(low=36, mid=36, high=36, confidence=1.0),
            estimated_state_item_count=Interval(
                low=416000000, mid=804000000, high=804000000 * 1.2, confidence=1.0
            ),
        ),
    )

    plan = planner.plan_certain(
        model_name="org.netflix.evcache",
        region="us-east-1",
        desires=inmemory_qps,
    )

    for candidate in plan:
        total_ram = (
            candidate.candidate_clusters.zonal[0].instance.ram_gib
            * candidate.candidate_clusters.zonal[0].count
        )

        assert total_ram > inmemory_qps.data_shape.estimated_state_size_gib.mid


def test_evcache_ondisk_high_disk_usage():
    high_disk_usage_rps = CapacityDesires(
        service_tier=0,
        query_pattern=QueryPattern(
            estimated_read_per_second=Interval(
                low=284, mid=7110000, high=7110000 * 1.2, confidence=1.0
            ),
            estimated_write_per_second=Interval(
                low=0, mid=2620000, high=2620000 * 1.2, confidence=1.0
            ),
            estimated_mean_write_size_bytes=Interval(
                low=12000, mid=12000, high=12000 * 1.2, confidence=1.0
            ),
            estimated_mean_read_size_bytes=Interval(
                low=16000, mid=16000, high=16000 * 1.2, confidence=1.0
            ),
        ),
        data_shape=DataShape(
            estimated_state_size_gib=Interval(
                low=2306867, mid=2306867, high=2306867, confidence=1.0
            ),
            estimated_state_item_count=Interval(
                low=132000000000,
                mid=132000000000,
                high=132000000000 * 1.2,
                confidence=1.0,
            ),
        ),
    )

    plan = planner.plan_certain(
        model_name="org.netflix.evcache",
        region="us-east-1",
        desires=high_disk_usage_rps,
    )

    for candidate in plan:
        if candidate.candidate_clusters.zonal[0].instance.drive is not None:
            total_disk = (
                candidate.candidate_clusters.zonal[0].instance.drive.size_gib
                * candidate.candidate_clusters.zonal[0].count
            )

            assert (
                total_disk > high_disk_usage_rps.data_shape.estimated_state_size_gib.mid
            )


def test_evcache_zero_item_count():
    zero_item_count_rps = CapacityDesires(
        service_tier=0,
        query_pattern=QueryPattern(
            estimated_read_per_second=Interval(
                low=1, mid=1, high=1 * 1.2, confidence=1.0
            ),
            estimated_write_per_second=Interval(
                low=1, mid=1, high=1 * 1.2, confidence=1.0
            ),
            estimated_mean_write_size_bytes=Interval(
                low=1, mid=1, high=1 * 1, confidence=1.0
            ),
            estimated_mean_read_size_bytes=Interval(
                low=1, mid=1, high=1 * 1, confidence=1.0
            ),
        ),
        data_shape=DataShape(
            estimated_state_size_gib=Interval(low=0, mid=0, high=0, confidence=1.0),
            estimated_state_item_count=Interval(low=0, mid=0, high=0, confidence=1.0),
        ),
    )

    plan = planner.plan_certain(
        model_name="org.netflix.evcache",
        region="us-east-1",
        desires=zero_item_count_rps,
    )

    for candidate in plan:
        if candidate.candidate_clusters.zonal[0].instance.drive is not None:
            total_ram = (
                candidate.candidate_clusters.zonal[0].instance.drive.size_gib
                * candidate.candidate_clusters.zonal[0].count
            )

            assert (
                total_ram > zero_item_count_rps.data_shape.estimated_state_size_gib.mid
            )
