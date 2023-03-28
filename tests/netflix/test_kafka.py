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
