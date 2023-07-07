from service_capacity_modeling.capacity_planner import planner
from service_capacity_modeling.hardware import shapes
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import DataShape
from service_capacity_modeling.interface import FixedInterval
from service_capacity_modeling.interface import Interval
from service_capacity_modeling.interface import QueryPattern
from service_capacity_modeling.models.common import working_set_from_drive_and_slo
from service_capacity_modeling.models.org.netflix import nflx_cockroachdb_capacity_model
from service_capacity_modeling.stats import dist_for_interval

simple_desire = CapacityDesires(
    service_tier=1,
    query_pattern=QueryPattern(
        estimated_read_per_second=Interval(
            low=100, mid=1000, high=10000, confidence=0.98
        ),
        estimated_write_per_second=Interval(
            low=100, mid=1000, high=10000, confidence=0.98
        ),
    ),
    data_shape=DataShape(
        estimated_state_size_gib=Interval(low=10, mid=100, high=1000, confidence=0.98),
    ),
)


def test_crdb_simple():
    plan = planner.plan(
        model_name="org.netflix.cockroachdb",
        region="us-east-1",
        desires=simple_desire,
    )

    lr = plan.least_regret[0]
    lr_cluster = lr.candidate_clusters.zonal[0]

    # Resulting cluster should not be too expensive
    assert 2000 < lr.candidate_clusters.total_annual_cost < 10_000

    # Should have enough disk space for around 80GiB of data in a single
    # replica (compression). Also that drive should be ephemeral
    assert lr_cluster.instance.drive is not None
    assert lr_cluster.count * lr_cluster.instance.drive.size_gib > 80

    # Should have enough CPU to handle 1000 QPS
    assert lr_cluster.count * lr_cluster.instance.cpu >= 4


def test_crdb_working_set():
    ephem = shapes.region("us-east-1").instances["i4i.xlarge"].drive
    ebs = shapes.region("us-east-1").drives["gp3"]
    super_slow_drive = ebs.copy(deep=True)
    # Simulate a very slow drive
    super_slow_drive.name = "slow"
    super_slow_drive.read_io_latency_ms = FixedInterval(
        low=5, mid=8, high=20, confidence=0.9
    )

    latency_sensitive = nflx_cockroachdb_capacity_model.default_desires(
        simple_desire, {}
    )
    results = {}
    for drive in (ephem, ebs, super_slow_drive):
        working_set = working_set_from_drive_and_slo(
            drive_read_latency_dist=dist_for_interval(drive.read_io_latency_ms),
            read_slo_latency_dist=dist_for_interval(
                latency_sensitive.query_pattern.read_latency_slo_ms
            ),
            estimated_working_set=None,
            # CRDB has looser latency SLOs but we still want a lot of the data
            # hot in cache. Target the 95th percentile of disk latency to
            # keep in RAM.
            target_percentile=0.95,
        ).mid
        results[drive.name] = working_set
    assert results["ephem"] < 0.05
    assert results["gp3"] < 0.05
    assert results["slow"] > 0.5


def test_crdb_footprint():
    space = CapacityDesires(
        service_tier=1,
        query_pattern=QueryPattern(
            estimated_read_per_second=Interval(
                low=100, mid=1000, high=10000, confidence=0.98
            ),
            estimated_write_per_second=Interval(
                low=100, mid=1000, high=10000, confidence=0.98
            ),
        ),
        data_shape=DataShape(
            estimated_state_size_gib=Interval(
                low=100, mid=1000, high=10000, confidence=0.98
            ),
        ),
    )

    plan = planner.plan(
        model_name="org.netflix.cockroachdb",
        region="us-east-1",
        desires=space,
    )

    lr = plan.least_regret[0]
    lr_cluster = lr.candidate_clusters.zonal[0]

    # Resulting cluster should not be too expensive
    assert 2000 < lr.candidate_clusters.total_annual_cost < 8000

    # Should have enough disk space for around 80GiB of data in a single
    # replica (compression). Also that drive should be ephemeral
    assert lr_cluster.instance.drive is not None
    assert lr_cluster.count * lr_cluster.instance.drive.size_gib > 800

    # Should have enough CPU to handle 1000 QPS
    assert lr_cluster.count * lr_cluster.instance.cpu >= 4
