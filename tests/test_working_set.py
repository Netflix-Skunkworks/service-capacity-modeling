from service_capacity_modeling.hardware import shapes
from service_capacity_modeling.interface import FixedInterval
from service_capacity_modeling.models.common import WorkingSetEstimator
from service_capacity_modeling.stats import dist_for_interval


def test_working_set():
    gp2_interval = shapes.region("us-east-1").drives["gp2"].read_io_latency_ms
    drive_gp2 = dist_for_interval(gp2_interval)

    ephem_drive = shapes.region("us-east-1").instances["m5d.2xlarge"].drive
    if ephem_drive is None:
        assert False

    ephem_interval = ephem_drive.read_io_latency_ms
    drive_ephem = dist_for_interval(ephem_interval)

    slo_dist_db = dist_for_interval(
        FixedInterval(
            minimum_value=0.1, low=0.4, mid=2, high=5, maximum_value=10, confidence=0.98
        )
    )

    slo_dist_cache = dist_for_interval(
        FixedInterval(low=0.2, mid=0.6, high=1, confidence=0.98)
    )

    target_percentile = 0.9
    estimator = WorkingSetEstimator()

    db_gp2_working_set = estimator.working_set_percent(
        drive_read_latency_dist=drive_gp2,
        read_slo_latency_dist=slo_dist_db,
        target_percentile=target_percentile,
    ).mid

    db_ephem_working_set = estimator.working_set_percent(
        drive_read_latency_dist=drive_ephem,
        read_slo_latency_dist=slo_dist_db,
        target_percentile=target_percentile,
    ).mid

    assert db_gp2_working_set > db_ephem_working_set

    # Cache latency requirements should be a lot more aggressive

    cache_gp2_working_set = estimator.working_set_percent(
        drive_read_latency_dist=drive_gp2,
        read_slo_latency_dist=slo_dist_cache,
        target_percentile=target_percentile,
    ).mid

    cache_ephem_working_set = estimator.working_set_percent(
        drive_read_latency_dist=drive_ephem,
        read_slo_latency_dist=slo_dist_cache,
        target_percentile=target_percentile,
    ).mid

    assert cache_gp2_working_set > cache_ephem_working_set
    assert cache_gp2_working_set >= 0.90
    assert 0.4 >= cache_ephem_working_set >= 0.01
    assert 0.5 >= db_gp2_working_set >= 0.25
    assert 0.2 >= db_ephem_working_set >= 0.01
