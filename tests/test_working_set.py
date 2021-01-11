from service_capacity_modeling.interface import FixedInterval
from service_capacity_modeling.models.common import WorkingSetEstimator
from service_capacity_modeling.stats import gamma_for_interval


def test_working_set():
    drive_gp2 = gamma_for_interval(
        FixedInterval(low=0.4, mid=0.6, high=2, confidence=0.98)
    )

    drive_ephem = gamma_for_interval(
        FixedInterval(low=0.2, mid=0.3, high=1, confidence=0.98)
    )

    slo_dist_db = gamma_for_interval(
        FixedInterval(low=0.4, mid=2, high=10, confidence=0.98)
    )

    slo_dist_cache = gamma_for_interval(
        FixedInterval(low=0.2, mid=0.5, high=5, confidence=0.98)
    )

    target_percentile = 0.10
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

    print(
        db_gp2_working_set,
        db_ephem_working_set,
        cache_gp2_working_set,
        cache_ephem_working_set,
    )

    assert cache_gp2_working_set >= 0.99
    assert 0.9 >= cache_ephem_working_set >= 0.20
    assert 0.6 >= db_gp2_working_set >= 0.10
    assert 0.3 >= db_ephem_working_set >= 0.01
