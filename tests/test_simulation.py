import numpy as np

from service_capacity_modeling.capacity_planner import model_desires
from service_capacity_modeling.interface import CapacityDesires
from service_capacity_modeling.interface import certain_float
from service_capacity_modeling.interface import certain_int
from service_capacity_modeling.interface import DataShape
from service_capacity_modeling.interface import Interval
from service_capacity_modeling.interface import QueryPattern
from service_capacity_modeling.stats import _beta_dist_from_interval
from service_capacity_modeling.stats import _gamma_dist_from_interval
from service_capacity_modeling.stats import beta_for_interval
from service_capacity_modeling.stats import gamma_for_interval


def test_gamma_lower():
    interval = Interval(low=1000, mid=10000, high=100000, confidence=0.980)
    g = gamma_for_interval(interval)

    assert abs(g.mean() - interval.mid) < 0.01

    rvs = g.rvs(10000)

    assert interval.minimum > 200
    assert min(rvs) >= interval.minimum

    p = np.percentile(rvs, [50, 95])
    assert p[0] > interval.low
    assert p[1] > interval.mid


def test_gamma_precise():
    interval = Interval(
        minimum_value=6000, low=8000, mid=10000, high=20000, confidence=1
    )
    g = gamma_for_interval(interval)

    assert abs(g.mean() - interval.mid) < 0.01

    rvs = g.rvs(100000)

    assert interval.minimum > 2000
    assert min(rvs) >= interval.minimum

    p = np.percentile(rvs, [50, 95])
    assert p[0] > 7000
    assert p[1] > interval.mid


def test_simulate_interval():
    int_1 = Interval(low=1000, mid=10000, high=100000, confidence=0.98)
    int_2 = certain_float(100)
    int_3 = Interval(low=0.1, mid=1, high=10, confidence=0.98)
    solns = [(1.0, 2), (0.05, 3), (1, 3)]
    for interval, soln in zip((int_1, int_2, int_3), solns):
        shape, d = _gamma_dist_from_interval(interval)
        assert soln[0] < shape < soln[1]
        assert abs(d.mean() - interval.mid) < 0.01


def test_beta_lower():
    interval = Interval(low=1000, mid=10000, high=100000, confidence=0.980)
    b = beta_for_interval(interval)

    assert abs(b.mean() - interval.mid) < 0.01

    rvs = b.rvs(10000)

    assert interval.minimum > 200
    assert min(rvs) >= interval.minimum

    p = np.percentile(rvs, [50, 95])
    assert p[0] > interval.low
    assert p[1] > interval.mid


def test_beta_precise():
    interval = Interval(
        minimum_value=6000, low=8000, mid=10000, high=20000, confidence=1
    )
    b = beta_for_interval(interval)

    assert abs(b.mean() - interval.mid) < 0.01

    rvs = b.rvs(10000)

    assert interval.minimum > 2000
    assert min(rvs) >= interval.minimum

    p = np.percentile(rvs, [50, 95])
    assert p[0] > interval.low
    assert p[1] > interval.mid


def test_simulate_interval_beta():
    # Skew right
    int_1 = Interval(low=1000, mid=10000, high=100000, confidence=0.98)
    # Point function
    int_2 = certain_float(100)
    # Skew right smaller
    int_3 = Interval(low=0.1, mid=1, high=10, confidence=0.98)
    # ~Uniform distribution
    int_4 = Interval(
        minimum_value=1,
        low=100,
        mid=1000,
        high=1900,
        maximum_value=2000,
        confidence=0.98,
    )
    solns = [(1, 10), (0, 3), (1, 10), (0, 2)]
    for interval, soln in zip((int_1, int_2, int_3, int_4), solns):
        (alpha, beta, root), d = _beta_dist_from_interval(interval)
        # Check that the delta/uniform function has roughly equal values
        # since deltas and uniforms should just spread over the interval
        if interval.mid in (100, 1000):
            assert abs(alpha - beta) < 0.01
        assert root.success
        assert soln[0] < alpha < soln[1]
        assert abs(d.mean() - interval.mid) < 0.01


desires = CapacityDesires(
    service_tier=1,
    query_pattern=QueryPattern(
        estimated_read_per_second=Interval(
            low=1000, mid=10000, high=100000, confidence=0.98
        ),
        estimated_write_per_second=Interval(
            low=500, mid=2000, high=90000, confidence=0.98
        ),
        estimated_mean_read_latency_ms=Interval(
            low=0.1, mid=1, high=10, confidence=0.98
        ),
        estimated_mean_write_latency_ms=certain_float(1),
    ),
    data_shape=DataShape(
        estimated_state_size_gib=certain_int(500),
        estimated_working_set_percent=certain_float(1),
    ),
)


def test_model_desires():
    models = model_desires(desires, 10)
    samples = set()
    for model in models:
        rps = model.query_pattern.estimated_read_per_second
        wps = model.query_pattern.estimated_read_per_second
        rl = model.query_pattern.estimated_mean_read_latency_ms

        assert rps.low == rps.mid == rps.high
        assert 500 < rps.mid < 100000
        assert 250 < wps.mid < 90000
        assert rl.mid > 0.05

        samples.add((rps.mid, wps.mid, rl.mid))

    assert len(samples) == 10
