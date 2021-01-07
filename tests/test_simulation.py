import numpy as np

from service_capacity_modeling.capacity_planner import model_desires
from service_capacity_modeling.models import CapacityDesires
from service_capacity_modeling.models import certain_float
from service_capacity_modeling.models import certain_int
from service_capacity_modeling.models import DataShape
from service_capacity_modeling.models import Interval
from service_capacity_modeling.models import QueryPattern
from service_capacity_modeling.stats import _gamma_dist_from_interval
from service_capacity_modeling.stats import gamma_for_interval


def test_gamma_lower():
    interval = Interval(low=1000, mid=10000, high=100000, confidence=0.90)
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

    rvs = g.rvs(10000)

    assert interval.minimum > 2000
    assert min(rvs) >= interval.minimum

    p = np.percentile(rvs, [50, 95])
    assert p[0] > interval.low
    assert p[1] > interval.mid


def test_simulate_interval():
    int_1 = Interval(low=1000, mid=10000, high=100000, confidence=0.9)
    int_2 = certain_float(100)
    int_3 = Interval(low=0.1, mid=1, high=10, confidence=0.9)
    solns = [(1.5, 2), (0.05, 0.5), (2.0, 3)]
    for interval, soln in zip((int_1, int_2, int_3), solns):
        shape, d = _gamma_dist_from_interval(interval)
        assert soln[0] < shape[0] < soln[1]
        assert abs(d.mean() - interval.mid) < 0.01


desires = CapacityDesires(
    service_tier=1,
    query_pattern=QueryPattern(
        estimated_read_per_second=Interval(
            low=1000, mid=10000, high=100000, confidence=0.9
        ),
        estimated_write_per_second=Interval(
            low=500, mid=2000, high=90000, confidence=0.9
        ),
        estimated_mean_read_latency_ms=Interval(
            low=0.1, mid=1, high=10, confidence=0.9
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
