import numpy as np

from service_capacity_modeling.models import Interval
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
