from functools import lru_cache
from typing import Tuple

import numpy as np
from scipy.optimize import fsolve
from scipy.special import gammainc as gammaf
from scipy.stats import gamma as gamma_dist
from scipy.stats import rv_continuous

from service_capacity_modeling.models import Interval


def _gamma_fn_from_params(low, mid, high, confidence):
    assert low <= mid <= high

    confidence = min(confidence, 0.95)
    confidence = max(confidence, 0.01)

    low_p = 0 + (1 - confidence) / 2.0
    high_p = 1 - (1 - confidence) / 2.0

    # cdf(x) = F(k) * gammaf(shape, x / scale)
    # mean = shape * scale
    # We know the value at two points of the cdf and the mean so we can
    # basically setup a system of equations of cdf(high) / cdf(low) = known
    # and mean = known
    #
    # Then we can use numeric methods to solve for the remaining shape parameter

    def f(k):
        zero = high / low
        return gammaf(k, high_p * k / mid) / gammaf(k, low_p * k / mid) - zero

    return f


def _gamma_dist_from_interval(
    interval: Interval, seed: float = 0xCAFE
) -> Tuple[float, rv_continuous]:
    # If we know cdf(high), cdf(low) and mean (mid) we can use an iterative
    # solver to find a possible gamma interval

    # Note we shift the lower bound and mean by the minimum (defaults to
    # half the lower bound so we don't end up with less than the minimum
    # estimate. This does distort the gamma but in a way that is useful for
    # capacity planning (sorta like a wet-bias in forcasting models)
    minimum = interval.minimum
    lower = interval.low - minimum
    mean = interval.mid - minimum

    f = _gamma_fn_from_params(lower, mean, interval.high, interval.confidence)
    shape = fsolve(f, 2)

    dist = gamma_dist(shape, loc=minimum, scale=(mean / shape))
    dist.random_state = np.random.RandomState(seed=seed)
    return (shape, dist)


# This can be expensive, so cache it
@lru_cache(maxsize=128)
def gamma_for_interval(interval: Interval, seed: float = 0xCAFE) -> rv_continuous:
    return _gamma_dist_from_interval(interval, seed=seed)[1]
