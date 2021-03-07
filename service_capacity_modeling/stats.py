from functools import lru_cache
from typing import Tuple

import numpy as np
from scipy.optimize import fsolve
from scipy.optimize import root
from scipy.special import betainc as betaf
from scipy.special import gammainc as gammaf
from scipy.stats import beta as beta_dist
from scipy.stats import gamma as gamma_dist
from scipy.stats import rv_continuous

from service_capacity_modeling.interface import Interval
from service_capacity_modeling.interface import IntervalModel


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


def _beta_fn_from_params(low, mid, high, confidence):
    assert low <= mid <= high < 1.0

    confidence = min(confidence, 0.95)
    confidence = max(confidence, 0.01)

    low_p = 0.0 + (1 - confidence) / 2.0
    high_p = 1.0 - (1 - confidence) / 2.0

    def f(a):
        zero = high / low
        return betaf(a, a / mid - a, high_p) / betaf(a, a / mid - a, low_p) - zero

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
    dist.random_state = np.random.default_rng(seed=seed)
    return (shape, dist)


def _beta_dist_from_interval(
    interval: Interval, seed: float = 0xCAFE
) -> Tuple[float, rv_continuous]:
    # If we know cdf(high), cdf(low) and mean (mid) we can use an iterative
    # solver to find a possible beta fit

    minimum = interval.minimum
    maximum = interval.maximum
    scale = maximum - minimum

    lower = (interval.low - minimum) / scale
    mean = (interval.mid - minimum) / scale
    upper = (interval.high - minimum) / scale

    f = _beta_fn_from_params(lower, mean, upper, interval.confidence)
    alpha = root(f, 2).x[0]

    dist = beta_dist(alpha, alpha / mean - alpha, loc=minimum, scale=scale)
    dist.random_state = np.random.default_rng(seed=seed)
    return (alpha, dist)


# This can be expensive, so cache it
@lru_cache(maxsize=128)
def _gamma_for_interval(interval: Interval, seed: float = 0xCAFE) -> rv_continuous:
    return _gamma_dist_from_interval(interval, seed=seed)[1]


def gamma_for_interval(interval: Interval, seed: float = 0xCAFE) -> rv_continuous:
    result = _gamma_for_interval(interval, seed)
    # Use the new Generator API instead of RandomState for ~20% speedup
    result.random_state = np.random.default_rng(seed=seed)
    return result


# This can be expensive, so cache it
@lru_cache(maxsize=128)
def _beta_for_interval(interval: Interval, seed: float = 0xCAFE) -> rv_continuous:
    return _beta_dist_from_interval(interval, seed=seed)[1]


def beta_for_interval(interval: Interval, seed: float = 0xCAFE) -> rv_continuous:
    result = _beta_for_interval(interval, seed)
    # Use the new Generator API instead of RandomState for ~20% speedup
    result.random_state = np.random.default_rng(seed=seed)
    return result


def dist_for_interval(interval: Interval, seed: float = 0xCAFE) -> rv_continuous:
    if interval.model_with == IntervalModel.beta:
        result = beta_for_interval(interval=interval, seed=seed)
    elif interval.model_with == IntervalModel.gamma:
        result = gamma_for_interval(interval=interval, seed=seed)
    else:
        result = beta_for_interval(interval=interval, seed=seed)
    return result
