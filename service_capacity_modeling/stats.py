from functools import lru_cache
from typing import Sequence
from typing import Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.optimize import OptimizeResult
from scipy.optimize import root
from scipy.special import gammainc as gammaf
from scipy.stats import beta as beta_dist
from scipy.stats import gamma as gamma_dist
from scipy.stats import rv_continuous

from service_capacity_modeling.interface import certain_float
from service_capacity_modeling.interface import Interval
from service_capacity_modeling.interface import IntervalModel

# Parameter estimation of various scipy distributions using
# See https://www.johndcook.com/quantiles_parameters.pdf for
# background

EPSILON = 0.001

# Gamma distribution G(alpha, beta) with mean alpha * beta


def _gamma_fn_from_params(low, mid, high, confidence):
    assert 0 < low <= mid <= high
    confidence = min(confidence, 0.99)
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
    interval: Interval, seed: int = 0xCAFE
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

    if lower == 0:
        lower = EPSILON

    f = _gamma_fn_from_params(lower, mean, interval.high, interval.confidence)
    result = root(f, 2)
    shape = result.x[0]

    dist = gamma_dist(shape, loc=minimum, scale=mean / shape)
    dist.random_state = np.random.default_rng(seed=seed)
    return (shape, dist)


# This can be expensive, so cache it
@lru_cache(maxsize=128)
def _gamma_for_interval(interval: Interval, seed: int = 0xCAFE) -> rv_continuous:
    return _gamma_dist_from_interval(interval, seed=seed)[1]


def gamma_for_interval(interval: Interval, seed: int = 0xCAFE) -> rv_continuous:
    result = _gamma_for_interval(interval, seed)
    # Use the new Generator API instead of RandomState for ~20% speedup
    result.random_state = np.random.default_rng(seed=seed)
    return result


# Beta distribution B(alpha, beta) with mean alpha / (alpha + beta)


def _beta_cost_fn_from_params(low, mid, high, confidence):
    assert low <= mid <= high < 1.0
    assert mid > 0

    # Assume symmetric percentiles were provided
    confidence = min(confidence, 0.99)
    confidence = max(confidence, 0.01)

    low_p = 0.0 + (1 - confidence) / 2.0
    high_p = 1.0 - (1 - confidence) / 2.0

    def cost(alpha):
        beta = alpha / mid - alpha
        if alpha == 0 or beta == 0:
            return float("inf")

        cost = (beta_dist.cdf(low, alpha, beta) - low_p) ** 2
        cost += (beta_dist.cdf(high, alpha, beta) - high_p) ** 2
        return cost

    return cost


def _beta_dist_from_interval(
    interval: Interval, seed: int = 0xCAFE
) -> Tuple[Tuple[float, float, OptimizeResult], rv_continuous]:
    # If we know cdf(high), cdf(low) and mean (mid) we can use an iterative
    # solver to find a possible beta fit

    if interval.minimum == interval.maximum:
        minimum = interval.low - EPSILON
        maximum = interval.high + EPSILON
        scale = maximum - minimum
    else:
        minimum = interval.minimum
        maximum = interval.maximum
        scale = maximum - minimum

    lower = (interval.low - minimum) / scale
    mean = (interval.mid - minimum) / scale
    upper = (interval.high - minimum) / scale

    f = _beta_cost_fn_from_params(lower, mean, upper, interval.confidence)
    result = minimize(f, x0=2, bounds=[(0.1, 40)])
    alpha = result.x[0]

    dist = beta_dist(alpha, alpha / mean - alpha, loc=minimum, scale=scale)
    dist.random_state = np.random.default_rng(seed=seed)
    return (alpha, alpha / mean - alpha, result), dist


# This can be expensive, so cache it
@lru_cache(maxsize=128)
def _beta_for_interval(interval: Interval, seed: int = 0xCAFE) -> rv_continuous:
    return _beta_dist_from_interval(interval, seed=seed)[1]


def beta_for_interval(interval: Interval, seed: int = 0xCAFE) -> rv_continuous:
    result = _beta_for_interval(interval, seed)
    # Use the new Generator API instead of RandomState for ~20% speedup
    result.random_state = np.random.default_rng(seed=seed)
    return result


def dist_for_interval(interval: Interval, seed: int = 0xCAFE) -> rv_continuous:
    if interval.model_with == IntervalModel.beta:
        result = beta_for_interval(interval=interval, seed=seed)
    elif interval.model_with == IntervalModel.gamma:
        result = gamma_for_interval(interval=interval, seed=seed)
    else:
        result = beta_for_interval(interval=interval, seed=seed)
    return result


def interval_percentile(
    interval: Interval, percentiles: Sequence[int]
) -> Sequence[Interval]:
    if interval.can_simulate:
        samples = dist_for_interval(interval).rvs(1028)
        p = np.percentile(samples, percentiles)
        return [certain_float(i) for i in p]
    else:
        return [interval] * len(percentiles)
