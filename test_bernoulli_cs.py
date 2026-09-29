from fractions import Fraction
import itertools
import math

import pytest
from scipy.optimize import brentq

from bernoulli_cs import (
    BetaBernoulliCS,
    beta_cs_interval,
    beta_mixture_log_lr,
)
from simulate_cs import precompute_cs_envelopes


def test_generic_beta_martingale_conditional_expectation_exact():
    a = Fraction(3, 2)
    b = Fraction(7, 5)
    p0 = Fraction(3, 7)

    for n in range(10):
        for successes in range(n + 1):
            failures = n - successes
            total = a + b + n
            success_factor = (a + successes) / (total * p0)
            failure_factor = (b + failures) / (total * (1 - p0))
            conditional_expectation = (
                p0 * success_factor + (1 - p0) * failure_factor
            )
            assert conditional_expectation == 1


def test_boundary_hypotheses_are_supermartingales_exact():
    a = Fraction(3, 2)
    b = Fraction(7, 5)

    for n in range(10):
        # At p0=0, only all-failure histories have positive probability.
        failure_under_p0 = (b + n) / (a + b + n)
        assert failure_under_p0 < 1

        # At p0=1, only all-success histories have positive probability.
        success_under_p1 = (a + n) / (a + b + n)
        assert success_under_p1 < 1


def test_all_short_paths_start_at_one_and_remain_nonnegative_in_log_domain():
    a, b, alpha = 1.3, 0.7, 0.05
    p0 = 0.37
    for length in range(1, 8):
        for path in itertools.product((0, 1), repeat=length):
            successes = sum(path)
            failures = length - successes
            log_lr = beta_mixture_log_lr(p0, successes, failures, a, b)
            assert math.isfinite(log_lr)
            assert log_lr >= -50.0


def test_zero_sample_is_whole_interval_and_has_log_lr_zero():
    for p0 in (0.0, 0.25, 0.75, 1.0):
        assert beta_mixture_log_lr(p0, 0, 0, 1.0, 1.0) == 0.0

    interval = beta_cs_interval(0, 0, 1.0, 1.0, 0.05)
    assert interval.low == 0.0
    assert interval.high == 1.0


@pytest.mark.parametrize(
    ("successes", "failures", "p0"),
    [
        (1, 0, 0.0),
        (5, 1, 0.0),
        (0, 1, 1.0),
        (1, 5, 1.0),
    ],
)
def test_impossible_observations_under_boundary_have_infinite_log_lr(
    successes, failures, p0
):
    assert beta_mixture_log_lr(p0, successes, failures, 1.0, 1.0) == math.inf


def test_all_success_and_all_failure_intervals_have_correct_endpoint_behavior():
    all_success = beta_cs_interval(10, 0, 1.0, 1.0, 0.05)
    assert all_success.low > 0.0
    assert all_success.high == 1.0
    assert math.isfinite(beta_mixture_log_lr(1.0, 10, 0, 1.0, 1.0))
    assert beta_mixture_log_lr(0.0, 10, 0, 1.0, 1.0) == math.inf

    all_failure = beta_cs_interval(0, 10, 1.0, 1.0, 0.05)
    assert all_failure.low == 0.0
    assert all_failure.high < 1.0
    assert math.isfinite(beta_mixture_log_lr(0.0, 0, 10, 1.0, 1.0))
    assert beta_mixture_log_lr(1.0, 0, 10, 1.0, 1.0) == math.inf


def test_bisection_brackets_bound_root_from_outside():
    a, b, alpha = 1.7, 0.9, 0.03
    threshold = -math.log(alpha)

    for successes, failures in [(1, 1), (3, 5), (10, 2), (37, 64)]:
        interval = beta_cs_interval(
            successes,
            failures,
            a,
            b,
            alpha,
            x_tolerance=1e-12,
            log_tolerance=0.0,
        )

        lower = brentq(
            lambda p: beta_mixture_log_lr(p, successes, failures, a, b) - threshold,
            0.0,
            successes / (successes + failures),
            xtol=1e-14,
            rtol=1e-14,
        )
        upper = brentq(
            lambda p: beta_mixture_log_lr(p, successes, failures, a, b) - threshold,
            successes / (successes + failures),
            1.0,
            xtol=1e-14,
            rtol=1e-14,
        )

        outer_low, inner_low = interval.lower_bracket
        inner_high, outer_high = interval.upper_bracket
        assert outer_low <= lower <= inner_low
        assert inner_high <= upper <= outer_high
        assert interval.high - interval.low >= upper - lower


def test_envelope_contains_every_mathematically_accepted_grid_point():
    a, b, alpha = 0.8, 1.2, 0.07
    threshold = -math.log(alpha)
    for successes, failures in [(1, 0), (0, 1), (4, 9), (20, 20), (50, 1)]:
        interval = beta_cs_interval(successes, failures, a, b, alpha)
        for index in range(1001):
            p0 = index / 1000
            accepted = (
                beta_mixture_log_lr(p0, successes, failures, a, b) <= threshold
            )
            if accepted:
                assert interval.low <= p0 <= interval.high


def test_running_intersection_is_monotone_and_append_equals_batch():
    observations = [1, 0, 0, 1, 1, 1, 0]
    online = BetaBernoulliCS(1.0, 1.0, 0.05)
    previous_low = 0.0
    previous_high = 1.0

    for observation in observations:
        online.update(observation)
        running = online.running_intersection
        assert running.low >= previous_low
        assert running.high <= previous_high
        previous_low, previous_high = running.low, running.high

    batch = BetaBernoulliCS.from_observations(observations, 1.0, 1.0, 0.05)
    assert online.current == batch.current
    assert online.running_intersection == batch.running_intersection
    assert online.n == batch.n


def test_current_inversion_need_not_be_nested():
    tracker = BetaBernoulliCS(1.0, 1.0, 0.05)
    tracker.update(1)
    first_low = tracker.current.low
    tracker.update(0)
    assert tracker.current.low < first_low


@pytest.mark.parametrize(
    ("a", "b", "alpha"),
    [
        (0.0, 1.0, 0.05),
        (1.0, -1.0, 0.05),
        (1.0, 1.0, 0.0),
        (1.0, 1.0, 1.0),
    ],
)
def test_invalid_prior_and_alpha_are_rejected(a, b, alpha):
    with pytest.raises(ValueError):
        beta_cs_interval(1, 1, a, b, alpha)


def test_invalid_observations_are_rejected():
    tracker = BetaBernoulliCS(1.0, 1.0, 0.05)
    with pytest.raises(ValueError):
        tracker.update(2)
    with pytest.raises(ValueError):
        tracker.update(-1)


def test_vectorized_simulation_table_matches_scalar_envelopes():
    a, b, alpha = 1.0, 1.0, 0.05
    low, high = precompute_cs_envelopes(20, a, b, alpha)
    for successes in range(21):
        for failures in range(21 - successes):
            interval = beta_cs_interval(successes, failures, a, b, alpha)
            n = successes + failures
            assert abs(low[n, successes] - interval.low) <= 2e-12
            assert abs(high[n, successes] - interval.high) <= 2e-12
            assert low[n, successes] <= interval.lower_bracket[1] + 2e-12
            assert high[n, successes] >= interval.upper_bracket[0] - 2e-12
