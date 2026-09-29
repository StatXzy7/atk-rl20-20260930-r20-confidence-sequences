import itertools
import math

import pytest

from bernoulli_cs import BetaBernoulliCS, log_beta, log_likelihood_ratio


def test_log_beta_matches_reciprocal_for_small_arguments():
    assert log_beta(1, 1) == pytest.approx(0.0)
    assert math.exp(log_beta(2, 3)) == pytest.approx(1.0 / 12.0)


@pytest.mark.parametrize("p0", [0.0, 0.37, 1.0])
@pytest.mark.parametrize("a,b", [(1.0, 1.0), (0.5, 2.0), (2.0, 0.5)])
def test_martingale_or_supermartingale_one_step_by_enumeration(p0, a, b):
    max_n = 7
    for n in range(max_n + 1):
        for history in itertools.product((0, 1), repeat=n):
            s = sum(history)
            f = n - s

            current = math.exp(log_likelihood_ratio(p0, s, f, a, b))
            after_success = math.exp(
                log_likelihood_ratio(p0, s + 1, f, a, b)
            )
            after_failure = math.exp(
                log_likelihood_ratio(p0, s, f + 1, a, b)
            )

            if 0.0 < p0 < 1.0:
                expected = p0 * after_success + (1.0 - p0) * after_failure
                assert expected == pytest.approx(current, rel=1e-10, abs=1e-10)
            elif p0 == 0.0:
                expected = after_failure if s == 0 else math.inf
                assert expected <= current * (1.0 + 1e-12) + 1e-12
            else:
                expected = after_success if f == 0 else math.inf
                assert expected <= current * (1.0 + 1e-12) + 1e-12


def test_zero_sample_is_full_parameter_space():
    model = BetaBernoulliCS(1, 1, 0.05)
    cs = model.confidence_set()
    assert (cs.low, cs.high, cs.empty) == (0.0, 1.0, False)
    assert all(cs.contains(p) for p in (0.0, 0.5, 1.0))


def test_boundary_likelihood_ratios():
    assert log_likelihood_ratio(0.0, 1, 0, 1, 1) == math.inf
    assert log_likelihood_ratio(1.0, 0, 1, 1, 1) == math.inf
    assert log_likelihood_ratio(0.0, 0, 5, 1, 1) == pytest.approx(
        math.log(1.0 / 6.0)
    )
    assert log_likelihood_ratio(1.0, 5, 0, 1, 1) == pytest.approx(
        math.log(1.0 / 6.0)
    )


@pytest.mark.parametrize("observations", [[0] * 50, [1] * 50])
def test_long_boundary_sequences_do_not_underflow(observations):
    model = BetaBernoulliCS(1, 1, 0.05)
    model.observe_batch(observations)
    cs = model.confidence_set()
    assert math.isfinite(cs.low)
    assert math.isfinite(cs.high)
    assert cs.numerical_tolerance <= 1e-9


def test_returned_endpoints_are_outward_brackets():
    model = BetaBernoulliCS(1, 1, 0.05)
    model.observe_batch([1, 0, 1, 1, 0, 1, 0, 0])
    cs = model.confidence_set()

    assert not model.accepts(cs.low)
    assert model.accepts(cs.lower_root_bracket[1])
    assert model.accepts(cs.upper_root_bracket[0])
    assert not model.accepts(cs.high)


def test_all_success_and_all_failure_brackets_include_boundary():
    success = BetaBernoulliCS(0.7, 1.3, 0.05)
    success.observe_batch([1] * 8)
    success_cs = success.confidence_set()
    assert not success.accepts(success_cs.low)
    assert success_cs.high == 1.0

    failure = BetaBernoulliCS(0.7, 1.3, 0.05)
    failure.observe_batch([0] * 8)
    failure_cs = failure.confidence_set()
    assert failure_cs.low == 0.0
    assert not failure.accepts(failure_cs.high)


@pytest.mark.parametrize("observation", [0, 1])
def test_matching_boundary_is_covered_over_ordinary_long_paths(observation):
    model = BetaBernoulliCS(1, 1, 0.05)
    model.observe_batch([observation] * 1000)
    cs = model.confidence_set()
    assert cs.contains(float(observation))


def test_incremental_and_batch_construction_agree():
    observations = [1, 0, 1, 1, 0, 0, 1]

    incremental = BetaBernoulliCS(0.5, 1.5, 0.10)
    batch = BetaBernoulliCS(0.5, 1.5, 0.10)
    batch.observe_batch(observations)

    for x in observations:
        incremental.observe(x)

    assert (
        incremental.successes,
        incremental.failures,
        incremental.n,
    ) == (batch.successes, batch.failures, batch.n)
    assert incremental.confidence_set() == batch.confidence_set()


def test_running_intersection_is_nested_and_batch_agrees():
    observations = [1, 1, 0, 0, 1, 0, 1, 1, 0]

    incremental = BetaBernoulliCS(1, 1, 0.05)
    previous_low = 0.0
    previous_high = 1.0

    for n, x in enumerate(observations, start=1):
        incremental.observe(x)
        batch = BetaBernoulliCS(1, 1, 0.05)
        batch.observe_batch(observations[:n])

        left = incremental.running_intersection()
        right = batch.running_intersection()

        assert left == right
        assert left.low >= previous_low
        assert left.high <= previous_high
        previous_low, previous_high = left.low, left.high


def test_invalid_construction_arguments_are_rejected():
    with pytest.raises(ValueError):
        BetaBernoulliCS(0, 1, 0.05)
    with pytest.raises(ValueError):
        BetaBernoulliCS(1, -1, 0.05)
    with pytest.raises(ValueError):
        BetaBernoulliCS(1, 1, 0.0)
    with pytest.raises(ValueError):
        BetaBernoulliCS(1, 1, 1.0)


def test_observations_must_be_binary():
    model = BetaBernoulliCS(1, 1, 0.05)
    with pytest.raises(ValueError):
        model.observe(2)
