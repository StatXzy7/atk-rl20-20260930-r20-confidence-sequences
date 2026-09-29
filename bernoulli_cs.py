"""Beta-mixture confidence sequences for Bernoulli observations.

The core implementation uses only the Python standard library.  NumPy and
SciPy are used only by the simulation script.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Iterable


def _log_beta(x: float, y: float) -> float:
    return math.lgamma(x) + math.lgamma(y) - math.lgamma(x + y)


def _check_prior_and_alpha(a: float, b: float, alpha: float) -> None:
    for name, value in (("a", a), ("b", b)):
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a finite positive number")
    if not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between 0 and 1")


def beta_mixture_log_lr(
    p0: float,
    successes: int,
    failures: int,
    a: float,
    b: float,
) -> float:
    """Logarithm of the Beta(a,b)-mixture likelihood ratio against p0.

    Bernoulli factors use the usual likelihood convention at the boundary:
    a zero exponent contributes 1.  At p0=0, any success makes the ratio
    infinite; at p0=1, any failure makes it infinite.

    The empty sample gives log LR 0 at every p0, including both boundaries.
    """
    _check_prior_and_alpha(a, b, 0.5)
    if not isinstance(successes, int) or not isinstance(failures, int):
        raise TypeError("successes and failures must be integers")
    if successes < 0 or failures < 0:
        raise ValueError("successes and failures must be nonnegative")
    if not math.isfinite(p0) or not 0.0 <= p0 <= 1.0:
        raise ValueError("p0 must be in [0, 1]")

    n = successes + failures
    if n == 0:
        return 0.0

    log_numerator = _log_beta(a + successes, b + failures) - _log_beta(a, b)

    if p0 == 0.0:
        return math.inf if successes > 0 else log_numerator
    if p0 == 1.0:
        return math.inf if failures > 0 else log_numerator

    return (
        log_numerator
        - successes * math.log(p0)
        - failures * math.log1p(-p0)
    )


@dataclass(frozen=True)
class CSInterval:
    """A numerically conservative confidence-set interval.

    ``low`` and ``high`` are an outer envelope.  ``lower_bracket`` and
    ``upper_bracket`` expose the final bisection brackets.  For a lower
    boundary the bracket is ordered as (outer, inner); for an upper boundary
    it is (inner, outer).
    """

    low: float
    high: float
    lower_bracket: tuple[float, float]
    upper_bracket: tuple[float, float]
    x_tolerance: float
    log_tolerance: float


def beta_cs_interval(
    successes: int,
    failures: int,
    a: float,
    b: float,
    alpha: float,
    x_tolerance: float = 1e-12,
    log_tolerance: float = 1e-10,
    max_iterations: int = 100,
) -> CSInterval:
    """Invert the Beta-mixture e-process at one sample count.

    The mathematical acceptance condition is ``log M_n(p0) <= log(1/alpha)``.
    ``log_tolerance`` is an explicitly allowed outward numerical slack;
    bisection points on the returned outer side are rejected even after that
    slack.  Hence ``[low, high]`` contains the mathematical acceptance set,
    assuming the libm log-beta error is no larger than ``log_tolerance``.
    """
    _check_prior_and_alpha(a, b, alpha)
    if not isinstance(successes, int) or not isinstance(failures, int):
        raise TypeError("successes and failures must be integers")
    if successes < 0 or failures < 0:
        raise ValueError("successes and failures must be nonnegative")
    if not math.isfinite(x_tolerance) or x_tolerance <= 0:
        raise ValueError("x_tolerance must be positive")
    if not math.isfinite(log_tolerance) or log_tolerance < 0:
        raise ValueError("log_tolerance must be nonnegative")
    if not isinstance(max_iterations, int) or max_iterations <= 0:
        raise ValueError("max_iterations must be a positive integer")

    n = successes + failures
    no_root = (0.0, 0.0), (1.0, 1.0)
    if n == 0:
        return CSInterval(0.0, 1.0, no_root[0], no_root[1], x_tolerance, log_tolerance)

    log_threshold = -math.log(alpha)

    def rejected_even_with_slack(p0: float) -> bool:
        log_lr = beta_mixture_log_lr(p0, successes, failures, a, b)
        if math.isnan(log_lr):
            raise FloatingPointError(f"non-finite log LR at p0={p0}")
        return log_lr > log_threshold + log_tolerance

    empirical_p = successes / n

    if successes > 0:
        outer, inner = 0.0, empirical_p
        for _ in range(max_iterations):
            if inner - outer <= x_tolerance:
                break
            middle = (outer + inner) / 2.0
            if middle == outer or middle == inner:
                break
            if rejected_even_with_slack(middle):
                outer = middle
            else:
                inner = middle
        low = outer
        lower_bracket = (outer, inner)
    else:
        low = 0.0
        lower_bracket = (0.0, 0.0)

    if failures > 0:
        inner, outer = empirical_p, 1.0
        for _ in range(max_iterations):
            if outer - inner <= x_tolerance:
                break
            middle = (inner + outer) / 2.0
            if middle == inner or middle == outer:
                break
            if rejected_even_with_slack(middle):
                outer = middle
            else:
                inner = middle
        high = outer
        upper_bracket = (inner, outer)
    else:
        high = 1.0
        upper_bracket = (1.0, 1.0)

    return CSInterval(
        low,
        high,
        lower_bracket,
        upper_bracket,
        x_tolerance,
        log_tolerance,
    )


class BetaBernoulliCS:
    """Online tracker for both current and running-intersection intervals."""

    def __init__(
        self,
        a: float,
        b: float,
        alpha: float,
        x_tolerance: float = 1e-12,
        log_tolerance: float = 1e-10,
        max_iterations: int = 100,
    ) -> None:
        _check_prior_and_alpha(a, b, alpha)
        self.a = float(a)
        self.b = float(b)
        self.alpha = float(alpha)
        self.x_tolerance = float(x_tolerance)
        self.log_tolerance = float(log_tolerance)
        self.max_iterations = max_iterations
        self._successes = 0
        self._failures = 0
        initial = beta_cs_interval(
            0,
            0,
            self.a,
            self.b,
            self.alpha,
            self.x_tolerance,
            self.log_tolerance,
            self.max_iterations,
        )
        self._current = initial
        self._running = initial

    @property
    def n(self) -> int:
        return self._successes + self._failures

    @property
    def successes(self) -> int:
        return self._successes

    @property
    def failures(self) -> int:
        return self._failures

    @property
    def current(self) -> CSInterval:
        return self._current

    @property
    def running_intersection(self) -> CSInterval:
        return self._running

    def update(self, observation: int) -> "BetaBernoulliCS":
        if observation not in (0, 1) or (
            isinstance(observation, float) and not observation.is_integer()
        ):
            raise ValueError("observations must be Bernoulli 0/1 values")
        observation = int(observation)

        if observation == 1:
            self._successes += 1
        else:
            self._failures += 1

        self._current = beta_cs_interval(
            self._successes,
            self._failures,
            self.a,
            self.b,
            self.alpha,
            self.x_tolerance,
            self.log_tolerance,
            self.max_iterations,
        )

        running = self._running
        lower_bracket = running.lower_bracket
        upper_bracket = running.upper_bracket
        new_low = running.low
        new_high = running.high

        if self._current.low > running.low:
            new_low = self._current.low
            lower_bracket = self._current.lower_bracket
        if self._current.high < running.high:
            new_high = self._current.high
            upper_bracket = self._current.upper_bracket

        self._running = replace(
            running,
            low=new_low,
            high=new_high,
            lower_bracket=lower_bracket,
            upper_bracket=upper_bracket,
        )
        return self

    def observe(self, observations: Iterable[int]) -> "BetaBernoulliCS":
        for observation in observations:
            self.update(observation)
        return self

    @classmethod
    def from_observations(
        cls,
        observations: Iterable[int],
        a: float,
        b: float,
        alpha: float,
        x_tolerance: float = 1e-12,
        log_tolerance: float = 1e-10,
        max_iterations: int = 100,
    ) -> "BetaBernoulliCS":
        tracker = cls(
            a,
            b,
            alpha,
            x_tolerance=x_tolerance,
            log_tolerance=log_tolerance,
            max_iterations=max_iterations,
        )
        tracker.observe(observations)
        return tracker
