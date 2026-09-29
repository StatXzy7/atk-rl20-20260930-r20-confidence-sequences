from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple


RootBracket = Tuple[float, float]


@dataclass(frozen=True)
class ConfidenceSet:
    low: float
    high: float
    empty: bool
    lower_root_bracket: RootBracket
    upper_root_bracket: RootBracket
    numerical_tolerance: float
    n: int
    successes: int
    failures: int

    def contains(self, p: float, tolerance: float = 0.0) -> bool:
        if self.empty:
            return False
        return self.low - tolerance <= float(p) <= self.high + tolerance


@dataclass(frozen=True)
class RunningIntersection:
    low: Optional[float]
    high: Optional[float]
    empty: bool
    n: int

    def contains(self, p: float, tolerance: float = 0.0) -> bool:
        if self.empty:
            return False
        return self.low - tolerance <= float(p) <= self.high + tolerance


def log_beta(a: float, b: float) -> float:
    a = float(a)
    b = float(b)
    if not (a > 0.0 and b > 0.0 and math.isfinite(a) and math.isfinite(b)):
        raise ValueError("Beta parameters a and b must be finite positive numbers")
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def log_marginal_evidence(
    successes: int,
    failures: int,
    a: float,
    b: float,
) -> float:
    _validate_counts(successes, failures)
    log_b0 = log_beta(a, b)
    return log_beta(a + successes, b + failures) - log_b0


def log_likelihood_ratio(
    p0: float,
    successes: int,
    failures: int,
    a: float,
    b: float,
) -> float:
    _validate_p0(p0)
    _validate_counts(successes, failures)
    _validate_prior(a, b)

    log_mixture = log_marginal_evidence(successes, failures, a, b)
    if p0 == 0.0:
        if successes > 0:
            return math.inf
        return log_mixture

    if p0 == 1.0:
        if failures > 0:
            return math.inf
        return log_mixture

    log_null = successes * math.log(p0) + failures * math.log1p(-p0)
    return log_mixture - log_null


class BetaBernoulliCS:
    def __init__(
        self,
        a: float,
        b: float,
        alpha: float,
        *,
        tol: float = 1e-10,
        max_iter: int = 60,
    ) -> None:
        _validate_prior(a, b)
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0, 1)")
        if not (0.0 < tol < 1.0):
            raise ValueError("tol must be in (0, 1)")
        if max_iter < 1:
            raise ValueError("max_iter must be positive")

        self.a = float(a)
        self.b = float(b)
        self.alpha = float(alpha)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.n = 0
        self.successes = 0
        self.failures = 0
        self._observations = []
        self._log_threshold = math.log(1.0 / self.alpha)
        self._running_low = 0.0
        self._running_high = 1.0
        self._running_empty = False
        self._running_dirty = True
        self._running_n = 0
        self._running_successes = 0
        self._running_failures = 0

    @property
    def log_threshold(self) -> float:
        return self._log_threshold

    def observe(self, x: int) -> None:
        if isinstance(x, bool) or x not in (0, 1):
            raise ValueError("observations must be 0 or 1")
        if x:
            self.successes += 1
        else:
            self.failures += 1
        self.n += 1
        self._observations.append(int(bool(x)))
        self._running_dirty = True

    def observe_batch(self, xs: Iterable[int]) -> None:
        for x in xs:
            self.observe(x)

    def accepts(self, p0: float, *, numerical_slack: float = 0.0) -> bool:
        value = log_likelihood_ratio(
            p0, self.successes, self.failures, self.a, self.b
        )
        threshold = self._log_threshold + numerical_slack
        if math.isinf(value) and value > 0:
            return False
        if value <= threshold:
            return True
        if not math.isfinite(value):
            return False
        return _safe_leq(value, threshold)

    def confidence_set(self) -> ConfidenceSet:
        return self._compute_confidence_set(
            self.successes, self.failures, self.n
        )

    def _compute_confidence_set(
        self, successes: int, failures: int, n: int
    ) -> ConfidenceSet:
        if n == 0:
            return ConfidenceSet(
                low=0.0,
                high=1.0,
                empty=False,
                lower_root_bracket=(0.0, 0.0),
                upper_root_bracket=(1.0, 1.0),
                numerical_tolerance=self.declared_tolerance(),
                n=0,
                successes=successes,
                failures=failures,
            )

        if failures == 0:
            if self._point_status(1.0, successes, failures) > 0:
                return self._full_set(successes, failures, n)
            lower = self._bisect_decreasing(
                0.0, 1.0, successes, failures
            )
            upper = (1.0, 1.0)
        elif successes == 0:
            if self._point_status(0.0, successes, failures) > 0:
                return self._full_set(successes, failures, n)
            lower = (0.0, 0.0)
            upper = self._bisect_increasing(
                0.0, 1.0, successes, failures
            )
        else:
            p_hat = successes / n
            if self._point_status(p_hat, successes, failures) > 0:
                return self._full_set(successes, failures, n)
            lower = self._bisect_lower(
                0.0, p_hat, successes, failures
            )
            upper = self._bisect_upper(
                p_hat, 1.0, successes, failures
            )

        if lower == (0.0, 1.0) or upper == (0.0, 1.0):
            return self._full_set(successes, failures, n)

        numerical_tolerance = max(
            self.declared_tolerance(),
            lower[1] - lower[0],
            upper[1] - upper[0],
        )

        return ConfidenceSet(
            low=lower[0],
            high=upper[1],
            empty=False,
            lower_root_bracket=lower,
            upper_root_bracket=upper,
            numerical_tolerance=numerical_tolerance,
            n=n,
            successes=successes,
            failures=failures,
        )

    def running_intersection(self) -> RunningIntersection:
        if self._running_dirty:
            self._update_running_intersection()

        if self._running_empty:
            return RunningIntersection(None, None, True, self.n)
        return RunningIntersection(
            self._running_low, self._running_high, False, self.n
        )

    def _update_running_intersection(self) -> None:
        if self._running_empty:
            self._running_dirty = False
            return

        low = self._running_low
        high = self._running_high
        successes = self._running_successes
        failures = self._running_failures

        for index in range(self._running_n, self.n):
            if self._observations[index]:
                successes += 1
            else:
                failures += 1

            prefix_n = index + 1
            current = self._compute_confidence_set(successes, failures, prefix_n)
            low = max(low, current.low)
            high = min(high, current.high)
            if low > high:
                self._running_empty = True
                self._running_low = math.nan
                self._running_high = math.nan
                self._running_n = self.n
                self._running_successes = successes
                self._running_failures = failures
                self._running_dirty = False
                return

        self._running_low = low
        self._running_high = high
        self._running_n = self.n
        self._running_successes = successes
        self._running_failures = failures
        self._running_dirty = False

    def declared_tolerance(self) -> float:
        return max(self.tol, 8.0 * math.ulp(1.0))

    def _bisect_lower(
        self, low: float, high: float, successes: int, failures: int
    ) -> RootBracket:
        if (
            self._point_status(low, successes, failures) <= 0
            or self._point_status(high, successes, failures) > 0
        ):
            return (0.0, 1.0)

        for _ in range(self.max_iter):
            if high - low <= self.declared_tolerance():
                break
            mid = (low + high) * 0.5
            if self._point_status(mid, successes, failures) <= 0:
                high = mid
            else:
                low = mid
        return (low, high)

    def _bisect_upper(
        self, low: float, high: float, successes: int, failures: int
    ) -> RootBracket:
        if (
            self._point_status(low, successes, failures) > 0
            or self._point_status(high, successes, failures) <= 0
        ):
            return (0.0, 1.0)

        for _ in range(self.max_iter):
            if high - low <= self.declared_tolerance():
                break
            mid = (low + high) * 0.5
            if self._point_status(mid, successes, failures) <= 0:
                low = mid
            else:
                high = mid
        return (low, high)

    def _bisect_increasing(
        self, low: float, high: float, successes: int, failures: int
    ) -> RootBracket:
        if (
            self._point_status(low, successes, failures) > 0
            or self._point_status(high, successes, failures) <= 0
        ):
            return (0.0, 1.0)

        for _ in range(self.max_iter):
            if high - low <= self.declared_tolerance():
                break
            mid = (low + high) * 0.5
            if self._point_status(mid, successes, failures) <= 0:
                low = mid
            else:
                high = mid
        return (low, high)

    def _bisect_decreasing(
        self, low: float, high: float, successes: int, failures: int
    ) -> RootBracket:
        if (
            self._point_status(low, successes, failures) <= 0
            or self._point_status(high, successes, failures) > 0
        ):
            return (0.0, 1.0)

        for _ in range(self.max_iter):
            if high - low <= self.declared_tolerance():
                break
            mid = (low + high) * 0.5
            if self._point_status(mid, successes, failures) <= 0:
                high = mid
            else:
                low = mid
        return (low, high)

    def _full_set(
        self, successes: int, failures: int, n: int
    ) -> ConfidenceSet:
        return ConfidenceSet(
            low=0.0,
            high=1.0,
            empty=False,
            lower_root_bracket=(0.0, 1.0),
            upper_root_bracket=(0.0, 1.0),
            numerical_tolerance=1.0,
            n=n,
            successes=successes,
            failures=failures,
        )

    def _point_status(
        self, p: float, successes: int, failures: int
    ) -> int:
        value = log_likelihood_ratio(
            p, successes, failures, self.a, self.b
        )
        threshold = self._log_threshold

        if math.isinf(value) and value > 0:
            return 1
        if value < threshold:
            return -1
        if value == threshold:
            return 0
        if not math.isfinite(value):
            return 1
        if _safe_leq(value, threshold):
            return -1
        return 1


def _safe_leq(value: float, threshold: float) -> bool:
    scale = max(1.0, abs(value), abs(threshold))
    return value <= threshold + 8.0 * math.ulp(scale)


def _validate_prior(a: float, b: float) -> None:
    a = float(a)
    b = float(b)
    if not (a > 0.0 and b > 0.0 and math.isfinite(a) and math.isfinite(b)):
        raise ValueError("a and b must be finite positive numbers")


def _validate_p0(p0: float) -> None:
    p0 = float(p0)
    if not (0.0 <= p0 <= 1.0):
        raise ValueError("p0 must be in [0, 1]")


def _validate_counts(successes: int, failures: int) -> None:
    if not isinstance(successes, int) or not isinstance(failures, int):
        raise TypeError("successes and failures must be integers")
    if successes < 0 or failures < 0:
        raise ValueError("counts must be nonnegative")
