from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.special import gammaln


@dataclass(frozen=True)
class CSResult:
    ever_excluded: np.ndarray
    stopped: np.ndarray
    stop_time: np.ndarray


def wilson_interval(s: np.ndarray, n: int, z: float = 1.959963984540054):
    p_hat = s / n
    denom = 1.0 + z * z / n
    center = (p_hat + z * z / (2.0 * n)) / denom
    margin = z * np.sqrt(p_hat * (1.0 - p_hat) / n + z * z / (4.0 * n * n)) / denom
    return center - margin, center + margin


def beta_cs_intervals(s: np.ndarray, f: np.ndarray, a: float, b: float, log_inv_alpha: float):
    n = s + f
    n_float = n.astype(np.float64)
    log_mix = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + gammaln(a + s.astype(np.float64))
        + gammaln(b + f.astype(np.float64))
        - gammaln(a + b + n_float)
    )

    low = np.zeros(n.shape, dtype=np.float64)
    high = np.ones(n.shape, dtype=np.float64)

    interior = (s > 0) & (f > 0)
    all_success = (f == 0) & (s > 0)
    all_failure = (s == 0) & (f > 0)

    p_hat = np.divide(s, n, out=np.full(n.shape, 0.5), where=n > 0)

    def status(p, mask):
        log_null = s[mask] * np.log(p) + f[mask] * np.log1p(-p)
        return log_mix[mask] - log_null - log_inv_alpha

    if np.any(interior):
        l = np.zeros(np.count_nonzero(interior), dtype=np.float64)
        r = p_hat[interior].copy()
        for _ in range(40):
            m = (l + r) * 0.5
            accepted = status(m, interior) <= 0.0
            r = np.where(accepted, m, r)
            l = np.where(accepted, l, m)
        low[interior] = l

        l = p_hat[interior].copy()
        r = np.ones_like(l)
        for _ in range(40):
            m = (l + r) * 0.5
            accepted = status(m, interior) <= 0.0
            l = np.where(accepted, m, l)
            r = np.where(accepted, r, m)
        high[interior] = r

    if np.any(all_success):
        l = np.zeros(np.count_nonzero(all_success), dtype=np.float64)
        r = np.ones_like(l)
        for _ in range(40):
            m = (l + r) * 0.5
            accepted = status(m, all_success) <= 0.0
            r = np.where(accepted, m, r)
            l = np.where(accepted, l, m)
        low[all_success] = l

    if np.any(all_failure):
        l = np.zeros(np.count_nonzero(all_failure), dtype=np.float64)
        r = np.ones_like(l)
        for _ in range(40):
            m = (l + r) * 0.5
            accepted = status(m, all_failure) <= 0.0
            l = np.where(accepted, m, l)
            r = np.where(accepted, r, m)
        high[all_failure] = r

    return low, high


def active_beta_cs_intervals(
    successes: np.ndarray,
    failures: np.ndarray,
    active: np.ndarray,
    a: float,
    b: float,
    log_inv_alpha: float,
):
    low = np.zeros_like(active, dtype=np.float64)
    high = np.ones_like(active, dtype=np.float64)
    if not np.any(active):
        return low, high

    active_low, active_high = beta_cs_intervals(
        successes[active], failures[active], a, b, log_inv_alpha
    )
    low[active] = active_low
    high[active] = active_high
    return low, high


def run_cs_experiment(
    rng: np.random.Generator,
    p: float,
    samples: int,
    horizon: int,
    a: float,
    b: float,
    alpha: float,
    rule: str,
):
    successes = np.zeros(samples, dtype=np.int64)
    failures = np.zeros(samples, dtype=np.int64)
    active = np.ones(samples, dtype=bool)
    ever_excluded = np.zeros(samples, dtype=bool)
    stopped = np.zeros(samples, dtype=bool)
    stop_time = np.full(samples, horizon + 1, dtype=np.int64)
    log_inv_alpha = math.log(1.0 / alpha)

    for t in range(1, horizon + 1):
        x = rng.random(samples) < p
        successes += x
        failures += ~x

        low, high = active_beta_cs_intervals(
            successes, failures, active, a, b, log_inv_alpha
        )

        inside = (low <= p) & (p <= high)
        ever_excluded |= ~inside

        if rule == "lower_above_half":
            trigger = low > 0.5
        elif rule == "upper_below_half":
            trigger = high < 0.5
        elif rule == "width":
            trigger = high - low < 0.25
        elif rule == "quarter":
            trigger = (low > 0.25) | (high < 0.75)
        else:
            raise ValueError(f"unknown rule: {rule}")

        newly_stopped = active & trigger
        stopped |= newly_stopped
        stop_time[newly_stopped] = t
        active &= ~trigger

    return CSResult(ever_excluded, stopped, stop_time)


def run_wilson_experiment(
    rng: np.random.Generator,
    p: float,
    samples: int,
    horizon: int,
):
    successes = np.zeros(samples, dtype=np.int64)
    ever_excluded = np.zeros(samples, dtype=bool)

    for t in range(1, horizon + 1):
        successes += rng.random(samples) < p
        low, high = wilson_interval(successes, t)
        ever_excluded |= (p < low) | (p > high)

    return ever_excluded


def wilson_estimate(x: np.ndarray):
    n = x.size
    k = int(np.count_nonzero(x))
    phat = k / n
    z = 1.959963984540054
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2.0 * n)) / denom
    margin = z * math.sqrt(phat * (1.0 - phat) / n + z * z / (4.0 * n * n)) / denom
    se = math.sqrt(phat * (1.0 - phat) / n) if 0.0 < phat < 1.0 else 0.0
    return {
        "events": k,
        "trials": n,
        "estimate": phat,
        "binomial_se": se,
        "wilson_95": [max(0.0, center - margin), min(1.0, center + margin)],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=50_000)
    parser.add_argument("--horizon", type=int, default=1_000)
    parser.add_argument("--a", type=float, default=1.0)
    parser.add_argument("--b", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--output", type=Path, default=Path("results/simulation.json"))
    args = parser.parse_args()

    settings = [
        (0.01, "upper_below_half"),
        (0.10, "upper_below_half"),
        (0.50, "width"),
        (0.90, "lower_above_half"),
        (0.99, "lower_above_half"),
    ]

    rng = np.random.default_rng(args.seed)
    cs_results = []
    for p, rule in settings:
        result = run_cs_experiment(
            rng,
            p,
            args.samples,
            args.horizon,
            args.a,
            args.b,
            args.alpha,
            rule,
        )
        stopped_count = int(np.count_nonzero(result.stopped))
        cs_results.append(
            {
                "p": p,
                "rule": rule,
                "exclusion": wilson_estimate(result.ever_excluded),
                "stopped_rate": stopped_count / args.samples,
                "mean_stop_time_when_stopped": (
                    float(np.mean(result.stop_time[result.stopped]))
                    if stopped_count
                    else None
                ),
            }
        )

    wilson_results = []
    for p, _ in settings:
        ever = run_wilson_experiment(rng, p, args.samples, args.horizon)
        wilson_results.append(
            {"p": p, "horizon": args.horizon, "exclusion": wilson_estimate(ever)}
        )

    payload = {
        "config": vars(args) | {"output": str(args.output)},
        "beta_confidence_sequence": cs_results,
        "fixed_time_wilson_repeated_review": wilson_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
