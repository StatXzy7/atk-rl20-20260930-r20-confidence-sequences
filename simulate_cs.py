"""Finite-horizon Monte Carlo checks for the Beta Bernoulli CS.

The simulation is evidence, not part of the coverage proof.  It checks a
finite collection of stopping rules and paths; Ville's inequality supplies the
finite-sample guarantee.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.special import betaln
from scipy.stats import beta as beta_dist


def precompute_cs_envelopes(
    n_max: int,
    a: float,
    b: float,
    alpha: float,
    x_tolerance: float = 1e-12,
    log_tolerance: float = 1e-10,
    bisection_iterations: int = 55,
) -> tuple[np.ndarray, np.ndarray]:
    n_values = np.arange(n_max + 1)
    s_values = np.arange(n_max + 1)
    n_grid, s_grid = np.meshgrid(n_values, s_values, indexing="ij")
    valid = (s_grid <= n_grid) & (n_grid > 0)
    successes = s_grid[valid]
    trials = n_grid[valid]
    failures = trials - successes

    log_numerator = (
        betaln(a + successes, b + failures) - betaln(a, b)
    )
    threshold = -math.log(alpha) + log_tolerance
    empirical_p = successes / trials

    low = np.full((n_max + 1, n_max + 1), np.nan)
    high = np.full((n_max + 1, n_max + 1), np.nan)
    low[0, 0] = 0.0
    high[0, 0] = 1.0

    active_lower = successes > 0
    outer = np.zeros_like(empirical_p, dtype=np.float64)
    inner = empirical_p.copy()
    for _ in range(bisection_iterations):
        p0 = np.where(active_lower, (outer + inner) / 2.0, 0.5)
        log_lr = (
            log_numerator
            - successes * np.log(p0)
            - failures * np.log1p(-p0)
        )
        rejected = active_lower & (log_lr > threshold)
        outer = np.where(rejected, p0, outer)
        inner = np.where(active_lower & ~rejected, p0, inner)

    low_flat = np.zeros_like(empirical_p)
    np.copyto(low_flat, outer, where=active_lower)
    low[valid] = low_flat

    active_upper = failures > 0
    inner = empirical_p.copy()
    outer = np.ones_like(empirical_p)
    for _ in range(bisection_iterations):
        p0 = np.where(active_upper, (inner + outer) / 2.0, 0.5)
        log_lr = (
            log_numerator
            - successes * np.log(p0)
            - failures * np.log1p(-p0)
        )
        rejected = active_upper & (log_lr > threshold)
        outer = np.where(rejected, p0, outer)
        inner = np.where(active_upper & ~rejected, p0, inner)

    high_flat = np.ones_like(empirical_p)
    np.copyto(high_flat, outer, where=active_upper)
    high[valid] = high_flat

    finite_indices = np.tril_indices(n_max + 1)
    if not np.all(np.isfinite(low[finite_indices])):
        raise FloatingPointError("non-finite lower endpoint in CS table")
    if not np.all(np.isfinite(high[finite_indices])):
        raise FloatingPointError("non-finite upper endpoint in CS table")
    return low, high


def precompute_cp_intervals(
    n_max: int, alpha: float
) -> tuple[np.ndarray, np.ndarray]:
    low = np.full((n_max + 1, n_max + 1), np.nan)
    high = np.full((n_max + 1, n_max + 1), np.nan)
    low[0, 0] = 0.0
    high[0, 0] = 1.0

    for n in range(1, n_max + 1):
        successes = np.arange(n + 1)
        failures = n - successes
        low[n, 0] = 0.0
        low[n, 1 : n + 1] = beta_dist.ppf(
            alpha / 2.0, successes[1:], failures[1:] + 1.0
        )
        high[n, n] = 1.0
        high[n, :n] = beta_dist.ppf(
            1.0 - alpha / 2.0, successes[:-1] + 1.0, failures[:-1]
        )
    return low, high


def binomial_standard_error(k: int, r: int) -> float:
    probability = k / r
    return math.sqrt(probability * (1.0 - probability) / r)


def simulate_cs(
    true_p_values: list[float],
    replicates: int,
    n_max: int,
    a: float,
    b: float,
    alpha: float,
    seed: int,
) -> list[dict[str, float | int]]:
    low, high = precompute_cs_envelopes(n_max, a, b, alpha)
    rng = np.random.default_rng(seed)
    results = []

    for true_p in true_p_values:
        successes = np.zeros(replicates, dtype=np.int64)
        ever_excluded = np.zeros(replicates, dtype=bool)

        width_active = np.ones(replicates, dtype=bool)
        direction_active = np.ones(replicates, dtype=bool)
        either_active = np.ones(replicates, dtype=bool)

        width_errors = 0
        direction_errors = 0
        either_errors = 0
        width_stops = 0
        direction_stops = 0
        either_stops = 0

        for n in range(1, n_max + 1):
            successes += rng.random(replicates) < true_p
            lower = low[n, successes]
            upper = high[n, successes]
            excluded = (true_p < lower) | (true_p > upper)
            ever_excluded |= excluded

            width_trigger = width_active & ((upper - lower) <= 0.30)
            direction_trigger = direction_active & (
                (lower >= 0.60) | (upper <= 0.40)
            )
            either_trigger = either_active & (
                width_trigger | direction_trigger
            )

            width_errors += int(np.count_nonzero(width_trigger & excluded))
            direction_errors += int(np.count_nonzero(direction_trigger & excluded))
            either_errors += int(np.count_nonzero(either_trigger & excluded))

            width_stops += int(np.count_nonzero(width_trigger))
            direction_stops += int(np.count_nonzero(direction_trigger))
            either_stops += int(np.count_nonzero(either_trigger))

            width_active &= ~width_trigger
            direction_active &= ~direction_trigger
            either_active &= ~either_trigger

        ever_k = int(np.count_nonzero(ever_excluded))
        results.append(
            {
                "p": true_p,
                "replicates": replicates,
                "horizon": n_max,
                "ever_excluded": ever_k,
                "ever_excluded_rate": ever_k / replicates,
                "ever_excluded_se": binomial_standard_error(ever_k, replicates),
                "width_rule_stops": width_stops,
                "width_rule_excluded_at_stop": width_errors,
                "width_rule_rate": width_errors / replicates,
                "width_rule_se": binomial_standard_error(width_errors, replicates),
                "direction_rule_stops": direction_stops,
                "direction_rule_excluded_at_stop": direction_errors,
                "direction_rule_rate": direction_errors / replicates,
                "direction_rule_se": binomial_standard_error(
                    direction_errors, replicates
                ),
                "either_rule_stops": either_stops,
                "either_rule_excluded_at_stop": either_errors,
                "either_rule_rate": either_errors / replicates,
                "either_rule_se": binomial_standard_error(either_errors, replicates),
            }
        )

    return results


def simulate_repeated_cp(
    true_p: float,
    replicates: int,
    n_max: int,
    alpha: float,
    seed: int,
) -> dict[str, float | int]:
    low, high = precompute_cp_intervals(n_max, alpha)
    rng = np.random.default_rng(seed + 777)
    successes = np.zeros(replicates, dtype=np.int64)
    ever_excluded = np.zeros(replicates, dtype=bool)

    for n in range(1, n_max + 1):
        successes += rng.random(replicates) < true_p
        excluded = (true_p < low[n, successes]) | (true_p > high[n, successes])
        ever_excluded |= excluded

    final_excluded = (
        (true_p < low[n_max, successes]) | (true_p > high[n_max, successes])
    )
    ever_k = int(np.count_nonzero(ever_excluded))
    final_k = int(np.count_nonzero(final_excluded))
    return {
        "p": true_p,
        "replicates": replicates,
        "fixed_horizon": n_max,
        "fixed_time_excluded": final_k,
        "fixed_time_rate": final_k / replicates,
        "fixed_time_se": binomial_standard_error(final_k, replicates),
        "repeated_check_excluded": ever_k,
        "repeated_check_rate": ever_k / replicates,
        "repeated_check_se": binomial_standard_error(ever_k, replicates),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=20_000)
    parser.add_argument("--cs-horizon", type=int, default=500)
    parser.add_argument("--cp-horizon", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--output", type=Path, default=Path("simulation_results.json"))
    args = parser.parse_args()

    result = {
        "config": {
            "prior": "Beta(1,1)",
            "a": 1.0,
            "b": 1.0,
            "alpha": 0.05,
            "cs_horizon": args.cs_horizon,
            "cp_horizon": args.cp_horizon,
            "replicates": args.replicates,
            "seed": args.seed,
            "width_rule": "stop when high-low <= 0.30",
            "direction_rule": (
                "stop when low >= 0.60 (declare p>0.5) or high <= 0.40 "
                "(declare p<0.5)"
            ),
            "mc_error": (
                "binomial standard error sqrt(q(1-q)/R); 1.96*SE gives an "
                "approximate 95% Monte Carlo interval"
            ),
        },
        "cs": simulate_cs(
            [0.0, 0.05, 0.20, 0.50, 0.80, 0.95, 1.0],
            args.replicates,
            args.cs_horizon,
            1.0,
            1.0,
            0.05,
            args.seed,
        ),
        "repeated_clopper_pearson": simulate_repeated_cp(
            0.50,
            args.replicates,
            args.cp_horizon,
            0.05,
            args.seed,
        ),
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
