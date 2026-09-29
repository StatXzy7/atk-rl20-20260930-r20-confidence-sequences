# Simulation results

These numbers are finite-simulation diagnostics, not a proof of coverage.

## Configuration

- Beta mixture: `a=1`, `b=1`
- Level: `alpha=0.05`
- Repetitions: 50,000 per row
- Maximum horizon: 1,000
- RNG seed: 20,260,930
- Monte Carlo uncertainty reported as binomial standard error and a two-sided 95% Wilson interval for the estimated exclusion probability.

Stopping rules are data-dependent and use the current CS:

- `upper_below_half`: stop when the current upper endpoint is below 0.5.
- `lower_above_half`: stop when the current lower endpoint is above 0.5.
- `width`: stop when current interval width is below 0.25.

For every replicate, the code also records whether the true `p` was excluded at any time up to stopping or the horizon.

## Beta confidence sequence

| True p | Stop rule | Any-time exclusions | Estimate | Binomial SE | 95% Wilson interval | Mean stop time |
|---:|---|---:|---:|---:|---|---:|
| 0.01 | `upper_below_half` | 644 / 50,000 | 0.01288 | 0.000504 | [0.01193, 0.01391] | 8.33 |
| 0.10 | `upper_below_half` | 1,125 / 50,000 | 0.02250 | 0.000663 | [0.02124, 0.02384] | 12.69 |
| 0.50 | `width` | 1,457 / 50,000 | 0.02914 | 0.000752 | [0.02770, 0.03065] | 164.57 |
| 0.90 | `lower_above_half` | 1,127 / 50,000 | 0.02254 | 0.000664 | [0.02128, 0.02388] | 12.70 |
| 0.99 | `lower_above_half` | 676 / 50,000 | 0.01352 | 0.000516 | [0.01254, 0.01457] | 8.34 |

The targeted one-sided rules can stop very early under extreme true probabilities. Their lower empirical exclusion rates are consistent with using a fixed `alpha=0.05` budget without spending the whole horizon; they are not estimates of a tighter universal bound.

## Control: repeated fixed-time Wilson intervals

At every time, the control uses a standard fixed-time 95% Wilson score interval and checks whether the true `p` has ever been outside. This does not control repeated inspection.

| True p | Any-time exclusions | Estimate | Binomial SE | 95% Wilson interval |
|---:|---:|---:|---:|---|
| 0.01 | 17,988 / 50,000 | 0.35976 | 0.002146 | [0.35556, 0.36398] |
| 0.10 | 24,942 / 50,000 | 0.49884 | 0.002236 | [0.49446, 0.50323] |
| 0.50 | 26,239 / 50,000 | 0.52478 | 0.002233 | [0.52040, 0.52916] |
| 0.90 | 24,510 / 50,000 | 0.49020 | 0.002236 | [0.48582, 0.49458] |
| 0.99 | 18,302 / 50,000 | 0.36604 | 0.002154 | [0.36183, 0.37027] |

The raw machine-readable output is in `results/simulation.json`.
