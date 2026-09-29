# Bernoulli beta-mixed confidence sequence

This repository implements an any-time Bernoulli confidence sequence from a fixed, pre-specified Beta(a,b) likelihood-ratio mixture.

- Derivation and coverage proof: `docs/bernoulli_confidence_sequence.md`
- Pure-Python implementation: `bernoulli_cs/confidence_sequence.py`
- Unit and boundary tests: `tests/test_confidence_sequence.py`
- Monte Carlo and fixed-time-interval control: `experiments/run_simulation.py`
- Reported results: `results/README.md` and `results/simulation.json`

## Quick start

```python
from bernoulli_cs import BetaBernoulliCS

monitor = BetaBernoulliCS(a=1.0, b=1.0, alpha=0.05)
monitor.observe(1)
monitor.observe_batch([0, 1, 1, 0])

cs = monitor.confidence_set()
print(cs.low, cs.high, cs.numerical_tolerance)

running = monitor.running_intersection()
print(running.low, running.high, running.empty)
```

## Tests and simulation

```bash
python -m pytest -q
python experiments/run_simulation.py --samples 50000 --horizon 1000
```
