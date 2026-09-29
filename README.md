# Bernoulli Confidence Sequence

固定 Beta(a,b) 混合似然比构造的 Bernoulli 任意时刻置信序列。

## 文件

- `bernoulli_cs.py`：标准库实现的 log-Beta 闭式、保守括区求根、在线/批量接口；
- `test_bernoulli_cs.py`：鞅恒等式、边界、数值括区、嵌套性和接口一致性测试；
- `simulate_cs.py`：NumPy/SciPy 蒙特卡洛和 Clopper–Pearson 对照；
- `DERIVATION.md`：完整推导、边界定义和数值包络说明；
- `RESULTS.md`：实际测试和模拟结果；
- `simulation_results.json`：一次默认运行的原始结果。

## 使用

```python
from bernoulli_cs import BetaBernoulliCS

tracker = BetaBernoulliCS(a=1.0, b=1.0, alpha=0.05)
tracker.observe([1, 0, 1, 1])

print(tracker.current.low, tracker.current.high)
print(tracker.running_intersection.low, tracker.running_intersection.high)
```

逐条追加：

```python
tracker = BetaBernoulliCS(1.0, 1.0, 0.05)
for x in [1, 0, 1, 1]:
    tracker.update(x)
```

一次批量输入与逐条追加调用同一数值核心，结果相同。

## 验证

```bash
python3 -m pytest -q
python3 simulate_cs.py \
  --replicates 20000 \
  --cs-horizon 500 \
  --cp-horizon 200 \
  --seed 20260930
```
