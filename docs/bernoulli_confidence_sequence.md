# Beta-mixed Bernoulli confidence sequence

## 1. 设定与目标

设

\[
X_t\mid p\sim \operatorname{Bernoulli}(p),\qquad X_t \text{ i.i.d.},\quad p\in[0,1],
\]

并记

\[
S_n=\sum_{t=1}^n X_t,\qquad F_n=n-S_n.
\]

目标是构造随时间更新的集合 \(C_n\)，使任意、可能依赖于到当时为止全部数据的停止时间 \(\tau\)（允许 \(\tau=\infty\)）都满足

\[
\Pr_p\!\left(p\in C_\tau\right)\ge 1-\alpha,
\]

以及整条路径上的同时保证

\[
\Pr_p\!\left(\exists n\ge 0:\ p\notin C_n\right)\le \alpha.
\]

固定时刻 95% 区间只控制单个时刻，不能在每个时刻重复查看后仍声称 95% 同时覆盖。

超参数 \(a,b>0\)、显著性水平 \(\alpha\in(0,1)\) 必须在看到数据前固定。实现默认使用 \(a=b=1\)，即均匀先验，但用户可构造任意 Beta 混合。

## 2. Beta 混合似然与闭式边际

取混合分布

\[
Q=\operatorname{Beta}(a,b),\qquad
q(u)=\frac{u^{a-1}(1-u)^{b-1}}{B(a,b)}.
\]

对候选原假设 \(p_0\in[0,1]\)，在样本 \((S_n,F_n)=(s,f)\) 上定义混合似然比

\[
L_n(p_0)
=
\int_{[0,1]}
\frac{p(X_1,\ldots,X_n\mid u)}
     {p(X_1,\ldots,X_n\mid p_0)}
q(u)\,du.
\]

对 \(0<p_0<1\)，

\[
L_n(p_0)
=
\frac{
\int_0^1 u^{a+s-1}(1-u)^{b+f-1}\,du/B(a,b)
}{
p_0^s(1-p_0)^f
}.
\]

用 Beta 函数 \(B(x,y)=\Gamma(x)\Gamma(y)/\Gamma(x+y)\)，得到闭式表达

\[
\boxed{
L_n(p_0)
=
\frac{B(a+s,b+f)}{B(a,b)\,p_0^s(1-p_0)^f}
}
\qquad (0<p_0<1).
\]

因此对数似然比为

\[
\log L_n(p_0)
=
\log B(a+s,b+f)-\log B(a,b)
-s\log p_0-f\log(1-p_0).
\]

代码用 `math.lgamma` 计算

\[
\log B(x,y)=\log\Gamma(x)+\log\Gamma(y)-\log\Gamma(x+y),
\]

从而在长序列中避免直接乘以极小概率导致的下溢。

### 边界 \(p_0=0,1\)

在 \(p_0=0\) 下，任何成功都是零概率事件；在 \(p_0=1\) 下，任何失败都是零概率事件。定义为

\[
L_n(0)=
\begin{cases}
\dfrac{B(a+s,b+f)}{B(a,b)},&s=0,\\
+\infty,&s>0,
\end{cases}
\]

以及

\[
L_n(1)=
\begin{cases}
\dfrac{B(a+s,b+f)}{B(a,b)},&f=0,\\
+\infty,&f>0.
\end{cases}
\]

全失败时 \(B(a,b+n)/B(a,b)\) 是 Beta(a,b) 对该全失败序列的预测概率；全成功同理。零样本时

\[
L_0(p_0)=\frac{B(a,b)}{B(a,b)}=1
\]

对所有 \(p_0\in[0,1]\) 都成立，因此初始集合是整个参数空间 \([0,1]\)。

## 3. 非负鞅或超鞅

对内部点 \(0<p_0<1\)，令

\[
Q(A)=\int_A q(u)\,du.
\]

则

\[
L_n(p_0)=\frac{dQ_n}{dP_{p_0,n}}(\mathcal F_n),
\]

其中 \(P_{p_0,n}\) 是前 \(n\) 次 Bernoulli 观测的分布，\(Q_n\) 是先对 \(u\sim Q\)、再令观测服从 Bernoulli(u) 的混合分布。

也可以直接用 Beta 后验预测验证。给定 \(s,f\)，后验为

\[
u\mid\mathcal F_n\sim\operatorname{Beta}(a+s,b+f),
\]

下一次成功的混合预测概率为

\[
\pi_{n+1}(1)=\frac{a+s}{a+b+n},
\qquad
\pi_{n+1}(0)=\frac{b+f}{a+b+n}.
\]

于是

\[
\frac{L_{n+1}(p_0)}{L_n(p_0)}
=
\begin{cases}
\dfrac{a+s}{(a+b+n)p_0},&X_{n+1}=1,\\[1ex]
\dfrac{b+f}{(a+b+n)(1-p_0)},&X_{n+1}=0.
\end{cases}
\]

在 \(P_{p_0}\) 下取条件期望：

\[
\begin{aligned}
E_{p_0}\!\left[\frac{L_{n+1}}{L_n}\mid\mathcal F_n\right]
&=
p_0\frac{a+s}{(a+b+n)p_0}
+(1-p_0)\frac{b+f}{(a+b+n)(1-p_0)}\\
&=\frac{a+s+b+f}{a+b+n}=1.
\end{aligned}
\]

所以 \(L_n(p_0)\) 在 \(0<p_0<1\) 时是以 1 为初值的非负鞅。

对边界 \(p_0=0\)：

- 若已经出现成功，原假设路径概率为零，\(L_n(0)=+\infty\)；
- 若尚未出现成功，则 \(L_n(0)=B(a,b+n)/B(a,b)\)，且
  \[
  E_0[L_{n+1}(0)\mid\mathcal F_n]=L_n(0)\frac{b+n}{a+b+n}\le L_n(0).
  \]

因此它在 \(p_0=0\) 下是非负超鞅。\(p_0=1\) 的论证完全对称。

## 4. Ville 不等式与反演

对非负（超）鞅 \(L_n(p_0)\)，Ville 不等式给出

\[
\Pr_{p_0}\!\left(\sup_{n\ge 0}L_n(p_0)\ge \frac1\alpha\right)\le \alpha.
\]

因而定义

\[
\boxed{
C_n(\alpha)
=
\left\{p_0\in[0,1]:
L_n(p_0)<\frac1\alpha
\right\}
=
\left\{p_0:\log L_n(p_0)<\log(1/\alpha)\right\}.
}
\]

对每个固定的真参数 \(p\)，

\[
\Pr_p\!\left(\exists n\ge 0:p\notin C_n\right)
\le
\Pr_p\!\left(\sup_{n\ge0}L_n(p)\ge1/\alpha\right)
\le\alpha.
\]

这个界不依赖于停止规则。特别地，若 \(\tau\) 根据 \(\mathcal F_\tau\) 任意选择，则

\[
\Pr_p(p\notin C_\tau)\le\alpha.
\]

数值实现中用 `<=` 与带容差判断来保留边界；额外包含等值点不会破坏覆盖界。

## 5. 反演集合的形状

对 \(0<p_0<1\)，

\[
\log L_n(p_0)
=
\log B(a+s,b+f)-\log B(a,b)
-s\log p_0-f\log(1-p_0).
\]

固定数据后，\(-s\log p_0-f\log(1-p_0)\) 是 \(p_0\) 的凸函数，因此对水平 \(\log(1/\alpha)\) 的下水平集是区间。实现返回：

- `s=f=0`：\([0,1]\)；
- `s>0,f>0`：两个根都在 \((0,1)\)；
- `s>0,f=0`：右端点固定为 1；
- `s=0,f>0`：左端点固定为 0。

对内部样本，MLE \(\hat p=s/n\) 处的似然比不大于 1。对全成功路径，\(L_n(1)=B(a+n,b)/B(a,b)\le1\)；对全失败路径，\(L_n(0)=B(a,b+n)/B(a,b)\le1\)。因为拒绝阈值 \(1/\alpha>1\)，所以这种 Beta 混合反演在 \(0<\alpha<1\) 下总是至少保留 MLE 或相容边界，单时刻理论集合不为空。若内部括区无法验证，数值实现会向外退回到 \([0,1]\)。

## 6. 数值求根与向外包络

令

\[
g(p)=\log L_n(p)-\log(1/\alpha).
\]

接受条件为 \(g(p)\le0\)。实现对每一侧维护带括区：

- 左根括区 \((\ell_L,\ell_H)\) 满足 \(\ell_L\) 在拒绝侧、\(\ell_H\) 在接受侧；
- 右根括区 \((r_L,r_H)\) 满足 \(r_L\) 在接受侧、\(r_H\) 在拒绝侧。

返回的可声明区间为

\[
[\ell_L,r_H].
\]

因此若真实根位于数值括区内，浮点误差只会向区间外侧扩张，不会把真实理论集合缩窄。返回对象还给出两个内部括区和

\[
\text{declared tolerance}
=
\max(\text{tol},\ \ell_H-\ell_L,\ r_H-r_L,\ 8\varepsilon_{\text{mach}}).
\]

默认 `tol=1e-10`，二分次数 `max_iter=60`。若内部括区符号检查失败（例如极端参数或浮点不可靠），实现保守退回 \([0,1]\)，而不是返回未验证的窄区间。

## 7. 单时刻集合与运行交集

`confidence_set()` 返回的是当前时刻的反演集合 \(C_n\)。它本身已经享有全路径覆盖：真参数在任何时刻被排除的概率不超过 \(\alpha\)。

`running_intersection()` 返回

\[
R_n=\bigcap_{k=0}^n C_k.
\]

二者不同：\(C_n\) 只使用当前反演，而 \(R_n\) 还记住所有此前集合。由定义，

\[
R_{n+1}=R_n\cap C_{n+1}\subseteq R_n,
\]

所以运行交集的左端点单调非降、右端点单调非升；若为空则继续为空。代码通过回放全部前缀来保证“逐个追加观测”和“一次批量输入后取最终运行交集”得到相同结果。若只需要当前集合，批量输入不会额外支付运行交集的回放成本。

运行交集也满足覆盖，因为

\[
\{\exists n:p\notin R_n\}
=
\{\exists n:p\notin C_n\}.
\]

## 8. 实现接口

```python
from bernoulli_cs import BetaBernoulliCS

model = BetaBernoulliCS(a=1.0, b=1.0, alpha=0.05)
model.observe(1)
model.observe_batch([0, 1, 1, 0])

cs = model.confidence_set()
print(cs.low, cs.high, cs.numerical_tolerance)

running = model.running_intersection()
print(running.low, running.high, running.empty)
```

公开函数还包括：

- `log_beta(a, b)`：log Beta 函数；
- `log_marginal_evidence(s, f, a, b)`：混合边际概率的对数；
- `log_likelihood_ratio(p0, s, f, a, b)`：任意 \(p_0\in[0,1]\) 的对数混合似然比；
- `accepts(p0)`：直接检验某个参数点是否在当前集合中。

## 9. 证明与模拟证据的地位

第 2–4 节给出的是覆盖保证的数学证明：它对任意停止时间和所有 \(n\) 同时成立。

后面的蒙特卡洛模拟只是有限样本的经验检查，用来发现实现错误并展示与重复固定时刻区间的差别。无论多少次模拟都不能替代上述 Ville 覆盖定理；模拟误差也必须用二项式抽样误差报告。
