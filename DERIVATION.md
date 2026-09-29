# Bernoulli Beta 混合置信序列：推导与实现

## 1. 设置

设观测 \(X_1,X_2,\ldots\) 在真实参数 \(p\) 下独立同分布于 Bernoulli\((p)\)。预先固定 Beta\((a,b)\)，\(a,b>0\)，以及 \(\alpha\in(0,1)\)。记

\[
S_n=\sum_{t=1}^n X_t,\qquad F_n=n-S_n.
\]

对每个待检验点 \(p_0\in[0,1]\)，定义 Beta 混合似然比

\[
M_n(p_0)=
\int_0^1
\prod_{t=1}^n
\frac{p_0^{X_t}(1-p_0)^{1-X_t}}
{q^{X_t}(1-q)^{1-X_t}}
\frac{q^{a-1}(1-q)^{b-1}}{B(a,b)}\,dq.
\]

## 2. Beta 函数闭式表达

合并相同因子：

\[
\prod_{t=1}^n q^{X_t}(1-q)^{1-X_t}
=q^{S_n}(1-q)^{F_n}.
\]

所以

\[
M_n(p_0)
=
\frac{B(a+S_n,b+F_n)}{B(a,b)}
p_0^{-S_n}(1-p_0)^{-F_n}.
\]

log 域表达式为

\[
\ell_n(p_0)=\log M_n(p_0)
=\log B(a+S_n,b+F_n)-\log B(a,b)
-S_n\log p_0-F_n\log(1-p_0),
\]

其中

\[
\log B(x,y)=\log\Gamma(x)+\log\Gamma(y)-\log\Gamma(x+y).
\]

实现使用 `math.lgamma`，避免 Beta 值、概率乘积和幂次的下溢。

## 3. 鞅、超鞅和边界

把似然比写成 \(M_n(p_0)=\prod_{t=1}^n Z_t(p_0)\)。给定历史 \(\mathcal F_{t-1}\)，

\[
Z_t(p_0)=
\begin{cases}
\dfrac{a+S_{t-1}}{(a+b+t-1)p_0},&X_t=1,\\[1.1ex]
\dfrac{b+F_{t-1}}{(a+b+t-1)(1-p_0)},&X_t=0.
\end{cases}
\]

当 \(0<p_0<1\) 时，在 \(P_{p_0}\) 下

\[
\begin{aligned}
E[Z_t(p_0)\mid\mathcal F_{t-1}]
&=
p_0\frac{a+S_{t-1}}{(a+b+t-1)p_0}
+(1-p_0)\frac{b+F_{t-1}}{(a+b+t-1)(1-p_0)}\\
&=\frac{a+b+S_{t-1}+F_{t-1}}{a+b+t-1}=1.
\end{aligned}
\]

因此 \(M_n(p_0)\) 是从 \(M_0(p_0)=1\) 出发的非负鞅。

边界定义如下：

- \(p_0=0\)：出现成功时 \(\ell_n(0)=+\infty\)；全失败时
  \[
  M_n(0)=\frac{B(a,b+n)}{B(a,b)}
  =\prod_{t=0}^{n-1}\frac{b+t}{a+b+t},
  \]
  其条件期望增量小于 1，是超鞅。
- \(p_0=1\)：出现失败时 \(\ell_n(1)=+\infty\)；全成功时
  \[
  M_n(1)=\frac{B(a+n,b)}{B(a,b)}
  =\prod_{t=0}^{n-1}\frac{a+t}{a+b+t},
  \]
  同样是超鞅。
- 零样本：对所有 \(p_0\)，\(M_0(p_0)=1\)。

概率为零的样本路径不参与边界参数下的期望；似然比在这些点按无穷处理。

## 4. Ville 不等式和反演集合

对从 1 出发的非负鞅，或期望不超过 1 的非负超鞅，Ville 不等式给出

\[
P_{p_0}\!\left(\exists n\ge 0:M_n(p_0)\ge\frac1\alpha\right)\le\alpha.
\]

时刻 \(n\) 的当前反演集合是

\[
C_n(\alpha)=\{p_0\in[0,1]:\ell_n(p_0)\le\log(1/\alpha)\}.
\]

若真实参数为 \(p\)，则

\[
P_p\!\left(\exists n\ge0:p\notin C_n(\alpha)\right)\le\alpha.
\]

因此停止时间可以由已观察数据任意决定。有限停止时间 \(\tau\) 满足

\[
P_p(p\notin C_\tau(\alpha))\le\alpha.
\]

## 5. 区间形状和特殊样本

对内部 \(p_0\)，

\[
\frac{\partial^2\ell_n}{\partial p_0^2}
=\frac{S_n}{p_0^2}+\frac{F_n}{(1-p_0)^2}\ge0,
\]

所以水平阈值反演出的是闭区间。经验频率 \(\widehat p_n=S_n/n\) 是内部最小点。

边界样本规则：

- 零样本：\(C_0=[0,1]\)；
- 全成功：区间为 \([L_n,1]\)；
- 全失败：区间为 \([0,U_n]\)；
- 既有成功又有失败：区间为 \([L_n,U_n]\)。

## 6. 当前集合、固定时刻区间和运行交集

当前集合 \(C_n\) 使用同一个非负鞅和同一个固定阈值 \(1/\alpha\)，因此已经是任意时刻联合有效的置信序列。普通固定时刻 95% 区间只承诺一个预先固定时刻的覆盖；每一步重复查看它并不能得到 95% 同时覆盖。

当前集合不保证逐时刻嵌套。运行交集

\[
R_n=\bigcap_{k=0}^n C_k
\]

单调嵌套，并保持路径覆盖：

\[
P_p(\exists n:p\notin R_n)\le
P_p(\exists n:p\notin C_n)\le\alpha.
\]

代码同时提供 `current` 和 `running_intersection`。

## 7. 保守括区求根

数学接受条件是 \(\ell_n(p_0)\le T\)，其中 \(T=\log(1/\alpha)\)。实现使用二分：

- 下根：从拒绝点 0 到接受点 \(\widehat p_n\)；
- 上根：从接受点 \(\widehat p_n\) 到拒绝点 1；
- 返回最终括区，而不只返回一个浮点估计。

容差是显式参数：

- `x_tolerance`：\(p\) 尺度二分宽度，默认 `1e-12`；
- `log_tolerance`：只有当
  \[
  \ell_n(p_0)>T+\texttt{log\_tolerance}
  \]
  才把外点判定为拒绝，默认 `1e-10`。

下端返回外端拒绝点，上端也返回外端拒绝点，所以数值区间是精确数学集合的向外包络，不会因近似把真实集合向内缩窄。可把 `log_tolerance` 设为 `0.0` 做无额外 log 域缓冲的二分。

## 8. 在线和批量接口

`BetaBernoulliCS.update(x)` 逐条追加；`observe(xs)` 和 `from_observations(xs, ...)` 批量输入。它们都调用同一个 `beta_cs_interval` 纯函数，因此相同序列、先验、\(\alpha\) 和容差产生完全相同的当前集合、运行交集和计数。

