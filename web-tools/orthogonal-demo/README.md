# orthogonal-demo：高维随机向量的最小夹角

交互式网页工具，演示"高维空间中随机向量近乎正交"的定量规律：

- **曲线 1**：固定 N，max ρ（两两点乘最大值 = 最小夹角的余弦）随 K 变化的期望水平，
  对比一阶近似 `2√(lnK/N)` 与 Gumbel 极值修正（中位数）。
- **曲线 2**：给定 (N, K)，max ρ 的近似密度（Gumbel 渐近），叠加单对点乘的精确
  beta 密度作对比。

理论来源：Cai, Fan, Jiang, *Distributions of Angles in Random Packing on Spheres*,
JMLR 14 (2013), arXiv:1306.0256。

## 用法

直接用浏览器打开 `index.html` 即可（ECharts 走 CDN，需要联网）。
如需完全离线，下载 `echarts.min.js` 到本目录并修改 `index.html` 中的 `<script src>`。

## 文件结构

- `index.html`：页面骨架
- `js/theory.js`：纯数学公式层（lgamma、beta 密度、Gumbel 极值分布、分位数），无 DOM 依赖
- `js/app.js`：UI 状态同步与 ECharts 渲染
- `css/style.css`：样式

---

## 数学背景

### 0. 问题设定

在 N 维单位球面 $S^{N-1}$ 上独立均匀地取 K 个向量 $x_1,\dots,x_K$，
记两两点乘（= 夹角余弦）

$$\rho_{ij} = \langle x_i, x_j \rangle = \cos\Theta_{ij}, \qquad M = \binom{K}{2} \text{ 对组合}$$

关心两个等价的极值量：

$$\max_{i<j} \rho_{ij} = \cos\Theta_{\min} \quad\text{（单侧，对应最小夹角）}, \qquad
\max_{i<j} |\rho_{ij}| \quad\text{（双侧，即压缩感知中的 mutual coherence）}$$

### 1. 单对点乘的精确分布（曲线 2 中的"精确 beta"线）

固定一个向量，另一个均匀随机，则 $\rho = \cos\theta$ 有**精确**密度（论文引理 6.1）：

$$g(\rho) = \frac{\Gamma(N/2)}{\sqrt{\pi}\,\Gamma\!\left(\frac{N-1}{2}\right)}\,(1-\rho^2)^{\frac{N-3}{2}}, \qquad |\rho| < 1$$

要点：

- 等价表述：$\rho^2 \sim \mathrm{Beta}\!\left(\tfrac12, \tfrac{N-1}{2}\right)$。
- 几何意义：$P(\rho > t)$ 就是球面被一个球冠截取的面积占比；因子 $(1-\rho^2)^{(N-3)/2}$
  表明 N 越大，质量越向"赤道"（$\rho=0$，即 90°）集中——这是"高维近乎正交"的根源。
- 大 N 近似：$\rho \approx \mathcal N(0,\,1/N)$，即散布 $\sigma \approx 1/\sqrt{N}$。
- 实现细节（`theory.js`）：归一化常数用 Lanczos 近似的 `lgamma` 在对数域计算，
  避免 N 很大时 $\Gamma$ 函数溢出。

### 2. 极值的渐近分布（Gumbel）

单对尾部在大 N 下有高斯型渐近 $P(\rho > t) \approx \frac{1}{t\sqrt{2\pi N}} e^{-N t^2/2}$。
M 对组合虽只是两两独立（不相互独立），但 Chen–Stein 方法可证明其极值行为与独立情形一致
（论文定理 5，条件 $\ln K = o(N)$）：

$$W = N\,(\max \rho)^2 - a(K), \qquad a(K) = 4\ln K - \ln\ln K$$

$$P(W \le y) \;\longrightarrow\; \exp\!\left(-\kappa\, e^{-y/2}\right), \qquad
\kappa = \begin{cases} \dfrac{1}{4\sqrt{2\pi}} & \text{单侧 } \max\rho \\[6pt]
\dfrac{1}{2\sqrt{2\pi}} & \text{双侧 } \max|\rho| \end{cases}$$

即 W 收敛到**标准 Gumbel 分布**（位置参数 $2\ln\kappa$、尺度 2）。
双侧常数是单侧的两倍：正负两条对称的尾巴各自竞逐极值，极端机会翻倍；
效果是把分布平移 $2\ln 2$，即 $(\max|\rho|)^2$ 比 $(\max\rho)^2$ 大约大 $2\ln 2 / N$。

> 注意：论文原文给出的是 $2N\log\sin\Theta_{\min} + 4\ln K - \ln\ln K$ 的极限
> $1 - e^{-\kappa e^{y/2}}$；由 $\log\sin\Theta_{\min} \approx -(\max\rho)^2/2$ 可知
> 它就是 $-W$ 的分布，整理后即上面的标准形式。

由此推出工具中使用的三个量：

**分位数函数**（对任意 $p \in (0,1)$；曲线 1 取 $p=0.5$ 即中位数，曲线 2 横轴上限取
$p=0.999$）：由 $F = p$ 反解
$y = -2\ln\!\left(\frac{-\ln p}{\kappa}\right)$，故

$$Q(p) \;=\; \sqrt{\frac{a(K) - 2\ln\!\left(\frac{-\ln p}{\kappa}\right)}{N}},
\qquad \text{中位数 } = Q(0.5)$$

**max ρ 的密度**（曲线 2 的"Gumbel 渐近"线）：对 $R = \sqrt{(W + a)/N}$ 做变量替换，
$f_R(r) = 2N r \cdot f_W(Nr^2 - a)$，其中 $f_W(y) = \frac{\kappa}{2} e^{-y/2} \exp(-\kappa e^{-y/2})$，
即

$$f_R(r) = N\kappa\, r\, e^{-\frac{N r^2 - a}{2}} \exp\!\left(-\kappa\, e^{-\frac{N r^2 - a}{2}}\right)$$

**近似均值**（统计栏）：标准 Gumbel（尺度 2）的均值为 $E[W] = 2(\gamma + \ln\kappa)$，
$\gamma \approx 0.5772$ 为 Euler–Mascheroni 常数，故

$$E[\max\rho] \approx \sqrt{\frac{a(K) + 2(\gamma + \ln\kappa)}{N}}$$

（严格说 $\sqrt{E[R^2]} \ge E[R]$，此式以 $\sqrt{E[R^2]}$ 代替 $E[R]$，仅作参考。）

### 3. 一阶近似（曲线 1 的"2√(lnK/N)"线）

把 M 对点乘当作独立 $\mathcal N(0, 1/N)$，用高斯最大值启发式：令
$M \cdot P(\rho > t) \approx 1$，只保留指数主项 $e^{-Nt^2/2}$，得

$$\frac{Nt^2}{2} \approx \ln M \approx 2\ln K \quad\Longrightarrow\quad
E[\max\rho] \approx \sqrt{\frac{2\ln M}{N}} \approx 2\sqrt{\frac{\ln K}{N}}$$

与 Gumbel 精确渐近对比：它丢掉了 $1/t$ 因子产生的 $-\ln\ln K$ 修正和分布位置常数，
因此**系统性偏高**（蒙特卡洛验证中约高 15%~25%）。它的价值在于形式极简，
一眼给出"数量级随 $\sqrt{\ln K / N}$ 缩放"的直觉。

### 4. 适用范围与失真

#### 4.1 渐近条件 $\ln K = o(N)$ 的含义

定理要求 K、N 同时趋向无穷时 **ln K 增长得比 N 慢一个量级**，即

$$ \frac{\ln K}{N} \longrightarrow 0 $$

注意它限制的是 **K 的对数**而非 K 本身，所以 K 其实可以非常大：

| K 随 N 的增长方式 | ln K | 是否满足 o(N) |
|---|---|---|
| $K = N^2$（多项式） | $2\ln N$ | ✓ |
| $K = e^{\sqrt{N}}$ | $\sqrt{N}$ | ✓ |
| $K = e^{0.3N}$（真指数） | $0.3N$ | ✗（比值为常数） |
| $K = e^{N^2}$ | $N^2$ | ✗（比值发散） |

**为什么需要这个条件**：由一阶近似 $\max\rho \approx 2\sqrt{\ln K / N}$ 可见，
$\ln K = o(N)$ 恰好等价于 $\max\rho \to 0$，即所有夹角收敛到 90°——这正是
"近乎正交"成立、且 Gumbel 推导（围绕 $\rho=0$ 展开）有效的区域。
论文据此划分三个渐近区域，各有不同的极限定理：

1. **$\ln K / N \to 0$（次指数区域）**：max ρ → 0，夹角全部挤在 90° 附近，
   $N(\max\rho)^2 - 4\ln K + \ln\ln K$ 收敛到 Gumbel。**本工具画的就是这个区域**。
2. **$\ln K / N \to \beta > 0$（指数区域）**：max ρ 不再趋于 0，而是收敛到正常数
   $\sqrt{1 - e^{-4\beta}}$，最小夹角趋于 $\arccos\sqrt{1-e^{-4\beta}} < 90°$
   （论文定理 6）——向量多到"装不下"，必然出现实质性的接近。本工具未覆盖。
3. **$\ln K / N \to \infty$（超指数区域）**：最小夹角 → 0°（论文定理 7）。

**对照实际场景**（LLM 词嵌入）：词表 $K \approx 10^5$、维数 $N = 4096$ 时
$\ln K / N \approx 11.5 / 4096 \approx 0.003$，深在次指数区域内部，
故 Gumbel 公式严格适用：任意两个词嵌入夹角大概率不小于
$\arccos\!\left(2\sqrt{11.5/4096}\right) \approx 83.9°$。

#### 4.2 小 K 失真

渐近定理还要求 K 足够大（$M = \binom{K}{2}$ 足够大，极值才进入 Gumbel 吸引域）。
实测 $K \gtrsim 20$ 时预测与蒙特卡洛误差已在 2% 以内；K 为个位数时明显失真
（极端例子 K=2：只有一对，单侧 max ρ 的中位数精确等于 0）。
若需对小 K 也精确，可用 $P(\max\rho \le t) \approx F_{\mathrm{beta}}(t)^M$
（精确 beta CDF 的 M 次方 + 独立性近似）数值求解，大 K 时与 Gumbel 自然衔接。

### 参考文献

- T. Cai, J. Fan, T. Jiang. *Distributions of Angles in Random Packing on Spheres*.
  JMLR 14 (2013). arXiv:1306.0256 —— 最小角/最大点乘的极限分布（本文核心）。
- T. Cai, T. Jiang. *Limiting laws of coherence of random matrices with applications
  to testing covariance structure and construction of compressed sensing matrices*.
  Ann. Statist. 39 (2011) —— 双侧 coherence 的极限定理。
