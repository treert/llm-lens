# 随机投影的保距性：Johnson–Lindenstrauss 引理

前置：[rmsnorm.md](rmsnorm.md) §2（$\chi^2$ 长度集中——本文与之共用同一套数学机制）、
[attention-score-distribution.md](attention-score-distribution.md)（$1/D$ 初始化与 $\div\sqrt{H}$ 缩放）。

一句话：**用 $N$ 个随机向量把 $D$ 维向量投影到 $N$ 维，只要缩放得当且 $N$ 足够大，
长度（进而距离、夹角）在单样本层面也几乎不变**——且所需 $N$ 与原维度 $D$ 无关，
只依赖点数和精度。

## 1. 设定、归一化约定与 LLM 的联系

随机矩阵 $R \in \mathbb{R}^{N \times D}$，元素 iid 零均值。方差取多少，决定"保"的是什么：

| 约定 | 元素方差 | 效果 |
|---|---|---|
| A（保距 / JL） | $1/N$ | $\mathbb{E}\|Rx\|^2 = \|x\|^2$，$N$ 大时逐样本保距（本文主线） |
| B（随机子空间） | $1/D$ | 行近单位正交，$R$ 近似"向随机 $N$ 维子空间投影"，长度缩 $N/D$（§5） |

两者只差整体常数：约定 A $= \sqrt{D/N} \times$ 约定 B。

### 与 LLM 初始化的关系：保距 ≠ 保分量方差

attention 的 $W_Q$ 也用 $1/D$ 缩放（约定 B），但目的不同：

$$q = W_Q A, \qquad \mathbb{E}\|q\|^2 = \frac{H}{D}\,\|A\|^2$$

向量长度缩为 $H/D$ 倍，**不是**保距变换。它保的是**分量方差**（$\sigma^2 \to \sigma^2$），
因为下游关心分量尺度（要点积、要进残差流）。若要保距需用 $1/H$ 缩放。

同一条"高维平方和集中"原理，三种缩放常数三种用途：

| 缩放 | 场景 | 保什么 |
|---|---|---|
| 元素方差 $1/N$ | JL 随机投影 | 向量长度 / 距离 / 夹角 |
| 元素方差 $1/D$ | 线性层、attention 初始化（一方固定） | 分量方差 |
| $\div\sqrt{H}$ | 点积（双方随机） | 分数方差 |

以下默认约定 A：$R_{ij} \overset{iid}{\sim} \mathcal{N}(0,\, 1/N)$，$x \in \mathbb{R}^D$ 为任意固定向量。

## 2. 期望层面：精确保距

**命题**：$\mathbb{E}\|Rx\|^2 = \|x\|^2$（不依赖高斯假设，零均值、方差 $1/N$、独立即可）。

**推导**：第 $i$ 个输出分量是输入的随机加权和：

$$(Rx)_i = \sum_{j=1}^{D} R_{ij}\, x_j$$

平方取期望，交叉项因独立性归零（$\mathbb{E}[R_{ij}R_{ik}] = \frac{1}{N}\delta_{jk}$）：

$$\mathbb{E}[(Rx)_i^2] = \sum_{j,k} \mathbb{E}[R_{ij}R_{ik}]\, x_j x_k
= \frac{1}{N}\sum_{j=1}^{D} x_j^2 = \frac{\|x\|^2}{N}$$

对 $i$ 求和（共 $N$ 项）：

$$\mathbb{E}\|Rx\|^2 = N\cdot\frac{\|x\|^2}{N} = \|x\|^2 \qquad\blacksquare$$

**推论（保内积与夹角）**：同法可得 $\mathbb{E}\langle Rx, Ry\rangle = \langle x, y\rangle$
（把 $x_j x_k$ 换成 $x_j y_k$）。
长度、距离（对 $x-y$ 用命题）、内积、夹角在期望层面全部精确保持。

## 3. 单样本层面：$\chi^2$ 集中

期望保距不够——随机投影要"有用"，需要**每次实现**都近似保距。这正是
[rmsnorm.md](rmsnorm.md) §2 的长度集中在这里的重演。

**命题**：对固定 $x$（条件于 $x$，随机性全在 $R$），

$$\|Rx\|^2 \;\overset{d}{=}\; \frac{\|x\|^2}{N}\, Z, \qquad Z \sim \chi^2(N)$$

**推导**：由 §2，$(Rx)_i$ 是独立高斯的加权和，故仍是高斯，$(Rx)_i \sim \mathcal{N}(0,\, \|x\|^2/N)$；
不同 $i$ 使用 $R$ 的不同行，相互独立。于是

$$\|Rx\|^2 = \sum_{i=1}^{N} (Rx)_i^2
= \frac{\|x\|^2}{N}\sum_{i=1}^{N} Z_i^2, \qquad Z_i \overset{iid}{\sim} \mathcal{N}(0,1)
\qquad\blacksquare$$

由 $\mathbb{E}[\chi^2(N)] = N$、$\mathrm{Var}(\chi^2(N)) = 2N$：

$$\mathrm{Var}(\|Rx\|^2) = \frac{\|x\|^4}{N^2}\cdot 2N = \frac{2\|x\|^4}{N},
\qquad \frac{\mathrm{std}(\|Rx\|^2)}{\mathbb{E}\|Rx\|^2} = \sqrt{\frac{2}{N}}$$

开方传到 $\|Rx\|$ 上（delta 方法）减半：相对涨落 $\approx 1/\sqrt{2N}$。
**与 rmsnorm.md §2 的涨落表是同一个表**，$D$ 换成 $N$ 即可：

| $N$ | 64 | 1024 | 4096 |
|---|---|---|---|
| $\|Rx\|$ 相对涨落 | 8.8% | 2.2% | 1.1% |

**尾概率**：$\chi^2$ 集中不等式（Laurent–Massart）给出，对 $0 < \varepsilon < 1$，

$$\Pr\!\left[\,\bigl|\|Rx\|^2 - \|x\|^2\bigr| \ge \varepsilon\|x\|^2\,\right]
\le 2\exp\!\left(-N\Bigl(\frac{\varepsilon^2}{4}-\frac{\varepsilon^3}{6}\Bigr)\right)$$

失败概率随 $N$ **指数**衰减——这是 §4 能 union bound 的关键。

## 4. JL 引理：$n$ 个点只需 $O(\varepsilon^{-2}\log n)$ 维

**JL 引理（1984）**：任给 $\mathbb{R}^D$ 中 $n$ 个点与 $0 < \varepsilon < 1$，只要

$$N \;\ge\; c\,\varepsilon^{-2}\ln n \qquad（c \text{ 为绝对常数}）$$

就存在线性映射（事实上随机 $R$ 即以高概率是）把所有两两距离保到 $(1 \pm \varepsilon)$ 内。

**证明骨架**：

1. 点对 $(u, v)$ 的距离 = 差向量的长度 $\|u-v\|$，对差向量用 §3 的尾概率，
   单对失败概率 $\le 2\exp\!\left(-N(\frac{\varepsilon^2}{4} - \frac{\varepsilon^3}{6})\right)$；
2. 共 $\binom{n}{2} \approx n^2/2$ 对，union bound：总失败概率
   $\le n^2 \exp\!\left(-N(\frac{\varepsilon^2}{4} - \frac{\varepsilon^3}{6})\right)$；
3. 取 $N \ge c\,\varepsilon^{-2}\ln n$ 使上式 $< 1$——不仅"存在"，而且
   **随机取一个 $R$ 就大概率成功**。

两个注记：

- **$D$ 彻底消失**：所需维度只看点数 $n$ 和精度 $\varepsilon$。$n$ 个点的"距离信息"本质上是
  $n^2$ 个数，指数级集中 + union bound 把代价压成 $\log n$；
- **最优性**：$N = \Omega(\varepsilon^{-2}\log n)$ 已被证明不可改进（Larsen & Nelson 2017），
  即随机投影做到了理论极限。工程变体有稀疏 JL、Fast JL（FFT 加速）等。

## 5. 等价视角：随机向量近正交

换约定 B（元素方差 $1/D$）看同一件事。$R$ 的第 $i$ 行 $r_i \in \mathbb{R}^D$：

- **行长度集中**：$\mathbb{E}\|r_i\|^2 = 1$，$\chi^2(D)$ 集中，相对涨落 $\sqrt{2/D}$；
- **行间近正交**：$i \ne j$ 时 $r_i \cdot r_j$ 是 $D$ 个独立乘积之和，
  $\mathrm{std} \approx 1/\sqrt{D}$ → 高维下两两内积 $\approx 0$；
- 于是 $RR^\top \approx I_N$：$R$ 近似"先向随机 $N$ 维子空间正交投影、再换正交坐标"。
  投影丢弃正交补方向，故 $\mathbb{E}\|Rx\|^2 = \frac{N}{D}\|x\|^2$——乘回 $\sqrt{D/N}$ 即回到约定 A。

更精确的谱刻画（Marchenko–Pastur）：$N, D \to \infty$、$N/D \to c \le 1$ 时，
$R$ 的奇异值落在 $[1-\sqrt{c},\; 1+\sqrt{c}]$ 内。$c \to 0$（即 $N \ll D$）时全部奇异值 $\to 1$：
随机矩阵在其值域上**一致地**近似等距。这正是 [qk-spectrum.md](qk-spectrum.md) §2
所用"随机矩阵谱集中"思想的另一化身。

## 6. 速查

- 单点保距：$\|Rx\|^2 \sim \frac{\|x\|^2}{N}\chi^2(N)$，相对涨落 $\sqrt{2/N}$（长度则 $1/\sqrt{2N}$）；
- $n$ 点保距：$N = O(\varepsilon^{-2}\log n)$，与 $D$ 无关，且不可改进；
- 结构性看法：随机行近正交（$RR^\top \approx I$），$c = N/D \to 0$ 时奇异值全 $\to 1$（MP）；
- 与 LLM 的联系：初始化用 $1/D$ 保分量方差，与 JL 保距差一个 $D/H$ 的缩放常数。
