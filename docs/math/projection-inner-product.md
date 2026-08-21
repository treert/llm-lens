# 投影后的点积：同一矩阵与两个矩阵

通用数学笔记：固定向量经随机矩阵投影后，点积的期望、方差与集中性，
核心对比是**两次投影用同一个矩阵**与**用两个独立矩阵**。
同一矩阵的保距（范数）主线见 `docs/jl-projection.md`；
两个矩阵 + 随机输入视角的注意力分数分布见 `docs/attention-score-distribution.md`；
$M = A^\top B$ 奇异值谱的极限律见 `docs/math/random-matrix-spectra.md`。

## 0. 设定与记号

- $x, y \in \mathbb{R}^D$：**固定**向量（随机性全在矩阵）， $s = \langle x, y\rangle$；
- $A, B \in \mathbb{R}^{H \times D}$：元素 iid $\mathcal{N}(0, \sigma_w^2)$， $A \perp B$；
- 投影后点积： $\tilde s = \langle Ax, Ay\rangle$（同一矩阵）或 $\langle Ax, By\rangle$（两个矩阵）；
- 两种常用缩放（即 `docs/jl-projection.md` §1 的约定 A/B）：
  - **JL 缩放** $\sigma_w^2 = 1/H$： $\mathbb{E}\tilde s = s$，保内积；
  - **LLM 缩放** $\sigma_w^2 = 1/D$：保分量方差，对应权重初始化
    （ $H \leftrightarrow d_k$， $D \leftrightarrow d_{\mathrm{model}}$， $c = H/D$）。

## 1. 核心恒等式：按行展开成 iid 求和

$$\tilde s = x^\top M y, \qquad M = A^\top B \in \mathbb{R}^{D \times D}$$

（同一矩阵即 $B = A$。）把 $M$ 按行拆成 $H$ 个秩-1 矩阵之和，
$\tilde s$ 随之拆成 **$H$ 个 iid 标量之和**：

$$\tilde s = \sum_{i=1}^{H} \langle a_i, x\rangle\,\langle b_i, y\rangle,
\qquad a_i, b_i \text{ 为 } A, B \text{ 的第 } i \text{ 行}$$

记 $U = \langle a, x\rangle$、 $V = \langle b, y\rangle$，它们是零均值联合高斯，全部信息在三矩：

$$\mathbb{E}[U^2] = \sigma_w^2\|x\|^2, \qquad
\mathbb{E}[V^2] = \sigma_w^2\|y\|^2, \qquad
\mathbb{E}[UV] = \begin{cases} \sigma_w^2\, s & a = b\ \text{（同一矩阵）} \\ 0 & a \perp b\ \text{（独立矩阵）} \end{cases}$$

iid 求和 ⇒ 期望、方差都 $\times H$； $H \to \infty$ 时 $\tilde s$ 渐近高斯（CLT）。
后面所有结论都由这一条推出。

## 2. 同一矩阵：期望保留，方差 $\sim 1/H$

$\mathbb{E}[A^\top A] = H\sigma_w^2 I_D$，故

$$\mathbb{E}[\tilde s] = H\sigma_w^2\, s$$

方差用 Isserlis 公式（零均值联合高斯： $\mathbb{E}[U^2V^2] = \mathbb{E}[U^2]\mathbb{E}[V^2] + 2(\mathbb{E}[UV])^2$）：

$$\mathrm{Var}(\tilde s) = H\sigma_w^4\bigl(\|x\|^2\|y\|^2 + s^2\bigr)$$

JL 缩放（ $\sigma_w^2 = 1/H$）下无偏，且

$$\mathrm{Std}(\tilde s) = \sqrt{\frac{\|x\|^2\|y\|^2 + s^2}{H}}
\;=\; \sqrt{\frac{1 + \cos^2\theta}{H}} \quad (\text{单位向量})$$

两个一致性检验：

- **特例 $x = y$**： $\tilde s = \|Ax\|^2$，公式退化为 $\mathrm{Var} = 2\|x\|^4/H$——
  正是 `docs/jl-projection.md` §3 的 $\chi^2$ 涨落；
- **集中不等式**：由范数版尾概率经极化恒等式
  $\langle x,y\rangle = \frac14(\|x{+}y\|^2 - \|x{-}y\|^2)$ 推出
  （ $|\varepsilon| \le 1$， $c$ 为绝对常数）：

$$\Pr\bigl(|\tilde s - s| \ge \varepsilon\,\|x\|\,\|y\|\bigr) \le 2e^{-c\varepsilon^2 H}$$

## 3. 两个独立矩阵：期望归零

$A \perp B$ 时 $\mathbb{E}[M] = \mathbb{E}[A^\top]\mathbb{E}[B] = 0$，故对**任意**固定的 $x, y$：

$$\mathbb{E}[\tilde s] = 0, \qquad
\mathrm{Var}(\tilde s) = H\sigma_w^4\,\|x\|^2\|y\|^2$$

原有点积的信息全部丢失，只剩零均值噪声。直觉：两次独立随机变换后，
两向量在低维空间的取向互不相关，逐分量乘积随机抵消。

**与随机输入视角的对偶**：`docs/attention-score-distribution.md` §1 是
"$W$ 固定、输入随机"，得 $\mathrm{Var}(s) = H D^2 \sigma_w^4 \sigma^4$；
本文是"输入固定、 $W$ 随机"。两者代入典型长度 $\|x\|^2 = \|y\|^2 = D\sigma^2$ 后
**公式吻合**——同一对象 $x^\top M y$ 的两种条件化，方差由同一组二阶矩决定。

## 4. 相关矩阵：信息保留程度 = 相关程度

两种极端之间可以插值。令 $B = \rho A + \sqrt{1-\rho^2}\,C$（ $C$ 独立同分布，
保证 $B$ 的元素方差仍是 $\sigma_w^2$），则 $\mathbb{E}[A^\top B] = \rho H\sigma_w^2 I_D$，

$$\mathbb{E}[\tilde s] = \rho\, H\sigma_w^2\, s, \qquad
\mathrm{Var}(\tilde s) = H\sigma_w^4\bigl(\|x\|^2\|y\|^2 + \rho^2 s^2\bigr)$$

（方差： $\tilde s = \rho\, x^\top A^\top A y + \sqrt{1-\rho^2}\, x^\top A^\top C y$，
两项协方差为 0，各自方差直接相加。）

单参数 $\rho$ 把两种情形统一： $\rho = 1$ 同一矩阵， $\rho = 0$ 独立矩阵。
**信号项随相关性线性衰减，噪声项始终都在**。

这是理解注意力的关键： $W_Q \ne W_K$（初始化时 $\rho = 0$）仍能传递信息，
是因为**训练让两矩阵相关**—— $M = W_Q^\top W_K$ 从零均值随机矩阵长出非零结构，
$q\cdot k$ 才编码语义相关性。初始化时的纯噪声形态见 §5 与
`docs/attention-score-distribution.md`。

**训练后 $\rho$ 怎么测**：玩具模型的标量 $\rho$ 有两个可操作推广——

1. 逐元素相关（ $\rho$ 的直接估计，用 $\mathrm{tr}(W_Q^\top W_K) = \langle W_Q, W_K\rangle_F$）：

$$\hat\rho = \frac{\langle W_Q, W_K\rangle_F}{\|W_Q\|_F\,\|W_K\|_F},
\qquad \text{初始化基线 } |\hat\rho| \lesssim 1/\sqrt{HD}$$

（基线推导：初始化时分子是 $HD$ 个独立零均值乘积之和，std $= \sqrt{HD}\,\sigma_w^2$；
分母 $\|W_Q\|_F\|W_K\|_F \approx HD\sigma_w^2$（ $\chi^2$ 集中）；比值即
$\hat\rho \sim \mathcal{N}(0,\, 1/\sqrt{HD})$。）

2. 各向同性信号系数： $\rho_{\mathrm{eff}} = \mathrm{tr}(M)\,/\,(DH\sigma_w^2)$
（ $M = \mu I + M'$ 分解中 $\mu = \mathrm{tr}(M)/D$ 对应玩具模型的 $\rho H\sigma_w^2$）。

但要警惕标量的盲区：恒等式 $\mathrm{tr}(M) = \sum_i \sigma_i\,(u_i \cdot v_i)$ 表明
$\hat\rho$ **只看见 $u_i \approx v_i$ 的对称成分**（相似性匹配头）；
强非对称头（induction 型， $u_i \perp v_i$）信号再强 $\hat\rho$ 也接近 0。
训练信号的主要载体是各向异性的低秩结构，完整的"相关性"是
$H \times H$ 对齐矩阵 $U_Q^\top U_K$——初始化近似随机正交（元素 $\sim 1/\sqrt{H}$），
训练后少数方向 $\cos \to \pm 1$。经验规律与实测方案见 `docs/qk-spectrum.md` §4、§6。

## 5. 公式总表

一般 $\sigma_w^2$（固定 $x, y$）：

| 情形 | $\mathbb{E}[\tilde s]$ | $\mathrm{Var}(\tilde s)$ |
|---|---|---|
| 同一 $A$（ $\rho = 1$） | $H\sigma_w^2\, s$ | $H\sigma_w^4(\|x\|^2\|y\|^2 + s^2)$ |
| 相关（ $0<\rho<1$） | $\rho H\sigma_w^2\, s$ | $H\sigma_w^4(\|x\|^2\|y\|^2 + \rho^2 s^2)$ |
| 独立 $A, B$（ $\rho = 0$） | $0$ | $H\sigma_w^4\|x\|^2\|y\|^2$ |

LLM 缩放（ $\sigma_w^2 = 1/D$，取典型长度 $\|x\|^2 = \|y\|^2 = D$， $c = H/D$）：

| 情形 | $\mathbb{E}[\tilde s]$ | $\mathrm{Var}(\tilde s)$ |
|---|---|---|
| 同一 $A$ | $c\, s$（**有偏**） | $H\bigl(1 + s^2/D^2\bigr) \approx H$ |
| 独立 $A, B$ | $0$ | $H$ |

注意：LLM 缩放下同一矩阵投影**不是无偏的**，系数是 $c = H/D$ 而非 1
（ $\mathbb{E}[A^\top A] = (H/D) I_D$；想保内积须用 JL 缩放 $1/H$，见
`docs/jl-projection.md` §1）。两种情形方差在同一量级（ $\sim H$），**区别全在期望**。

**注意力分数的 $\div\sqrt{d_k}$**：初始化时 $W_Q \perp W_K$（ $\rho = 0$），
$\mathrm{Var}(q\cdot k) = d_k\,\sigma_w^4\|x\|^2\|y\|^2 \approx d_k$——
$q\cdot k$ 是 $d_k$ 个 iid 零均值项之和，方差随维数线性累积，
除以 $\sqrt{d_k}$ 把它归一到 $O(1)$，防止 softmax 饱和。
（精确分布形状见 `docs/attention-score-distribution.md` §1。）

**与 JL 投影的目标差异**：JL 要保距（ $\tilde s \approx s$、理想 $M \approx I$），
方差是要消灭的敌人；注意力恰好相反——它不打算保留朴素内积 $s$，
而是**学一个新内积** $\langle x, y\rangle_M = x^\top M y$。
把 $M$ 分解为 $\mu I + M'$： $\mu$ 成分才是"保留 $s$"的部分
（相似性头里显著，induction 头里 $\approx 0$），主导项 $x^\top M' y$ 与 $s$
基本无关，由 $x, y$ 在 $M$ 学出的匹配方向上的对齐决定。
所以"分数与投影前内积关系不大"不是缺陷而是设计使然：
$M$ 编码任务学出的关系匹配，不是几何相似度； $\|M\|_F$ 超基线增长
（方差变大）正是信号容量的增长，见 `docs/qk-spectrum.md` §4。

## 6. 谱视角：点积 = 谱 × 取向

对 $M$ 做 SVD（ $M = U\Sigma V^\top$， $\mathrm{rank}(M) \le H$），记 $\hat x = U^\top x$、 $\hat y = V^\top y$：

$$\tilde s = \sum_{r=1}^{H} \sigma_r\,\hat x_r\,\hat y_r$$

- **同一矩阵**： $M \succeq 0$， $U = V$， $\sigma_r = \lambda_r \ge 0$。
  坐标在同一组基下逐分量相乘，各方向"同向"相加，期望
  $\mathbb{E}\tilde s = (\mathbb{E}\,\mathrm{tr}\,M / D)\, s$——JL 缩放下 $\mathrm{tr}\,M/D \to 1$；
- **两个矩阵**： $U, V$ 渐近独立随机取向， $\hat x_r$ 与 $\hat y_r$ 相互错开、
  符号随机，求和抵消到零均值。

$M$ 奇异值谱的极限律：同一矩阵为 Wishart（ $MP_c$），
两个矩阵为自由乘积 $MP_c \boxtimes MP_c$，见 `docs/math/random-matrix-spectra.md` §1。
谱的矩决定点积的典型涨落（如 $\mathrm{Var} \propto \mathbb{E}\|M\|_F^2$），
$x, y$ 相对奇异向量的取向决定个体偏差。

**信噪比**（JL 缩放、单位向量、相关参数 $\rho$）：

$$\mathrm{SNR} = \frac{|\mathbb{E}\tilde s|}{\mathrm{Std}(\tilde s)}
= \frac{\rho\,|s|}{\sqrt{(1 + \rho^2 s^2)/H}} \;\le\; \rho|s|\sqrt{H}$$

同一矩阵时 $\mathrm{SNR} \sim \sqrt{H}$（可用）；独立矩阵时信号恒零（不可用）。
初始化注意力即后者；训练的本质是把 $\rho$（ $M$ 的结构）从零拉起来。

## 7. 实用推论

- 想保内积必须对**所有**向量用同一个投影矩阵——近似最近邻、LSH 的一致性要求；
- 降方差三条路：增大 $H$（std $\propto 1/\sqrt{H}$）、多次独立投影取平均、
  结构化投影（FJLT， $O(D\log D)$）；稀疏投影只省计算、不降方差；
- 检验"$M$ 是否仍是随机矩阵"的快速判据： $\mathrm{tr}(M)/D$ 是否显著非零、
  谱是否偏离 MP 系——可直接用于诊断训练后的 $W_Q^\top W_K$
  （谱基线见 `docs/math/random-matrix-spectra.md`，实测方案见 `docs/qk-spectrum.md`）。

## 参考文献

- W. B. Johnson, J. Lindenstrauss, *Extensions of Lipschitz mappings into a Hilbert space*, 1984；
- S. Dasgupta, A. Gupta, *An elementary proof of a theorem of Johnson and Lindenstrauss*,
  Random Structures & Algorithms, 2003——两页纸的初等证明；
- R. Vershynin, *High-Dimensional Probability*, Cambridge 2018——集中不等式与
  sub-exponential 工具的系统处理。
