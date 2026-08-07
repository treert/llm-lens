# 自由概率速览：$\boxtimes$ 是什么

面向本项目的读者：不需要完整学自由概率，只需要理解
"$M = W_Q^\top W_K$ 的谱为什么是 $MP_c \boxtimes MP_c$"。
本文用最少的概念解释 $\boxtimes$，全部结论都有数值验证（见 `web-tools/spectrum-demo/`）。

## 1. 从经典概率的类比出发

经典概率里，**独立**随机变量 $X, Y$ 的组合有标准工具：

| 组合 | 分布 | 计算技巧 |
|---|---|---|
| 和 $X + Y$ | 卷积 $\mu * \nu$ | 特征函数（Fourier 变换）**相乘**：$\varphi_{X+Y} = \varphi_X \varphi_Y$ |
| 积 $XY$（正值变量） | 乘性卷积 | Mellin 变换相乘 |

"变换把卷积变成普通乘法"是核心结构。自由概率把这套结构搬到**矩阵**世界，
但需要先回答：矩阵的"独立"是什么？

## 2. 矩阵为什么不能直接用经典概率

两个经典随机变量 $X, Y$ 独立时，一切联合矩可拆分：
$\mathbb{E}[X^m Y^n] = \mathbb{E}[X^m]\mathbb{E}[Y^n]$。

矩阵不行：**乘法不对易**。谱分布只由 $\mathrm{tr}(X^k)$ 这类"矩"决定，
而 $\mathrm{tr}(XYXY)$ 这类交错矩无法从 $X$、$Y$ 各自的谱拆出来——
除非两个矩阵的特征向量"完全不同时对齐"。多一个条件不够：光有"独立"（条目独立采样）
还不够，关键是**特征向量的相对取向**。

## 3. 自由独立性：特征向量毫无对齐

**自由（free）独立性**是经典独立性在非交换世界的对应物。直觉表述：

> $X, Y$ 自由 ⇔ $X$ 的特征向量系与 $Y$ 的特征向量系"随机取向"，
> 没有任何方向同时对齐两者。

严格定义（Speicher）：对任意交错的多项式 $p_1, q_1, p_2, q_2, \dots$，只要每个
$p_i(X)$、$q_i(Y)$ 都中心化（$\varphi(p_i(X)) = 0$，$\varphi$ = 归一化迹），就有

$$\varphi\bigl(p_1(X)\, q_1(Y)\, p_2(X)\, q_2(Y) \cdots\bigr) = 0$$

这个"交错积的迹必为零"的规则足以把任意交错矩递归地拆成各自矩的多项式。例如：

$$\varphi(XYXY) = \varphi(X^2)\,\varphi(Y)^2 + \varphi(X)^2\,\varphi(Y^2) - \varphi(X)^2 \varphi(Y)^2$$

（经典独立会给 $\varphi(X^2)\varphi(Y^2)$，不同！这就是两种"独立"的可观测差别。）

**为什么大随机矩阵天然满足**：Voiculescu 定理——独立高斯（Ginibre/GOE）矩阵之间、
或固定矩阵与 Haar 随机旋转的矩阵之间，维数 $\to \infty$ 时渐近自由。
$W_Q, W_K$ 初始化时条目 iid 高斯，正落在这个范畴。

## 4. $\boxplus$ 与 $\boxtimes$：自由版卷积

自由的 $X, Y$（谱分布 $\mu, \nu$）：

- **和**的谱：自由加性卷积 $\mu \boxplus \nu$，用 **R-变换**计算：$R_{\mu \boxplus \nu} = R_\mu + R_\nu$；
- **积**的谱：自由乘性卷积 $\mu \boxtimes \nu$，用 **S-变换**计算：

$$S_{\mu \boxtimes \nu}(z) = S_\mu(z)\, S_\nu(z)$$

与经典对照：特征函数 ↔ R-变换（对应和），Mellin 变换 ↔ **S-变换（对应积）**。

$\boxtimes$ 显然**交换、结合**（$S$ 相乘满足）——所以"两个因子换序，极限谱曲线相同"
是显然的代数事实。

## 5. S-变换：定义与 MP 律的例子

对 $[0,\infty)$ 上均值非零的分布 $\mu$，设矩 $m_n = \int x^n d\mu$，
矩生成函数 $\psi(z) = \sum_{n\ge1} m_n z^n$，$\chi$ 为其逆函数，则

$$S_\mu(z) = \frac{1+z}{z}\,\chi_\mu(z)$$

计算流程：矩 → $\psi$ → 求逆得 $\chi$ → 得 $S$；反之给定 $S$ 可反解矩生成函数。

**Marchenko-Pastur 律 $MP_c$**（单 Wishart 的谱，均值归一为 1）：

$$S_{MP_c}(z) = \frac{1}{1+cz}$$

于是两个自由 Wishart 之积（$M = A^\top B$ 的非零平方奇异值）的谱：

$$S_{MP_c \boxtimes MP_c}(z) = \frac{1}{(1+cz)^2}$$

## 6. 从 S-变换到密度：三次方程怎么来的

记 $\mathcal{M}(w) = 1 + \psi(w)$（含零阶矩）。由 $S(z) = \frac{1+z}{z}\chi(z)$
得 $\chi(z) = \frac{zS(z)}{1+z}$，代入 $z = \mathcal{M} - 1$ 与
$S(z) = \dfrac{1}{(1+cz)^2}$：

$$w = \chi(\mathcal{M}-1) = \frac{(\mathcal{M}-1)\,S(\mathcal{M}-1)}{\mathcal{M}}
= \frac{\mathcal{M}-1}{\mathcal{M}\,(1 + c(\mathcal{M}-1))^2}$$

（对一般的 $MP_a \boxtimes MP_b$：$w = \dfrac{\mathcal{M}-1}{\mathcal{M}(1-a+a\mathcal{M})(1-b+b\mathcal{M})}$。）

这是 $\mathcal{M}$ 的**三次代数方程**。给定 $x$，令 $w = 1/x$ 解三次方程：
支撑内恰有一对共轭复根，密度

$$p(x) = \frac{|\mathrm{Im}\,\mathcal{M}(1/x)|}{\pi x}$$

$c = 1$ 时方程退化为 $w\mathcal{M}^3 = \mathcal{M} - 1$，即 Fuss-Catalan 数生成函数的
方程 $\mathcal{M} = 1 + w\mathcal{M}^3$——这就是"$FC_2$ 的矩是 Fuss-Catalan 数"的来源。
数值实现见 `web-tools/spectrum-demo/js/theory.js`（Cardano 求根）。

## 7. 常用矩公式（手算够用）

记均值 $\bar\mu = m_1(\mu)$ 等。由 S-变换展开系数可得：

$$m_1(\mu \boxtimes \nu) = \bar\mu\,\bar\nu$$
$$m_2(\mu \boxtimes \nu) = m_2(\mu)\,\bar\nu^2 + \bar\mu^2\,m_2(\nu) - \bar\mu^2\bar\nu^2$$

代入两个 $MP_c$（$m_1 = 1$，$m_2 = 1+c$）：乘积谱均值 1、$m_2 = 1 + 2c$、**方差 $2c$**。
代入 $MP_c \boxtimes MP_1$：均值 1、方差 $1 + c$（$AB^\top$ 摆法 ÷c 后的曲线）。

## 8. 使用边界

- $\boxtimes$ 描述的是 $H, D \to \infty$、$H/D \to c$ 的**渐近极限**；有限维有涨落
  （如谱最大值可略超支撑右端）；
- 要求渐近自由：独立高斯 ✓；训练后的 $W_Q, W_K$ 一般不自由——
  此时 $MP_c \boxtimes MP_c$ 是**初始化基线**，实测谱的偏离正是训练效应
  （见 `docs/qk-spectrum.md` §4、§6）；
- $M = A^\top B$ 换成 $AB^\top$（乘法维度互换）极限律会变，见
  `docs/math/random-matrix-spectra.md`（曲线与验证工具见 `web-tools/spectrum-demo/`）。

## 参考文献

- D. Voiculescu, *Limit laws for random matrices and free products*, Invent. Math. 1991
  ——渐近自由的基本定理；
- A. Nica, R. Speicher, *Lectures on the Combinatorics of Free Probability*, Cambridge 2006
  ——S-变换与矩的系统性处理；
- T. Tao, *Topics in Random Matrix Theory*, AMS 2012（§2.5 自由概率入门）——
  最友好的读本。
