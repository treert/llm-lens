# 高斯矩阵与其乘积的奇异值谱：MP 律、Fuss-Catalan 与自由乘积

通用数学笔记。交互式曲线工具见 `web-tools/spectrum-demo/`；
$\boxtimes$（自由乘性卷积）的概念入门见 `docs/math/free-probability.md`；
在 QK 谱分析中的应用见 `docs/qk-spectrum.md` §3。

## 0. 设定与记号

- $A, B$：$H \times D$ 独立高斯矩阵，条目 iid $\mathcal{N}(0, \sigma_w^2)$；
  初始化缩放取 $\sigma_w^2 = 1/D$，此时均值归一为 1；
- $c = H/D \le 1$：大维极限下所有谱只依赖比值 $c$；
- $\lambda = \sigma_w^2 D$：尺度因子（归一即 $\lambda = 1$）。奇异值随矩阵倍数线性缩放
  （$\sigma(\alpha M) = \alpha\,\sigma(M)$），故 $\sigma^2$ 谱横轴：单矩阵乘 $\lambda$、
  乘积乘 $\lambda^2$（收缩维数不同）——**形状不变，仅整体伸缩**；
- 三种对象：单矩阵 $A$；$D \times D$ 乘积 $A^\top B$（QK 分析的形式）；
  $H \times H$ 乘积 $AB^\top$（乘法维度互换）。

## 1. 公式总表（σ² 轴，含 λ 的一般形式）

| 对象 | 极限律 | 均值 | 方差 | $x_-$（左端） | $x_+$（右端） |
|---|---|---|---|---|---|
| $A$ | $MP_c$ | $\lambda$ | $c\lambda^2$ | $\lambda(1-\sqrt{c})^2$ | $\lambda(1+\sqrt{c})^2$ |
| $A^\top B$ 非零 | $MP_c \boxtimes MP_c$ | $\lambda^2$ | $2c\lambda^4$ | $\lambda^2 / g(M_-)$ | $\lambda^2 / g(M_+)$ |
| $AB^\top$ | $c \cdot (MP_c \boxtimes MP_1)$ | $c\lambda^2$ | $c^2(1+c)\lambda^4$ | $0$ | $c\lambda^2 \cdot \dfrac{M_+^2(1-c+cM_+)}{M_+-1}$ |

其中

$$g(M) = \frac{M-1}{M\,(1-c+cM)^2}, \qquad
M_\pm = \frac{3c \pm \sqrt{c^2+8c}}{4c}
\;\;(\text{方程 } 2cM^2 - 3cM + (c-1) = 0\text{ 的两根})$$

$AB^\top$ 行的 $M_+ = \dfrac{4c - 1 + \sqrt{1+8c}}{4c}$
（方程 $2cM^2 + (1-4c)M - 2(1-c) = 0$ 的正根；$M_-$ 根为负，对应 $x_- = 0$）。

补充说明：

- **$c = 1$ 特例**：$A^\top B$ 与 $AB^\top$ 统一为 **Fuss-Catalan 分布 $FC_2$**
  （$M_+ = 3/2$，$x_+ = 27/4$，以 Fuss-Catalan 数 $1, 1, 3, 12, 55, \dots$ 为矩）；
- **$\sigma$ 轴版本**：$s_\pm = \sqrt{x_\pm}$（如 $MP_c$ 为 $s_\pm = \sqrt{\lambda}(1 \pm \sqrt{c})$）；
- **$\sigma$ 轴的均值/方差无初等闭式**（$E[\sigma] \ne \sqrt{E[\sigma^2]}$），可数值积分；
  唯一干净的特例是 $c = 1$ 单矩阵（quarter-circle）：
  $E[\sigma] = 8/(3\pi) \approx 0.8488$，$\mathrm{Var} = 1 - 64/(9\pi^2) \approx 0.2795$；
- **支撑特例数值**（$\lambda = 1$）：

| $c$ | $MP_c$ | $MP_c \boxtimes MP_c$ | $MP_c \boxtimes MP_1$（$AB^\top$ ÷c） |
|---|---|---|---|
| 1 | $[0,\; 4]$ | $[0,\; 6.75]$ | $[0,\; 6.75]$ |
| 1/4 | $[0.25,\; 2.25]$ | $[0.136,\; 3.098]$ | $[0,\; 4.848]$ |
| 1/16 | $[0.563,\; 1.563]$ | $[0.440,\; 1.871]$ | $[0,\; 4.237]$ |
| $10^{-3}$ | $[0.938,\; 1.064]$ | $[0.913,\; 1.092]$ | $[0,\; 4.004]$ |

## 2. 单矩阵：Marchenko-Pastur 律

$AA^\top$ 是（归一化的）Wishart 矩阵，其特征值 = $A$ 的平方奇异值，渐近密度：

$$p(x) = \frac{\sqrt{(x_+ - x)(x - x_-)}}{2\pi c\, x}, \qquad
x_\pm = (1 \pm \sqrt{c})^2$$

$c \to 0$ 时收缩到 $x = 1$ 附近的窄带（$A$ 接近部分等距）；$c = 1$ 时支撑 $[0, 4]$，
在 0 处按 $x^{-1/2}$ 发散（$\sigma$ 轴视角即 quarter-circle 律 $p(s) = \sqrt{4-s^2}/\pi$）。

## 3. 乘积：恒等链与自由乘积

### 3.1 $D \times D$ 摆法 $M = A^\top B$

平方奇异值先写成回文形，再用 Sylvester（$XY$ 与 $YX$ 非零特征值相同）折叠：

$$\sigma^2(M) = \mathrm{eig}(MM^\top)
= \mathrm{eig}\bigl(\underbrace{A^\top B B^\top A}_{\text{回文},\; D \times D}\bigr)
\;\xrightarrow{\text{折叠}}\;
\mathrm{eig}\bigl((BB^\top)(AA^\top)\bigr)\quad(H \times H)$$

回文形消失、变成**两个 $H \times H$ Wishart 之积**——两者渐近自由、各带参数 $c$，
故极限律为 $MP_c \boxtimes MP_c$。注意 $(AA^\top)(BB^\top)$ 虽非对称，
但两个半正定矩阵之积的特征值必为实非负
（相似于对称半正定矩阵 $(AA^\top)^{1/2}(BB^\top)(AA^\top)^{1/2}$）。

### 3.2 $H \times H$ 摆法 $\widetilde M = AB^\top$

$$\sigma^2(\widetilde M)
= \mathrm{eig}\bigl(\underbrace{A\, B^\top B\, A^\top}_{\text{回文},\; H \times H}\bigr)$$

差别在中间那块：$B^\top B$ 是 $D \times D$ 而**秩只有 $H$**。
折叠给出非零谱 $= \mathrm{eig}_{\ne 0}\bigl((A^\top A)(B^\top B)\bigr)$——两个**秩亏**
Wishart 之积；等价地，把 $B^\top B$ 压缩到它的 $H$ 维行空间看，它是一个 $MP_c$ 的
"总体协方差"，而 $A$ 限制到该空间是**方形** $H \times H$ 高斯——方形广义 Wishart
给出 $MP_1 \boxtimes MP_c$，再乘整体尺度 $c$
（$\mathbb{E}\|AB^\top\|_F^2 = H^2/D$ vs $\mathbb{E}\|A^\top B\|_F^2 = H$，恰差 $1/c$）。
**两种摆法极限律不同的根源就在这里**。

### 3.3 自由乘积的矩

自由性给出 $\varphi(XYXY) = \varphi(X^2) + \varphi(Y^2) - 1$（均值 1 时），
故 $MP_c \boxtimes MP_c$ 二阶矩 $1 + 2c$、方差 $2c$；$MP_c \boxtimes MP_1$ 方差 $1 + c$。
乘积谱在所有 $c$ 下都比单矩阵谱（方差 $c$）更宽。

## 4. 密度的三次方程解法

$MP_c$ 的 S-变换为 $S(z) = 1/(1+cz)$；自由乘性卷积的 S-变换相乘。对一般的
$MP_a \boxtimes MP_b$，矩生成函数 $\mathcal{M}(w) = \sum_{n\ge0} m_n w^n$ 满足三次方程

$$w\,ab\,\mathcal{M}^3 + w(a+b-2ab)\,\mathcal{M}^2
+ \bigl(w(1-a)(1-b) - 1\bigr)\mathcal{M} + 1 = 0, \qquad w = 1/x$$

$c = 1$（即 $a = b = 1$）时退化为 $w\mathcal{M}^3 = \mathcal{M} - 1$，
正是 Fuss-Catalan 生成函数 $\mathcal{M} = 1 + w\mathcal{M}^3$ 的方程——
这就是"$FC_2$ 的矩是 Fuss-Catalan 数"的来源。

支撑内 $x$ 对应一对共轭复根，密度 $p(x) = |\mathrm{Im}\,\mathcal{M}(1/x)| / (\pi x)$；
支撑外三根皆实，密度为 0。数值实现用 Cardano 公式（实系数运算，无需复数，
见 `web-tools/spectrum-demo/js/theory.js`）。

支撑端点来自分支点条件 $dw/d\mathcal{M} = 0$，即 §1 表中的 $M_\pm$ 公式。

## 5. 左端点行为（两轴）

$c < 1$ 时单矩阵谱与 $A^\top B$ 谱左端 $> 0$（0 附近有空隙）；$AB^\top$ 谱左端**贴 0
且发散**——对三次方程在 $x \to 0^+$ 展开得

$$p(x) \sim \frac{1}{\pi\sqrt{(1-c)\,x}} \qquad (c<1)$$

$c \to 1$ 时空隙收拢，各谱在 0 处分别按 $x^{-1/2}$（MP）、$x^{-2/3}$（乘积）发散——
每多一个矩阵因子，谱更宽、近零奇点更强。

$\sigma$ 轴做变量替换 $p_\sigma(s) = 2s\,p_{\sigma^2}(s^2)$，0 点行为改变：
MP（$c=1$）与 $AB^\top$（$c<1$）变为**有限值**（$2/\pi$、$2/(\pi\sqrt{1-c})$），
乘积（$c=1$）仍发散但减弱为 $s^{-1/3}$。右端点在两轴下都按平方根式归零。

## 6. AB 与 BA：特征值、奇异值与乘法维度

"AB 与 BA 是否相同"要分三层回答：

1. **特征值（固定矩阵）**：永远相同——Sylvester 恒等式 $\det(I - XY) = \det(I - YX)$
   保证 $XY$ 与 $YX$ 的非零特征值逐值相等；$D \times D$ 形式只是多垫 $D - H$ 个零
   （ESD 含 $(1-c)\delta_0$ 原子）。
2. **奇异值（固定矩阵）**：一般**不同**。$\sigma^2$ 看的是 $XY(YX)^\top$ 型乘积的
   特征值，Sylvester 用不上。方阵反例：$A = \mathrm{diag}(2,1)$、
   $B = \begin{pmatrix}1&1\\0&1\end{pmatrix}$ 给出
   $\sigma(AB) \approx \{2.92, 0.68\}$、$\sigma(BA) \approx \{2.29, 0.87\}$。
   例外：互为转置的比较（$A^\top B$ vs $B^\top A = (A^\top B)^\top$）奇异值精确相同——
   那是转置的性质，不是换序的性质。
3. **渐近谱密度**：iid 高斯 ensemble 下各有确定的极限律，且两种摆法**不同**
   （见 §1 总表；推导见 §3）。

对照单矩阵可见乘法的代价：$D \times D$ 摆法保持均值 1、方差翻倍（$c \to 2c$）；
$H \times H$ 摆法均值缩到 $c$，相对展宽 $\mathrm{var}/\mathrm{mean}^2$
从 $c$ 涨到 $(1+c)$——乘法维度越"压缩"，谱相对越宽。

## 7. 数值验证

全部结论经过随机矩阵蒙特卡洛对照（脚本 `web-tools/spectrum-demo/test/plot_fuss_catalan.py`，
输出 `output/fuss-catalan.png`；$H{=}128, D{=}2048$，30 次试验）：

- $\sigma^2(A^\top B)$ 非零均值 0.999、方差 0.126（理论 1、$2c = 0.125$）；
- $\sigma^2(AB^\top)/c$ 与 $MP_c \boxtimes MP_1$ 系综分位数逐位吻合，
  方差 1.067（理论 $1+c = 1.0625$）；
- $\mathrm{eig}(AB^\top)$ 与 $\mathrm{eig}(BA^\top)$ 非零部分逐值差
  $\sim 5\times10^{-12}$（Sylvester）；
- $\sigma^2(A^\top B)$ 非零部分与 $\mathrm{eig}\bigl((AA^\top)(BB^\top)\bigr)$
  逐值差 $\sim 4\times10^{-8}$（相对误差 $\sim 10^{-14}$，§3.1 恒等链）；
- 三次方程密度的锚点值与 node 自检交叉对照到 $10^{-9}$
  （`web-tools/spectrum-demo/test/theory-selftest.js`）。
