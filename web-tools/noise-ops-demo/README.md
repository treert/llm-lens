# noise-ops-demo：噪音向量的基本运算（分布视角）

交互式网页工具，从**概率密度、均值、方差**三个视角演示高斯噪音向量的三种基本运算——
加、数乘、点积（LLM 线性代数的"三原色"：矩阵投影 = 每行一个点积组成新向量）。

两个面板：

1. **按元素运算**（标量视角）：同一批样本对 $x$、$y$ 逐元素做 $x$、$x+y$、$x \circ y$、$x^2$
   四种运算，四条精确理论密度同图对比，选中项叠加蒙特卡洛直方图。直观看：
   相加保持正态；按元素乘变成尖峰重尾的乘积正态；平方把质量压到正半轴、均值漂离 0。
2. **求和：点积与长度**（向量视角）：点积（双方随机 / 一方固定 $\pm 1$）、$\|x\|^2$，
   以及投影点积（attention 分数：先各自 $H$ 维投影再点积）四个模式，
   直方图 + 精确理论曲线。拖 $D$ 从 1 到 4096 可看乘积正态 → Laplace → 正态的
   中心极限过程，以及 $\|x\|^2$ 的长度集中（RMSNorm 的前提）；
   两种点积模式的理论/样本方差对比即回答"$\sigma^2 = 1/D$ 能否让点积方差为 1"。

## 用法

直接用浏览器打开 `index.html` 即可（ECharts 走 CDN，需要联网）。

- 两个面板参数各自独立、互不影响：面板一有分量方差 $\sigma^2$、样本量 $N$；
  面板二有维数 $D$（滑块走 2 的幂 1~8192，输入框同范围）与分量方差 $\sigma^2$
  （预设 $1$ / $1/D$ / $1/\sqrt{D}$ 随 $D$ 联动，也可自定义）；
- 全局仅共享随机种子（影响复现性，不影响分布结论）；
- 理论曲线随参数即时更新；蒙特卡洛不自动采样——点各面板参数行的「采样」
  才生成样本并叠加直方图，参数变更后旧样本失效（按钮高亮提示重新采样）。

## 文件结构

- `index.html`：页面骨架
- `js/theory.js`：理论分布层（各运算的精确密度、均值、方差；log 域 Bessel $K_\nu$），无 DOM 依赖
- `js/sampler.js`：蒙特卡洛采样层（mulberry32 + Box–Muller、直方图、样本矩），无 DOM 依赖
- `js/app.js`：UI 状态同步与 ECharts 渲染
- `css/style.css`：样式
- `test/theory-selftest.js`：node 自检（`node test/theory-selftest.js`）

---

## 数学背景

设定：$x$、$y$ 为 $D$ 维向量，各分量独立，$x_i \sim \mathcal{N}(0, \sigma^2)$，$y_i \sim \mathcal{N}(0, \sigma^2)$。

### 1. 按元素运算

| 运算 | 分布 | 均值 | 方差 | 仍为正态？ |
|---|---|---|---|---|
| $x$ | $\mathcal{N}(0, \sigma^2)$ | 0 | $\sigma^2$ | 是 |
| $x+y$ | $\mathcal{N}(0, 2\sigma^2)$ | 0 | $\sigma_x^2 + \sigma_y^2$ | 是（独立正态之和仍正态，方差相加） |
| $x \circ y$ | 乘积正态 | 0 | $\sigma_x^2\sigma_y^2$ | **否** |
| $x^2$ | $\sigma^2\chi^2_1$ | $\sigma^2$ | $2\sigma^4$ | **否** |

**乘积正态**（normal product distribution）：对 $z = x \cdot y$ 作特征函数
$\varphi(t) = \mathbb{E}[e^{itxy}] = (1 + \sigma_x^2\sigma_y^2 t^2)^{-1/2}$，反演得

$$f(z) = \frac{1}{\pi\,\sigma_x\sigma_y}\,K_0\!\left(\frac{|z|}{\sigma_x\sigma_y}\right)$$

$K_0$ 为第二类修正 Bessel 函数。性质：$z=0$ 处对数发散（$K_0(u) \sim -\ln(u/2)$，
尖峰**无上界**但可积——发散极慢，$z = 10^{-300}$ 时密度也仅 $\sim 10^2/\sigma^2$ 量级）；
指数重尾；峰度 9（正态为 3）。零均值独立时

$$\mathrm{Var}(xy) = \mathbb{E}[x^2]\,\mathbb{E}[y^2] = \sigma_x^2\sigma_y^2 \quad\text{（方差的乘积）}$$

**平方**：$z = x^2$ 是 $\Gamma(1/2,\; 2\sigma^2)$，即 $\sigma^2$ 倍的 $\chi^2_1$：

$$f(z) = \frac{1}{\sigma\sqrt{2\pi z}}\,e^{-z/(2\sigma^2)},\quad z>0$$

注意均值 $\mathbb{E}[x^2] = \sigma^2 \ne 0$：自乘把关于 0 对称的噪音变成恒正的"能量"，
均值整体漂移到 $\sigma^2$——这是与"两个独立噪音相乘"（均值仍为 0）的本质区别。

**尖峰有没有上界？** 没有——两个尖峰都是真实的数学奇异，但方式不同：
乘积正态是对数奇异（极慢），$\chi^2_1$ 是幂律奇异 $z^{-1/2}$（较快），两者都可积。
$\chi^2$ 的奇异性随自由度消失：

| 自由度 $k$ | $f(0)$ | 形状 |
|---|---|---|
| 1 | $+\infty$（$z^{-1/2}$ 奇异） | 尖峰在 0 |
| 2 | $1/2$（有限） | 从有限值指数衰减 |
| $\ge 3$ | 0 | 峰离开原点（$z > 0$ 处） |

对应面板二的 $\|x\|^2 \sim \sigma^2\chi^2(D)$：$D=1$ 尖峰顶天、$D=2$ 从有限值衰减、
$D \ge 3$ 峰离开原点。
直方图在尖峰处仍与理论线吻合：箱宽有限，显示的是箱内平均密度，总是有限值。

### 2. 点积：先按元素乘，再求和

速览（推导见下文）：

| 运算 | 分布 | 均值 | 方差 | 仍为正态？ |
|---|---|---|---|---|
| $x \cdot y$（双方随机） | Bessel-K 型，$\nu = (D-1)/2$ | 0 | $D\sigma_x^2\sigma_y^2$ | 否（$D=1$ 乘积正态、$D=2$ Laplace；$D$ 大由 CLT 趋近正态） |
| $x \cdot v$（一方固定，$v_i = \pm 1$） | $\mathcal{N}(0, D\sigma^2)$ | 0 | $D\sigma^2$ | **是**（精确正态） |
| $\|x\|^2$（长度平方） | $\sigma^2\chi^2(D)$ | $D\sigma^2$ | $2D\sigma^4$ | 否（只有正半轴；$D$ 大趋近正态） |
| $(W_Q A)\cdot(W_K B)/\sqrt{H}$（attention 分数，$\sigma_w^2 = 1/D$） | $H$ 维双方随机点积 $\div\sqrt{H}$ | 0 | $\sigma^4$（与 $D$、$H$ 无关） | 否（形状只看 $H$） |

投影点积即 attention 分数的初始化分布；训练后取决于 $W_Q^\top W_K$ 的奇异值谱，
完整讨论见 `docs/attention-score-distribution.md`。

**双方随机** $z = \sum_i x_i y_i$：特征函数 $\varphi(t) = (1 + a^2 t^2)^{-D/2}$
（$a = \sigma_x\sigma_y$），反演得

$$f(z) = \frac{(|z|/2a)^{\nu}\,K_\nu(|z|/a)}{\sqrt{\pi}\,\Gamma(D/2)\,a},
\qquad \nu = \frac{D-1}{2}$$

- 均值 0，**方差 $D\sigma_x^2\sigma_y^2$**；
- $D=1$ 退化为乘积正态；$D=2$ 时 $K_{1/2}$ 有初等式，恰好是 **Laplace$(0, a)$**；
- $D$ 增大由 CLT 趋于 $\mathcal{N}(0, D\sigma_x^2\sigma_y^2)$——求和"抹平"了乘积正态的尖峰。

实现细节（`theory.js`）：$K_\nu$ 全程在 log 域计算（$\nu$ 为整数或半整数），
整数阶用 Abramowitz & Stegun 9.8 有理逼近取 log，半整数阶从初等式
$K_{1/2}(x) = \sqrt{\pi/2x}\,e^{-x}$ 出发，统一前向递推
$\log K_{\nu+1} = \mathrm{logaddexp}\bigl(\log K_{\nu-1},\; \log(2\nu/x) + \log K_\nu\bigr)$
（$K$ 对 $\nu$ 前向递推稳定）。
log 域保证 $D=4096$、$|z|/a$ 很大时 $e^{-x}$ 不下溢。

**一方固定** $z = \sum_i v_i x_i$（$v$ 为给定向量，非新鲜噪音）：独立正态的加权和仍正态，

$$z \sim \mathcal{N}(0,\; \sigma^2\|v\|^2)$$

取 $v_i \in \{\pm 1\}$ 时 $\|v\|^2 = D$，即 $\mathcal{N}(0, D\sigma^2)$。不发生"方差相乘"——
固定向量的分量只是把独立噪音重新线性组合。

**长度平方** $\|x\|^2 = \sum_i x_i^2 \sim \sigma^2\chi^2(D)$：均值 $D\sigma^2$，
方差 $2D\sigma^4$，相对涨落 $\sqrt{2/D}$。高维时 $\|x\|$ 集中到 $\sigma\sqrt{D}$ 附近——
这是 RMSNorm 能工作的前提（§4 详解）。

### 3. 关键问题：$\sigma^2 = 1/D$ 时点积方差是不是 1？

$$\mathrm{Var}(x\cdot y) = D\,\sigma_x^2\sigma_y^2$$

- **双方皆随机**，$\sigma^2 = 1/D$：$\mathrm{Var} = D \cdot (1/D)^2 =$ **$1/D$** ❌
  （缩过头，差一个因子 $D$；要方差为 1 需 $\sigma^2 = 1/\sqrt{D}$）
- **一方固定**（$\|v\|^2 = D$），$\sigma^2 = 1/D$：$\mathrm{Var} = D \cdot (1/D) =$ **1** ✓

与 LLM 的对应：

- **线性层初始化** $W_{ij} \sim \mathcal{N}(0, 1/D_{in})$：输入 $x$ 对本次前向传播是"给定"的
  （一方固定情形），若输入分量方差为 1，输出分量方差 $= D_{in} \cdot (1/D_{in}) \cdot 1 = 1$，
  逐层保持方差稳定——这正是 Xavier / Kaiming 初始化的思路。
- **Attention 的 $q \cdot k$**：$q$、$k$ 都来自数据、皆视为随机（双方随机情形），
  分量方差 $\approx 1$ 时点积方差 $= D_k$，故 softmax 前要除以 $\sqrt{D_k}$。

两种缩放因子 $1/D$ 与 $1/\sqrt{D}$ 的分野，本质就是点积中"一方固定还是双方皆随机"；
面板二的两种点积模式可直接对比这两种情形（理论方差与蒙特卡洛实测同屏显示）。

### 4. 附：RMSNorm 与注意力分数

RMSNorm 的原理（校准性 + 稳定性）及它与注意力分数的方差链联动
（RMSNorm → 分量方差归一 → 投影保持 → 点积 → $\div\sqrt{H}$ → 分数方差恒 1），
已移至独立文档：[`docs/rmsnorm.md`](../../docs/rmsnorm.md)。
