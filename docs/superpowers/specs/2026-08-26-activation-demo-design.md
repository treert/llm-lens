# activation-demo 设计：LLM 激活函数可视化

日期：2026-08-26
状态：已确认（用户于 2026-08-26 批准方案 B 与本设计）

## 1. 背景与目标

`web-tools/` 新增一个交互式工具 `activation-demo/`，从原理层面展示 LLM 中的激活函数。
不做"更漂亮的 Wikipedia"（只画 $y=f(x)$ 曲线），而是双视角：

1. **函数与导数形状**：饱和区、梯度消失、ReLU 死区、GELU/SiLU 的"平滑 ReLU"本质；
2. **分布视角**（项目招牌风格）：高斯噪音过激活函数后分布如何变形——理论密度 vs
   蒙特卡洛直方图对照，延续 noise-ops-demo 的交互模式。

GLU 门控族（SwiGLU 等二维输入）**不在本期范围**，留作后续迭代；本期用静态对照表交代
"主流模型用了什么"并注明门控变体后续覆盖。

## 2. 函数清单（11 个）

| id | 名称 | 公式 | 参数 | 单调性 |
|---|---|---|---|---|
| sigmoid | Sigmoid | $\sigma(x)=1/(1+e^{-x})$ | — | 单调 |
| tanh | Tanh | $\tanh(x)$ | — | 单调 |
| relu | ReLU | $\max(0,x)$ | — | 弱单调， $y=0$ 处点质量 |
| leaky-relu | Leaky ReLU | $x>0?x:\alpha x$ | $\alpha$ 默认 0.01，范围 0.01–0.3 | 单调 |
| elu | ELU | $x>0?x:\alpha(e^x-1)$ | $\alpha$ 默认 1，范围 0.5–2 | 单调 |
| softplus | Softplus | $\ln(1+e^x)$ | — | 单调 |
| gelu-exact | GELU（精确） | $x\,\Phi(x)$ | — | **非单调**（ $x\approx-0.75$ 处极小 $\approx-0.170$） |
| gelu-tanh | GELU（tanh 近似） | $\frac{x}{2}\bigl(1+\tanh\sqrt{2/\pi}\,(x+0.044715x^3)\bigr)$ | — | **非单调** |
| silu | SiLU / Swish | $x\,\sigma(\beta x)$ | $\beta$ 默认 1，范围 0.1–5 | 由数值检测决定（ $\beta$ 小时近线性单调，大时趋 ReLU） |
| mish | Mish | $x\tanh(\mathrm{softplus}(x))$ | — | **非单调** |
| relu2 | ReLU² | $\max(0,x)^2$ | — | 弱单调， $y=0$ 处点质量 |

单调性**不写死在注册表里**：`theory.js` 运行时数值定位临界点（ $f'(x)=0$ 的根，
导数符号扫描 + 二分），有几段单调段就按几段处理。

## 3. 文件结构

```
web-tools/activation-demo/
  index.html        页面骨架（ECharts 走 CDN）
  css/style.css
  js/functions.js   激活函数注册表：id/中文名/LaTeX 公式/fn/dfn/参数规格，无 DOM
  js/theory.js      输出密度（求逆+多原像求和）、Gauss–Hermite 矩、临界点定位，无 DOM
  js/sampler.js     mulberry32 + Box–Muller、直方图（沿用 noise-ops-demo 模式），无 DOM
  js/app.js         UI 状态同步与 ECharts 渲染
  test/selftest.js  node 自检（node test/selftest.js）
  README.md         用法 + 数学背景
```

另需：根目录 `index.html` 工具列表登记一张卡片。

## 4. 面板一：函数与导数

- 复选框勾选函数叠加显示（默认选 relu / gelu-exact / silu / tanh），每条曲线配中文名 +
  公式图例；
- 视图切换：只看函数 / 只看导数 / 上下双图联动（共享 x 轴缩放）；
- x 范围滑块（默认 $[-5,5]$，可调至 $[-10,10]$）；
- 参数滑块随勾选动态出现：leaky-relu 的 $\alpha$、elu 的 $\alpha$、silu 的 $\beta$；
- 面板底部静态对照表（只展示不联动）：

| 模型 | FFN 激活 | 备注 |
|---|---|---|
| GPT-2 / GPT-3 | GELU（tanh 近似） | OpenAI 系惯例 |
| BERT | GELU（精确） | |
| LLaMA / Qwen / Kimi-K3 | SwiGLU | 门控变体，后续面板覆盖 |
| Gemma | GeGLU | 门控变体，后续面板覆盖 |
| 早期 MLP/CNN | ReLU / Tanh | |

## 5. 面板二：噪音过激活的分布

输入 $x\sim\mathcal N(0,\sigma^2)$，看 $y=f(x)$ 的分布。

### 5.1 交互

- 下拉选**一个**激活函数（默认 gelu-exact——非单调、密度有可积奇点，最有看头）；
- $\sigma^2$ 滑块 + 输入框（0.1–10，默认 1）； $N$ 样本量选择（1e4–1e6，默认 1e5）；
- 全局随机种子输入（影响复现性，不影响分布结论）；
- 蒙特卡洛**不自动采样**：点「采样」按钮生成样本叠加直方图；参数变更后旧样本失效，
  按钮高亮提示重新采样（与 noise-ops-demo 一致）；
- 理论曲线随参数即时更新；统计行显示理论均值/方差（Gauss–Hermite）vs
  样本均值/方差；relu / relu2 额外显示点质量 $P(y=0)=0.5$ 的理论值 vs 直方图 0 号 bin 实测比例。

### 5.2 理论输出密度

按单调段分段处理， $y$ 的原像集合 $\{x_i : f(x_i)=y\}$ 在每段内二分/Brent 求逆：

$$f_Y(y)=\sum_i \frac{f_X(x_i)}{|f'(x_i)|}$$

- **单调函数**：单一原像，变量替换；
- **非单调函数**（gelu-exact、gelu-tanh、mish、部分 $\beta$ 的 silu）：临界点预先数值
  定位（导数符号扫描 + 二分），原像数随 $y$ 变化；极小值附近 $f'\to 0$ 密度发散——
  **可积奇点**，是本期要展示的现象之一，显示时截断封顶并在 README 说明；
- **relu / relu2**： $y=0$ 处为点质量 $P(x\le 0)=0.5$（输入零均值），连续密度只覆盖
  $y>0$ 且积分为 0.5；图上用单独尖峰标注 + 直方图 0 号 bin 对照，直观展示
  "ReLU 产生精确稀疏"。

### 5.3 理论矩

不做密度积分，直接对输入分布 Gauss–Hermite 求积（64 点）：

$$\mathbb E[g(x)]\approx\frac{1}{\sqrt\pi}\sum_i w_i\,g\bigl(\sigma\sqrt2\,t_i\bigr)$$

分别取 $g=f$ 与 $g=f^2$ 得均值与二阶矩，方差 $=\mathbb E[f^2]-\mathbb E[f]^2$。

### 5.4 数值细节

- $\Phi(x)$：Abramowitz–Stegun 7.1.26 误差函数逼近；
- sigmoid 按符号分两支避免 $e^x$ 上溢；softplus 用 $\max(x,0)+\mathrm{log1p}(e^{-|x|})$；
- 求逆扫描区间随 $\sigma$ 自适应（ $\pm 8\sigma$ 外加函数定义域余量）；
- 原像搜索的网格扫描 + 二分，容差 1e-12 量级。

## 6. 数据流

`functions.js` 注册表是唯一事实源：
面板一直接调用 `fn`/`dfn` 画曲线；面板二 `theory.js` 消费同一注册表算临界点/密度/矩；
`sampler.js` 只负责采样与直方图；`app.js` 汇总渲染。增删函数只改 `functions.js` 一个文件。

## 7. 错误与边界处理

- $\sigma^2$、 $N$ 等输入非法：clamp 到合法范围并刷新显示值；
- 非单调函数奇点附近理论密度截断显示（图内标注），不影响直方图对照；
- ECharts CDN 离线：README 注明需联网（与现有 demo 一致）。

## 8. 测试

`test/selftest.js`（node 直接运行，失败非零退出）：

1. 每个函数的 `dfn` 对 `fn` 做中心差分校验（相对容差 1e-5，ReLU 折点处跳过）；
2. 输出密度归一化： $\int f_Y\,dy + P_{\text{atom}} \approx 1$（数值积分，容差 1e-3）；
3. Gauss–Hermite 矩 vs 蒙特卡洛矩（ $N=2\times10^5$，相对容差 5%）；
4. 单调函数的求逆自洽： $f(g(y))=y$；
5. 临界点定位校验：gelu-exact 极小点应在 $x\approx-0.75$ 附近且 $f$ 值 $\approx-0.170$。

## 9. 文档与登记

- `README.md`：用法、文件结构、数学背景（变量替换与多原像求和、可积奇点、
  ReLU 点质量与稀疏、Gauss–Hermite 求积、各函数在 LLM 中的来历）；
- 根目录 `index.html` 工具列表加卡片（标签建议：激活函数 / 分布视角 / 理论对照）；
- markdown 行内公式空格遵守仓库规范（`$` 前为文字或全角标点需加空格），
  写完跑 `python tools/fix_md_math_spacing.py` 检查。

---

## 10. 补充设计：面板三 GLU 门控族（2026-08-26 用户批准，视图组合=热力图+分布+切片联动）

### 10.1 数学框架

门控输出 $y=u\cdot g(v)$： $u$ 被门控支， $g(v)$ 门支。输入建模为相关系数 $\rho$
的联合高斯（LLM 初始化基线 $\rho=0$）， $u,v$ 各自方差均为 $\sigma^2$。
由 $u\mid v\sim\mathcal N(\rho v,\ \sigma^2(1-\rho^2))$：

$$f_Y(y)=\int \varphi(v;\sigma)\,\frac{1}{|g(v)|}\,
\varphi\!\left(\frac{y}{g(v)};\ \rho v,\ \sigma^2(1-\rho^2)\right)dv$$

（对 $v$ 一维 Gauss–Hermite 求积即得精确理论密度。）

矩一维化： $\mathbb E[y]=\rho\,\mathbb E[v\,g(v)]$，
$\mathbb E[y^2]=\mathbb E[g(v)^2(\sigma^2(1-\rho^2)+\rho^2v^2)]$。

**条件分布切片**： $y\mid v_0$ 恰为精确高斯
$\mathcal N\bigl(\rho v_0 g(v_0),\ \sigma^2(1-\rho^2)g(v_0)^2\bigr)$——
边缘分布 = 各 $v$ 切片高斯的加权混合。

**门函数四个**（复用注册表条目）：SwiGLU(silu，默认) / GLU(sigmoid) / GEGLU(gelu-exact) / ReGLU(relu)。
ReGLU 有 0.5 点质量（ $v\le 0$ 门关闭 $y\equiv0$）；SwiGLU 在 $\rho=0$ 时
$y=0$ 处有乘积正态型对数尖峰（Bessel-K，可积），自检用 ReGLU $\rho=0$、 $y>0$
的闭式 $f(y)=K_0(y/\sigma^2)/(2\pi\sigma^2)$ 校验（复用 noise-ops-demo 的 logBesselK）。

### 10.2 布局与交互

- **A 门控地形热力图**（左）： $[-4\sigma,4\sigma]^2$ 网格热力 $z=u\cdot g(v)$
  （发散色负蓝正红），叠加 1/2/3σ 联合高斯等高线椭圆
  （参数式 $u=k\sigma\cos t,\ v=k\sigma(\rho\cos t+\sqrt{1-\rho^2}\sin t)$）；
  点击取 $v_0$ 并画水平虚线。
- **B 切片视图**（右，与热力图并排）：两种模式切换——
  「分布切片」灰底边缘密度 + 彩色条件高斯（ $g(v_0)\approx0$ 时提示门关闭）；
  「曲面切片」 $y=u\cdot g(v_0)$ 直线族（多次点击累积，可清空）。
- **C 输出分布图**（通栏）：理论密度 + 采样直方图 + 统计行，
  交互与面板二一致（ $\rho$ 滑块 −0.95~0.95 默认 0、 $\sigma^2$ 、 $N$ 、采样/重置、种子共享）。

### 10.3 实现拆解（8 任务，TDD + 逐步提交）

1. `functions.js` 加 `ActFns.gates` 注册；2. `sampler.js` 加 `sampleBivariate`/`applyGate`；
3. `theory.js` 加 GLU 段（`gluOutputDensity`/`gluOutputMoments`/`gluConditional`/`gluBinMass`/`gaussEllipse`）；
4. 面板三 HTML/CSS（热力图+切片并排、分布通栏）；5. 热力图+等高线+点击；
6. 切片双模式；7. 边缘分布+采样统计；8. README/根卡片更新。

自检新增：GLU 矩 vs MC（ $\rho\in\{-0.5,0,0.7\}\times$ 4 门）、条件分布矩 vs 条件采样、
ReGLU $K_0$ 闭式、区间质量 vs MC、椭圆参数式校验。
