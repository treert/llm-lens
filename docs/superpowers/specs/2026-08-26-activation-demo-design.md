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
