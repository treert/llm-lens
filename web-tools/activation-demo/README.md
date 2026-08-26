# activation-demo：LLM 激活函数——曲线、导数与噪音分布

交互式网页工具，从**函数形状**与**输出分布**两个视角理解 LLM 常见激活函数。

三个面板：

1. **函数与导数**：11 个激活函数叠加对比——经典（Sigmoid、Tanh、ReLU、Leaky ReLU、
   ELU、Softplus）与 LLM 现役（GELU 精确/tanh 近似双版本、SiLU/Swish、Mish、ReLU²），
   可切换只看函数 / 只看导数 / 双图联动；带参函数（Leaky ReLU 的 $\alpha$、ELU 的
   $\alpha$、SiLU 的 $\beta$）有滑块。附主流模型的激活函数对照表
   （SwiGLU 等门控变体留待后续面板）。
2. **噪音过激活的分布**：输入 $x\sim\mathcal N(0,\sigma^2)$，理论输出密度
   （变量替换 + 多原像求和）与蒙特卡洛直方图对照，Gauss–Hermite 理论矩 vs 样本矩。
   ReLU 族展示 $y=0$ 处**点质量**（精确稀疏），GELU/SiLU/Mish 展示非单调极小值处的
   **可积密度奇点**，饱和函数展示质量向两端的堆积。
3. **GLU 门控族（二维）**： $y=u\cdot g(v)$， $(u,v)$ 为相关系数 $\rho$ 的联合高斯。
   门控地形热力图（点击选切片 $v_0$）+ 切片视图（条件高斯分布 / 曲面斜率族）+
   边缘输出分布（梯形积分理论密度 vs 蒙特卡洛）。覆盖 SwiGLU / GLU / GEGLU / ReGLU
   四门——SwiGLU 在 $y=0$ 处有乘积正态型对数尖峰，ReGLU 有 0.5 点质量，
   $\mathbb E[y]=\rho\,\mathbb E[v\,g(v)]$ 演示"相关性即门控注意力"。

## 用法

直接用浏览器打开 `index.html` 即可（ECharts 走 CDN，需要联网）。

- 理论曲线随参数即时更新；蒙特卡洛**不自动采样**——点面板二的「采样」叠加直方图，
  参数变更后旧样本失效（按钮高亮提示重新采样）；
- 「固定种子」控制复现性（影响样本，不影响理论结论）；
- 数值自检：`node test/selftest.js`（解析导数差分、密度闭式抽查、
  区间理论质量 vs 蒙特卡洛、Gauss–Hermite 矩校验、临界点定位）。

## 文件结构

- `index.html`：页面骨架
- `js/functions.js`：激活函数注册表（fn/dfn/参数规格/说明），无 DOM 依赖
- `js/theory.js`：输出密度（单调段切分 + 二分求逆 + 多原像求和）、
  Gauss–Hermite 矩、临界点定位，无 DOM 依赖
- `js/sampler.js`：mulberry32 + Box–Muller、直方图、样本矩，无 DOM 依赖
- `js/app.js`：UI 状态同步与 ECharts 渲染
- `test/selftest.js`：node 自检

---

## 数学背景

设定： $x\sim\mathcal N(0,\sigma^2)$， $y=f(x)$ 为逐元素激活。

### 1. 输出密度：变量替换与多原像求和

在 $f$ 的每个单调段上， $y$ 的原像 $x_i$ 满足 $f(x_i)=y$，密度为

$$f_Y(y)=\sum_i \frac{f_X(x_i)}{|f'(x_i)|}$$

- 单调函数只有一个原像（如 sigmoid 的原像是 logit： $x=\ln\frac{y}{1-y}$）；
- GELU/SiLU/Mish 在负半轴有一个极小值点 $c$： $f(c)<y<0$ 时有两个原像，
  $y<f(c)$ 时没有， $y>0$ 时一个；
- ReLU 是退化情形（见 §3）。

实现上（`theory.js`）先数值定位临界点（ $f'=0$：符号扫描 2001 点 + 二分）切出单调段，
再对每个 $y$ 逐段二分求逆（80 次迭代）。单调性不写死——SiLU 的 $\beta$ 很小时
函数近线性单调， $\beta$ 大时出现凹陷，全靠运行时检测。

### 2. 可积奇点

极小值附近 $f'\to 0$，密度公式的分母趋于 0： $f_Y$ 在 $y=f(c)$ 处发散。
这是**可积奇点**（van Hove 型）： $f(x)-f(c)\approx\frac12 f''(c)(x-c)^2$，故

$$f_Y(y)\sim (y-f(c))^{-1/2},\qquad y\to f(c)^+$$

平方根奇异可积。图上表现为尖峰（网格未必恰好命中奇点，峰高有限但可任意大）。
GELU 在 $\sigma=1$ 时峰在 $y\approx-0.170$（ $c\approx-0.752$），SiLU（ $\beta=1$）
在 $y\approx-0.278$（ $c\approx-1.278$）。

### 3. ReLU 的点质量与精确稀疏

ReLU 把 $x\le 0$ 的一半质量全部压到 $y=0$：输出分布是"原子（atom）+ 连续密度"
的混合，

$$P(y=0)=\Phi(0)=\frac12,\qquad f_Y(y)=\varphi(y;\sigma),\ y>0$$

连续密度只积到 $1/2$。图上用红色虚线单独标注点质量，并与直方图中
"恰好为 0"的样本比例对照——这就是 ReLU 产生**精确稀疏激活**的几何图像：
负半轴的连续质量塌缩成一个点。顺带均值从 0 漂到 $\sigma/\sqrt{2\pi}$，
方差 $\frac{\sigma^2}{2}(1-1/\pi)$。

ReLU² 同理（点质量同为 $1/2$），正半轴再经平方把密度拉向 0：
均值 $=\sigma^2/2$，分布尖峰重尾。

### 4. 理论矩：Gauss–Hermite 求积

均值/方差不走密度积分，直接对输入分布求积（64 点）：

$$\mathbb E[g(x)]\approx\frac{1}{\sqrt\pi}\sum_{i=1}^{64} w_i\,g\bigl(\sigma\sqrt2\,t_i\bigr)$$

取 $g=f$ 与 $g=f^2$ 得 $\mathbb E[y]$ 与 $\mathbb E[y^2]$，
方差 $=\mathbb E[y^2]-\mathbb E[y]^2$。节点/权重用 Numerical Recipes 的 gauher
现场算（归一化物理学家 Hermite 多项式的 Newton 迭代，初值用渐近递推）。

GELU 均值有闭式可校验：由 Stein 引理 $\mathbb E[x\,g(x)]=\sigma^2\mathbb E[g'(x)]$
取 $g=\Phi$（ $g'=\varphi$），

$$\mathbb E[\mathrm{gelu}(x)]=\sigma^2\int\varphi(x)\,\frac{1}{\sigma}\varphi\!\left(\frac{x}{\sigma}\right)dx
=\frac{\sigma^2}{\sqrt{2\pi(1+\sigma^2)}},
\qquad \sigma=1 \Rightarrow \frac{1}{2\sqrt\pi}\approx 0.2821$$

自检以此校验求积正确性。

### 5. 各函数在 LLM 中的来历

- **Sigmoid / Tanh**：RNN 时代标配。两端饱和梯度 $\to 0$ 是深层训练的大敌；
  但 sigmoid 以"门"的形式活在 SiLU/SwiGLU 里。tanh 为奇函数保持零均值，
  分布图上两者的质量都被挤到区间两端（边界密度对数发散，可积）。
- **ReLU 族**：计算零成本 + 正半轴梯度恒 1，但负半轴梯度恒 0（死区）；
  输出精确稀疏（§3）。Leaky ReLU 用斜率 $\alpha$ 换无死区；
  ReLU² 在部分新模型中回归（稀疏 + 更陡的正半轴）。
- **ELU / Softplus**：平滑化尝试。ELU 负半轴指数饱和到 $-\alpha$ 把均值拉回，
  曾号称自归一化；Softplus 是 ReLU 的 $C^\infty$ 平滑版。
- **GELU**：BERT/GPT 系标配。用高斯 CDF 当"软门"： $x\,\Phi(x)$ 可解释为
  "输入 × 输入为正的概率"。tanh 近似（GPT-2 实际使用）与精确版肉眼难辨——
  面板一勾选两个版本叠加可见差异量级。
- **SiLU / Swish**： $x\,\sigma(\beta x)$，LLaMA/Qwen/Kimi 等 SwiGLU 的门支。
  $\beta$ 连续插值： $\beta\to 0$ 趋线性 $x/2$， $\beta\to\infty$ 趋 ReLU。
- **Mish**：与 SiLU 形状相似，负半支极小值更浅。

门控变体（SwiGLU、GeGLU 等 $\mathrm{GLU}(x)=(xW)\odot g(xV)$ 双分支结构）
是二维输入的，见面板三与下一节。

### 6. GLU 门控族： $y = u\cdot g(v)$

现代 LLM 的 FFN 激活不是标量函数而是双分支门控：两个投影
$u=xW_1$、 $v=xW_2$ 按元素相乘 $u\odot g(v)$。输入建模为相关系数 $\rho$
的联合高斯（初始化时两投影独立， $\rho=0$ 是随机基线； $u$、 $v$ 各自方差 $\sigma^2$）。

**条件分布是精确高斯**： $u\mid v\sim\mathcal N(\rho v,\ \sigma^2(1-\rho^2))$，故

$$y\mid v_0 \sim \mathcal N\bigl(\rho v_0\,g(v_0),\ \sigma^2(1-\rho^2)\,g(v_0)^2\bigr)$$

边缘分布就是各 $v$ 切片高斯的加权混合（面板三「分布切片」可视）：

$$f_Y(y)=\int \varphi(v;\sigma)\,\frac{1}{|g(v)|}\,
\varphi\!\left(\frac{y}{g(v)}-\rho v;\ \sigma^2(1-\rho^2)\right)dv$$

实现上（`theory.js`）对 $v$ 做梯形积分（2001 箱中点法）：被积函数在 ReGLU
门等处有尖峰（峰值在 $v\approx y$），GH 求积收敛差，梯形精度 $\sim 10^{-8}$
（与 ReGLU 的 $K_0$ 闭式对照验证）。

**矩一维化**（64 点 GH 足够光滑）：

$$\mathbb E[y]=\rho\,\mathbb E[v\,g(v)],\qquad
\mathbb E[y^2]=\mathbb E\bigl[g(v)^2\,(\sigma^2(1-\rho^2)+\rho^2v^2)\bigr]$$

第一式是面板三的主角：**输出均值的方向完全由 $\rho$ 的符号决定**——
正相关时"门开（ $v$ 大）信号（ $u$ ）也大"，输出均值漂正；负相关压负。
初始化基线 $\rho=0$ 均值恒 0；训练学出的投影相关，就是门控的"注意力"。

**四个门的分布形态**（ $\rho=0$）：

- **SwiGLU**： $v\approx 0$ 时 $g(v)\approx v/2$， $y\approx u\cdot v/2$ 趋近乘积正态——
  $y=0$ 处有 Bessel-$K_0$ 型**对数尖峰**（可积，与 noise-ops-demo 呼应），无点质量；
- **ReGLU**： $v\le 0$ 门关闭， $y\equiv 0$ → **0.5 点质量**（与 ReLU 面板呼应），
  且 $\rho=0$ 时 $y>0$ 支有闭式 $f(y)=K_0(y/\sigma^2)/(2\pi\sigma^2)$（自检校验）；
- **GLU(σ)**：门恒正，输出符号跟随 $u$，分布在 $y=0$ 附近平滑；
- **GEGLU**：介于 SwiGLU 与 GLU 之间，负 $v$ 门近似关但不严格为 0。

**门控地形热力图**： $z=u\cdot g(v)$ 发散色（蓝负红正）+ 联合高斯 1/2/3σ
等高线椭圆（参数式 $u=k\sigma\cos t,\ v=k\sigma(\rho\cos t+\sqrt{1-\rho^2}\sin t)$，
$\rho$ 控制旋转扁缩）。SwiGLU 的地形一目了然：上半平面门开（ $z$ 随 $u$ 渐变），
下半平面门关（ $z\approx 0$ 全白）——等高线椭圆的质心落在哪个区域，
决定了输出分布的形态。
