# proj-dot-demo：双随机投影的点积分布（Variance-Gamma 与夹角的遗忘）

交互式网页工具，演示注意力 score 的随机初始化基线：单位向量 $A, B \in \mathbb{R}^D$
（$\rho = \cos(A,B)$ 精确受控）经过独立高斯矩阵 $M_a, M_b \in \mathbb{R}^{H\times D}$
（元素方差 $1/D$，初始化缩放）投影后的点积

$$s = (M_a A)\cdot(M_b B) = A^\top M B,\qquad M := M_a^\top M_b$$

这是 $(xW_Q)\cdot(yW_K) = x^\top(W_QW_K^\top)\,y$ 的标量版；`spectrum-demo` 看 $M$ 的
奇异值谱，这里看 $s$ 本身的密度。

**三种随机方案，两种结论：**

1. **方案一（固定 $A,B$，随机 $M_a,M_b$）——精确可解，且与 $\rho$ 严格无关。**
   $u = M_aA \sim \mathcal N(0,\beta I_H)$、$v = M_bB \sim \mathcal N(0,\beta I_H)$ 且独立
   （$\beta = 1/D$），故 $s = \sum_{k=1}^H u_kv_k$ 是对称 **Variance-Gamma（Bessel-K）分布**，
   特征函数 $(1+\beta^2t^2)^{-H/2}$：

   $$f(s) = \frac{2\,|s|^{\nu} K_\nu(|s|/\beta)}{(2\beta)^{\nu+1}\sqrt{\pi}\,\Gamma(H/2)},
   \qquad \nu = \tfrac{H-1}{2}$$

   只依赖 $(\|A\|,\|B\|,H,\beta)$——$M$ 双各向同性，$A,B$ 可被各自独立旋转，相对夹角
   不进分布（演示中调 $\rho$ 滑块，样本甚至不重抽）。特例：$H=1$ 时原点对数发散；
   $H=2$ 时恰为 Laplace 分布；$H\to\infty$ 高斯化。方差 $H\beta^2$，**超额峰度 $6/H$**。

2. **方案二（固定 $M_a,M_b$，随机 $A,B$）——能感知 $\rho$，但敏感度是实例随机的。**
   条件矩（大维 concentrate 近似）：

   $$\mathbb E[s] = \rho\,\mathrm{tr}M/D,\qquad
   \mathrm{Var}(s) = \bigl[2\rho^2\|M_s\|_F^2 + (1-\rho^2)\|M\|_F^2\bigr]/D^2$$

   对随机无关的 $M_a,M_b$：$\mathrm{tr}M = \langle M_a, M_b\rangle_F$ 本身均值 0、
   std $\sqrt{H/D}$（**换种子方向随机翻转**），而 $\|M\|_F^2 \approx H$ concentrate——
   均值偏移相对涨落只有 $\sim\rho/\sqrt{D}$，被维度压没（默认参数下 $\approx 0.01\sigma$，
   直方图上看不见，统计表的「均值信噪比」行可见）。

3. **方案三（固定相关矩阵 $M_b = \alpha M_a + \sqrt{1-\alpha^2}\,G$）——$\rho$ 效应
   成为确定性信号。** $\mathrm{tr}M \approx \alpha H$，均值偏移 $\rho\alpha H/D$ 的信噪比
   $\sim\rho\alpha\sqrt{H}$ 随维度**增强**——对应训练后 $W_QW_K^\top$ 脱离各向同性基线。

**两方案的关系**：$\rho=0$ 时方案二对矩阵系综取平均（页面上「退火平均」开关，24 批矩阵
合并）后与方案一**严格同一分布**；$\rho\neq0$ 时方案一严格不动，方案二的偏移系综平均为 0。
自由概率语言：随机无关的 $M_a,M_b$ 使 $M$ 与 $(A,B)$ 的取向渐近自由，QK 训练的本质
就是打破这种自由性。对照恒等式：单投影保内积 $\mathbb E[(PA)\cdot(PB)] = \sigma^2H(A\cdot B)$，
双投影遗忘内积 $\mathbb E[(M_aA)\cdot(M_bB)] = 0$（随机初始化的注意力对 token 相关性是盲的）。

## 用法

直接用浏览器打开 `index.html` 即可（ECharts 走 CDN，需要联网）。

- **方案切换**：一 / 二 / 三（相关矩阵）。方案二、三可设矩阵种子、「换一批矩阵」
  观察 $\mathrm{tr}M$ 变号；「退火平均」展示与方案一的恒等。
- **手动采样**：打开页面不采样（只画理论曲线）；改方案/H/D/样本数/种子/退火/显式
  只标脏并即时刷新理论曲线（直方图沿用已缓存样本），点**运行采样**才真正重采样。
  $\rho$ 不进样本 key（方案一/三不换样本、方案二只改实例均值标注），故 $\rho$ 滑块
  保持即时重渲染——这正是"分布与 ρ 无关"的演示点。未采样或参数已改时按钮旁有提示。
- **H、D 滑杆**（对数刻度）：投影维与输入维，$c = H/D$ 在附注中显示。
- **$\rho$ 滑块**：方案一下分布纹丝不动（核心现象）；方案二、三下观察均值微移。
- **α 滑块**（方案三）：矩阵相关强度，$\mathrm{tr}M \approx \alpha H$。
- **显式矩阵采样**（方案一）：取 $A=e_1$、$B=\rho e_1+\sqrt{1-\rho^2}e_2$ 显式混合
  （双各向同性下不失一般性）——代码里 $\rho$ 出现，但 $\rho g_1+\sqrt{1-\rho^2}g_2$
  与 $g_1$ 同分布且与 $u$ 独立：$\rho$ 在单次实现中存在，在分布中消失。
- **横轴切换**：归一 $t = s/(\sqrt H/D)$（跨参数比较形状，只剩峰度 $6/H$）或原始 $s$。
- 统计表：样本均值/标准差/超额峰度 vs VG 理论 vs 实例高斯；方案二、三附加
  $\mathrm{tr}M$、$\|M\|_F^2$、均值信噪比。

## 文件结构

- `index.html`：页面骨架与公式块
- `js/theory.js`：纯数学层（对数域 Bessel-K：偶数 $H$ 走半整数阶初等和、奇数 $H$ 走
  缩放递推；VG 密度与矩；方案二条件矩），无 DOM 依赖
- `js/app.js`：种子化 PRNG（mulberry32 + Box-Muller）、蒙特卡洛采样（分块异步）、
  矩阵不变量（$\mathrm{tr}M$、$\|M\|_F^2 = \langle M_aM_a^\top, M_bM_b^\top\rangle_F$、
  $\mathrm{tr}(M^2) = \mathrm{tr}((M_aM_b^\top)^2)$，$O(H^2D)$ 免构造 $D\times D$ 的 $M$）、
  ECharts 渲染
- `css/style.css`：样式
- `test/theory-selftest.js`：theory.js 的 node 自检
  （`node web-tools/proj-dot-demo/test/theory-selftest.js`）
- `test/plot_proj_dot_density.py`：Python 蒙特卡洛对照（四子图），
  输出 `<仓库根>/output/proj-dot-density.png`

## 数值验证

- `test/theory-selftest.js`：Bessel-K 锚点值（$K_0$、$K_1$、$K_2$、$K_3$ 与半整数阶
  闭式）、整数阶递推恒等式、VG 密度的特例恒等（$H=2$ 逐点等于 Laplace、$H=1$ 等于
  $K_0$ 形式、峰值 $\Gamma((H{-}1)/2)/(2\beta\sqrt\pi\Gamma(H/2))$）、归一化与解析矩
  （$H\beta^2$、$3H(H{+}2)\beta^4$，对数网格数值积分）、$H=64$ 与高斯的 Edgeworth
  量级偏差、方案二矩公式；
- Python 侧蒙特卡洛对照见 `test/plot_proj_dot_density.py`：方案一两种 $\rho$ 统计一致
  且落在 VG 曲线上；方案二同一批矩阵的均值微移 $\rho\,\mathrm{tr}M/D$ 与样本均值吻合；
  方案三 $\mathrm{tr}M \approx \alpha H$ 的大偏移与实例高斯曲线吻合。
