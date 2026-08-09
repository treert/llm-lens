# proj-len-demo：瓶颈投影链的长度平方分布（伽马、K_ν 乘积与深度对数正态）

交互式网页工具，演示单位向量 $x \in \mathbb{R}^D$ 经过**瓶颈块链**后的长度平方

$$s = \|(AB)^L x\|^2,\qquad B \in \mathbb{R}^{M\times D}\ (\text{方差 }1/D),\quad
A \in \mathbb{R}^{D\times M}\ (\text{方差 }1/M)$$

的分布：$A$、$B$ 为独立高斯矩阵，配对 fan-in（$1/D$ 下投、$1/M$ 上投）使
$\mathbb E\|ABx\|^2 = \|x\|^2$——**长度期望逐块保值（均值恒 1）**。$M \in [1, 4D]$
控制瓶颈宽度。这是 FFN / 适配器 / LoRA 式"降维-升维"结构的初始化零模型；
观测目标可选完整链 $y=(AB)^Lx$ 或中间层 $z=Bx$（压缩分布）。
姊妹篇：`spectrum-demo`（奇异值谱）、`proj-dot-demo`（投影点积）。

**核心结构：卡方因子化（精确）。** 中间层 $\|Bx\|^2 = (\|x\|^2/D)\,\chi^2_M$；
条件于 $z$，$\|Az\|^2 = (\|z\|^2/M)\,\chi^2_D$ 且卡方部分与 $\|z\|^2$ 独立，递推得

$$s = \prod_{l=1}^{L}\frac{\chi^2_M\,\chi^2_D}{DM}
\qquad(\text{2L 个独立卡方因子})$$

与 $x$ 的方向严格无关（各向同性——与 proj-dot-demo 的"夹角遗忘"同源）。

**三条主线：**

1. **中间层 $z=Bx$：伽马分布** $\mathrm{var}_B\cdot\chi^2_M$（$M{=}1$ 原点 $s^{-1/2}$
   发散；$M{=}2$ 指数分布；大 $M$ 高斯化）。$B$ 的元素方差可选 $\mathrm{var}_B = 1/D$
   （配对下投，均值 $M/D$——瓶颈压缩把典型长度压到 $\sqrt{M/D}$）或
   $\mathrm{var}_B = 1/M$（均值归一 $=1$、方差 $2/M$，与完整链同刻度，只剩 $M$
   控制的涨落）。
2. **单块 $AB$：不同形状伽马乘积 = 广义 Bessel-$K_\nu$ 闭式**
   $f(s) = \frac{2s^{\bar k-1}K_{k_1-k_2}(2\sqrt{s/\theta_1\theta_2})}
   {\Gamma(k_1)\Gamma(k_2)(\theta_1\theta_2)^{\bar k}}$
   （$k_1{=}M/2$、$\theta_1{=}2/D$、$k_2{=}D/2$、$\theta_2{=}2/M$、$\bar k{=}(k_1{+}k_2)/2$，
   $\nu = (M{-}D)/2$ 为任意实数阶，积分表示 + 正向递推在 log 域计算）。
   **均值恰为 1**（配对 fan-in 保长度）；右尾 $e^{-2\sqrt{s/\theta_1\theta_2}}$
   是拉伸指数——重尾化的第一层（对数纵轴下肉眼可见与伽马直线的差别）。
3. **$L\ge2$：深度对数正态。** $\ln s \approx \mathcal N(\mu,\sigma^2)$，
   $\mu = L[\psi(M/2){-}\ln(M/2) + \psi(D/2){-}\ln(D/2)] \approx -L(1/M{+}1/D)$，
   $\sigma^2 = L[\psi'(M/2){+}\psi'(D/2)]$。
   **均值恒 1，中位数 $\approx e^{-L(1/M+1/D)}$ 指数萎缩**——均值被重尾撑着，
   典型样本塌缩；瓶颈越窄（$M$ 小）萎缩越快。这给了 LayerNorm 存在意义的
   随机矩阵注脚：Norm 压掉的正是这种深度方向的对数正态涨落。

**quenched 对照（方案二：固定矩阵链，随机 $x$ 球面均匀）。** $s = x^\top W x$，
$W = P^\top P$（$P$ = 合成链；用结合律 $P = A(BA)^{L-1}B$ 合成，$BA$ 仅 $M{\times}M$，
避免 $O(LD^3)$）。球面二次型精确矩：

$$\mathbb E_x[s] = \mathrm{tr}W/D,\qquad
\mathrm{Var}_x(s) = \frac{2[\mathrm{tr}(W^2) - (\mathrm{tr}W)^2/D]}{D(D+2)}$$

**self-averaging：实例内方差 ≈ annealed 方差的大部分**，换一批矩阵直方图形状
不动、仅中心平移；实例中心 $\mathrm{tr}W/D$ 的跳动（Wick 全方差律）：

$$\text{中间层：}\ \mathrm{Var}(\mathrm{tr}W/D) = \frac{2M\,\mathrm{var}_B^2}{D};\qquad
\text{单块：}\ \mathrm{Var}(\mathrm{tr}W/D) = \frac{2M+4D+2}{MD^2}
\ \ (M{=}D \text{ 时} \approx 6/D^2)$$

随维度消失——因为 $\mathrm{tr}W = \|P\|_F^2$ 是**平方和**（强自平均），对照
proj-dot-demo 的 $\mathrm{tr}M = \langle M_a,M_b\rangle_F$ 是**符号和**
（涨落 $\sqrt{H/D}$ 不消失）。同为二次型，长度统计稳健、点积均值脆弱。

## 用法

直接用浏览器打开 `index.html` 即可（ECharts 走 CDN，需要联网）。

- **方案切换**：一（随机矩阵，annealed）/ 二（固定链，quenched）。方案二可换种子
  观察直方图不动、中心微跳。
- **手动采样**：打开页面不采样（只画理论曲线）；改方案/观测/方差/L/M/D/样本数/
  种子也只标脏并即时刷新理论曲线（直方图沿用已缓存样本），点**运行采样**才
  真正重采样——避免大维度下每次拖滑杆都触发重算。未采样或参数与缓存样本
  不一致时按钮旁有黄色提示。
- **观测切换**：完整链 $y=(AB)^Lx$ / 中间层 $z=Bx$。
- **中间层方差**（仅观测中间层时可选）：$1/D$（均值 $M/D$）/ $1/M$（均值归一）。
- **块数 L**：1（单块，$K_\nu$ 闭式）/ 2–16（对数正态近似；无初等闭式）。
- **M、D 滑杆**（对数刻度）：M 为瓶颈维，范围 [1, 4D]（上限随 D 收拢）；
  $M/D$ 在附注中显示。
- **横轴**：归一 ÷均值 或原始 $s$；线性或对数刻度（深链重尾建议对数）。
- **纵轴**：线性或对数（看拉伸指数尾）。
- 标线：均值（$1$ 或 $M/D$，灰虚线）与中位数 $\approx e^{-L(1/M+1/D)}$（橙点线）
  随 $L$ 分离——本演示最有冲击力的画面。
- 统计表：样本均值/标准差/中位数/$\ln s$ 矩 vs 理论；方案二附加 $\mathrm{tr}W/D$
  与中心跳动幅度。

## 文件结构

- `index.html`：页面骨架与公式块
- `js/theory.js`：纯数学层（Lanczos Γ、digamma/trigamma 渐近展开、任意阶
  Bessel-$K_\nu$（积分表示 + log 域 logaddexp 递推）、伽马/$K_\nu$ 乘积密度、
  链矩与对数域参数、quenched 球面矩、Marsaglia–Tsang 伽马/卡方采样），无 DOM 依赖
- `js/app.js`：种子化 PRNG、卡方因子采样（方案一，MT 采样每层 O(1)）、
  链合成 $P = A(BA)^{L-1}B$（结合律）+ 球面采样（方案二）、ECharts 渲染（对数坐标）
- `css/style.css`：样式
- `test/theory-selftest.js`：theory.js 的 node 自检
  （`node web-tools/proj-len-demo/test/theory-selftest.js`）
- `test/plot_proj_len_density.py`：Python 蒙特卡洛对照（四子图），
  输出 `<仓库根>/output/proj-len-density.png`

## 数值验证

- `test/theory-selftest.js`：$\psi/\psi'$ 锚点；任意阶 $K_\nu$ 对照
  （整数阶对 NR 系数版 $K_0/K_1$、半整数闭式 $K_{1/2},K_{3/2},K_{7/2}$、
  大阶渐近 $K_{40}(0.05)$）；伽马密度特例；$K_\nu$ 乘积密度的等形状退化、
  归一化与解析矩 $\mathbb E[P^m] = \frac{\Gamma(k_1{+}m)\Gamma(k_2{+}m)}
  {\Gamma(k_1)\Gamma(k_2)}(\theta_1\theta_2)^m$（对数网格数值积分）；链矩公式；
  quenched 球面矩与中心跳动；MT 采样的蒙特卡洛矩（均值/方差/$\mathbb E\ln$ 对 $\psi$）；
- Python 侧对照：中间层均值 $0.2500$ vs $M/D{=}0.25$（$\mathrm{var}_B{=}1/M$ 时
  $1.0006$ vs 恒 $1$、std $0.17691$ vs $\sqrt{2/M}{=}0.17678$）；单块均值 $1.0008$ vs 恒 $1$、
  std $0.19878$ vs 理论 $0.19826$；$(AB)^8$ 样本中位数 $0.8596$ vs LN 近似
  $0.8548$，均值/中位数 $1.162$ vs $e^{L(1/M+1/D)} = 1.169$；quenched 三个种子的
  实例均值逐一吻合 $\mathrm{tr}W/D$、实例 std 吻合球面公式，中心跳动（±0.01）与
  $\sqrt{(2M{+}4D{+}2)/(MD^2)} = 0.0166$ 同量级——self-averaging 成立。
