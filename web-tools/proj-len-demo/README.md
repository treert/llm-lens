# proj-len-demo：瓶颈投影块的长度平方分布（伽马与 K_ν 乘积卡方）

交互式网页工具，演示单位向量 $x \in \mathbb{R}^D$ 经过**单块瓶颈**后的长度平方

$$s = \|ABx\|^2,\qquad B \in \mathbb{R}^{M\times D}\ (\text{方差 }1/D),\quad
A \in \mathbb{R}^{D\times M}\ (\text{方差 }1/M)$$

的分布： $A$、 $B$ 为独立高斯矩阵，配对 fan-in（ $1/D$ 下投、 $1/M$ 上投）使
$\mathbb E\|ABx\|^2 = \|x\|^2$——**长度期望保值（均值恒 1）**。 $M \in [1, 4D]$
控制瓶颈宽度（可小于也可大于 $D$）。这是 FFN / 适配器 / LoRA 式"降维-升维"结构的
初始化零模型；观测目标可选完整块 $y=ABx$ 或中间层 $z=Bx$（压缩分布）。
姊妹篇：`spectrum-demo`（奇异值谱）、`proj-dot-demo`（投影点积）。

**核心结构：卡方因子化（精确）。** 中间层 $\|Bx\|^2 = \mathrm{var}_B\,\chi^2_M$；
条件于 $z=Bx$， $\|Az\|^2 = (\|z\|^2/M)\,\chi^2_D$ 且卡方部分与 $\|z\|^2$ 独立，故

$$s = \frac{\chi^2_M\,\chi^2_D}{DM}\qquad(\text{两个独立卡方因子})$$

与 $x$ 的方向严格无关（各向同性——与 proj-dot-demo 的"夹角遗忘"同源）。

**两条主线：**

1. **中间层 $z=Bx$：伽马分布** $\mathrm{var}_B\cdot\chi^2_M$（ $M{=}1$ 原点 $s^{-1/2}$
   发散； $M{=}2$ 指数分布；大 $M$ 高斯化）。 $B$ 的元素方差可选 $\mathrm{var}_B = 1/D$
   （配对下投，均值 $M/D$——瓶颈压缩把典型长度压到 $\sqrt{M/D}$）或
   $\mathrm{var}_B = 1/M$（均值归一 $=1$、方差 $2/M$，与完整块同刻度，只剩 $M$
   控制的涨落）。
2. **完整块 $AB$：不同形状伽马乘积 = 广义 Bessel-$K_\nu$ 闭式（精确）**
   $f(s) = \frac{2s^{\bar k-1}K_{k_1-k_2}(2\sqrt{s/\theta_1\theta_2})}
   {\Gamma(k_1)\Gamma(k_2)(\theta_1\theta_2)^{\bar k}}$
   （ $k_1{=}M/2$、 $\theta_1{=}2/D$、 $k_2{=}D/2$、 $\theta_2{=}2/M$、 $\bar k{=}(k_1{+}k_2)/2$，
   $\nu = (M{-}D)/2$ 为任意实数阶，积分表示 + 正向递推在 log 域计算）。
   **均值恰为 1**（配对 fan-in 保长度）；方差 $2/M+2/D+4/(MD)$；中位数
   $\approx e^{-(1/M+1/D)} < 1$——均值被重尾（上投 $A$ 注入的 $2/D$ 涨落）撑着，
   典型样本偏小。右尾 $e^{-2\sqrt{s/\theta_1\theta_2}}$ 是拉伸指数（对数纵轴下
   肉眼可见与伽马直线的差别）。

**quenched 对照（方案二：固定矩阵，随机 $x$ 球面均匀）。** $s = x^\top W x$，
$W = P^\top P$（ $P=AB$）。球面二次型精确矩：

$$\mathbb E_x[s] = \mathrm{tr}W/D,\qquad
\mathrm{Var}_x(s) = \frac{2[\mathrm{tr}(W^2) - (\mathrm{tr}W)^2/D]}{D(D+2)}$$

**self-averaging：实例内方差 ≈ annealed 方差的大部分**，换一批矩阵直方图形状
不动、仅中心平移；实例中心 $\mathrm{tr}W/D$ 的跳动（Wick 全方差律）：

$$\text{中间层：}\ \mathrm{Var}(\mathrm{tr}W/D) = \frac{2M\,\mathrm{var}_B^2}{D};\qquad
\text{完整块：}\ \mathrm{Var}(\mathrm{tr}W/D) = \frac{2M+4D+2}{MD^2}
\ \ (M{=}D \text{ 时} \approx 6/D^2)$$

随维度消失——因为 $\mathrm{tr}W = \|P\|_F^2$ 是**平方和**（强自平均），对照
proj-dot-demo 的 $\mathrm{tr}M = \langle M_a,M_b\rangle_F$ 是**符号和**
（涨落 $\sqrt{H/D}$ 不消失）。同为二次型，长度统计稳健、点积均值脆弱。

**实例分布的形状：右偏，不是高斯。** 固定 $W$ 后秩 $\le M$，在特征基下
$s=\sum_{i=1}^{M}\lambda_i u_i^2$（ $u$ 为球面坐标）——约 $M$ 个有效自由度的加权
卡方和，**正偏：众数 < 中位数 < 均值 $\mathrm{tr}W/D$**，偏度
$\gamma_1\sim\sqrt{8/M_{\mathrm{eff}}}$， $M_{\mathrm{eff}}=(\mathrm{tr}W)^2/\mathrm{tr}(W^2)$。
红虚线"实例高斯"只匹配前两阶矩、对称、峰钉在均值，所以直方图的峰相对它系统性
偏左 $\approx\sigma\gamma_1/2$、右尾更厚——这不是采样误差，换种子只平移中心、
偏度方向不变（ $M{=}64,D{=}256$ 时偏移约 $0.04$，肉眼可见）。紫曲线"实例偏态"
把精确形状（中间层伽马 / 完整块 $K_\nu$）标准化后**仿射变换到实例矩**：

$$f(s)=\frac1b\,f_0\!\Big(c+\frac{s-\mathrm{tr}W/D}{b}\Big),\qquad
b=\sqrt{\mathrm{Var}_{\mathrm{inst}}/\mathrm{Var}_0}$$

（ $c$、 $\mathrm{Var}_0$ 为 annealed 均值/方差）——保留右偏形状、峰位对齐直方图。
$\alpha>0$ 时特征值更分散、 $M_{\mathrm{eff}}$ 更小，偏得更明显。

## 用法

直接用浏览器打开 `index.html` 即可（ECharts 走 CDN，需要联网）。

- **方案切换**：一（随机矩阵，annealed）/ 二（固定矩阵，quenched）。方案二可换种子
  观察直方图不动、中心微跳。
- **观测切换**：完整块 $y=ABx$ / 中间层 $z=Bx$。
- **中间层方差**（仅观测中间层时可选）： $1/D$（均值 $M/D$）/ $1/M$（均值归一）。
- **矩阵相关 α**（仅方案二完整块可选）： $A = \alpha\sqrt{D/M}\,B^\top + \sqrt{1-\alpha^2}\,G$
  （ $G$ 为 $D{\times}M$ 高斯、方差 $1/M$；方差配平 $\alpha^2\frac{D}{M}\frac{1}{D}+(1{-}\alpha^2)\frac{1}{M}=\frac{1}{M}$，
  $A$ 的边缘仍是各向同性高斯，但与 $B$ 相关）。 $\alpha{=}0$ 独立（现状，trW/D 自平均到 1）；
  $\alpha\to1$ 时 $A\approx\sqrt{D/M}B^\top$， $P=AB\approx\sqrt{D/M}B^\top B$，
  $\mathrm{tr}W/D$ **显著偏离 1**（如 $M{=}64,D{=}256,\alpha{=}0.9$ 时 ≈1.24）——
  "上投矩阵学出了下投矩阵的结构"，打破自平均、实例曲线偏离理论乘积伽马（此时理论曲线
  标"α>0 矩阵相关，仅参考"）。对应 LoRA/FFN 训练后上下投矩阵产生的结构关联。
- **手动采样**：打开页面不采样（只画理论曲线）；改方案/观测/方差/M/D/样本数/种子
  只标脏并即时刷新理论曲线（直方图沿用已缓存样本），点**运行采样**才真正重采样。
- **分片流式采样**：采样与方案二的矩阵处理都按固定时间片（~24ms/块，按实测自适应
  块大小）切片、片间让出主线程，并把已采样本流式重绘——大维度下 UI 不卡，
  直方图随采样逐步收敛。
- **暂停 / 重置**：采样中可**暂停**（保留已采样本、按钮变"继续"）、**重置**
  （清空已采样本与矩阵缓存回"尚未采样"）。采样过程中改动任何采样参数会**自动重置**；
  横轴/纵轴/曲线勾选不触发重置（只重绘）。状态文本实时显示"采样中 count / N 样本…"。
- **M、D 滑杆**（对数刻度）：M 为瓶颈维，范围 [1, 4D]（上限随 D 收拢）；D ≤ 8192。
- **方案二不构造大矩阵（按 min(M,D) 选侧）**：完整块 $P=AB$ 是 $D{\times}D$ 但秩
  $\le\min(M,D)$，所以——
  - **$M\le D$（瓶颈）**：不构造 $P$。采样分步 $z=Bx$、 $\|Az\|^2$（ $O(MD)$/样本）；
    实例矩用 $M{\times}M$ 小矩阵 $G=(A^\top A)(BB^\top)$： $\mathrm{tr}W=\mathrm{tr}G$、
    $\mathrm{tr}(W^2)=\mathrm{tr}(G^2)$（迹轮换）， $O(M^2D)$。
  - **$M>D$（宽瓶颈）**：构造 $P=AB$（ $D{\times}D$，比 $M{\times}M$ 小），
    $\mathrm{tr}W=\|P\|_F^2$、 $\mathrm{tr}(W^2)=\|PP^\top\|_F^2$；采样直接用 $P$。
  - 内存估算（中间层 $8(MD{+}M^2)$、瓶颈侧 $8(2MD{+}3M^2)$、宽瓶颈侧 $8(2MD{+}2D^2)$
    字节）超 **2 GB** 时禁用"运行采样"并提示。 $D{=}8192$ 配小 $M$ 也可行。
- **实例矩的 Hutchinson 随机估计**：精确算 $\mathrm{tr}W$、 $\mathrm{tr}(W^2)$ 要
  $O(\min(M,D)^3)$（Gram/小矩阵），大矩阵时会让直方图空等上百秒。预估精确实例矩
  超 **5 秒**（按 $\sim10^8$ flops/s 粗估）就改用 **Hutchinson 迹估计**：只作用
  矩阵-向量乘法（ $\mathrm{tr}W=\mathbb E_v\|ABv\|^2$、 $\mathrm{tr}(W^2)=\mathbb E_v\|Wv\|^2$，
  $v$ 标准正态）， $O(k\cdot MD)$、探针数自适应到 ~3 秒，几秒出近似值、采样即开始。
  实例曲线/统计表标注"Hutchinson 估计"。 $k{=}64$ 时 trW 误差 ~0.2%、trW2 ~0.3%。
- **横轴**：归一 ÷均值 或原始 $s$；线性或对数刻度（重尾建议对数）。
- **纵轴**：线性或对数（看拉伸指数尾）。
- 曲线勾选：精确理论（蓝）、高斯近似（灰虚）、实例高斯（红，对称、峰在均值）
  与**实例偏态**（紫，精确形状仿射到实例矩，峰位贴合直方图——实例分布右偏，
  众数 < 均值，见上节）。
- 标线：均值（ $1$ 或 $M/D$，灰虚线）与中位数 $\approx e^{-(1/M+1/D)}$（橙点线）。
- 统计表：样本均值/标准差/中位数/$\ln s$ 矩 vs 理论；方案二附加 $\mathrm{tr}W/D$
  与中心跳动幅度。

## 文件结构

- `index.html`：页面骨架与公式块
- `js/theory.js`：纯数学层（Lanczos Γ、digamma/trigamma 渐近展开、任意阶
  Bessel-$K_\nu$（积分表示 + log 域 logaddexp 递推）、伽马/$K_\nu$ 乘积密度、
  块矩与对数域参数、quenched 球面矩、Marsaglia–Tsang 伽马/卡方采样），无 DOM 依赖
- `js/app.js`：种子化 PRNG、卡方因子采样（方案一，MT 采样 O(1)/样本）、
  方案二双侧矩阵处理（瓶颈侧 M×M 小矩阵 / 宽瓶颈侧 D×D 的 P，均分片）+ 分步球面采样、
  ECharts 渲染（对数坐标）
- `css/style.css`：样式
- `test/theory-selftest.js`：theory.js 的 node 自检
  （`node web-tools/proj-len-demo/test/theory-selftest.js`）
- `test/plot_proj_len_density.py`：Python 蒙特卡洛对照，
  输出 `<仓库根>/output/proj-len-density.png`

## 数值验证

- `test/theory-selftest.js`： $\psi/\psi'$ 锚点；任意阶 $K_\nu$ 对照
  （整数阶对 NR 系数版 $K_0/K_1$、半整数闭式 $K_{1/2},K_{3/2},K_{7/2}$、
  大阶渐近 $K_{40}(0.05)$）；伽马密度特例； $K_\nu$ 乘积密度的等形状退化、
  归一化与解析矩 $\mathbb E[P^m] = \frac{\Gamma(k_1{+}m)\Gamma(k_2{+}m)}
  {\Gamma(k_1)\Gamma(k_2)}(\theta_1\theta_2)^m $（对数网格数值积分）；块矩公式；
  quenched 球面矩与中心跳动；MT 采样的蒙特卡洛矩（均值/方差/$\mathbb E\ln$ 对 $\psi$）；
- 双侧实例矩（Node 数值对照）： $M\le D$ 时小矩阵 $G=(A^\top A)(BB^\top)$ 与
  $M>D$ 时显式 $P=AB$ 的 $\mathrm{tr}W$、 $\mathrm{tr}(W^2)$ 相对误差 $\sim10^{-14}$；
  分步采样 $\|A(Bx)\|^2$ 对照显式 $\|Px\|^2$ 最大相对误差 $\sim10^{-15}$；
- 浏览器实测：瓶颈侧 $D{=}8192$、 $M{=}512$ 完整块样本均值 $0.9996$、trW/D $0.9997$；
  宽瓶颈侧 $D{=}1024$、 $M{=}4096$ 样本均值 $1.0002$、trW/D $1.0002$——实例中心均贴合。
