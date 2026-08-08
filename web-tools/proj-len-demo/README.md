# proj-len-demo：随机投影链的长度平方分布（卡方、乘积卡方与深度对数正态）

交互式网页工具，演示单位向量 $X \in \mathbb{R}^D$ 经过高斯矩阵链后的长度平方

$$s = \|B_{L-1}\cdots B_1 A\,X\|^2$$

的分布：$A$ 为 $H\times D$（元素方差 $1/D$，初始化缩放），$B_l$ 为 $H\times H$
（方差 $1/H$，fan-in），$L$ = 矩阵总数。这是残差流上每层变换的初始化统计的
零模型；姊妹篇：`spectrum-demo`（奇异值谱）、`proj-dot-demo`（投影点积）。

**核心结构：卡方因子化（精确）。** 条件于上一层输出 $Y$，下一层
$\|BY\|^2 = (\|Y\|^2/H)\,\chi^2_H$，且卡方部分与 $\|Y\|^2$ 独立，递推得

$$s = \frac{1}{D}\,\chi^2_H \cdot \prod_{l=1}^{L-1}\frac{\chi^2_H}{H}
\qquad(\text{L 个独立卡方因子})$$

与 $X$ 的方向严格无关（各向同性——与 proj-dot-demo 的"夹角遗忘"同源）。

**三条主线：**

1. **$L=1$：伽马分布** $\chi^2_H/D$（$H{=}1$ 原点 $s^{-1/2}$ 发散；$H{=}2$ 指数分布；
   大 $H$ 高斯化）。均值 $c = H/D$：fan-in 缩放下 $\mathbb E\|BX\|^2 = \|X\|^2$ 逐层保值。
2. **$L=2$：独立伽马乘积 = Bessel-$K_0$ 闭式**
   $f(s) = \frac{2s^{k-1}K_0(2\sqrt{s/\theta_1\theta_2})}{\Gamma(k)^2(\theta_1\theta_2)^k}$
   （$k=H/2$，$\theta_1=2/D$，$\theta_2=2/H$）。均值仍 $c$，相对方差从 $2/H$ 涨到
   $\approx 4/H$；右尾 $e^{-2\sqrt{s/\theta_1\theta_2}}$ 是拉伸指数——重尾化的第一层
   （对数纵轴下肉眼可见与伽马直线的差别）。
3. **$L\ge3$：深度对数正态。** $\ln s \approx \mathcal N(\mu,\sigma^2)$，
   $\mu = -\ln D + L(\ln2 + \psi(H/2)) - (L-1)\ln H$，$\sigma^2 = L\,\psi'(H/2)$。
   **均值 $c$ 不变，中位数 $\approx c\,e^{-L/H}$ 指数萎缩**——均值被重尾撑着，
   典型样本塌缩（均值/中位数 $\approx e^{L/H}$）。这给了 LayerNorm 存在意义的
   随机矩阵注脚：Norm 压掉的正是这种深度方向的对数正态涨落；也与 dynamical
   isometry 文献相连（高斯矩阵谱宽 $4\sqrt c$ 不 concentrate，逐层涨落不可避免）。

**quenched 对照（方案二：固定矩阵链，随机 $X$ 球面均匀）。** $s = X^\top W X$，
$W = M^\top M$（$M$ = 合成链）。球面二次型精确矩：

$$\mathbb E_X[s] = \mathrm{tr}W/D,\qquad
\mathrm{Var}_X(s) = \frac{2[\mathrm{tr}(W^2) - (\mathrm{tr}W)^2/D]}{D(D+2)}$$

**self-averaging：实例内方差 ≈ annealed 方差的大部分**（球面约束下谱加权的
$(1{+}c)$ 修正被 $-(\mathrm{tr}W)^2/D$ 项抵消），换一批矩阵直方图形状不动、
仅中心平移；实例中心 $\mathrm{tr}W/D$ 的跳动（$L$ 层精确，逐层全方差律递推）

$$\mathrm{Var}(\mathrm{tr}W/D) = \frac{L(L-1+2c)}{D^2}
\qquad(\text{逐层全方差律递推，谱二阶矩 } m_2 = L{+}c)$$

（$L{=}1$ 即 $2H/D^3$；随 $L$ 平方增长——深链时平方和的自平均也变差）

随维度消失——因为 $\mathrm{tr}W = \|M\|_F^2$ 是**平方和**（强自平均），对照
proj-dot-demo 的 $\mathrm{tr}M = \langle M_a,M_b\rangle_F$ 是**符号和**
（涨落 $\sqrt{H/D}$ 不消失）。同为二次型，长度统计稳健、点积均值脆弱。

## 用法

直接用浏览器打开 `index.html` 即可（ECharts 走 CDN，需要联网）。

- **方案切换**：一（随机矩阵，annealed）/ 二（固定链，quenched）。方案二可换种子
  观察直方图不动、中心微跳。
- **层数 L**：1（卡方）/ 2（$K_0$ 乘积）/ 3–16（对数正态近似；L≥3 无初等闭式，
  精确密度是 Meijer G）。
- **H、D 滑杆**（对数刻度），$c = H/D$ 在附注中显示。
- **横轴**：归一 $\div c$ 或原始 $s$；线性或对数刻度（深链重尾建议对数）。
- **纵轴**：线性或对数（看拉伸指数尾）。
- 标线：均值 $c$（灰虚线）与中位数 $c\,e^{-L/H}$（橙点线）随 $L$ 分离——
  本演示最有冲击力的画面。
- 统计表：样本均值/标准差/中位数/$\ln s$ 矩 vs 理论；方案二附加 $\mathrm{tr}W/D$
  与中心跳动幅度 $\sqrt{2H/D^3}$。

## 文件结构

- `index.html`：页面骨架与公式块
- `js/theory.js`：纯数学层（Lanczos Γ、NR 系数 Bessel-$K_{0,1}$、digamma/trigamma
  渐近展开、伽马/$K_0$ 乘积密度、链矩与对数域参数、quenched 球面矩），无 DOM 依赖
- `js/app.js`：种子化 PRNG、卡方因子采样（方案一）、链合成 $M = B\cdots BA$
  （$O(LH^2D)$ 一次）+ 球面采样（每样本 $O(HD)$）、ECharts 渲染（对数坐标）
- `css/style.css`：样式
- `test/theory-selftest.js`：theory.js 的 node 自检
  （`node web-tools/proj-len-demo/test/theory-selftest.js`）
- `test/plot_proj_len_density.py`：Python 蒙特卡洛对照（四子图），
  输出 `<仓库根>/output/proj-len-density.png`

## 数值验证

- `test/theory-selftest.js`：$\psi/\psi'$ 锚点（$-\gamma$、$\pi^2/6$、$\pi^2/2$ 等）、
  伽马密度特例（$H{=}2$ 逐点等于指数分布、原点行为）、$K_0$ 乘积密度的特例恒等与
  归一化/解析矩（$\mathbb E[P^m] = [\Gamma(k{+}m)/\Gamma(k)]^2(\theta_1\theta_2)^m$，
  对数网格数值积分）、链矩公式、quenched 球面矩精确值；
- Python 侧对照：$L{=}1,2$ 均值/方差与理论吻合；$L{=}8$ 样本中位数 $0.2216$ vs
  LN 近似 $0.2205$，均值/中位数 $1.129$ vs $e^{L/H} = 1.133$；quenched 三个种子的
  实例均值逐一吻合 $\mathrm{tr}W/D$、实例 std 吻合球面公式，中心跳动（±0.008）与
  $\sqrt{L(L{-}1{+}2c)}/D = 0.0068$ 同量级——self-averaging 成立。
