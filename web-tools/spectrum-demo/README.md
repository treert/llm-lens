# spectrum-demo：随机矩阵的奇异值谱（MP 律与 Fuss-Catalan）

交互式网页工具，展示三条理论密度曲线的对比：

- **单矩阵**： $H \times D$ 高斯矩阵 $A$（元素方差 $1/D$，即初始化缩放）的平方奇异值
  服从 **Marchenko-Pastur 律** $MP_c$；
- **乘积（D×D 摆法）**： $M = A^\top B$（ $A, B$ 独立）的非零平方奇异值服从**自由乘性卷积**
  $MP_c \boxtimes MP_c$； $c = 1$ 时即 **Fuss-Catalan 分布** $FC_2$，支撑 $[0, 27/4]$；
- **乘积（H×H 摆法）**： $\widetilde{M} = AB^\top$ 的平方奇异值服从
  $c \cdot (MP_c \boxtimes MP_1)$（图中为 ÷c 归一曲线）——同一对矩阵换乘法维度，
  极限律尺度和形状都改变。

大维极限下三者都只依赖比值 $c = H/D$。归一口径下均值均为 1，乘积把方差从 $c$ 拉宽到 $2c$。
对应 `docs/qk-spectrum.md` §3 的初始化基线（ $QK$ 谱的随机对照）。

**数学背景（MP 律、自由乘积、三次方程解法、支撑端点公式、AB/BA 辨析）已提取到
`docs/math/random-matrix-spectra.md`； $\boxtimes$ 的概念入门见
`docs/math/free-probability.md`。**

只画理论曲线，无蒙特卡洛（模拟验证在 Python 侧做，见 `test/plot_fuss_catalan.py`）。

## 用法

直接用浏览器打开 `index.html` 即可（ECharts 走 CDN，需要联网）。

- **c = H/D 滑杆**（对数刻度 $10^{-3} \sim 1$）与数值输入框联动；预设按钮：
  $c=1$（ $FC_2$）、 $1/4$、 $1/16$、 $128/7168 \approx 0.018$（LLM 注意力头的典型比值）。
- **$\sigma^2 / \sigma$ 轴切换**： $\sigma^2$ 轴（平方奇异值）是理论的自然变量；
  $\sigma$ 轴做变量替换 $p_\sigma(s) = 2s\,p_{\sigma^2}(s^2)$，此视角下 $c=1$ 的
  单矩阵谱即 quarter-circle 律 $p(s) = \sqrt{4-s^2}/\pi$。
- **曲线显隐与横轴切换**（图表面板顶部）：三条曲线可独立显隐
  （未归一时各曲线横轴量级悬殊，适合单看某一条），σ²/σ 轴切换同在一处；
  ABᵀ 紫线为 $\sigma^2(AB^\top)/c$ 的密度 $MP_c \boxtimes MP_1$，
  $c=1$ 时与红线重合（都退化为 $FC_2$），可作自洽性对照。
- **元素方差 $\sigma_w^2$ 选项**：演示未归一（ $\sigma_w^2 = 1/\sqrt D$ 或 $1$，按 $D = 2048$）
  的效果——各曲线形状不变、仅整体伸缩，但单矩阵乘 $\lambda = \sigma_w^2 D$、
  乘积乘 $\lambda^2$（收缩维数不同），三者量级悬殊，直观说明归一化的必要性；
- 竖直点线为各分布的支撑端点；统计表给出均值、方差、左右端点、 $\sigma_{\max}$ 对比，
  下方公式块列出全部闭式（含尺度规则）。

## 文件结构

- `index.html`：页面骨架
- `js/theory.js`：纯数学公式层（MP 密度、乘积谱密度的三次方程解法、支撑端点），无 DOM 依赖
- `js/app.js`：UI 状态同步与 ECharts 渲染
- `css/style.css`：样式
- `test/theory-selftest.js`：theory.js 的 node 自检
  （`node web-tools/spectrum-demo/test/theory-selftest.js`）
- `test/plot_fuss_catalan.py`：Python 蒙特卡洛对照（随机矩阵直方图 vs 理论曲线），
  输出 `<仓库根>/output/fuss-catalan.png`

## 数值验证

- `test/theory-selftest.js`：锚点值（与 Python/numpy 独立实现交叉对照到 $10^{-9}$）、
  归一化、解析矩（ $1$、 $1+2c$、 $1+c$）、支撑端点、支撑外密度恒零、
  $c=1$ 时 $AB^\top$ 曲线与 $FC_2$ 逐点重合；
- Python 侧蒙特卡洛对照（随机矩阵直方图 vs 理论曲线）见 `test/plot_fuss_catalan.py`，
  输出 `<仓库根>/output/fuss-catalan.png`。
