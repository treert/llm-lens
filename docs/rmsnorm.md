# RMSNorm 为什么能工作

RMSNorm 把 $D$ 维向量除以其均方根（再乘可学增益 $g$）：

$$\mathrm{RMSNorm}(x) = \frac{x}{\mathrm{RMS}(x)}\,g, \qquad
\mathrm{RMS}(x) = \sqrt{\frac{\|x\|^2}{D}}$$

设输入 $x$ 的分量近似 iid、零均值、分量方差 $\sigma^2$。RMSNorm 能成立依赖
长度平方 $\|x\|^2 \sim \sigma^2\chi^2(D)$ 给出的两个数字
（演示见 `web-tools/noise-ops-demo` 面板二「长度平方」模式）：

## 1. 校准性（期望层面，精确）

RMS 是分量方差的良好估计：$\mathbb{E}[\|x\|^2/D] = \sigma^2$（长度平方的均值 = 分量方差 $\times\ D$）。

更进一步，归一化后分量的方差**精确**为 1，不需要任何近似——
由分量的交换对称性（$\mathbb{E}[x_i^2/\|x\|^2]$ 对 $i$ 相同，且对 $i$ 求和为 1）：

$$\mathbb{E}\!\left[\frac{x_i^2}{\|x\|^2}\right] = \frac{1}{D}
\;\Longrightarrow\;
\mathbb{E}\!\left[\left(\frac{x_i}{\mathrm{RMS}(x)}\right)^{\!2}\right] = 1$$

注意 $\mathrm{RMS}(x)$ 本身是随机变量，但对称性使该恒等式与分布细节无关
（各分量 iid 即可，甚至不需要高斯假设）。

"精确为 1"是期望意义（跨分量/跨样本平均）；
单个样本的归一化质量由下一节的稳定性保证。

## 2. 稳定性（单样本层面，渐近）

RMS 逐样本几乎恒定。由 $\mathrm{Var}(\|x\|^2) = 2D\sigma^4$，$\|x\|^2/D$ 的相对标准差为
$\sqrt{2/D}$；开方传到 RMS 上（delta 方法）再减半：

$$\frac{\mathrm{std}(\mathrm{RMS})}{\mathbb{E}[\mathrm{RMS}]} \approx \frac{1}{\sqrt{2D}}$$

| $D$ | 64 | 1024 | 4096 |
|---|---|---|---|
| RMS 相对涨落 | 8.8% | 2.2% | 1.1% |

$D$ 越大，每个样本被除以的因子越接近同一个常数 $\sigma$——这就是
"高维向量长度集中"（$\|x\| \approx \sigma\sqrt{D}$）在归一化中的表现：
RMSNorm 在高维几乎等价于除以一个常数，却又对分量尺度的真实变化
（$\sigma$ 漂移）保持自校准——这正是它比"固定除数"更鲁棒的原因。

## 3. 与注意力分数的联动：一条方差恒 1 的链

pre-norm 架构的 attention 分支：$x \to$ RMSNorm $\to W_Q/W_K$ 投影 $\to$ 点积 $\to \div\sqrt{H}$。
逐环看分量方差（投影与点积的规则推导见
[attention-score-distribution.md](attention-score-distribution.md) §1–§2）：

| 环节 | 运算 | 情形 | 方差流 |
|---|---|---|---|
| RMSNorm | $x\,/\,\sqrt{\|x\|^2/D}$ | — | $\sigma^2 \to$ **1**（精确） |
| 投影 $q = W_Q A$，$k = W_K B$ | $H \times D$ 矩阵，$\sigma_w^2 = 1/D$ | 一方固定 | $1 \to$ **1** |
| 点积 $s = q \cdot k$ | $H$ 维求和 | 双方随机 | $\to H$ |
| 缩放 | $s\,/\,\sqrt{H}$ | — | $\to$ **1** |

即初始化时注意力分数的方差稳定为 1，且每一环都不随维度失控：
RMSNorm 把上游任意尺度的输入先归一，投影保持方差，
点积的膨胀被 $\sqrt{H}$ 缩放精确抵消。

三条严格性注记：

- RMSNorm 后分量不再是高斯，且因共享 RMS 因子有 $O(1/D)$ 的弱相关；
  方差链分析只依赖二阶矩（$\mathbb{E}[x_i^2]$、分量近似不相关），结论不受影响。
- LayerNorm 同理（先减均值再多一步中心化，方差结论相同）。
- "方差恒 1"只是初始化时的结论（$\sigma_w^2 = 1/D$ 对应"平"的谱）。
  训练后 $M = W_Q^\top W_K$ 的奇异值谱一般变得集中，
  $\mathrm{Var}(s) = \sigma^4\|M\|_F^2$ 由谱决定、通常偏离 1（均值 0 不受影响）——
  见 [attention-score-distribution.md](attention-score-distribution.md) §4。

## 4. 输出侧与残差主干

方差链的另半段：分数经 softmax 加权、$W_O$ 投影、残差累加回到主干。

### 4.1 $W_O$：多头拼接的 $\times M$ 与 $1/(MH)$

$M$ 个头（各 $H$ 维）拼接成 $MH$ 维后过 $W_O$（$D \times MH$），"一方固定"情形：

$$\mathrm{Var}(y_i) = MH \cdot \sigma_{w_O}^2 \cdot \sigma_o^2
\;\Longrightarrow\;
\sigma_{w_O}^2 = \frac{1}{MH} = \frac{1}{M}\times\frac{1}{H}$$

- 只按 $1/H$（单头维度）初始化才会放大 $M$ 倍；标准规则 $\sigma_w^2 = 1/D_{in}$
  对 $W_O$ 的扇入就是 $MH$，$1/M$ 内含其中，无需额外处理；
- **扇入是列数不是行数**：标准架构 $MH = D_{model}$ 只是数值巧合。
  反例：PaLM（$d_{model}=18432$、$48\times 256 = 12288$）、$MH > D_{model}$ 的小模型设计、
  GQA（$W_Q$/$W_K$/$W_V$/$W_O$ 扇入互不相同）——一律逐矩阵按列数初始化；
- softmax 加权留了余量：$o_h = \sum_j \alpha_j v_j$（$\alpha_j \ge 0$、$\sum_j \alpha_j = 1$），
  $\mathrm{Var}(o_h) = \sigma_v^2 \sum_j \alpha_j^2 \le \sigma_v^2$，
  均匀注意时缩到 $1/n$，one-hot 才不变。

### 4.2 残差累加与 $1/\sqrt{2N}$

残差流是累加结构（每 block 两个分支，共 $2N$ 个）：

$$y_L = x + \sum_l f_l(x), \qquad \mathrm{Var}(y_L) \approx \mathrm{Var}(x) + 2N\cdot v$$

随深度线性放大。GPT-2 的对策是把每个分支末层（$W_O$、MLP 输出投影）初始化乘
$1/\sqrt{2N}$，使累加结果回到 $O(1)$。

**$N$ 是总层数（常数），不是层号**：所有层共享同一缩放因子。效果是**同层内**的
差异——残差末层的元素比其他矩阵（$W_Q$、MLP 上投影等）小 $1/\sqrt{2N}$，
而非"越深的层缩得越多"。统一缩放即可，是因为总方差 ≈ 各分支贡献**之和**
（近似不相关）：每个累加项出同一份折扣，总和即回 $O(1)$，项数已计入常数 $2N$。
因此该缩放也体现在训练后权重的层内统计上（$W_O$/下投影元素方差系统性小于
$W_Q$/上投影，偏离倍数反映训练对初始化比例的改写）。注意：

- 这是 **pre-norm 特有**的需求：主干无归一化，只能初始化时预除；
  post-norm 每层末尾 LayerNorm 事后压回，对初始化容错（代价在梯度侧——
  深层 post-norm 训练不稳，这正是转向 pre-norm 的原因）；
- 只是初始化技巧：训练后权重漂移，主干方差由训练动态重新决定；
- Kimi K3 的 AttnRes 把跨 block 累加换成 softmax 凸组合，方差有界、
  不再需要随深度的修正，见 [kimi-k3.md](kimi-k3.md)「AttnRes」一节。

### 4.3 两条备注

- **方差归一 ≠ 保距**：本文所有规则只保持分量二阶矩（最弱一档约束）。
  更强的一档是保距/等距：正交初始化、dynamical isometry、
  残差分支零初始化（ReZero、把 $W_O$ 初始化为 0 使分支近似恒等映射）。
- **工程现实**：Megatron 系常用全局固定 std（如 0.02，等效扇入 2500）+ 输出层缩放，
  不逐矩阵按扇入。能粗放的前提是失配在同数量级内，
  且有 pre-norm 归一化、Adam 逐参数归一、softmax 温度容忍三重兜底；
  失配差出数量级照样发散（熵坍缩/logit 发散，见 [qk-spectrum.md](qk-spectrum.md) §3）。

## 5. 演示对应

- `web-tools/noise-ops-demo` 面板一「x²」模式：看 $\mathbb{E}[x^2] = \sigma^2$（平方的均值 = 方差本身）；
- 面板二「长度平方 $\|x\|^2$」模式：看卡方分布与长度集中（均值 $D\sigma^2$、相对涨落 $\sqrt{2/D}$）；
- 面板二「投影点积」模式：看 $\div\sqrt{H}$ 后分数方差恒为 $\sigma^4$（$\sigma=1$ 时即 1），
  拖 $H$ 只剩形状变化——就是 §3 链条末环的直观演示。
