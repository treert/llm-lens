# RoPE 的数学：旋转群表示、多频率展开与外推边界

通用数学笔记：位置（一个低信息量、高精确、与语义无关的整数）如何经由旋转编码
进入点积注意力，以及长上下文外推困难的数学根源。
注意力分数的**无位置**分量（内容匹配）见 `attention-score-distribution.md` 与
`qk-spectrum.md`；本文分析位置依赖的结构。K3 的 MLA 部分旋转参数见 `kimi-k3.md`。

## 0. 设定与记号

- 头维 $H$，2D 通道数 $H/2$；位置 $i, j \in \mathbb{Z}$，相对位置 $m = j - i$；
- 单通道旋转 $r(\phi) = \begin{pmatrix}\cos\phi & -\sin\phi \\ \sin\phi & \cos\phi\end{pmatrix}$，
  整体旋转 $R_i = \mathrm{blockdiag}\bigl(r(i\theta_1), \dots, r(i\theta_{H/2})\bigr) \in \mathbb{R}^{H \times H}$；
- 频率几何分布 $\theta_r = b^{-2(r-1)/H}$，base $b$（经典值 $10^4$；各模型见 config 的
  `rope_theta`，长上下文模型另有 `rope_scaling`）；
- 作用方式：$q_i = R_i W_Q x_i$，$k_j = R_j W_K x_j$。

## 1. 核心恒等式：点积只依赖相对位置

旋转群的基本同态（整数加法群 $\to SO(2)$）：$r(\phi)^\top r(\psi) = r(\psi - \phi)$，
分块对角后 $R_i^\top R_j = R_{j-i}$。于是分数

$$s(i, j) = q_i \cdot k_j = x_i^\top W_Q^\top R_i^\top R_j\, W_K x_j
= x_i^\top \underbrace{W_Q^\top R_{m}\, W_K}_{M(m)}\, x_j, \qquad m = j - i$$

两个直接推论：

- **相对性**：绝对位置被消去（$R_i^\top$ 吸收 $i$），分数天然平移不变；
- **精确性**：位置差 $m$ 变成**精确相位差** $m\theta_r$——恒等式层面无近似。
  位置的代数结构（相邻差 1、差可累加）被无噪声地表示为旋转复合；
  旋转是正交变换，不改变向量范数——位置以**相位**形式存在，不与内容竞争能量。

$m = 0$ 时 $R_0 = I$，$M(0) = M$ 即 `qk-spectrum.md` 分析的无位置矩阵。

## 2. 单通道几何：相位差编码

第 $r$ 个 2D 通道，记 $u = (W_Q x_i)^{(r)} \in \mathbb{R}^2$、$v = (W_K x_j)^{(r)}$，
极坐标 $u = |u|(\cos\alpha, \sin\alpha)$、$v = |v|(\cos\beta, \sin\beta)$：

$$u^\top r(m\theta_r)\, v = |u|\,|v|\,\cos\bigl(m\theta_r + \beta - \alpha\bigr)$$

**位置进相位，内容定振幅** $|u||v|$。单通道是周期的——位置只能确定到模
$2\pi/\theta_r$，不唯一；唯一性靠多通道组合（下节）。

## 3. 多频率展开：有限傅里叶级数

固定内容 $(x_i, x_j)$ 与头权重，分数作为 $m$ 的函数是 **$H/2$ 项有限傅里叶级数**：

$$s(m) = \sum_{r=1}^{H/2} A_r \cos\bigl(m\theta_r + \phi_r\bigr),
\qquad A_r = |u_r||v_r|,\ \phi_r = \beta_r - \alpha_r$$

几何分布的频率 $\theta_r$（从 $\approx 1$ 到 $\approx b^{-1}$）相当于**多进制展开**：
高频通道分辨相邻位置（周期几个 token），低频通道覆盖远程（周期 $2\pi b$）。
信息量对账：位置只需 $\log_2 L$ 比特，$H/2$ 个频率通道绰绰有余；
语义与位置在同一向量中**频分复用**——各通道的相位跑位置、振幅跑内容。

各通道振幅 $A_r$ 由权重与内容决定 → 训练可以选择"哪个头用哪些频率"：
用高频通道为主的头是局部头，给低频通道大振幅的头能做远程匹配。

## 4. 唯一性范围与外推混叠

- **无混叠范围** $\approx$ 最长周期 $T_{\max} = 2\pi/\theta_{\min} \approx 2\pi b$
  （$b = 10^4$ 时约 6.3 万 token）；$m$ 与 $m + T_r$ 在通道 $r$ 上同相位；
- **真正的外推失败机制更细**：训练只见过 $m \in [0, L_{\text{train}}]$，
  低频通道在训练窗口内走不完一个周期——外推时这些通道的相位是
  **训练分布外（OOD）的输入**，分数行为失控，而不是"缓慢算错"；
- K3：`max_position_embeddings = 1{,}048{,}576$（`kimi-k3.md`），远超经典
  base 的无混叠范围，必然依赖 `rope_scaling`（具体方案见模型 config.json）。

## 5. 衰减界：局部性先验

RoPE 原论文（Su et al. 2021）证明：对零均值 iid 向量，RoPE 内积的期望**包络随
$|m|$ 衰减**（把 $\sum_r h_r e^{im\theta_r}$ 型求和用 Abel 变换界住，
得 $\sim O(1/m)$ 包络乘以通道振幅）。含义：

- 初始化自带"关注附近"的归纳偏置——局部注意力是白送的；
- 但这是期望/包络陈述而非硬约束：远程匹配头（如 induction 头需要
  attend 回数百 token 前的位置）靠训练把特定通道的振幅 $A_r$ 拉大可实现。

## 6. 与加性位置编码的对比

加性 PE（$x + p_i$，正弦或可学习）代入双线性形，四项分离：

$$s(i,j) = \underbrace{x_i^\top M x_j}_{\text{内容}\times\text{内容}}
+ \underbrace{x_i^\top M p_j + p_i^\top M x_j}_{\text{内容}\times\text{位置}}
+ \underbrace{p_i^\top M p_j}_{\text{位置}\times\text{位置}}$$

| | 加性 PE | RoPE |
|---|---|---|
| 位置进入方式 | 相加（与内容并列） | 相乘（旋转内容） |
| 分数结构 | 四项可加分离 | 耦合在 $M(m)$ 中 |
| 相对位置 | 需额外设计 | 恒等式免费给出 |
| 绝对位置参数表 | 有（长度受限） | 无（任意 $m$ 统一处理） |

加性方案里"位置与语义无关"靠**双线性可加性**实现（各占独立项）；
RoPE 靠**正交旋转**实现（相位与振幅分治）。两条路线回答了同一个问题。

## 7. 长上下文扩展：PI / NTK / YaRN

共同数学本质：**重参数化频率映射** $\theta_r \to f(\theta_r)$，使训练域内的
相位分布覆盖推理所需的位置域：

| 方案 | 频率映射 | 代价/特点 |
|---|---|---|
| PI（位置插值） | $m \to m/\lambda$，全通道线性压缩 | 近距分辨率损失，需微调恢复 |
| NTK-aware | 提高 base $b \to b'$，非均匀压缩 | 保高频、压低频，免微调但有限 |
| YaRN | 分频段：高频不动、低频插值、斜坡过渡 | 加 softmax 温度修正补偿 logit 尺度 |

温度修正的由来：缩放频率改变了 $s(m)$ 的涨落幅度（§3 的傅里叶级数振幅分布变化），
logit 方差漂移，softmax 熵随之漂移——需乘温度补偿回初始化时的方差水平
（呼应 `attention-score-distribution.md` §1 的方差归一逻辑）。

## 8. 部分旋转（MLA）：空分复用 vs 频分复用

K3 的 MLA（`kimi-k3.md`）：每头 QK 维度 = `qk_nope_head_dim` 128（内容，不旋转）
+ `qk_rope_head_dim` 64（位置，旋转）。点积分块可加：

$$q \cdot k = \underbrace{q_{\text{nope}} \cdot k_{\text{nope}}}_{\text{纯内容}}
+ \underbrace{(R_i q_{\text{rope}}) \cdot (R_j k_{\text{rope}})}_{\text{纯位置调制}}$$

——§6 加性 PE 的"分离"优点与 RoPE 的"相对位置"优点的合体：

- 位置只占用**专用子空间**（空分复用），而非标准 RoPE 的同维不同频率（频分复用）；
- nope 部分的 $M$ 谱分析**完全不受位置污染**——`qk-spectrum.md` 的谱框架
  在 nope 子空间上精确成立；
- rope 分量在 MLA 中由各头共享的 kv 潜变量升维得到（`kv_lora_rank=512`），
  位置行为跨头同质性高——各头的分工主要体现在 nope（内容）部分。

## 9. 静态分析切入点

RoPE 本身是与权重无关的纯数学对象（旋转矩阵可按频率直接构造），
但"头如何利用位置通道"可从权重静态检验：

- **位置敏感度曲线**：$M(m) = W_Q^\top R_m W_K$ 的谱范数 $\sigma_1(M(m))$
  随 $m$ 的变化（$m \in [-L, 0]$，因果注意力只用过去）；
- **位置头指纹**：曲线在 $m = -1$ 附近的尖锐峰 → previous-token 型头；
  平坦曲线 → 内容主导头（位置通道振幅小）；
- **MLA 分工检验**：分别对 nope / rope 子空间做上述分析，
  对比各头内容矩阵谱的离散度 vs 位置行为的同质度（§8 的预测）。

全程只读权重 + 构造旋转矩阵，符合项目静态分析原则（`kimi-k3.md`：
MLA 投影为 bf16 可直接读）。

## 参考文献

- J. Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding*, 2021
  ——RoPE 原论文，含相对位置恒等式与远程衰减界；
- S. Chen et al., *Extending Context Window of Large Language Models via Positional
  Interpolation*, 2023——PI 方案与外推 OOD 分析；
- B. Peng et al., *YaRN: Efficient Context Window Extension of Large Language Models*,
  2023——分频段插值与温度修正；
- DeepSeek-AI, *DeepSeek-V2/V3 Technical Report*——MLA 的部分旋转
  （nope/rope 分解）设计，K3 沿用。
