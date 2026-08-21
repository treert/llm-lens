# QK 谱：训练后的注意力分数由什么决定

前置：[attention-score-distribution.md](attention-score-distribution.md)（分数分布推导）、
[rmsnorm.md](rmsnorm.md) §3（初始化时的方差链）。

本文回答：噪音输入的 scaled 分数方差恒为 1，那**有语义关系的两个向量**分数一般多大？
并给出用开源权重静态验证的方案。

## 1. 谱分解框架

合并两次投影： $s = A^\top M B$， $M = W_Q^\top W_K$（ $D \times D$， $\mathrm{rank}(M) \le H$）。
对 $M$ 做 SVD：

$$s = \sum_{i=1}^{r} \sigma_i\,(u_i\cdot A)\,(v_i\cdot B)$$

- **噪音对**：各方向投影独立零均值 → $\mathbb{E}[s] = 0$、
  $\mathrm{Var}(s) = \sigma^4\|M\|_F^2$， $\div\sqrt{H}$ 后初始化时方差恰为 1——
  这只描述了**波动底噪**；
- **有关系的对**：训练把匹配结构编进 $M$ 的顶部奇异方向，相关对在大 $\sigma_i$ 方向上
  投影同号 → 分数获得**系统性偏移**（均值 $\ne 0$），形态为"语义信号 + $\pm 1$ 量级底噪"；
- softmax 只看**相对差**：匹配分数要比其余 key 高出几到十几个单位才接近 one-hot，
  底噪 $\mathrm{std} = 1$（scaled）即是注意力的"分辨率"。

量级锚点：归一化后 $\|A\| \approx \sqrt{D}$，单奇异项理论上限 $\sigma_i \cdot D$；
实际 embedding 能量分散，投影只有几个单位。实证上训练后模型的 attention logit
普遍长到**几十**的范围（见 §4 的熵坍缩文献）。

## 2. 三矩阵视角： $W_Q$、 $W_K$ 各自 SVD 与对齐矩阵

把两个投影矩阵各自 SVD（ $W_Q = U_Q\Sigma_Q V_Q^\top$、 $W_K = U_K\Sigma_K V_K^\top$）代入：

$$M = V_Q\,\bigl[\Sigma_Q\,(U_Q^\top U_K)\,\Sigma_K\bigr]\,V_K^\top$$

括号内仅 $H \times H$， $M$ 的非零奇异值恰好等于它的奇异值。三个推论：

1. **计算捷径**：无需构造 $D \times D$ 的 $M$——分别 SVD 两个 $H \times D$，
   再 SVD 一个 $H \times H$ 即可（§6.1 的步骤据此写）；
2. **容器**： $M$ 的左奇异向量住在 $W_Q$ 的行空间、右奇异向量住在 $W_K$ 的行空间，
   两个 $H$ 维子空间界定了 $M$ 的全部活动范围；
3. **谱的两个独立来源**：谱由 $\Sigma_Q$、 $\Sigma_K$（各自能量分布）与
   **对齐矩阵** $U_Q^\top U_K$（元素 = 两组奇异向量间的余弦）共同决定。
   各自重尾化与相互对齐是两种独立的训练效应——单独看 $W_Q$、 $W_K$ 的谱
   推不出 $M$ 的谱，必须三个 SVD 合看。

初始化基线： $U_Q^\top U_K \approx$ 随机正交（元素 $\sim 1/\sqrt{H}$），配合平谱
得到 §3 的 Fuss-Catalan；训练后预期少数方向 $\cos \to \pm 1$
（q 的某方向专门匹配 k 的某方向）， $M$ 的有效秩本质上由对齐矩阵的有效秩决定。

**对称性探针**：若某头学成"相似性匹配"（找与自己像的）， $W_Q \approx W_K$、
$M$ 近对称半正定（ $u_i \approx v_i$）；若是"序列匹配"（A 之后跟 B，induction 型），
$M$ 强非对称。指标 $\mathrm{sym}(M) = \|M - M^\top\|_F\,/\,\|M\|_F = 2\|M_a\|_F/\|M\|_F$
（ $M_a$ 为反对称部分，值域 $[0, 2]$），其中
$\|M-M^\top\|_F^2 = 2\sum_i\sigma_i^2 - 2\sum_{i,j}\sigma_i\sigma_j\,(u_i\cdot v_j)(u_j\cdot v_i)$。
两个读数锚点：相似性头 $\mathrm{sym}(M) \approx 0$（ $u_i \cdot v_i \approx \pm 1$）；
序列匹配头 $\mathrm{sym}(M) \approx \sqrt{2}$——对应 $\langle M, M^\top\rangle_F = 0$
（对称、反对称能量相等），典型充分条件是左右奇异子空间正交
（ $u_i \perp v_j$ 对所有 $i, j$，逐奇异项即 $u_i \cdot v_i \approx 0$）。

## 3. 初始化时的谱（基线）

- "全部奇异值 = 1"只在均值意义下成立： $\mathbb{E}\|M\|_F^2 = H$（ $\sigma_w^2 = 1/D$），
  $H$ 个非零奇异值平方平均摊到 $\sim 1$；
- 实际谱有 $O(1)$ 展宽： $M$ 是两个高斯矩阵的乘积，平方奇异值服从 **Fuss-Catalan 分布**
  （两个 Marchenko-Pastur 律的自由乘积， $H = D$ 时支撑 $[0,\; 27/4]$）；
- $\mathrm{rank}(M) \le H$， $H \ll D$ 时 $D$ 个奇异值中至少 $D-H$ 个**精确为 0**；
- 谱曲线的精确形式（ $MP_c$ 自由乘积、乘法维度 $D\times D$ vs $H\times H$ 的影响、
  AB/BA 辨析）统一维护在 `docs/math/random-matrix-spectra.md`；
  交互对比工具见 `web-tools/spectrum-demo/`。

## 4. 训练后的经验规律（文献共识）

没有像初始化那样精确的普适分布律，但多项独立研究指向一致的定性图像——
**从"均值 1 的紧支撑随机谱"演化为"重尾、低有效秩、顶部被任务结构主导"**：

1. **低有效秩 / 重尾**
   - Hu et al. 2021（LoRA）：实测大模型 $W_Q$、 $W_V$ 更新矩阵 intrinsic rank 很低，
     低秩微调由此成立；
   - Martin & Mahoney（heavy-tailed self-regularization / WeightWatcher）：
     训练良好层的权重谱呈幂律尾（ $\alpha \approx 2 \sim 4$），偏离随机矩阵谱
     （测的是 $W$ 本身，QK 乘积上定性一致）；
   - Elhage et al. 2021（transformer circuits）：QK 矩阵少数大奇异值对应可解释
     结构（如 induction head 的"匹配前文"方向），其余方向接近噪音水平。
2. **范数增长，logit 方差远超 1**
   - Zhai et al. 2023（σReparam）：训练中 $\|W_Q^\top W_K\|$ 谱范数持续增大，
     与注意力熵坍缩相关；
   - Dehghani et al. 2023（ViT-22B）：因 attention logit 发散引入 QK-LayerNorm；
     Gemma-2/3、Qwen3、OLMo-2 等现役模型标配 QK-norm——工业界默认
     "不压制 $\|M\|$ 会长"。即训练后 $\sigma^4\|M\|_F^2$ 通常明显大于初始化值，
     $\div\sqrt{H}$ 不再归一。
3. **头间差异极大**
   - Michel et al. 2019、Voita et al. 2019：大量 head 训练后接近冗余、可剪枝，
     少数 head 谱高度集中——"平均谱"意义有限，须逐头分析。

## 5. 与头维 $H$ 的关系

- $\div\sqrt{H}$ 已把 $H$ 从**量级**中消掉（噪音底方差恒 1，信号由学出的 $\sigma_i$ 决定）；
- $H$ 是**容量上限**： $\mathrm{rank}(M) \le H$，决定一个头最多能在多少个正交方向上
  编码匹配特征——影响"能分辨多少种关系"，而非"匹配分数多大"。

## 6. 开源权重验证方案

全程静态分析（只读 safetensors + numpy），不跑模型。

### 6.1 计算步骤

1. 读 $W_Q$、 $W_K$，按头切成 per-head 的 $H \times D$ 矩阵
   （存储通常是 $(n_{heads} \cdot H,\; D)$，需 reshape；GQA/MQA 注意 K 头数 < Q 头数，
   每个 K 头被一组 Q 头共享）；
2. 逐层逐头取 $M_h$ 的奇异值，**不显式构造 $D \times D$ 的 $M$**（§2 的捷径）：
   分别 SVD $W_{Qh}$、 $W_{Kh}$（各 $H \times D$），
   再 SVD $H \times H$ 矩阵 $\Sigma_Q(U_Q^\top U_K)\Sigma_K$，其奇异值即所求；
3. 指标（每层每头）：
   - 奇异值谱 $\{\sigma_i\}$；
   - $\|M\|_F^2 = \sum_i \sigma_i^2$（= 未缩放噪音分数方差 $/\ \sigma^4$）；
   - 谱范数 $\sigma_1$（最大 logit 偏移的控制量）；
   - 有效秩 $r_{eff} = \bigl(\sum_i \sigma_i\bigr)^2 \big/ \sum_i \sigma_i^2$；
   - 对齐矩阵 $U_Q^\top U_K$：元素分布与有效秩（§2）；
   - 对称性 $\mathrm{sym}(M)$ 与 $u_i\cdot v_i$ 分布（§2）；
4. 基线对比：同形状随机高斯 $W_Q$、 $W_K$（ $\sigma_w^2 = 1/D$）生成 $M_{rand}$，
   对照其 Fuss-Catalan 谱——看训练后谱偏离基线多少。

### 6.2 要检验的预测

| 预测 | 来源 |
|---|---|
| 非零奇异值 $\le H$ 个，其余精确为 0 | $\mathrm{rank}(M) \le H$ |
| 谱重尾化： $\sigma_1$ 明显大于体部， $r_{eff} \ll H$ | LoRA / WeightWatcher / circuits |
| $\|M\|_F^2 > H$（多数头），即 $\div\sqrt{H}$ 不再归一 | 熵坍缩 / QK-norm 文献 |
| 头间差异大：部分头谱仍接近随机基线 | head 剪枝研究 |
| 层间有规律（如中层 induction 头谱最集中） | circuits |
| 对齐矩阵结构化： $U_Q^\top U_K$ 少数元素 $\to \pm 1$ | §2，circuits |
| 对称性分化：相似性头 $\mathrm{sym}(M)\approx 0$、序列头 $\approx\sqrt{2}$ | §2 |
| GQA 组内右奇异子空间重叠显著高于跨组 | §6.3 |

### 6.3 GQA/MQA：共享 $W_K$ 的指纹

GQA 中 $G$ 个 Q 头共享一个 K 头（MQA 为全共享极限）：组内
$M_i = W_{Qi}^\top W_K$ 的右奇异向量被迫住在**同一个** $H$ 维 K 行空间——
MHA 中每头独立的方向库变成组内共享，产生容量竞争。可检验指纹：

- 组内各 $M_i$ 的右奇异子空间两两重叠（principal angle 小），跨组则近似随机；
- $W_K$ 谱被摊平（一个 K 空间尽量覆盖更多方向）或同组 Q 谱被挤尖
  （被迫共享少数匹配方向、功能同质化），或二者混合；
- 组内分化集中在左奇异侧与奇异值上，右奇异侧趋同——
  "组内 Q 头冗余度高"可与 head 剪枝研究互证；
- MQA 极限可能逼出 superposition：多个匹配方向叠进同一 K 方向（ $H$ 维硬上限）。

**K3 注意**：MLA 不是 GQA，但精神类似——96 头共享 `kv_lora_rank=512` 的 K 潜库
（`kv_a` 压缩、`kv_b` 各头升维）。 $W_K$ 的有效形式是两段投影之积，
低秩约束体现在潜空间而非 per-head 矩阵上，§6.1 步骤 1 的 reshape 需按此处理，
不能套用普通 per-head 切法。

### 6.4 注意事项

- **带 QK-norm 的模型**（Qwen3、Gemma-2/3、OLMo-2 等）： $q$、 $k$ 在头内被再归一，
  $\|M\|$ 的绝对尺度不代表 logit 尺度，结论需注明；建议同时分析一个
  不带 QK-norm 的对照模型（如 Qwen2.5、Llama-3）；
- RoPE 不影响本分析（作用在 $q$、 $k$ 上的旋转不改变逐头 $M$ 的奇异值结构
  对"关系匹配"的编码方式，但注意 RoPE 使分数依赖相对位置，
  本文的 $s$ 是"无位置"分量）；
- 权重目录只读；结果图写 `output/`；模型路径走 `llm_lens.get_model_dir()`。
