# QK 谱：训练后的注意力分数由什么决定

前置：[attention-score-distribution.md](attention-score-distribution.md)（分数分布推导）、
[rmsnorm.md](rmsnorm.md) §3（初始化时的方差链）。

本文回答：噪音输入的 scaled 分数方差恒为 1，那**有语义关系的两个向量**分数一般多大？
并给出用开源权重静态验证的方案。

## 1. 谱分解框架

合并两次投影：s = Aᵀ M B，M = W_QᵀW_K（D×D，rank(M) ≤ H）。对 M 做 SVD：

$$s = \sum_{i=1}^{r} \sigma_i\,(u_i\cdot A)\,(v_i\cdot B)$$

- **噪音对**：各方向投影独立零均值 → E[s] = 0，Var(s) = σ⁴‖M‖_F²，
  ÷√H 后初始化时方差恰为 1——这只描述了**波动底噪**；
- **有关系的对**：训练把匹配结构编进 M 的顶部奇异方向，相关对在大 σ_i 方向上
  投影同号 → 分数获得**系统性偏移**（均值 ≠ 0），形态为"语义信号 + ±1 量级底噪"；
- softmax 只看**相对差**：匹配分数要比其余 key 高出几到十几个单位才接近 one-hot，
  底噪 std = 1（scaled）即是注意力的"分辨率"。

量级锚点：归一化后 ‖A‖ ≈ √D，单奇异项理论上限 σ_i·D；实际 embedding 能量分散，
投影只有几个单位。实证上训练后模型的 attention logit 普遍长到**几十**的范围
（见 §3 的熵坍缩文献）。

## 2. 初始化时的谱（基线）

- "全部奇异值 = 1"只在均值意义下成立：E‖M‖_F² = H（σ_w² = 1/D），
  H 个非零奇异值平方平均摊到 ~1；
- 实际谱有 O(1) 展宽：M 是两个高斯矩阵的乘积，平方奇异值服从 **Fuss-Catalan 分布**
  （两个 Marchenko-Pastur 律的自由乘积，H = D 时支撑 [0, 27/4]）；
- rank(M) ≤ H，H ≪ D 时 D 个奇异值中至少 D−H 个**精确为 0**。

## 3. 训练后的经验规律（文献共识）

没有像初始化那样精确的普适分布律，但多项独立研究指向一致的定性图像——
**从"均值 1 的紧支撑随机谱"演化为"重尾、低有效秩、顶部被任务结构主导"**：

1. **低有效秩 / 重尾**
   - Hu et al. 2021（LoRA）：实测大模型 W_Q、W_V 更新矩阵 intrinsic rank 很低，
     低秩微调由此成立；
   - Martin & Mahoney（heavy-tailed self-regularization / WeightWatcher）：
     训练良好层的权重谱呈幂律尾（α ≈ 2~4），偏离随机矩阵谱（测的是 W 本身，
     QK 乘积上定性一致）；
   - Elhage et al. 2021（transformer circuits）：QK 矩阵少数大奇异值对应可解释
     结构（如 induction head 的"匹配前文"方向），其余方向接近噪音水平。
2. **范数增长，logit 方差远超 1**
   - Zhai et al. 2023（σReparam）：训练中 ‖W_QᵀW_K‖ 谱范数持续增大，
     与注意力熵坍缩相关；
   - Dehghani et al. 2023（ViT-22B）：因 attention logit 发散引入 QK-LayerNorm；
     Gemma-2/3、Qwen3、OLMo-2 等现役模型标配 QK-norm——工业界默认
     "不压制 ‖M‖ 会长"。即训练后 σ⁴‖M‖_F² 通常明显大于初始化值，÷√H 不再归一。
3. **头间差异极大**
   - Michel et al. 2019、Voita et al. 2019：大量 head 训练后接近冗余、可剪枝，
     少数 head 谱高度集中——"平均谱"意义有限，须逐头分析。

## 4. 与头维 H 的关系

- ÷√H 已把 H 从**量级**中消掉（噪音底方差恒 1，信号由学出的 σ_i 决定）；
- H 是**容量上限**：rank(M) ≤ H，决定一个头最多能在多少个正交方向上
  编码匹配特征——影响"能分辨多少种关系"，而非"匹配分数多大"。

## 5. 开源权重验证方案

全程静态分析（只读 safetensors + numpy），不跑模型。

### 5.1 计算步骤

1. 读 `W_Q`、`W_K`，按头切成 per-head 的 H×D 矩阵
   （存储通常是 (n_heads·H, D)，需 reshape；GQA/MQA 注意 K 头数 < Q 头数，
   每个 K 头被一组 Q 头共享）；
2. 逐层逐头算 M_h = W_Qhᵀ W_Kh（D×D，rank ≤ H），SVD 取奇异值；
   D 大时用随机化 SVD 或只算前若干奇异值即可；
3. 指标（每层每头）：
   - 奇异值谱 {σ_i}；
   - ‖M‖_F² = Σσᵢ²（= 未缩放噪音分数方差 / σ⁴）；
   - 谱范数 σ₁（最大 logit 偏移的控制量）；
   - 有效秩 r_eff = (Σσᵢ)² / Σσᵢ²；
4. 基线对比：同形状随机高斯 W_Q、W_K（σ_w² = 1/D）生成 M_rand，
   对照其 Fuss-Catalan 谱——看训练后谱偏离基线多少。

### 5.2 要检验的预测

| 预测 | 来源 |
|---|---|
| 非零奇异值 ≤ H 个，其余精确为 0 | rank(M) ≤ H |
| 谱重尾化：σ₁ 明显大于体部，r_eff ≪ H | LoRA / WeightWatcher / circuits |
| ‖M‖_F² > H（多数头），即 ÷√H 不再归一 | 熵坍缩 / QK-norm 文献 |
| 头间差异大：部分头谱仍接近随机基线 | head 剪枝研究 |
| 层间有规律（如中层 induction 头谱最集中） | circuits |

### 5.3 注意事项

- **带 QK-norm 的模型**（Qwen3、Gemma-2/3、OLMo-2 等）：q、k 在头内被再归一，
  ‖M‖ 的绝对尺度不代表 logit 尺度，结论需注明；建议同时分析一个
  不带 QK-norm 的对照模型（如 Qwen2.5、Llama-3）；
- RoPE 不影响本分析（作用在 q、k 上的旋转不改变逐头 M 的奇异值结构
  对"关系匹配"的编码方式，但注意 RoPE 使分数依赖相对位置，
  本文的 s 是"无位置"分量）；
- 权重目录只读；结果图写 `output/`；模型路径走 `llm_lens.get_model_dir()`。
