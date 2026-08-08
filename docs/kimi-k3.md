# Kimi-K3 模型结构笔记

> 信息来源:`config.json` 与 `model.safetensors.index.json`(只读解析)。
> 本文只记录对分析有用的结构性事实。

## 总体

| 项 | 值 |
| --- | --- |
| 模型类型 | `KimiK3ForConditionalGeneration`(多模态:语言模型 + 视觉塔) |
| 语言模型架构 | `kimi_linear`(KimiLinear:混合线性注意力 + MoE) |
| 权重分片 | 96 个 safetensors 分片 |
| 张量总数 | 497,220(其中约 49.4 万是路由专家的量化权重) |
| 索引记录总大小 | 约 1453.66 GiB |
| 词表大小 | 163,840(tiktoken 系 tokenizer,见 `tiktoken.model`) |

## 语言模型关键参数

| 参数 | 值 |
| --- | --- |
| 层数 | 93(config 中层号 1~93;第 1 层为 dense MLP,其余 92 层为 MoE) |
| hidden_size | 7168 |
| 注意力布局 | 24 层 MLA 全注意力 + 69 层 KDA 线性注意力,约每 4 层插入 1 层全注意力 |
| MLA | q_lora_rank=1536,kv_lora_rank=512,qk_nope_head_dim=128,qk_rope_head_dim=64,v_head_dim=128,96 头,带输出门控(g_proj) |
| KDA | 96 头,head_dim=128,short_conv kernel=4,全秩门控 |
| MoE | 896 个路由专家 + 2 个共享专家,每 token 选 16 个;sigmoid 路由(noaux_tc,带 e_score_correction_bias) |
| 专家维度 | moe_intermediate_size=3072,routed_expert_hidden_size=3584(latent MoE,带 norm) |
| dense MLP intermediate_size | 33792(仅第 1 层) |
| 上下文 | max_position_embeddings=1,048,576 |
| 其他 | 每层有 `self_attention_res_proj/norm`、`mlp_res_proj/norm` 残差分支(attn_res_block_size=12;AttnRes,见下节) |

## 位置编码:NoPE

语言模型**不使用 RoPE**,位置信息完全靠 KDA 层的 short_conv(kernel=4)与注意力因果结构提供:

- `config.json` 中 `mla_use_nope=true`;建模代码 `modeling_kimi_linear.py` 的 MLA 模块
  `assert self.use_nope` 且 `self.rotary_emb = None`,整个语言模型没有任何 rotary 模块;
- 注意 `qk_rope_head_dim=64` 的维度**结构上仍然存在**(`q_b_proj` 每头 192 = nope 128 + "rope" 64,
  `kv_a_proj_with_mqa` 576 = 512 + 64),只是不施加任何旋转,作为普通维度参与打分;
  其中 K 侧这 64 维由 `kv_a` 直出、96 头共享;
- 对权重静态分析的含义:attention score 无需剥离位置分量,可直接分解为
  nope 128 维(经 `kv_b` 升维)+ "rope" 64 维(`kv_a` 直出、全头共享)两块;
- 视觉塔另有自己的位置编码(`pos_emb_type=divided_fixed`,可学习插值位置嵌入),与语言模型无关。

## AttnRes(注意力残差)

传统残差 $h_l = h_{l-1} + f_{l-1}(h_{l-1})$ 对所有历史层等权累加,浅层信号被逐层稀释。
AttnRes(Kimi 2026.3,arXiv:2603.15031)把跨 block 的聚合改成**可学习的 softmax 凸组合**:

- 每 `attn_res_block_size / 2 = 6` 层存一个 block 表示 $V_i$(embedding 算第 0 个);
- 每层在 attention 前、MLP 前各做一次融合(对应权重 `self_attention_res_proj/norm`、
  `mlp_res_proj/norm`):$\mathrm{logits}_i = w \cdot \mathrm{RMSNorm}(V_i)$,
  $h = \sum_i \mathrm{softmax}(\mathrm{logits})_i\, V_i$,
  其中 $w$ 是 `res_proj` 的 7168 维可学习向量;
- block 内部仍是经典残差累加。

方差意义:凸组合(权重非负、和为 1)的方差有界
$\mathrm{Var}\!\left(\sum_i \alpha_i V_i\right) \le \max_i \mathrm{Var}(V_i)$,
不随深度线性增长——GPT-2 式 $1/\sqrt{2N}$ 初始化修正(见
[rmsnorm.md](rmsnorm.md) §4.2)在架构层面不再必要;
block 内 ~6 层的累加只剩常数级放大,被 RMSNorm 与训练吸收。
`res_norm` 同时抹平各 block 的尺度差异,使 logits 只反映方向匹配。

## 权重命名与量化分布

所有语言模型权重带 `language_model.` 前缀。

### bf16 直接可读(未被量化)

| 权重 | 形状/数量 | 说明 |
| --- | --- | --- |
| `language_model.model.embed_tokens.weight` | 163840 × 7168 | 词嵌入 |
| `language_model.lm_head.weight` | 163840 × 7168 | LM Head,与词嵌入**不共享**(`tie_word_embeddings=false`) |
| `...layers.N.self_attn.{q_a,q_b,kv_a,kv_b}_proj` 等 | 24 层 | MLA 全注意力层的低秩投影 |
| `...layers.N.self_attn.{q,k,v}_proj`、`{q,k,v}_conv1d`、`f_a/f_b_proj`、`A_log`、`dt_bias`、`b_proj`、`o_norm` | 69 层 | KDA 线性注意力参数 |
| `...layers.N.self_attn.{o_proj,g_proj}` | 93 层 | 所有层共有的输出投影与输出门控 |
| `...layers.N.block_sparse_moe.gate.weight` | 92 层,896 × 7168 | MoE 路由器 |
| `...layers.N.block_sparse_moe.shared_experts.{gate,up,down}_proj` | 92 层 | 2 个共享专家 |
| `...layers.N.block_sparse_moe.routed_expert_{up,down}_proj`、`routed_expert_norm` | 92 层 | latent MoE 的降维/升维投影 |
| `vision_tower.*`、`mm_projector.*` | 27 层 ViT | 视觉塔(hidden 1024,patch 14)与多模态投影(→7168) |

### mxfp4 量化(需反量化后才能分析)

| 权重 | 数量 | 说明 |
| --- | --- | --- |
| `...block_sparse_moe.experts.E.{w1,w2,w3}.weight_packed` | 92 层 × 896 专家 × 3 | 打包后的 4bit 权重 |
| `...block_sparse_moe.experts.E.{w1,w2,w3}.weight_scale` | 同上 | 每组(group_size=32)的缩放系数 |

量化格式:`compressed-tensors` 的 `mxfp4-pack-quantized`,对称量化,group_size=32。

**含义:绝大多数"有趣"的矩阵(词嵌入、LM Head、注意力投影、路由器、共享专家)都是 bf16,可以用 safetensors + numpy(ml_dtypes)直接读取分析;只有路由专家需要先反量化。**

## 分析切入点建议

1. **词嵌入 / LM Head**:向量夹角(余弦相似度)分布、词嵌入与 LM Head 对应行的相关性、各向异性(平均向量、主成分占比)。
2. **注意力投影矩阵**:奇异值谱、有效秩;MLA 低秩分解(q_a→q_b、kv_a→kv_b)的实际秩与 config 中 rank 的对比。
3. **KDA 线性注意力**:`A_log`(衰减率)、`dt_bias`、conv1d 核的分布。
4. **MoE 路由器**:`gate.weight` 行向量(每个专家一个 7168 维向量)之间的夹角分布——专家是否"正交分化";`e_score_correction_bias` 的分布。
5. **路由专家**:需先按 mxfp4 格式反量化(注意 group_size=32、共享 scale 的打包方式),之后可做专家间相似度等分析;数据量大(单专家 w1/w3 约 3584×3072),注意按需读取单个专家而不是整层加载。
6. **AttnRes**:逐层读 `res_proj` 向量(7168 维)的范数分布,对比初始化尺度看训练漂移;结合 `res_norm` 增益估计各 block 的 logits 尺度(softmax 是趋于均匀还是已分化出偏好)。
