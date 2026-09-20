# DeepSeek-V4.1-Flash 模型结构笔记

> 信息来源:`config.json`、`model.safetensors.index.json` 与各分片头部(只读解析),
> 架构语义参照模型自带的最小推理代码 `inference/model.py`;
> 外部参照为官方技术报告 **DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression**
> (DeepSeek-AI,arXiv:2609.19969v1,2026-09-17),凡引用处标注「官方」。
> 本文只记录对分析有用的结构性事实。
> 分析笔记(实证结果)见 `docs/ds_v4_1_flash/` 子目录:
> [ds_v4_1_flash/weight-moments.md](ds_v4_1_flash/weight-moments.md)(权重矩分析:初始化基线 vs 训练后实测)、
> [ds_v4_1_flash/cosine-distribution.md](ds_v4_1_flash/cosine-distribution.md)(token 向量两两余弦分布 vs 高维近正交理论)、
> [ds_v4_1_flash/norm-gains.md](ds_v4_1_flash/norm-gains.md)(RMSNorm 增益:尺度信息的真实载体)。

## 总体

| 项 | 值 |
| --- | --- |
| 模型类型 | `DeepseekV41ForCausalLM`(多模态:语言模型 + ViT 视觉塔 + aligner) |
| 结构范式(官方) | **非对称 CED(Causal Encoder–Decoder)**:L0–L19 作 causal encoder、L20–L39 作 decoder;decoder 的 global KV 不由各层隐状态生成,统一由 encoder 末层输出(H₂₀)经 **L20 的 compressor** 投影得到(全 decoder 共享一份) |
| 注意力机制(官方) | **CSA2(Compressed Sparse Attention 2)**:Full / Reindex / Reuse 三种静态模式跨层复用 main KV 与检索结果(逐层对应见"注意力"节) |
| KV cache 精度 | 主 KV / 压缩 latent 为 **FP4**(MXFP4:16 通道一个 E4M3 scale),SWA KV 保留 **FP8**;长程 KV ≈ 890 B/token(推导见"长程 KV 显存账") |
| 权重分片 | 48 个 safetensors 分片 |
| 张量总数 | 96,085(约 4.7 万是路由专家的 FP4 打包权重,4.76 万是 UE8M0 缩放系数) |
| 索引记录总大小 | 约 475.24 GiB |
| 词表大小 | 129,280(HF tokenizer,见 `tokenizer.json`) |
| dtype 分布 | BF16 520、F32 387、F8_E4M3 357、F8_E8M0 47,589、I8(FP4 打包)47,232 |

## 语言模型关键参数(text_config)

| 参数 | 值 |
| --- | --- |
| 层数 | 40(另有 3 层 DSpark MTP,checkpoint 前缀 `mtp.0/1/2`) |
| hidden_size | 5120 |
| 注意力 | latent attention:64 头 × head_dim 512(448 nope + 64 rope),Q 低秩(q_lora_rank=1280),单 KV 头(wkv 5120→512,K=V 共享同一向量),逐头 attn_sink,分组低秩输出(o_groups=8 × o_lora_rank=1024) |
| 稀疏注意力 | 滑窗 128 + 压缩 KV 稀疏检索(index_topk=512);压缩比按层:`compress_ratios` 共 43 项(40 主干 + 3 MTP),L0–L1=0(纯滑窗)、L2–L19=2、L20–L39=1、MTP=0 |
| MoE | 384 个路由专家 + 1 个共享专家,每 token 选 6 个;moe_intermediate_size=2304 |
| 路由 | scoring_func=sqrtsoftplus($\sqrt{\mathrm{softplus}(x)}$),noaux_tc 校正偏置(bias;图像 token 另用 bias_vl),norm_topk_prob,routed_scaling_factor=1.5 |
| Hyper-Connections | hc_mult=4 路并行残差流,可学习混合(Sinkhorn 20 次迭代,hc_eps=1e-6) |
| Engram | 第 1、14 层各挂一个 n-gram 哈希嵌入模块(见下节) |
| 上下文 | max_position_embeddings=1,048,576;RoPE 只加在末 64 维,YaRN(factor=16,原始 65,536) |
| 其他 | rms_norm_eps=1e-20,initializer_range=0.02,tie_word_embeddings=false,swiglu_limit=10.0 |
| 优化器分配(官方) | RMSNorm 及非矩阵参数 → **AdamW**(wd=0.1);主干线性矩阵 → Muon;Q/K 矩阵 → **Head-wise Muon**(按注意力头各配预条件器);Engram/Embedding/Prediction Head → **Nesterov Momentum + Sinkhorn Balancing**(只需一份 momentum buffer,避免 Adam 的两套状态) |

## 注意力:latent attention + 滑窗/压缩稀疏检索

每个注意力层(命名 `layers.N.attn.*`):

- **Q 低秩**:`wq_a`(5120→1280,FP8)→ `q_norm`(RMSNorm 1280)→ `wq_b`(1280→32768=64×512,FP8);
- **单 KV 头**:`wkv`(5120→512,FP8)→ `kv_norm`(RMSNorm 512);每个位置只有**一个** 512 维
  latent 向量,64 个注意力头共享,K 与 V 都是它(打分用全部 512 维,加权求和也是它);
- **RoPE**:只对 Q/KV 的末 64 维(qk_rope_head_dim)施加;前 448 维是 NoPE;
- **attn_sink**:64 维 fp32,逐头的注意力 sink logit(softmax 分母多一项);
- **两个 KV 来源**拼成一次稀疏注意力:本层滑窗(128 个原始 KV)+ 共享压缩 KV 缓存中
  每 query 检索的 top-512 个压缩位置。压缩与检索只由少数"源层"完成,其余层复用:
  - compressor 在 `kv_source_layer_ids={2,8,14,20}`(`compressor.wkv/wgate` 5120→512,`compressor.norm`),
    按 compress_ratio 把连续 token 压成一个 latent;
  - indexer 在 `index_source_layer_ids={2,8,14,20,24,28,32,36}`(32 头 × 128 维:
    `indexer.wq_b` 1280→4096 FP8、`indexer.wk` 512→128、`indexer.k_norm`、`indexer.weights_proj` 5120→32),
    为每个 query 选出 512 个压缩位置;候选块由 candidate_source_layer=20 提供(2048 块 × 8);
- **输出投影分组低秩**:输出先 reshape 成 8 组(每组 8 头 × 512 = 4096 维),
  `wo_a` 块对角(每组 4096→1024,FP8,convert.py 反量化后按 einsum 用),`wo_b`(8192→5120,FP8);
  输出后对 rope 尾维做**逆旋转**(因为 V 向量里带着 RoPE 旋转);
- RoPE 每层只有一组频率:compress_ratio=0 的层(L0、L1、MTP)禁用 YaRN、用基础 rope_theta=10000;
  compress_ratio>0 的层(L2–L39)**整层**(Q、SWA 滑窗 KV、压缩 KV 共用同一组 freqs_cis)用
  compress_rope_theta=160000 + YaRN——名字里的 "compress" 有误导性;压缩 KV 的"位置间隔更大"
  是靠旋转位置取**组首 token**(第 j 组取位置 j×m)体现的,不是单独的 theta。

**官方名称与模式(arXiv:2609.19969,2026-09-17)**:这套机制在官方技术报告里叫
**CSA2(Compressed Sparse Attention 2)**,与 **CED(Causal Encoder–Decoder)** 非对称结构配套——
L0–L19 是 causal encoder、L20–L39 是 decoder,decoder 的 global KV 不由各层自己的隐状态生成,
统一由 encoder 末层输出 H₂₀ 经 **L20 的 compressor** 投影得到——checkpoint 中只有 {2, 8, 14, 20}
有 compressor 权重,L21–L39 没有产生 global KV 的权重、只读 L20 的缓存,层间差异由各自的 Q 承担
(报告公式 C_l = H_{L/2}·W_l^KV 的逐层投影在本 checkpoint 退化为一份共享投影);
本机代码即"L20 的 compressor 吃第 20 层注意力子层的输入"。
逐层三模式与本地集合的对应关系:

| CSA2 模式 | 做什么 | 本机对应 |
| --- | --- | --- |
| **Full** | 自产 main KV;indexer K 由 main KV 投影;跑 indexer 产出 Top-K(唯一产出可复用 cache 的模式) | `kv_source_layer_ids` = {2, 8, 14, 20} |
| **Reindex** | main KV 与 indexer K 复用前序层,用本层 indexer Q 重算 Top-K | `index_source_layer_ids` 中非 Full 的 {24, 28, 32, 36} |
| **Reuse** | main KV、indexer K、Top-K 全复用,不跑 indexer | 其余 30 层(L3–7、9–13、15–19、21–23、25–27、29–31、33–35、37–39) |

压缩率 m=1(L20–L39)即报告所称"不压缩 main KV"的特例。相关精度:**主 KV/压缩 latent 用 FP4**(MXFP4,
16 通道一个 E4M3 scale),**SWA KV 保留 FP8**;**SWA Bounded Replay** 只 replay 最近 128 个 token 并把
SWA 截断在该段内,从而把 SWA KV 从持久化 cache 中移除;全局 KV ≈ 890 B/token(≈ V4-Flash 的 1/4)。

## 长程 KV 显存账:官方"890 B/token"的推导验证

官方报告给出全局 KV cache 常驻 HBM 的占用 ≈ **890 B/token**(≈ DeepSeek-V4-Flash 的 1/4)。
这个数字可以由本机的 config + 推理代码完全推出,不需要额外假设:

| 项 | 通道 | 数值格式 | 缩放 | 字节/条目 |
| --- | --- | --- | --- | --- |
| main KV / 压缩 latent | 512 | FP4 E2M1(0.5 B/通道) | 每 16 通道 1 个 E4M3 scale → 32 B | **288** |
| indexer K | 128 | FP4 E2M1 | 每 32 通道 1 个 E8M0 scale → 4 B | **68** |

每 token 的条目数:encoder 的 3 组 Full 各按 m=2 摊(3 × 1/2 = 1.5),decoder 20 层共享 1 份且 m=1
(1 × 1 = 1),合计 **2.5 条/token**。

```
encoder  3 × 1/2 × 356 = 534 B/token
decoder  1 ×  1  × 356 = 356 B/token
                        ─────
Global KV cache         = 890 B/token
```

官方口径的对照:DeepSeek-V4-Flash 为 **3514 B/token**,890 / 3514 = 25.3% ≈ **1/4**(即降低 74.7%,
同显存理论上可容纳 ≈ 3.95 倍上下文)。注意口径是**随序列长度增长的那部分**:SWA KV 本身大小不变,
只是靠 SWA Bounded Replay 被移出持久化存储(持久化 KV 因此降到上一代的 ≈ 1/8)。

对应代码:`fp4_act_quant(latent, 16, True, scale_dtype=torch.float8_e4m3fn)`(压缩 KV,见
`Attention._compress_kv`)与 `fp4_act_quant(k, fp4_block_size=32, True)`(indexer K,默认 E8M0)。

三个前提都与代码一致:L0/L1 与 MTP 的 `compress_ratio=0`、不产生长程 KV;indexer K 只由 4 个
Full 层(`Indexer.owns_k = layer_id in kv_source_layers`)产出,其余层读 `shared_attn.index_k`;
SWA KV(每层 128 槽 × 512 维 FP8)是与序列长度无关的 per-sequence 环形缓冲
(40 × 128 × 512 × 1 B ≈ 2.6 MB/序列),不计入 per-token 口径,持久化时又被 SWA Bounded Replay 省掉。

附带观察:两者都是 1 B/scale,但精度取向不同——压缩 latent 用 E4M3(每 16 通道,scale 更密、
非 2 的幂),indexer K 用 E8M0(每 32 通道、纯 2 的幂)。前者要反量化后参与加权求和,后者只用于
Top-K 排序,这个分工与"长程粗、局部细"的整体取向一致。

## Hyper-Connections(HC)

残差流不是单向量而是 hc_mult=4 份并行副本(命名 `layers.N.hc_{attn,ffn}_{fn,base,scale}`,fp32):

- 每个子层(attention/FFN)前后各有一次可学习混合:`hc_pre` 把 4 份压成子层输入,
  `hc_post` 把输出扩回 4 份并经 `comb` 矩阵(Sinkhorn 成双随机)混入残差;
- 混合系数由流本身算出:`hc_fn`(24×20480,20480=4×5120)投影 + `hc_base`(24)+ `hc_scale`(3);
- 子层算出的系数给**下一个**子层用(attention 产的 pre_mix 给本层 FFN 用,FFN 产的给下一层 attention);
- 24 = (2 + hc_mult) × hc_mult:pre(4)+ post(4)+ comb(16)。

上述"系数后移"的写法就是官方报告里的 **Single-Pass mHC**:输入混合改用上一层产出的系数
(X_{l+1} = B_l·X_l + C_l·F_l(A_{l-1}·X_l),而非常规的 A_l),使每个 hidden tile 一遍即可同时用于
系数预测与输入混合;部署期再融成 Mega-mHC kernel,激活内存流量由 (4n+4)d 降到 (2n+2)d(理论下界,n=hc_mult=4)。

## Engram(n-gram 哈希嵌入)

第 1、14 层各有一个 Engram 模块(命名 `layers.N.engram.*`),把 n-gram(≤4-gram)哈希查表结果
门控地写入残差流:

- `engram.embed.weight`:FP8 E4M3 的 **3.84 亿 × 256** 巨型哈希表(engram_num_embeddings 两层分别为
  384,006,168 / 384,016,682;engram_vocab_size=1600 万经哈希压缩到 compressed_vocab_size=99,092 再散列),
  每层一张,合计约 196 GB——**占全模型近一半体积**,逐行 1×32 UE8M0 缩放;
- `engram.wkv`(6144→25600,FP8):把 (max_ngram-1)×n_heads=24 个 256 维哈希行(6144)映射为
  5×5120:4 份 HC 副本各一个 key + 1 个共享 value;
- `q_weight`/`k_weight`([4,5120],bf16,初始化全 1):门控 = sigmoid(signed_sqrt(归一化点积)),
  即"流与 n-gram key 匹配程度"决定写入强度;
- 图像 token 不参与 n-gram(engram_mask 关闭门控)。

**V4.1 相对原 Engram 论文(《Conditional Memory via Scalable Lookup》)的调整(官方)**:

- 删除门控之后的短因果卷积(原为 `SiLU(Conv1D(RMSNorm(Ṽ))) + Ṽ`),理由是其收益不足以覆盖推理复杂度;
- 多残差流下**所有分支共享记忆表与 Value 投影,但各分支有自己的 Key 通路**:本机对应 `wkv`
  一次投出 4 份 key + 1 份共享 value,再由 `q_weight · k_weight` 逐分支加权;
- 推理时记忆表主要驻留 **Host Memory**,由后台 RDMA 预取;第二个模块放在第 14 层也留出了更长的
  通信隐藏窗口(第一个放 L1 是为了尽早替代静态局部模式的重建);
- 本机门控为 `sigmoid(copysign(sqrt(|dot|), dot))`(signed-sqrt,代码注释「matching the training kernel」),
  比报告里写的 σ(dot/√d) 多一层 signed-sqrt。

## MoE 与路由

- 每层 `ffn.gate.weight`(384×5120,bf16)+ `gate.bias` + `gate.bias_vl`(fp32,noaux_tc 校正;
  图像 span 内的 token 用 bias_vl,`noaux_tc_for_vl`);
- 打分 $\sqrt{\mathrm{softplus}(x/\tau)}$;bias 只参与选专家,不进权重;权重 = 原始分数归一化 × 1.5;
- 路由专家 SwiGLU(w1/w3:5120→2304,w2:2304→5120),**FP4 E2M1 量化**(见下节);
- 1 个共享专家(同维度,FP8 量化),每 token 必过;
- Expert 前向带 swiglu_limit=10 的 clamp(来自训练侧 fp8/fp4 激活值域约束)。

## DSpark MTP(多 token 预测/草稿)

`num_nextn_predict_layers=3`,checkpoint 前缀 `mtp.0/1/2`(对应 layer_id 40/41/42),
每层是一个完整 Block(注意力 + MoE,128 个路由专家选 3),embed/head 与主干共享:

- `mtp.0.main_proj`(15360→5120,FP8)+ `main_norm`:把 target 层(37/38/39)的**注意力输入**
  (非输出)拼接投影成草稿流输入;
- 注意力为 DSparkAttention:block_size=5 的块级草稿,窗口外只看本块;
- `mtp.2`(末段)独有 `norm`、`markov_head`(embed/head 各 129280×256,bf16,一阶马尔可夫 logit 偏置)、
  `confidence_head.proj`(5376→1,给草稿 token 打置信度);noise_token_id=128799。

## 视觉塔

- `vision.*`:32 层 ViT(hidden 1024,16 头,patch 14 → patch_embed.proj 1024×588=3·14·14,
  MLP SwiGLU 融合 w1 [5632,1024],w2 [1024,2816]),带 bias,标准 pre-norm,bf16 未量化;
- `aligner.w1/w2`(9216→5120→5120,带 bias):3×3 邻域(downsample_ratio=3)拼 9216 维投到语言模型;
- `image_start/image_end/image_newline`:5120 维可学习的图像 span 边界嵌入;image_token_id=129264。

## 权重命名与量化分布

权重名**没有** `model.` 前缀(Kimi-K3 有 `language_model.` 前缀,注意区分)。

### 直接可读(BF16/F32,未量化)

| 权重 | 形状/数量 | 说明 |
| --- | --- | --- |
| `embed.weight` / `head.weight` | 129280 × 5120 | 词嵌入 / LM Head,**不共享** |
| `norm.weight` | 5120 | lm_head 前的最终 RMSNorm |
| `layers.N.attn_norm/ffn_norm.weight` | 40 层 | 子层入口 RMSNorm(HC pre 之后) |
| `layers.N.attn.{q_norm,kv_norm}.weight` | 40 层 | Q 低秩/KV latent 后的 RMSNorm |
| `layers.N.ffn.gate.{weight,bias,bias_vl}` | 40 层 | MoE 路由器 |
| `layers.N.hc_*` | 40 层,fp32 | Hyper-Connections 混合参数 |
| `layers.N.attn.attn_sink` | 40 层,fp32,64 维 | 注意力 sink |
| `layers.N.attn.{compressor,indexer}.*` | 仅源层 | 压缩/检索模块(多为 bf16,仅 indexer.wq_b 为 FP8) |
| `mtp.*`、`vision.*`、`aligner.*`、`image_*` | | DSpark 草稿层(个别 FP8)与视觉塔(全 bf16) |

### FP8 E4M3 + UE8M0 块量化(32×32 块)

`wq_a/wq_b/wkv/wo_a/wo_b`(40 层)、`ffn.shared_experts.w{1,2,3}`(40 层)、
`engram.wkv`、`indexer.wq_b`(8 层)、mtp 对应项、`mtp.0.main_proj`。
反量化:`weight(fp8 查表) × scale(UE8M0=2^(e-127))` 逐 32×32 块广播
(`llm_lens.dequant_block`,已与 torch 官方实现对拍逐值一致)。

### FP4 E2M1 路由专家(I8 打包 + UE8M0 行块 1×32)

`layers.N.ffn.experts.E.w{1,2,3}.weight`(I8,列数折半:2304×2560 = 2304×5120 个 FP4)
+ `.scale`(F8_E8M0,2304×160)。打包:每字节低半字节为偶数列、高半字节为奇数列;
值表 {0,±0.5,±1,±1.5,±2,±3,±4,±6}(`llm_lens.unpack_fp4_e2m1`,已与官方
`inference/convert.py` 的查表解码对拍逐值一致)。mtp 的 128 专家同格式。

**含义:词嵌入、LM Head、全部 norm、路由器、HC、视觉塔都是 bf16/fp32 可直接分析;
注意力投影与共享专家需 FP8 块反量化(一行代码);路由专家需 FP4 解包 + 反量化。**
最大的单体权重是两张 engram 哈希表(各 ~92 GiB),只建议按行抽样读取。

## 分析切入点建议

1. **词嵌入 / LM Head**:余弦分布、配对相关性、各向异性(已做,见子目录笔记)。
2. **注意力**:`wkv`(5120→512)与 `wq_a`/`wq_b` 的奇异值谱;单 KV 头下 64 头 Q 的共享/分化程度
   (各头 Q 行与共享 K 的有效 logit 尺度);`attn_sink` 的逐层分布。
3. **稀疏检索**:indexer 的 `wk`/`wq_b` 有效秩;`weights_proj`(5120→32)行范数——哪些通道决定检索打分。
4. **MoE 路由器**:`gate.weight` 行向量夹角分布(专家是否正交分化);`bias`/`bias_vl` 分布及二者差异
   (文本 vs 图像 token 的专家偏好)。
5. **路由专家(FP4)**:反量化后做矩/死行分析,与共享专家对比(已抽样做初步矩分析);
   注意单专家 w1 反量化后 2304×5120×4B ≈ 47 MB,按需逐个读。
6. **Hyper-Connections**:`hc_fn` 的奇异结构;`hc_base`/`hc_scale` 逐层分布;Sinkhorn 后的 comb
   矩阵离单位阵多远(残差是否已显著跨副本混合)。
7. **Engram**:哈希表行的范数/余弦分布(抽样);`q_weight·k_weight` 增益画像;n-gram 门的典型强度。
8. **DSpark MTP**:`markov_head` 与主 embed/head 的关系(一阶马尔可夫 vs 语义);`confidence_head` 权重范数。
