# 推理流程全景:Prefill、Decode 与 KV Cache

定位:推理部署知识的第一篇。回答三个问题——一次文本生成在模型内部到底
发生了什么?为什么推理天然分成两个阶段?为什么所有推理系统都在围绕
KV Cache 做文章?

后续篇章([10-flashattention.md](10-flashattention.md)、资源估算、batching、量化)
都以本文的概念为地基。

## 1. 推理与训练的区别

| | 训练 | 推理 |
|---|---|---|
| 目标 | 更新权重 | 权重冻结,只前向 |
| 反向传播 | 有,激活要全存 | 无 |
| 输入 | 整段序列并行 | prompt 并行,生成部分**逐 token 串行** |
| 显存大头 | 权重 + 梯度 + 优化器状态 + 激活 | 权重 + KV Cache |
| 重复性 | 同样的数据反复过 | 每个 token 都要过一次完整模型 |

推理的特殊之处全在最后一行:**每生成 1 个 token,都要把几十 GB 的权重
完整过一遍**。理解这句话,就理解了推理优化 80% 的动机。

## 2. 一次生成的完整链路

以类 Llama 架构为例,从 prompt 到第一个生成 token:

```
prompt → tokenize → ids[N]
       → embedding 查表 → x[N, d]
       → ×L 个 transformer block:
             attn: RMSNorm → q/k/v 投影 → RoPE → attention → o 投影 → 残差相加
             mlp : RMSNorm → gate/up 投影 → SwiGLU → down 投影 → 残差相加
       → final RMSNorm
       → lm_head(与 embedding 常共享权重)→ logits[N, V]
       → 取最后一个位置的 logits → 采样 → 第一个生成 token
```

之后进入循环:把新 token 拼进去,再跑一次完整模型,采样下一个,直到
EOS 或达到 max_tokens。

两个细节:

- prefill 时虽然每个位置都会算出 logits,但**只有最后一个位置的被使用**
  (其余位置是训练的产物,推理时白算,但无法避免——它们的 KV 是需要的);
- 采样前的温度缩放 / top-k / top-p 过滤只作用在这一个位置上
  (temperature 的数学见 [08-kernel-math.md](08-kernel-math.md) §2、§4)。

## 3. 两个阶段:prefill 与 decode

```python
ids = tokenize(prompt)                 # 长度 N
kv   = empty_cache()

# ---- prefill:一次并行处理整个 prompt ----
logits = model(ids, cache=kv)          # N 个 token 一起过模型,顺手填满 KV Cache
next_id = sample(logits[-1])           # 第一个生成 token

# ---- decode:逐 token 自回归 ----
while next_id != EOS and 未超限:
    logits = model([next_id], cache=kv)  # 每步只有 1 个 token 进模型
    next_id = sample(logits)
    yield next_id
```

| | prefill | decode |
|---|---|---|
| 输入 | 整个 prompt($N$ 个 token 并行) | 上一步生成的 1 个 token |
| 输出 | 第一个生成 token + 填满的 KV Cache | 每步 1 个 token |
| 矩阵形态 | GEMM($N \times d$ 乘 $d \times d$) | GEMV($1 \times d$ 乘 $d \times d$) |
| 瓶颈 | **算力**(compute-bound) | **显存带宽**(memory-bound) |
| 对应 attention kernel | FlashAttention | Flash-Decoding(见 [10-flashattention.md](10-flashattention.md) §7) |
| 用户感受 | 决定 TTFT(首 token 延迟) | 决定 TPOT(每 token 间隔) |

## 4. KV Cache 的诞生:一个数学观察

第 $t$ 步生成时,attention 需要什么?新 token 的 query $q_t$ 要与**全部历史**
的 key 做点积,再用权重对**全部历史**的 value 加权:

$$\mathrm{attn}_t = \sum_{i \le t} \mathrm{softmax}\big(q_t k_i / \sqrt{d}\big)\, v_i$$

关键在于:历史 token 的 $k_i, v_i$ 只由 $x_1, \dots, x_i$ 决定(causal mask
保证后面的 token 影响不到它),所以在第 $i$ 步算出的 $k_i, v_i$,到第
$t > i$ 步时**逐比特不变**。于是:

- $K, V$:算过一次就缓存起来,之后每步只做新 token 的投影再 append
  ——这就是 KV Cache;
- $Q$:只有当前步用,用完即弃,**不缓存**;
- 代价:不做缓存的话,第 $t$ 步要对前 $t$ 个 token 重算全部投影,
  总计算量从 $O(N)$ 次投影膨胀到 $O(N^2)$ 次——生成越长越慢,不可接受。

由此顺带解释两个之前见过的结论:

- decode 阶段 $q$ 长度为 1,所有历史 KV 都合法可见,**无需 causal mask**
  ([08-kernel-math.md](08-kernel-math.md) §2);
- decode 的 attention 是"1 个 query 扫长 KV",并行度要从 KV 维切分获得
  ——正是 Flash-Decoding 的 Split-K([10-flashattention.md](10-flashattention.md) §7)。

## 5. 为什么 decode 慢:带宽瓶颈的一笔账

decode 每生成 1 个 token,都要从 HBM 读一遍几乎全部权重(以及越来越长的
KV Cache),却只做一个 token 的计算。算一笔账(7B 模型,FP16,权重 14 GB):

- 每 token 计算量 ≈ $2 \times 7\,\text{B} = 14$ GFLOP;
- 算术强度 = $14\,\text{GFLOP} / 14\,\text{GB} \approx 1$ FLOP/byte;
- H100 的算力/带宽比 ≈ $989\,\text{TFLOPS} / 3.35\,\text{TB/s} \approx 295$
  FLOP/byte ——算术强度差了近 300 倍,**算力几乎全程闲置,速度完全由带宽决定**;
- 单请求 decode 的理论上限 ≈ 带宽 / 权重大小 ≈ $3.35\,\text{TB/s} / 14\,\text{GB}
  \approx 240$ tok/s(还没算 KV Cache 的读取,实际更低);
- 换消费级显卡(4090,带宽约 1 TB/s),上限直接掉到 ~70 tok/s。

prefill 完全不同:$N$ 个 token 并行,同样的权重读一遍算了 $N$ 份,算术强度
$\approx N$ FLOP/byte——prompt 稍长就超过硬件平衡点,瓶颈在 Tensor Core
算力,GPU 吃得满满当当。

**推论**:单请求 decode 喂不饱 GPU,唯一的出路是让一次权重读取服务更多
请求——这就是 batching 的根本动机。batch 内 $B$ 个请求共享同一份权重读取,
算术强度 ×$B$,吞吐近线性提升,直到撞上算力瓶颈或显存瓶颈。生产系统的
continuous batching 等机制(后续篇章)都是围绕"如何高效凑 batch"展开。

## 6. KV Cache 的显存账(预告)

KV Cache 随 batch × 序列长度**线性增长**,公式:

$$\text{bytes} = \underbrace{2}_{K,V} \times L_{\text{层}} \times
n_{\text{kv头}} \times d_{\text{头}} \times \text{seqlen} \times
\text{每元素字节} \times \text{batch}$$

例:Llama-3-8B($L=32$,GQA 8 个 KV 头,$d_{\text{头}}=128$,FP16):

- 每 token:$2 \times 32 \times 8 \times 128 \times 2\,\text{B} = 128$ KB;
- 单请求 8K 上下文 ≈ 1 GB;128K 上下文 ≈ **16 GB——比权重本身还大**。

这就是为什么 KV Cache 是长上下文与并发的头号约束,也是 MQA/GQA/MLA、
KV 量化、PagedAttention 等一整族优化的靶子(资源估算篇会系统展开)。

## 7. 性能指标词汇表

| 指标 | 含义 | 由谁决定 |
|---|---|---|
| TTFT | Time To First Token,首 token 延迟 | prefill 速度(含排队) |
| TPOT / ITL | 相邻 token 间隔 | decode 速度 |
| 端到端延迟 | TTFT + 生成长度 × TPOT | — |
| Throughput | 系统总 tokens/s | batch 效率 |
| 显存水位 | 权重 + KV Cache + 激活与碎片 | 模型与并发配置 |

体验参照:人阅读速度约每秒几个到十几个 token,所以 TPOT 只需 < 100 ms
就"跟得上看";TTFT 则直接是"点了发送多久有反应"的体感。

## 8. 小结

- 推理 = prefill(并行,算力瓶颈,决定 TTFT)+ decode(逐 token 串行,
  带宽瓶颈,决定 TPOT);
- KV Cache 的存在依据是"历史 K/V 在后续步骤中逐比特不变",它把 $O(N^2)$
  的重算变成 append,代价是随长度线性增长的显存;
- decode 的算术强度极低,单请求喂不饱 GPU,batching 是吞吐的第一杠杆;
- 优化地图由此展开:FlashAttention / Flash-Decoding 管两个阶段的 attention
  效率(见 [10-flashattention.md](10-flashattention.md)),量化管权重与 KV 的体积,
  batching 与调度管 GPU 利用率,并行管"单卡放不下"的问题。
