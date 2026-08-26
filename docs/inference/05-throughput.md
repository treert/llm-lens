# 吞吐优化:让 GPU 永远有活干

前置:[01-inference-pipeline.md](01-inference-pipeline.md)(两阶段与瓶颈)、
[02-resource-estimation.md](02-resource-estimation.md)(KV 显存账)。
目标:理解 serving 系统把吞吐再抬一个数量级的四大机制——
continuous batching、PagedAttention、chunked prefill、prefix caching,
以及单请求加速的 speculative decoding。

## 1. 问题的起点:朴素批处理的三种浪费

回忆 [01-inference-pipeline.md](01-inference-pipeline.md) §5:batching 是 decode
吞吐的第一杠杆(一次权重读取服务 $B$ 个请求)。但朴素的"凑满一批、
跑完再放下一批"有三种浪费:

1. **长度不齐**:同批请求生成长度不同,先结束的必须等最长的,
   GPU 陪跑空转(padding 或等待);
2. **新请求排队**:batch 一旦开跑,新请求必须等整批结束;
3. **prefill 卡顿**:一个长 prompt 的 prefill 插进来,所有 decode 中的
   请求的 TPOT 被拖长。

serving 系统的吞吐优化,就是逐一消灭这三种浪费。

## 2. Continuous batching:迭代级调度

Orca(2022)提出的方案,也是 vLLM 的调度核心:**调度粒度从"请求"
细化到"单次迭代"(每生成一个 token 的那步前向)**。

- 每次迭代前,调度器重新决定本步跑哪些请求:有请求结束(EOS/超长)
  就**立即撤出**,队列里的新请求**立即插入**;
- 不再有"一批"的概念,GPU 上永远跑着当前所有活跃请求的混合体;
- 相对朴素批处理,吞吐提升可达数倍到一个数量级(请求长度差异越大,
  收益越大)。

代价:每步都要面对"batch 内各请求长度、位置都不同"的杂乱局面——
attention kernel 必须支持变长(varlen),这正是 FlashAttention 的
varlen 接口与 Flash-Decoding 存在的原因([10-flashattention.md](10-flashattention.md)
§7);而 KV 的杂乱存放则引出下一个机制。

## 3. PagedAttention:像 OS 管内存一样管 KV

连续 batching 下,每个请求的 KV 随生成不断增长、长度事先未知。若按
"最大长度"预分配连续显存,碎片与预留浪费可达 60%+。vLLM 的
PagedAttention 借鉴操作系统虚拟内存:

- KV 按固定大小的 **页(block,如 16 个 token)** 分配,页在显存中
  不必连续;
- 每个请求维护一张 **block table**(页表),kernel 顺着页表指针把离散页
  gather 进 SRAM 计算;
- 新 token 只需 append,增长按页按需分配,几乎零浪费;
- **Copy-on-Write 共享**:同一前缀的多个序列(平行采样 n>1、beam
  search、多轮对话的历史)共享物理页,写时才复制——一份前缀 KV
  服务多个分支。

效果:KV 浪费从 60%+ 降到 ~4%,同等显存下可并发的请求数翻倍以上。
注意 PagedAttention 是**显存管理**机制,与 FlashAttention 的**计算**
机制正交,现代 kernel 把两者合一(分页 gather + FA 计算)。

## 4. Chunked prefill:不让长 prompt 卡住所有人

一个 8K token 的 prefill 可能耗时数百 ms——若它独占一次前向,所有
decode 请求的 TPOT 立刻出现肉眼可见的卡顿。方案:

- 把 prefill 切成固定预算的**块**(如每步最多 2048 个 token,
  vLLM 的 `max_num_batched_tokens`);
- 每次迭代 = 若干 decode 请求 + 一块 prefill,**混合排布**;
- decode 几乎不受干扰(每步预算里 prefill 只占固定份额),prefill
  则利用 decode 剩余的算力细水长流;
- 数学依据:prefill 是算力瓶颈、decode 是带宽瓶颈,两者混合恰好
  互补——一次前向里 GEMM(算力)与 GEMV(带宽)各取所需。

相关进阶:**PD 分离(prefill-decode disaggregation)** 把两阶段拆到
不同的 GPU 池独立扩缩容,prefill 节点算完把 KV 传给 decode 节点。
生产级系统(Mooncake 等)的做法,单机用户了解即可。

## 5. Prefix caching:跨请求复用 KV

很多场景前缀天然重复:相同的 system prompt、多轮对话的历史、
Agent 反复携带的工具描述。既然 KV 只由前缀决定
([01-inference-pipeline.md](01-inference-pipeline.md) §4),那**相同前缀的 KV
可以直接复用,跳过重复 prefill**:

- vLLM 的 APC(Automatic Prefix Caching):按 block 哈希缓存历史 KV;
- SGLang 的 RadixAttention:用**基数树(radix tree)**组织缓存,
  按最长前缀匹配,天然契合树状对话结构;
- 命中时 TTFT 大幅下降(prefill 量归零),这在 Agent 场景收益巨大
  (每轮 prefix 占比常超过 90%)。

## 6. Speculative decoding:单请求也能加速

前面的机制都在优化"多请求",speculative decoding(投机解码)优化的是
**单请求 decode 延迟**,其依据是一个美妙的观察:

> decode 是带宽瓶颈:一次前向读全部权重。那么**一次前向算 1 个 token
> 还是算 k 个 token,成本几乎相同**(带宽没变,算力本来就闲置)。

**草稿-验证**流程:

1. 用一个便宜得多的**草稿模型**(或草稿头)串行猜出接下来 $k$ 个 token;
2. 目标大模型一次前向,并行计算这 $k$ 个位置各自的 logits
   (像一个小型 prefill);
3. 逐个比对:接受与目标模型一致的最长前缀,第一个分歧处以目标模型
   的分布修正,后面的全部丢弃。

**无损性**:配合 rejection sampling,输出分布与目标模型**严格相同**——
这不是近似加速,是精确加速(同 online softmax 一样属于"数学等价变换
换硬件利用率"家族)。注意验证阶段要算多个候选位置,attention 需要
特殊的树形/多 token mask——这正是 08-kernel-math.md §2 说"decode 无需
causal mask"时标注"Speculative Decoding 除外"的原因。

**加速比** ≈ 每步平均接受的 token 数,实践中 1.5~3×:

- 关键在于**接受率**:文本越可预测(代码、固定格式、翻译),接受率越高;
  高温采样接受率下降;
- 草稿从哪来:小模型(同系列 1B 给 8B 起草)、Medusa(目标模型加多个
  预测头)、EAGLE(在特征层起草,接受率更高)、自起草(n-gram/prompt
  lookup,适合重复性文本)、MTP(DeepSeek-V3 训练时自带多 token 预测头);
- vLLM/SGLang 均有开关;对 16 GB 卡的单人场景,这是最值得玩的加速项。

## 7. 各机制的优化对象总览

| 机制 | 消灭的浪费 | 主要收益 |
|---|---|---|
| Continuous batching | 请求间等待 | 吞吐 ×数倍 |
| PagedAttention | KV 碎片与预留 | 同显存并发 ×2+ |
| Chunked prefill | prefill 卡 decode | TPOT 平稳 |
| Prefix caching | 重复前缀的重算 | TTFT ↓(Agent 场景巨大) |
| PD 分离 | 两阶段互相干扰 | 生产集群效率 |
| Speculative decoding | 单请求算力闲置 | 单请求延迟 1.5~3× |

## 8. 小结

- 吞吐优化的主线:**batching 之后,一切围绕"调度"与"KV 复用"**;
- 单人用户主要受益:speculative decoding(延迟)、prefix caching
  (多轮对话 TTFT);
- 这些机制都是 vLLM/SGLang 的开箱功能,理解原理的价值在于**会调参**:
  `max_num_batched_tokens`、`max_num_seqs`、KV 池大小、投机参数,
  每个都对应本文的某笔账。
