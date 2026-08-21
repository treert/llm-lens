# 资源估算:显存与速度的定量账

前置:[01-inference-pipeline.md](01-inference-pipeline.md)(两阶段、KV Cache 的诞生)。
目标:给定一张显卡,定量回答三个问题——**放不放得下?能开多长上下文?
大概跑多快?**

本文实例使用两台真实机器:

| 机器 | GPU | 显存 | 显存带宽 | 架构 |
|---|---|---|---|---|
| 工作机 | RTX 5070 Ti | 16 GB | ~896 GB/s (GDDR7) | Blackwell,SM120,支持 FP8/FP4 |
| 家里机 | RTX 4080 | 16 GB | ~717 GB/s (GDDR6X) | Ada,SM89,支持 FP8 |

## 0. 总账

$$\text{总显存} = \underbrace{\text{权重}}_{\text{固定}} +
\underbrace{\text{KV Cache}}_{\text{随 batch}\times\text{seqlen 线性增长}} +
\underbrace{\text{激活与临时 buffer}}_{\text{随 prefill chunk 大小}} +
\underbrace{\text{运行时开销}}_{\text{CUDA context、框架、碎片}}$$

权重是"门票",KV Cache 是"日常开销",后两项是"手续费"。16 GB 卡上
手续费约占 1~1.5 GB,**可用预算按 ~14.5 GB 规划比较稳**(Windows 下显卡
还要驱动显示器,再扣一点)。

## 1. 权重显存

$$\text{权重字节} = \text{参数量} \times \text{每参数字节}$$

| 精度 | 每参数字节 | 说明 |
|---|---|---|
| FP16 / BF16 | 2 | 训练与高精度推理的标准 |
| FP8 / INT8 | 1 | Blackwell/Ada 有 FP8 硬件加速;INT8 多为 W8A8 |
| INT4(GPTQ/AWQ/GGUF Q4) | ~0.55~0.6 | 含 scale/zero-point 等元数据,比理论 0.5 略大 |

按档位换算(实际文件大小,含少量嵌入与元数据):

| 模型 | FP16 | Q8 | Q4 |
|---|---|---|---|
| 7B~8B | 15~16 GB | ~8.5 GB | 4.5~5 GB |
| 13B~14B | 27~28 GB | ~14.5 GB | 8~9 GB |
| 32B | ~64 GB | ~33 GB | 18~19 GB |
| 70B | ~140 GB | ~72 GB | ~40 GB |

对 16 GB 卡的第一结论:**8B 必须量化(FP16 权重本身就顶满),14B Q4 是
甜点位,32B Q4 放不下**(Q3 约 14 GB 但 KV 空间被挤光,不实用;除非
CPU offload,见 §5)。

## 2. KV Cache 显存

公式(推导见 [01-inference-pipeline.md](01-inference-pipeline.md) §6):

$$\text{bytes/token} = \underbrace{2}_{K,V} \times L_{\text{层}} \times
n_{\text{kv头}} \times d_{\text{头}} \times \text{每元素字节}$$

**架构差异极大**,同一量级模型可差近 10 倍(FP16 计):

| 模型 | 层数 | KV 头 | 头维 | bytes/token | 备注 |
|---|---|---|---|---|---|
| Llama-2-7B(MHA) | 32 | 32 | 128 | 512 KB | 老架构,KV 头 = Q 头 |
| Llama-3-8B(GQA) | 32 | 8 | 128 | 128 KB | GQA 把 KV 头砍到 1/4 |
| Qwen3-8B(GQA) | 36 | 8 | 128 | 144 KB | 层数更多 |
| Qwen3-14B(GQA) | 40 | 8 | 128 | 160 KB | |
| Qwen3-32B(GQA) | 64 | 8 | 128 | 256 KB | |
| DeepSeek-V3 / Kimi(MLA) | 61 | — | — | ~69 KB | 缓存压缩潜向量(512+64 维),与头数无关 |

换算成"每 GB 能装的 token 数":Llama-3-8B 约 8K tokens/GB,MLA 模型约
15K tokens/GB,MHA 老模型只有 2K tokens/GB。

并发时要再乘 batch:10 个并发请求、各 8K 上下文,Llama-3-8B 就是
$10 \times 8192 \times 128\,\text{KB} \approx 10$ GB。**长上下文 × 高并发
是显存的第一杀手**,这也是为什么生产系统围绕 KV Cache 做足了文章
(PagedAttention 分页管理、KV 量化、MLA、滑动窗口……后续篇章展开)。

## 3. 激活与运行时开销

- **激活**:前向过程中算子之间的中间结果张量(norm 输出、attention 输出、
  MLP 中间量、残差流),是"数据"而非阶段概念。训练要全存下来供反向
  传播,是训练显存大头;推理只前向、用过即弃,只需容纳**单次前向的
  工作集**,体积与一次前向处理的 token 数成正比:decode 时只有
  $O(\text{batch} \times d)$,几 MB,可忽略;
  prefill 与 chunk 大小成正比——框架用 chunked prefill 把每次前向限制在
  2048~8192 个 token,激活 + attention 临时量控制在几百 MB ~ 2 GB。
- **logits 临时量**:$[\text{token数}, V]$ 很大($V \approx 128K$);
  推理框架只对需要采样的位置算 logits,规避了这一项。
- **运行时固定开销**:CUDA context 与驱动 ~0.5 GB;框架常驻与 kernel
  模块数百 MB;CUDA Graph 捕获缓冲数百 MB;再留碎片余量。
- vLLM 用 `gpu_memory_utilization`(默认 0.90)统一记账:启动时先加载权重、
  跑一遍 profile,**把剩余显存全部预分配为 KV block 池**,日志会直接报告
  KV 池能容纳多少 token——这是最省心的"自动估算"。

## 4. 速度估算:带宽定 decode,算力定 prefill

decode 是带宽瓶颈([01-inference-pipeline.md](01-inference-pipeline.md) §5),
单请求理论上限:

$$\text{tok/s} \approx \frac{\text{显存带宽}}{\text{每 token 读取字节}}
\approx \frac{\text{带宽}}{\text{权重字节}}$$

| 场景 | 4080 (717 GB/s) | 5070 Ti (896 GB/s) |
|---|---|---|
| 8B FP16(16 GB) | ~45 tok/s | ~56 tok/s |
| 8B Q4(5 GB) | ~143 tok/s | ~179 tok/s |
| 14B Q4(9 GB) | ~80 tok/s | ~100 tok/s |

注意三点:

- 这是**上界**。量化模型要额外做反量化计算,且 KV Cache 读取随上下文
  增长而加重,实际打 5~7 折:8B Q4 实际约 50~110 tok/s(4080 靠低端、
  5070 Ti 靠高端),对个人使用绰绰有余(人阅读速度 < 20 tok/s)。
- 量化在省显存的同时还**提速**——读取字节直接减半/再减半。这是消费卡
  上量化几乎无代价的另一面。
- 5070 Ti 带宽比 4080 高约 25%,decode 直接快 25%;FP16 tensor 算力
  两者同档(~200 TFLOPS),但 Blackwell 的 FP8/FP4 算力翻倍,跑 FP8
  模型时 prefill 优势更大。

prefill 是算力瓶颈,理论上限 ≈ $\text{TFLOPS} / (2 \times \text{参数量})$,
8B 模型在 ~200 TFLOPS 卡上约 12K tok/s,实际 30%~60%(4~7K tok/s):
一个 2K token 的 prompt,首 token 延迟约 0.3~0.5 s。

## 5. 16 GB 卡的部署菜单(按 §0 总账实算)

可用预算 ~14.5 GB。逐案推演(decode 速度按 5070 Ti 估,4080 乘 0.8 折算):

**案例 A:8B Q4(如 Llama-3-8B / Qwen3-8B GGUF Q4_K_M,~5 GB)** — 甜点
- 剩 ~9.5 GB 给 KV + 开销;Llama-3-8B @128 KB/token → 约 70K+ tokens 的
  KV 空间,单人 32K 上下文随便用;
- decode ~60-110 tok/s。**结论:单人日常使用首选**。

**案例 B:8B FP16(16 GB)** — 不可行
- 权重顶满,KV 无处安放。要么 Q8(~8.5 GB,剩 ~6 GB → ~44K tokens),
  要么 Q4。

**案例 C:14B Q4(如 Qwen3-14B,~9 GB)** — 可行,质量/速度折中
- 剩 ~5.5 GB → @160 KB/token ≈ 34K tokens;decode ~55-70 tok/s(打折后);
- 适合"想要更好回答质量,上下文不极端长"的场景。

**案例 D:32B(Q4 ~18-19 GB)** — 单卡放不下
- 出路一:llama.cpp **CPU offload**——部分层放内存用 CPU 算,其余放显卡。
  容量解决了,但每 token 都要过内存总线(DDR5 双通道 ~100 GB/s,比显存
  慢一个量级),速度掉到个位数 tok/s,只适合离线批处理;
- 出路二:选 MoE 模型(总参数大但激活少,如 30B-A3B),decode 时按激活
  参数量读数据,速度红利真实;但容量不是免费的——30B 的 Q4_K_M 仍约
  17 GB,超过 16 GB 卡的预算,要 Q3/IQ4 级进一步压缩才勉强放得下,
  KV 余量随之紧张。

**案例 E:双卡?** 4080 与 5070 Ti 在两台机器上,组不了 NVLink;
单机双卡 16+16 GB 走 PCIe 张量并行可以放下 32B Q4,但通信开销吃掉部分
收益——属于"并行与分布式"篇的话题。

### 经验法则

1. 权重(量化后)≤ 显存的 60%,剩下留给 KV 与开销;
2. 单人场景先保证"权重放得下 + 32K 上下文",再谈并发;
3. 模型档位升一档(8B→14B)的智力收益,通常大于量化降一档(Q8→Q4)
  的质量损失——**16 GB 卡上 14B Q4 往往优于 8B Q8**;
4. 长上下文需求优先选 GQA/MLA 架构的模型,同等显存 KV 容量差 4~8 倍。

## 6. 实测验证(部署后花 5 分钟对账)

- `nvidia-smi`:看 `memory.used` 是否符合预期;Windows 下注意 WDDM
  与显示器占用。
- llama.cpp:启动日志直接打印各 buffer 大小与 `llama_kv_cache` 容量;
  生成时打印 `eval time` 的 tok/s。
- vLLM:启动日志的 `GPU KV cache size: ... tokens` 就是 KV 池容量;
  `Maximum concurrency for ... tokens per request` 是并发估算。
- 与 §4 的理论上界对比:实际值若远低于 5 折,通常是 offload 错配、
  上下文过长导致 KV 读取占比上升,或批大小不合适。

## 7. 小结

- 总账 = 权重(门票)+ KV(随规模线性)+ 激活(chunk 可控)+ 固定开销
  (~1.5 GB);
- 16 GB 卡的甜点位:**8B Q4 随便用,14B Q4 是质量上限,32B 需要
  MoE 或 offload 妥协**;
- decode 看带宽,5070 Ti 比 4080 快约 25%;prefill 看算力,两卡同档;
- 上下文与并发是 KV 的乘数,架构(GQA/MLA)决定 KV 的单价。
