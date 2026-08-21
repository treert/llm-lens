# 推理部署(Inference & Serving)笔记

目标:从原理层面理解大模型推理部署,最终能在自己的机器上定制部署开源模型
——会选框架、会估算资源、会调参数,必要时看得懂/改得动 kernel。

与仓库其他文档的分工:

- `docs/rmsnorm.md`、`docs/attention-score-distribution.md` 等偏**权重/分布**视角:
  回答"数学上为什么成立";
- 本目录偏**系统/工程**视角:同一个算子在 GPU 上怎么跑、为什么慢、怎么省。

## 笔记索引

**部署主线**(建议按序):

1. [01-inference-pipeline.md](01-inference-pipeline.md):推理流程全景——prefill/decode
   两阶段、自回归循环、KV Cache 的诞生、带宽瓶颈的定量账、TTFT/TPOT 指标。
2. [02-resource-estimation.md](02-resource-estimation.md):资源估算——显存总账
   (权重/KV/激活/开销)、常见架构的 KV 单价、带宽定 decode 的速度上界,
   实例基于本仓库两台 16 GB 卡(RTX 5070 Ti / RTX 4080)。
3. [03-deployment-basics.md](03-deployment-basics.md):单机部署实操——推理引擎内部
   结构、llama.cpp / vLLM / SGLang 选型、GGUF 文件名解读、离线批跑 vs
   在线 serving、Windows 双机实操路径与常见坑。
4. [04-quantization.md](04-quantization.md):量化——分组与 scale/zero-point 的数学、
   GGUF K-quant / GPTQ / AWQ / FP8 的分工、W4A16 vs W8A8、KV Cache 量化、
   选型法则(Q4_K_M 起步,质量敏感升档,生产上 FP8)。
5. [05-throughput.md](05-throughput.md):吞吐优化——continuous batching、
   PagedAttention、chunked prefill、prefix caching、PD 分离、
   speculative decoding(无损加速)。
6. [06-parallelism.md](06-parallelism.md):并行与分布式——TP/PP/DP/EP 各自切什么、
   通信原语与互联带宽账、消费卡的现实边界、MoE 的单卡红利。
7. [07-gpu-hardware.md](07-gpu-hardware.md):硬件常识——roofline 模型、存储层级表、
   Tensor Core 数据类型代际、消费卡 vs 数据中心卡、读规格书清单。

**算子数学线**(kernel 视角,按序):

8. [08-kernel-math.md](08-kernel-math.md):核心算子(RMSNorm / Softmax / Causal Mask /
   Sampling)的数学性质与 Kernel 优化逻辑——"数学等价变换换硬件利用率"。
9. [09-online-softmax.md](09-online-softmax.md):$(m, \ell)$ 流式更新的完整推导、
   手算例子、可结合 merge、LSE 及其数值性质。
10. [10-flashattention.md](10-flashattention.md):GPU 存储层级与内存墙、tiling kernel
    结构、复杂度账、FA1→FA4 演进、反向重算、Flash-Decoding 与 Ring Attention。

**进阶**:

11. [11-kernel-dev.md](11-kernel-dev.md):Kernel 定制——CUDA/Triton 心智模型、
    RMSNorm 的 Triton 实现、FlashAttention 源码导读地图、profiling 四件套。

**动手**:

12. [12-deployment-hands-on.md](12-deployment-hands-on.md):实操手册——llama.cpp
    安装、Qwen3-8B Q4_K_M 起服务、llama-bench 对账、14B/投机解码/WSL2+vLLM
    进阶、故障排查与实测记录模板。

原则:每个主题先讲清"数学/机制上为什么",再落到"工程上怎么做",
能实测的尽量用本机或小模型实测验证。
