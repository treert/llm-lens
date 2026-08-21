# (进阶)Kernel 定制:从读懂到改写

前置:[08-kernel-math.md](08-kernel-math.md)、[10-flashattention.md](10-flashattention.md)、
[07-gpu-hardware.md](07-gpu-hardware.md)。
目标:知道什么时候需要自己碰 kernel、CUDA/Triton 的心智模型、
读 FlashAttention 级别源码的路线图,以及 profiling 工作流。

> 本篇是地图不是教程:给出路径与关键点,具体语法跟官方教程走。

## 1. 什么时候需要,什么时候不需要

**不需要**(覆盖 95% 场景):主流算子(vLLM/FA 的 kernel)已被极致优化,
你的瓶颈几乎永远在显存、带宽、调度,而不是算子本身。先用
[02-resource-estimation.md](02-resource-estimation.md) 的账定位瓶颈。

**需要**的合理场景:

- 把一串 pointwise 操作融合成一个 kernel(省中间结果的 HBM 往返);
- 新研究 idea 没有现成算子(新的注意力变体、新的量化格式);
- 复现/理解论文——**学习目的本身就是最正当的理由**。

## 2. CUDA 心智模型(最小集)

- **三级并行**:grid → block(CTA,落在一个 SM 上)→ thread;
  32 线程 = 1 warp,**同 warp 内线程锁步执行**(分支发散 = 性能杀手,
  回顾 Gumbel-Max 消灭分支的意义,[08-kernel-math.md](08-kernel-math.md) §4);
- **访存纪律**:同 warp 访问连续地址(coalescing)才能用满带宽;
  用 128-bit 向量加载(`float4`);片上 SRAM 显式管理(`__shared__`);
- **规约模式**:warp 内用 `__shfl_xor` 蝴蝶交换,32 个数 5 步求和
  (FA2 的 softmax 规约就是这么做的);
- **occupancy**:每 SM 常驻的 warp 数,决定能否隐藏访存延迟;
  寄存器用多了 → occupancy 下降 → 延迟藏不住(FA4 软算 exp 时
  "寄存器溢出反噬性能"就是这个,[10-flashattention.md](10-flashattention.md) §5)。

一个 RMSNorm CUDA kernel 的思路正好串起以上全部:每 block 负责一行
(token),线程向量加载 → 块内两级规约(warp shuffle + shared mem)
求 $\sum x^2$ → `rsqrt` → 缩放写回。与 [08-kernel-math.md](08-kernel-math.md) §1
完全对应。

## 3. Triton:Python 语法,接近 CUDA 的性能

Triton 把并行单位从 thread 抬到 **block**:你写"对这个块做这些运算",
线程映射、coalescing、shared memory 由编译器处理。RMSNorm 全核:

```python
@triton.jit
def rmsnorm_kernel(X, W, Y, D: tl.constexpr, eps: tl.constexpr):
    row = tl.program_id(0)                 # 一个 program 处理一行
    cols = tl.arange(0, D)                 # D 取 2 的幂,不足补 mask
    x = tl.load(X + row * D + cols)        # 整行读入寄存器/SRAM
    ms = tl.sum(x * x, axis=0) / D         # 块内规约:均方
    y = x * tl.rsqrt(ms + eps)             # 归一化
    tl.store(Y + row * D + cols, y * tl.load(W + cols))
```

对照 [08-kernel-math.md](08-kernel-math.md) §1:一次遍历、无均值、块内闭环——
Triton 代码几乎是数学公式的直译。学习路径:官方 tutorials
(vector add → fused softmax → matmul)正好覆盖本目录的算子。

**Triton vs CUDA**:原型/研究与多数生产算子 Triton 足够(vLLM 大量
kernel 是 Triton);压榨最后 20%(FA3/FA4 级、硬件新特性 TMA/wgmma)
仍需 CUDA + CUTLASS。

## 4. torch.compile:不手写也能融合

`torch.compile(model)` 的 inductor 后端自动做:pointwise 链融合、
生成 Triton kernel、CUDA Graph 捕获(消除 launch 开销)。适合:

- 自己拼的研究代码,先 compile 再谈手写;
- 观察它生成的 Triton 代码,是极好的学习材料(看它融合了你哪几行)。

局限:attention 等已有高度优化实现的算子,compile 不会超越 FA;
动态 shape(varlen)场景需要额外处理。

## 5. FlashAttention 源码导读地图

按 [10-flashattention.md](10-flashattention.md) 的概念找代码,建议顺序:

1. **Triton 版 tutorial**(triton repo `fused-attention.py`):几百行,
   外 Q 内 KV 循环、 $(m, \ell, \tilde O)$ 三元组、除 $\ell$ 收尾——
   与本文档 §2 伪代码逐行对应,先读这个;
2. **FA2 官方 CUDA**(flash-attention repo):CUTLASS 风格,
   看 grid 维度的并行化与 warp 分工如何落地(本文档 §4);
3. **vLLM 的 paged attention kernel**:在 FA 基础上叠加 block table
   gather,看 PagedAttention 与 FA 如何合一;
4. **FA3/FA4(Hopper/Blackwell)**:persistent kernel、TMA、warp
   专门化——读懂前两步后再挑战,重点是调度器而非数学。

## 6. Profiling 工作流

从粗到细四件套:

1. **nvidia-smi**:`utilization.gpu` 高但 `utilization.memory` 也高?
   工具太粗,只能看趋势;
2. **torch.profiler / 框架自带指标**:算子级耗时与显存;vLLM 暴露
   Prometheus 指标(吞吐、KV 池水位、batch 组成);
3. **Nsight Systems(nsys)**:时间线视角——kernel 之间有没有空泡
   (launch 间隙、同步等待),决定要不要 CUDA Graph;
4. **Nsight Compute(ncu)**:单 kernel 解剖——直接告诉你
   "DRAM throughput 87%, SM busy 23%"(memory-bound 实锤)还是相反,
   以及 coalescing、occupancy 诊断。

判断口诀:**DRAM 忙 SM 闲 → memory-bound,去省访存;SM 忙 DRAM 闲 →
compute-bound,去降 FLOPs 或换 tensor core 路径**。

## 7. 建议实践路线(结合本仓库)

1. Triton 写 RMSNorm,与 `torch.nn.functional` 对拍数值;
2. 写 fused RoPE(回顾 `cos_sin_cache` 查表,[08-kernel-math.md](08-kernel-math.md) §5);
3. 用 ncu 对比自己写的 RMSNorm 与框架版本,看 DRAM 利用率差在哪;
4. 读 Triton 版 fused-attention,把 [10-flashattention.md](10-flashattention.md)
   每个概念标注到代码行;
5. (长线)给 vLLM 提一个小优化 PR。

本仓库的权重分析工具(`py-src/llm_lens`)可以负责提供测试输入:
直接从 safetensors 抽出真实 RMSNorm 权重与激活量级,比对数值更真实。

## 8. 小结

- 先算账定位瓶颈,绝大多数时候不需要写 kernel;
- 学习路径:Triton(天)→ torch.compile 生成码(读)→ Triton FA(周)
  → CUDA/CUTLASS(月);
- profiling 四件套:nvidia-smi → torch.profiler → nsys → ncu;
- 写 kernel 的终极检验标准还是那句:**访存少了没有,并行度高了没有**。
