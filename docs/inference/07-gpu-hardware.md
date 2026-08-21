# 硬件常识:为 LLM 推理读 GPU 规格书

前置:[01-inference-pipeline.md](01-inference-pipeline.md) §5(带宽瓶颈的账)。
目标:给"挑卡/估性能"建立定量直觉——roofline 模型、存储层级、
Tensor Core 数据类型,以及消费卡与数据中心卡的差异。

## 1. Roofline:一张图看懂所有瓶颈

定义**算术强度**(arithmetic intensity):

$$I = \frac{\text{FLOPs}}{\text{访存字节数}}$$

硬件有一个**平衡点**(ridge point)= 峰值算力 / 峰值带宽。 $I$ 低于它,
性能被带宽封顶(memory-bound);高于它,被算力封顶(compute-bound)。

| 硬件 | FP16 tensor 算力 | 显存带宽 | 平衡点(FLOP/byte) |
|---|---|---|---|
| RTX 4080 | ~195 TFLOPS | 717 GB/s | **~272** |
| RTX 5070 Ti | ~176 TFLOPS | 896 GB/s | ~196 |
| RTX 4090 | ~330 TFLOPS | 1.0 TB/s | ~330 |
| H100 SXM | ~989 TFLOPS | 3.35 TB/s | ~295 |

回顾推理两阶段([01-inference-pipeline.md](01-inference-pipeline.md)):

- **decode**:$I \approx 1$(每读 2 字节权重做 ~2 次乘加),离平衡点差
  200~300 倍 → 深度带宽瓶颈,跑带宽就是跑钱;
- **prefill**:$I \approx N$(每步 $N$ 个 token 复用同一遍权重),
  $N$ 达到几百就进入算力区 → 跑算力;
- batching 的本质:把 decode 的 $I$ 乘以 $B$,向平衡点靠拢。

所有推理优化几乎都能在这张图上定位:量化(降低字节数 → 同带宽下
吞吐升高)、FlashAttention(把 softmax 的访存从 HBM 提到 SRAM →
提高有效 $I$)、speculative decoding(带宽不变,$I$ × $k$)。

## 2. 存储层级:优化就是"让数据往上一层"

以 H100 为例的数量级(消费卡同构,绝对值不同):

| 层级 | 容量 | 带宽 | 延迟 |
|---|---|---|---|
| 寄存器 | 每 SM 256 KB | 最高 | ~1 cycle 级 |
| SRAM(shared memory) | 每 SM ~228 KB | 全 GPU 聚合 ~30 TB/s | ~20-30 cycles |
| L2 | 50 MB | ~12 TB/s | ~200-300 cycles |
| HBM | 80 GB | 3.35 TB/s | ~400-800 cycles |
| 主机内存(DDR5) | 数百 GB | ~100 GB/s | 微秒级 |
| SSD | TB 级 | ~7 GB/s(NVMe) | 数十微秒 |

相邻层级带宽差 3~10 倍,跨 HBM 与内存差 30 倍。本目录的所有技术
都是这张表的应用题:

- FlashAttention:$S, P$ 从 HBM 提到 SRAM/寄存器
  ([10-flashattention.md](10-flashattention.md));
- Online softmax / LSE:状态从 $O(N^2)$ 压到 $O(N)$,为的能留在上层
  ([09-online-softmax.md](09-online-softmax.md));
- KV Cache:用 HBM 空间换重算;PagedAttention 管的就是这一层;
- CPU offload / 模型热切换:再往下两层(内存、SSD)换容量,代价是
  30 倍、1000 倍的带宽落差——所以 offload 只适合"否则完全跑不了"。

## 3. Tensor Core 与数据类型

Tensor Core 是 SM 内的矩阵乘专用单元($D = AB + C$ 一条指令完成一个小
矩阵块),各代支持的数据类型就是算力梯度的来源:

| 架构(代表) | 新增类型 | 相对 FP16 吞吐 |
|---|---|---|
| Volta(V100,2017) | FP16 | 1× |
| Ampere(A100/30 系) | BF16、TF32、INT8 | 1× |
| Hopper(H100) | FP8(E4M3/E5M2) | 2× |
| Blackwell(B200/50 系) | FP4 | 4× |

要点:

- **BF16 vs FP16**:同吞吐;BF16 范围大精度低(8 位指数 7 位尾数),
  训练爱用;推理两者皆可([08-kernel-math.md](08-kernel-math.md) 开头的讨论);
- **FP8 是推理甜点**:精度足够、算力翻倍、权重减半,Ada(40 系)起
  消费卡也有(见 [04-quantization.md](04-quantization.md) §7);
- **FP4 目前主要服务于 Blackwell 生态**(NVFP4 等),模型与 kernel
  在快速跟进;
- TF32/FP64 对 LLM 推理无关紧要——数据中心卡的 FP64 优势是给
  科学计算的,买卡时不用为它付钱。

## 4. SFU:被低估的小单元

超越函数(exp、rsqrt、sin/cos)不走 ALU/Tensor Core,走每 SM 仅 16 个
的 SFU(vs 128 个 FP32 单元)。当 Tensor Core 把 GEMM 越推越快,softmax
里的 $e^x$ 就可能成为短板——FA4 用 ALU 多项式逼近分流 10%~25% 的
exp。细节见 [08-kernel-math.md](08-kernel-math.md) §5 与
[10-flashattention.md](10-flashattention.md) §5。RoPE 的 `cos_sin_cache` 查表
也是绕开 SFU 的经典操作。

## 5. 消费卡 vs 数据中心卡

| 维度 | 数据中心(H100/B200) | 消费(40/50 系) |
|---|---|---|
| 显存 | 80~192 GB HBM | 16~32 GB GDDR |
| 带宽 | 3.35~8 TB/s | 0.7~1.8 TB/s |
| 多卡互联 | NVLink + NVSwitch | 无 NVLink,PCIe 且 P2P 受限 |
| FP8/FP4 | 满血 | 有(吞吐相同比例),生态适配稍慢 |
| 可靠性/形态 | ECC、被动散热、7×24 | 风冷、桌面 |
| 每 GB 显存价格 | 高 5~10 倍 | 便宜 |

结论:**单人/小团队推理,消费卡性价比极高**;差距主要在多卡扩展性
(NVLink)与显存容量,而不是单卡算力。

## 6. 代表 GPU 速查(推理视角)

| 卡 | 显存 | 带宽 | FP16 tensor | FP8 | FP4 | 定位 |
|---|---|---|---|---|---|---|
| RTX 4080 | 16 GB | 717 GB/s | ~195 T | ✓ | ✗ | 甜点老卡 |
| RTX 5070 Ti | 16 GB | 896 GB/s | ~176 T | ✓ | ✓ | 带宽更高,架构更新 |
| RTX 4090 | 24 GB | 1.0 TB/s | ~330 T | ✓ | ✗ | 上代单卡王 |
| RTX 5090 | 32 GB | 1.79 TB/s | ~419 T | ✓ | ✓ | 消费旗舰 |
| A100 | 80 GB | 2.0 TB/s | 312 T | ✗ | ✗ | 老将,无 FP8 |
| H100 | 80 GB | 3.35 TB/s | 989 T | ✓ | ✗ | 训练/生产主力 |

(稠密算力,不含 sparsity;数值为公版标称量级。)

## 7. 我们的两张卡点评

- **5070 Ti vs 4080**:带宽 +25%(896 vs 717)→ decode 直接快 25%;
  FP16 tensor 反而 4080 略高(195 vs 176)→ FP16 prefill 两者相当;
  5070 Ti 多了 FP4 与更新的编解码/调度特性,战未来;
- 16 GB 显存是共同的硬约束,所以本目录的资源估算与量化章节才是
  日常主角;
- 若未来升级:**显存容量 > 带宽 > FP16 算力**,按这个顺序看新卡
  (24~32 GB 档能让 32B Q4 落地,体验跃迁)。

## 8. 读规格书清单

为一台推理机/一张卡做判断时:

1. **显存容量**——决定"装得下什么"(门票,最重要);
2. **显存带宽**——决定 decode 速度上限;
3. **FP16/BF16 tensor TFLOPS**——决定 prefill 上限;
4. **FP8/FP4 支持**——决定能否吃新一代量化红利;
5. **多卡互联**——只有要多卡时才看(NVLink 有无)。

忽略:游戏特性(光追、DLSS)、FP64、显示输出规格。

## 9. 小结

- Roofline 一统江湖:decode 在带宽区,prefill 在算力区,batch/量化/
  投机解码都是在图上挪位置;
- 存储层级表是全部优化的地图:往上爬一层就是 3~10 倍;
- 消费卡单卡性价比极高,瓶颈在容量与互联;
- 挑卡口诀:容量 > 带宽 > FP16 算力 > FP8/FP4。
