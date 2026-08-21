# FlashAttention:IO-Aware 的精确注意力

前置:[09-online-softmax.md](09-online-softmax.md)(本文直接使用其
$(m, \ell)$ 流式更新与 LSE 合并公式)。

## 0. 术语速查

| 术语 | 含义 |
|---|---|
| SM | 流多处理器,GPU 的计算单元(H100 有 132 个) |
| CTA / thread block | 调度到**一个** SM 上执行的一组线程 |
| warp | CTA 内 32 线程的物理执行组;warp group = 4 个 warp(Hopper 引入) |
| HBM | 全局显存,容量大带宽相对低(H100:80 GB,约 3 TB/s) |
| L2 cache | 全 GPU 共享(H100:50 MB,约 12 TB/s) |
| SRAM / shared memory | 每个 SM 片上,CTA 内共享(每 SM 约 200+ KB,带宽比 HBM 高一个量级) |
| 寄存器(RF) | 每线程私有,最快最小 |
| Tensor Core | SM 内矩阵乘专用单元(MMA 指令),算力远超通用 FP32 单元 |
| SFU | 特殊函数单元(exp/rsqrt/sin/cos),每 SM 仅 16 个(vs 128 个 FP32 单元) |

## 1. 动机:注意力的瓶颈是搬运,不是计算

标准实现按数学公式逐步物化中间矩阵($N$ = 序列长度,$d$ = head dim):

$$S = QK^\top\;(N \times N) \;\to\; P = \mathrm{softmax}(S)\;(N \times N)
\;\to\; O = PV\;(N \times d)$$

$S$ 和 $P$ 两个 $N \times N$ 矩阵要在 HBM 里写入再读出若干趟:

- $N = 8K$、FP16 时,一个 $S$ 就是 $8192^2 \times 2\,\text{B} = 128$ MB;
- 每个头、每层都有一对 $S, P$;
- safe softmax 的 3-pass 结构(见 09-online-softmax.md §1)让这些矩阵被反复搬运。

矩阵乘由 Tensor Core 执行,极快;而 softmax、mask、dropout 这些"夹心"操作
全都 memory-bound。结果:**计算单元大量时间在等数据**——这就是"内存墙"。

关键点:FlashAttention **不改变数学结果**(除浮点求和顺序差异外与标准
attention 逐元素等价),它不是稀疏/低秩之类的近似方法;它改的是**数据在
存储层级间的流动方式**。

## 2. 核心结构:tiling + online softmax

思想一句话:把 Q、K、V 沿序列维切成小块(tile),让一个 tile 的计算全程在
SRAM/寄存器内完成;softmax 的全局依赖用 online softmax 的运行状态化解,
$S$、 $P$ 永远不落 HBM。

每个 Q tile 沿 KV 方向流式扫描,寄存器里长驻未归一化三元组
$(m, \ell, \tilde O)$:

```
m = -inf, ℓ = 0, Õ = 0
for kv_block in K/V 的所有 tile:        # 内层循环
    S = Q_tile @ kv_block.Kᵀ / √d       # tile 内点积(含 √dk 缩放)
    m_new  = max(m, rowmax(S))
    α      = exp(m - m_new)             # 对历史的修正因子
    P̃      = exp(S - m_new)             # 未归一化的注意力权重(在寄存器里)
    ℓ      = ℓ * α + rowsum(P̃)
    Õ      = Õ * α + P̃ @ kv_block.V
    m      = m_new
O   = Õ / ℓ                              # 循环结束后才归一化,写 HBM 一次
LSE = m + log ℓ                          # 一并写出,留给反向/合并用
```

其中 $\tilde O$ 的更新利用的正是 online softmax 的可结合 merge
(见 09-online-softmax.md §5):旧累加按 $e^{m_{\text{old}}-m_{\text{new}}}$ 缩放后,
把新块的未归一化 $PV$ 直接加进去。 $P$ 从未被显式物化。

**Causal Mask 的块稀疏处理**(训练/prefill 阶段):按 tile 行列索引分类——

- 对角线以上的 tile(严格的 $j > i$):整块跳过,不算也不从 HBM 搬;
- 对角线以下的 tile:正常算,零掩码开销;
- 压对角线的边界 tile:在 SRAM 内逐元素判定,上三角写 $-\infty$
  (FP16 下用最小安全值 $-65504$)。

掩码矩阵本身零访存,且两次 GEMM 的计算/访存量各省约一半。
decode 阶段 query 长度只有 1,所有历史 KV 都该可见,**无需任何掩码**。

## 3. 复杂度账

设 SRAM 容量为 $M$(按元素计):

| 指标 | 标准实现 | FlashAttention |
|---|---|---|
| 显存占用 | $O(N^2)$(物化 $S, P$) | $O(Nd + N)$(只存 $O$ 和 LSE) |
| HBM 访存量 | $\Theta(Nd + N^2)$,常数大(多趟读写 $S,P$) | $\Theta(N^2 d^2 / M)$,典型小 3~10 倍 |
| FLOPs | 相同 | 相同(+少量 rescaling 开销) |

- 显存从 $O(N^2)$ 降到 $O(N)$,这是长上下文可行的前提($N=128K$ 时
  $N^2$ 个 FP16 元素是 32 GB,根本放不下);
- 访存量同阶但常数显著更小,实测端到端训练提速 2~4 倍(FA 论文数据);
- 代价:反向传播需要**重算**(recomputation)$S$ 和 $P$,FLOPs 略增,
  但省下的 HBM 流量远大于多算的(见 §6)。

## 4. FA1 → FA2:把循环顺序反过来

**FA1(2022)**:外层循环 KV tile、内层循环 Q tile;grid 只有
$(\text{batch}, \text{head})$ 维,即 $B \times H$ 个 CTA。两个问题:

- **并行度不足**:长上下文、小 batch 时 $B \times H$ 远小于 SM 数,
  大量 SM 闲置;
- **Q 状态的 HBM round-trip**:外层每扫完一个 KV tile,所有 Q tile 的
  部分状态 $(m, \ell, \tilde O)$ 都要写回 HBM,下一轮再读回来。

**FA2(2023)**:外层循环 Q tile、内层循环 KV tile。每个 CTA 认领一个
Q tile,像流水线一样吞入历史 KV:

$$\text{grid} = (\lceil N_q / B_M \rceil,\ \text{batch},\ \text{head})$$

- 并行维度多了序列维,SM 能跑满;
- $(m, \ell, \tilde O)$ 全程驻留寄存器——Tensor Core 的 MMA 指令
  ($D = AB + C$)输出本来就落在 warp 的寄存器(C-fragment)里,
  $\tilde O$ 直接在累加器上原地累加;只在结束时做一次 $\tilde O / \ell$
  并写 HBM 一次;
- warp 间按 Q 行切分、各自独立,不需要经 SRAM 拼结果,消除了 barrier
  同步。

**FA2 的软肋:L2 依赖"齐步走"**。每个 Q-tile 的 CTA 都独立去 HBM 读全量
KV,理论流量放大 $\lceil N_q / B_M \rceil$ 倍;实际靠 L2 兜底——同一调度
波次的 CTA 几乎同时访问同一 KV tile,第一个 CTA 拉进 L2 后其余命中
(L2 带宽约为 HBM 的 4 倍)。但这是**期望值层面的侥幸**:

- 硬件调度器一旦出现波次断层、SM 占用不均,命中率下滑;
- 长上下文 KV 是 GB 级,远超 L2 容量(H100 50 MB),命中率从约 99%
  退化到 80% 以下,性能曲线可见劣化(渐变,不是悬崖)。

## 5. FA3 → FA4:软件定义调度 + 硬件新特性

针对 FA2 的软肋和新一代硬件,演进方向是**把访存从"碰运气"变成"合约"**:

- **Persistent kernel + tile scheduler(FA3/FA4)**:CTA 常驻 SM,自己从
  全局 work queue 领任务,绕过硬件 CTA 调度器(消除波次断层);软件层
  主动编排访问顺序(相邻 CTA 访问相邻 KV tile、按 L2 容量做 swizzle),
  让 L2 命中从偶然变成设计。
- **TMA multicast(Hopper 起)**:一个 cluster 内多个 CTA 共享一次 HBM
  读取,硬件保证只搬一趟——KV 重读的抑制从"L2 软期望"升格为硬件合约。
- **Warp 专门化**:FA3 按 warp group 分工(producer 发 TMA 异步搬数 /
  consumer 跑 WGMMA + softmax,GEMM 与 softmax 流水线交叠);FA4 细化到
  warp 级(MMA、Softmax×2 组、Correction、Epilogue、Load 各司其职),
  多级流水线把 softmax 藏进 GEMM 的阴影里。
- **O 的落脚点**:HBM(FA1)→ 寄存器(FA2/FA3)→ TMEM(FA4,Blackwell
  的张量内存,$P$ 也可直接放 TMEM 作为 PV-GEMM 输入,免去寄存器/SRAM 中转)。
- **FA4 的 SFU 瓶颈**:Tensor Core 代际翻倍而 SFU 未同比例扩展,算 $e^x$
  的时间逼近 $QK^\top$ GEMM。FA4 让 75%~90% 的 exp 走原生 SFU 指令
  (MUFU.EX2),其余 10%~25% 用通用 ALU 多项式逼近分流(全软算会寄存器
  溢出反噬)。背景见 [08-kernel-math.md](08-kernel-math.md) §5(SFU 一节)。

## 6. 反向传播:用 LSE 重算 P

反传需要前向的 $P$(例如 $dV = P^\top dO$)。存下 $N \times N$ 的 $P$ 显然
违背初衷,FA 的做法:

- 前向只额外写 $O(N)$ 大小的 LSE 数组(每行一个标量);
- 反传时在片上重算局部 $s = q \cdot k / \sqrt d$,然后
  $P = e^{s - \mathrm{LSE}}$——利用 LSE 形式**化除法为减法**,一步指数即得
  (数学推导见 09-online-softmax.md §6);
- 多付了重算的 FLOPs,但避免了 $P$ 的 $O(N^2)$ 写读,整体仍然快得多。

这是"计算换访存"的典型案例:重计算提高算术强度,恰好顺应硬件趋势。

## 7. 推理侧变体:Flash-Decoding(Split-K)

decode 阶段每个请求只有 1 个 query,若一个 CTA 独自从头扫几十万 token 的
KV Cache,SM 几乎全部闲置。Flash-Decoding 把 KV 切成若干段分给多个 CTA
并行:

1. 各段独立流式算出局部 $(\tilde O_t, \mathrm{LSE}_t)$,写回 HBM;
2. 一个小型归约 kernel 按 online softmax 的 merge 公式合并:

$$L = \log \sum_t e^{\mathrm{LSE}_t}, \qquad
O = \sum_t e^{\mathrm{LSE}_t - L}\, \tilde O_t$$

每段只需写回 1 个 LSE 标量而不是 $(m, \ell)$ 两个,归约通信量减半。

## 8. 在推理系统里的位置

- **与 KV Cache**:prefill 阶段的 attention 是标准 FA 形态($N_q$ 大);
  decode 阶段是 Flash-Decoding 形态($N_q = 1$,本质是批量 GEMV)。
- **与 PagedAttention(vLLM)**:分页 KV Cache 解决的是**显存管理**
  (碎片化、按需分配),FA/Flash-Decoding 解决的是**计算与访存**;
  vLLM 的 attention kernel 顺着 block table 的页表指针 gather 离散的
  KV 块进 SRAM 计算,两者是正交叠加的关系。
- **多卡长上下文:Ring Attention**:上下文并行把序列切段分卡,环形拓扑
  P2P 轮转 K/V 切片,Q 不动;各卡凭 online softmax 增量修正本地
  $(m, \ell, \tilde O)$,结果与全局 attention 数值一致,且计算与 KV
  传输 overlap,避开昂贵的 All-Reduce。

## 9. 一句话回顾

FlashAttention 的主线只有一条:**承认"数据搬运比计算贵"的现实,用
online softmax 把 softmax 的全局依赖改写成流式可合并的局部状态,从而让
$S, P$ 两个大矩阵永远留在片上**。后续各版本(FA2 的循环反转、FA3/FA4 的
persistent kernel 与 warp 专门化)都是在同一张图上,把调度与流水线做得
更贴合具体硬件。

相关:[09-online-softmax.md](09-online-softmax.md)、
[08-kernel-math.md](08-kernel-math.md)(RMSNorm/Sampling 等其他算子的同类优化)、
[../attention-score-distribution.md](../attention-score-distribution.md)
(注意力分数的分布视角)。
