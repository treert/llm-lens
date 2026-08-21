# 核心算子的数学与 Kernel 优化

> 整理自星航管线《AI Infra入门:大模型中的数学与Infra优化》(KM 平台),
> 并与本仓库相关笔记交叉引用。

核心观点:推理侧的 Infra 优化,本质是用**数学等价变换**(或经验证的精度妥协、
架构简化),换取更好的访存局部性、并行度与 kernel 融合空间。本文按算子逐一拆解。
其中 Online Softmax 与 FlashAttention 两条线已独立成篇,本文只留摘要:

- [09-online-softmax.md](09-online-softmax.md):递推推导、手算例子、merge 结合律、LSE
- [10-flashattention.md](10-flashattention.md):存储层级、tiling kernel、FA1→FA4 演进

## 1. RMSNorm:砍掉均值的归一化

数学上为什么成立(高维长度集中、逐样本 RMS 几乎恒定)见
[../rmsnorm.md](../rmsnorm.md)。这里只列 Infra 视角的收益:

- **统计量减半**:LayerNorm 需要 $\mu$ 和 $\sigma^2$ 两个统计量;RMSNorm 只需
  $\sum x_i^2$,一次单向规约即可,寄存器/规约状态/ALU 指令都更省
  (消除了逐元素减均值)。
- **HBM 趟数其实都能做到 1-pass**:LayerNorm 可用
  $\mathrm{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$ 在一次遍历中同时累计
  $\sum x$ 与 $\sum x^2$。但在 FP16/BF16 下两项接近时会发生灾难性抵消
  (catastrophic cancellation),实际 kernel 用 FP32 累加或 Welford 算法保底。
  所以 RMSNorm 的真实优势是**寄存器压力与指令数**,而不是 HBM 趟数。
- **配套架构选择**:现代 LLM 的 Linear 几乎全部去掉 bias(q/k/v/o_proj、
  gate/up/down_proj)。解释有三:PaLM 论文称大模型训练更稳定;bias 的位移作用
  会被下一个 Norm 吸收(SwiGLU 门控也提供类似自由度);少一次 add 与 bias load。
- eps 取 `config.rms_norm_eps`(通常 1e-5 或 1e-6),仅防除零。

Post-Norm → Pre-Norm 的演进(梯度无损回传 vs 表征坍塌)与本文主题正交,从略;
要点:Pre-Norm 深层分支贡献相对主干越来越小,是"砍掉最后几层掉点不明显"
这类现象的根源。

## 2. Softmax 的数值安全外衣

$$\mathrm{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

| 手段 | 作用 | 生效位置 | 阶段 |
|---|---|---|---|
| $-\max(z)$ | 防 $e^z$ 上溢(safe) | 所有 softmax | 训练+推理 |
| $\div\sqrt{d_k}$ | 防点积方差膨胀 → softmax 退化 hard → 梯度消失 | 仅 attention | 训练+推理 |
| 上三角 $+(-\infty)$ | 因果掩码 | 仅 attention | 训练 + prefill;decode 天然满足 |
| $\div T$ | 控制采样多样性 | 仅 sampling | 仅推理 |

- **减 max 是恒等变换**:$e^{z_i - M} / \sum_j e^{z_j - M}$ 与原始式数学相等,
  且 $z_i - M \le 0$ 保证最大项为 $e^0 = 1$,上溢消失(下溢到 0 通常无害)。
- **方差膨胀**:在 $q_i, k_i$ 独立、零均值、单位方差的理想假设下,
  $\mathrm{Var}(q \cdot k) = d_k$;除以 $\sqrt{d_k}$ 把 logits 方差拉回 1。
  假设在真实网络中不严格成立,但初始化第一步稳住量级后,训练会自行适应
  (本仓库的实测视角见 [../attention-score-distribution.md](../attention-score-distribution.md))。
- **Causal Mask 的工程进化**:早期在 HBM 里生成 $L \times L$ 掩码矩阵再相加,
  长序列下是访存灾难;现在由 FlashAttention 在 tile 调度层分类处理
  (跳过 / 免掩码 / 边界块片上判定),掩码访存归零,计算与访存近减半,
  详见 [10-flashattention.md](10-flashattention.md) §2。decode 阶段 query 长度为 1,
  所有历史 KV 都该可见,**无需显式掩码**(例外:speculative decoding 的
  验证步一次并行算多个候选位置,需树形/多 token mask,
  见 [05-throughput.md](05-throughput.md) §6)。
- **Temperature**:$\mathrm{softmax}(z / T)$,$T<1$ 分布变尖锐(趋于确定性),
  $T>1$ 变平坦(多样性高)。

## 3. Online Softmax 与 FlashAttention(摘要)

朴素 safe softmax 是 3-pass(求 max → 求分母 → 归一化),pass 之间有数据依赖,
中间矩阵必须驻留 HBM。Online Softmax 维护运行状态 $(m, \ell)$,新块到来时
用缩放因子 $e^{m_{\text{old}} - m_{\text{new}}}$ 一次性修正历史,一趟遍历即得
全局统计量;且该操作满足结合律,可以任意并行合并
(推导与手算例子见 [09-online-softmax.md](09-online-softmax.md))。

FlashAttention 把这套"边走边修正"推广到整条 attention 流水线:tiling +
寄存器长驻未归一化三元组 $(m, \ell, \tilde O)$,$S$、 $P$ 永远不落 HBM,
显存从 $O(N^2)$ 降到 $O(N)$,且**数学上仍是精确注意力**
(机制与 FA1→FA4 演进见 [10-flashattention.md](10-flashattention.md))。

LSE($\mathrm{LSE} = m + \log \ell$)把两个 FP32 状态压成一个标量,
$\mathrm{softmax}(z_i) = e^{z_i - \mathrm{LSE}}$ 化除法为减法,支撑
反向重算 $P$、Flash-Decoding 的 Split-K 合并、Ring Attention 多卡缝合
(性质与合并公式见 [09-online-softmax.md](09-online-softmax.md) §6)。

## 4. Sampling:Gumbel-Max 把串行变并行

- **Multinomial(轮盘赌)**:抽 $u \sim U(0,1)$ 后扫概率前缀和,对 128K+ 词表
  做 prefix-sum 串行依赖重、同步开销大,GPU 不友好。
- **Gumbel-Max Trick**:给对数概率加标准 Gumbel 噪声再取最大值,与按 $p$
  采样数学等价:
  $$\arg\max_i (\ln p_i + g_i), \qquad g_i = -\ln(-\ln U_i)$$
- **vLLM 的实现变体**:采 $q_i \sim \mathrm{Exp}(1)$,直接算
  `probs.div_(q).argmax()`。由 $\ln$ 单调性 $\arg\max(p/q) = \arg\max(\ln p - \ln q)$,
  而 $q = -\ln U$ 时 $-\ln q$ 正是 Gumbel 噪声。
- **Infra 价值**:全程 element-wise + 一次 argmax 规约,无任何数据依赖;
  贪心采样 = 不除噪声,两种采样统一执行流,消灭分支发散;词表沿 TP 切分时,
  采样退化为满足结合律的 All-Reduce(MAX with index),通信量从 $O(V)$ 降到
  $O(\text{world\_size})$。

## 5. SFU:超越函数的物理瓶颈

$1/\sqrt{x}$(rsqrt)与 $e^x$(exp2)在 GPU 上都脱离 ALU 乘加流水线,交给每个
SM 内专门的 **SFU(特殊函数单元)** 以查表 + 多项式插值逼近。代价:SFU 数量远
少于通用单元(Hopper/Blackwell 每 SM 128 个 FP32 单元 vs 16 个 SFU)。两个推论:

- **RoPE 查表**:vLLM 在引擎初始化时预计算 `cos_sin_cache`
  ($[\text{max\_position}, \text{rotary\_dim}]$,几 MB 到几十 MB),运行时按 position
  索引,用小表换掉每次 forward 数百万次 sin/cos——经典空间换算力。
- **FA4 的软硬协同分流**:Tensor Core 代际翻倍而 SFU 未同比例扩展,$e^x$ 的耗时
  逼近 $QK^\top$ GEMM,成为流水线瓶颈。FA4 保留 75%~90% 走原生 MUFU.EX2,
  其余 10%~25% 用 CUDA Core 做多项式逼近(全软算会寄存器溢出反噬性能),
  让 SFU 与 ALU 在超越函数上并行运转(见 [10-flashattention.md](10-flashattention.md) §5)。

## 小结

| 算子 | 数学手段 | 换来的东西 |
|---|---|---|
| RMSNorm | 砍均值(实验验证平移贡献小) | 寄存器/规约状态/ALU 指令减半 |
| Safe Softmax | 减 max(恒等) | 防上溢,任意 logits 可算 |
| $\div\sqrt{d_k}$ | 方差性质 $\mathrm{Var}(X/c) = \mathrm{Var}(X)/c^2$ | 梯度舒适区,训练稳定 |
| Causal Mask 块稀疏 | $e^{-\infty} = 0$ | 掩码零访存,计算/访存近减半 |
| Online Softmax / FA | 运行状态 + rescaling(恒等) | 3-pass → 1-pass,$S,P$ 不落 HBM |
| LSE | $\log$ 化除为减(恒等) | 反传重算、Split-K、Ring 合并 |
| Gumbel-Max | 加噪取 max ≡ 按概率采样 | 串行前缀和 → 全并行,TP 友好 |
| RoPE 查表 / FA4 分流 | 预计算 / 多项式逼近(精度妥协) | 绕开 SFU 吞吐瓶颈 |
