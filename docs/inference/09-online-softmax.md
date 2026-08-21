# Online Softmax:把"算全局"改成"边走边修正"

前置:了解 safe softmax(减最大值防溢出)即可,见
[08-kernel-math.md](08-kernel-math.md) §2。本文是 [10-flashattention.md](10-flashattention.md)
的数学基础。

## 1. 要解决的问题

给定 logits 向量 $z = [z_1, \dots, z_N]$,safe softmax 分三步:

1. 遍历一遍,求最大值 $M = \max_j z_j$
2. 再遍历一遍,求指数和 $\ell = \sum_j e^{z_j - M}$
3. 第三次遍历,输出 $y_i = e^{z_i - M} / \ell$

问题不在计算量,而在**数据依赖**:第 2 步必须等第 1 步完全结束才能开始,
第 3 步又依赖前两步。这意味着 $z$ 必须完整地放在某个可读回的存储里
(GPU 上是 HBM 显存),被反复搬运 3 趟。在"算力远快于显存带宽"的今天,
这 3 趟搬运就是主要耗时——memory-bound。

能不能只遍历一趟,就同时拿到 $M$ 和 $\ell$?答案是 Online Softmax。

## 2. 单元素递推

维护两个运行状态:

- $m$:到目前为止见过的最大值(running max)
- $\ell$:到目前为止的指数和,**以当前 $m$ 为基准**(running sum)

初始 $m = -\infty$,$\ell = 0$。新元素 $z_i$ 到来时:

$$m_{\text{new}} = \max(m_{\text{old}},\, z_i)$$

$$\ell_{\text{new}} = \ell_{\text{old}} \cdot e^{m_{\text{old}} - m_{\text{new}}}
+ e^{z_i - m_{\text{new}}}$$

**推导**:$\ell_{\text{new}}$ 按定义是以 $m_{\text{new}}$ 为基准的、前 $i$ 个元素的
指数和。把旧元素和新元素拆开,再给旧元素的基准"换底":

$$\ell_{\text{new}} = \sum_{j \le i} e^{z_j - m_{\text{new}}}
= \sum_{j < i} e^{z_j - m_{\text{old}}} \cdot e^{m_{\text{old}} - m_{\text{new}}}
+ e^{z_i - m_{\text{new}}}
= \ell_{\text{old}} \cdot e^{m_{\text{old}} - m_{\text{new}}} + e^{z_i - m_{\text{new}}}$$

关键一步就是 $e^{z_j - m_{\text{new}}} = e^{z_j - m_{\text{old}}} \cdot
e^{m_{\text{old}} - m_{\text{new}}}$:基准从 $m_{\text{old}}$ 换到 $m_{\text{new}}$,
所有旧项只需统一乘一个缩放因子(rescaling factor)。这纯粹是指数律恒等变形,
没有任何近似。

直观理解:如果新元素没破纪录($m_{\text{new}} = m_{\text{old}}$),缩放因子是
$e^0 = 1$,直接累加;如果破了纪录,旧的累加和统一乘一个小于 1 的衰减系数,
等价于"从一开始就按新的最大值算"。

## 3. 手算一遍

$z = [2, 5, 1]$:

| 步 | $z_i$ | $m$ | $\ell$ 更新 | $\ell$ |
|---|---|---|---|---|
| 初始 | — | $-\infty$ | — | $0$ |
| 1 | $2$ | $2$ | $0 \cdot 0 + e^{2-2}$ | $1$ |
| 2 | $5$ | $5$ | $1 \cdot e^{2-5} + e^{5-5}$ | $e^{-3} + 1 \approx 1.0498$ |
| 3 | $1$ | $5$ | $1.0498 \cdot e^0 + e^{1-5}$ | $e^{-3} + 1 + e^{-4} \approx 1.0681$ |

对比全局算法:$M = 5$,$\ell = e^{2-5} + e^{5-5} + e^{1-5} = e^{-3} + 1 + e^{-4}$,
完全一致。最终输出时再遍历一遍 $y_i = e^{z_i - M} / \ell$:
$[0.0466,\ 0.9362,\ 0.0171]$,和为 1。

注意 online 版并没有消除"最后输出还要再看一遍数据"这件事——它的价值在于
**$M$ 和 $\ell$ 两个统计量可以流式地、一趟地算出来**,且中间状态只有 2 个数。
这个性质在分块和并行场景下才真正发光(§5、§6)。

## 4. 分块(向量)版本

实际 kernel 不会逐个元素处理,而是一次读入一个块(block/tile)。对块 $B$:

$$m_{\text{new}} = \max\big(m_{\text{old}},\ \max_{j \in B} z_j\big)$$

$$\ell_{\text{new}} = \ell_{\text{old}} \cdot e^{m_{\text{old}} - m_{\text{new}}}
+ \sum_{j \in B} e^{z_j - m_{\text{new}}}$$

形式与单元素版完全相同:块内先求局部最大值,再与历史合并。

## 5. 合并两个部分结果(merge)

online 更新是**可结合**的,这是它能并行化的根本原因。设两组不相交数据各自
流式算出 $(m_a, \ell_a)$ 和 $(m_b, \ell_b)$,合并:

$$m = \max(m_a, m_b), \qquad
\ell = \ell_a \cdot e^{m_a - m} + \ell_b \cdot e^{m_b - m}$$

若还各自带了未归一化的加权和 $\tilde O_t = \sum_{j \in t} e^{z_j - m_t} v_j$
(attention 里的 $PV$ 部分),则:

$$\tilde O = \tilde O_a \cdot e^{m_a - m} + \tilde O_b \cdot e^{m_b - m},
\qquad O = \tilde O / \ell$$

这个 merge 操作满足结合律,因此可以任意二叉树式归约——这正是
Flash-Decoding 的 Split-K(把长 KV 切成多段给多个 SM 并行,最后合并)和
Ring Attention(多卡各算一段再缝合)的数学基础,详见
[10-flashattention.md](10-flashattention.md) §7、§8。

## 6. LSE:一个标量顶两个

定义 Log-Sum-Exp:

$$\mathrm{LSE}(z) = \log \sum_j e^{z_j} = m + \log \ell$$

(第二个等号:提出 $e^m$,$\log \sum e^{z_j} = \log\big(e^m \sum e^{z_j - m}\big)
= m + \log \ell $,这就是工程上防溢出的 LSE 计算法。)

于是 softmax 可以写成单行:

$$\mathrm{softmax}(z_i) = \frac{e^{z_i - m}}{\ell}
= e^{z_i - m - \log \ell} = e^{z_i - \mathrm{LSE}}$$

**存 LSE 而不是 $(m, \ell)$ 的理由**:

- 存储与带宽减半:1 个 FP32 标量 vs 2 个。FlashAttention 前向出口处每个
  query 行都要写一次,累积可观。
- 恢复 $P$ 更便宜:$P = e^{s - \mathrm{LSE}}$ 是一次减法加一次 exp;拆存
  $(m, \ell)$ 则要减 $m$ 再除以 $\ell$,且 GPU 上除法指令更贵。
- merge 形式简洁:$\mathrm{LSE} = \log\big(e^{\mathrm{LSE}_a}
  + e^{\mathrm{LSE}_b}\big)$,对应输出 $O = \sum_t e^{\mathrm{LSE}_t - \mathrm{LSE}} O_t $。

**数值性质**(都很实用):

- $\ell \ge 1$(最大项贡献 $e^0 = 1$)$\Rightarrow \log \ell \ge 0
  \Rightarrow \mathrm{LSE} \ge m $;
- 于是 $z_i - \mathrm{LSE} \le z_i - m \le 0$:用 LSE 做指数**永不溢出**;
- $\ell \le N$,即 $\log \ell \le \log N$,LSE 自身量级可控;
- 下溢(特别小的 $z_i$ 使 $e^{z_i - m}$ 变 0)无害:softmax 里本来就是
  可忽略的概率。

## 7. 用在哪

| 场景 | online/LSE 的角色 |
|---|---|
| FlashAttention 前向 | 沿 KV 方向流式合并 tile,$(m, \ell, \tilde O)$ 长驻寄存器 |
| FlashAttention 反向 | 不存 $N \times N$ 的 $P$,只存 $O(N)$ 的 LSE,反传时 $P = e^{s - \mathrm{LSE}}$ 片上重算 |
| Flash-Decoding | decode 阶段 Split-K,多 SM 各算一段,按 §5 merge |
| Ring Attention / 上下文并行 | 多卡各持一段 KV,P2P 传递,本地增量修正 |
| 训练侧 log-softmax / CE loss | LSE 本身就是交叉熵里的 $\log \sum e^{z}$ 项 |

工程实现细节(GPU 存储层级、tile 调度、版本演进)见
[10-flashattention.md](10-flashattention.md)。
