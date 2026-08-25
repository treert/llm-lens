# LoRA:用低秩旁路撬动大模型

前置:[02-training-memory.md](02-training-memory.md)(全量微调的显存墙)、
[03-full-finetuning.md](03-full-finetuning.md)(SFT 机制)。
目标:理解 LoRA 的数学(为什么是 $W+BA$、为什么是低秩)、参数量账、
关键超参数(r、 $\alpha$、target modules)的选择依据,以及 merge 与变体。

## 1. 低秩假设:微调的更新"住"在低维子空间

LoRA(Hu et al., 2021)的出发点是一个经验观察(其前作 Aghajanyan
et al., 2021 的"内在维度"实验):**过参数化大模型在下游任务上微调时,
权重的有效更新量 $\Delta W$ 秩很低**——尽管 $W$ 是 $d \times d$ 的大
矩阵,真正需要的变化方向可能只有几到几十维。

直觉解释:预训练已经把通用表示学好了,适配新任务只需在少数方向上
"拧一拧",而不是重写整个矩阵。这也与 01 篇的表面对齐假说呼应:
微调改的是表达方式,不是知识体系,改动天然小。

## 2. 数学:冻结 $W$,训练 $BA$

对任意权重矩阵 $W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$,
LoRA 把它冻结,增量用两个 skinny 矩阵的低秩乘积表示:

$$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} BA,
\qquad B \in \mathbb{R}^{d_{\text{out}} \times r},\quad
A \in \mathbb{R}^{r \times d_{\text{in}}},\quad r \ll \min(d_{\text{in}}, d_{\text{out}})$$

前向变成(同一输入走两条路,相加):

$$h = W_0 x + \frac{\alpha}{r} B A x$$

反向时 $W_0$ 不存梯度,梯度只流向 $A$、 $B$——**可训练参数从
$d_{\text{in}} d_{\text{out}}$ 降到 $r(d_{\text{in}} + d_{\text{out}})$**。

参数量账($d = 4096$ 的方阵,r=16):

$$d^2 = 16.8\text{M} \quad\rightarrow\quad r \cdot 2d = 131\text{K}\quad(\text{0.78\%})$$

整个 Llama-3-8B 所有线性层都插 r=16 的 LoRA,可训练参数约 1 亿
(~1.2%),按 02 篇的 16 字节/参数法则,梯度+优化器+主权重总共
~1.6 GB——对比全量微调的 112 GB,这就是 LoRA 的全部意义。

**初始化**:$A$ 用高斯随机初始化,$B$ 初始化为**零**——于是训练起点
$\Delta W = 0$,模型行为与原模型严格一致,微调从零扰动开始生长。
(反过来的初始化 $A{=}0, B{=}\text{随机}$ 数学上等价,约定俗成用前者。)

## 3. $\alpha/r$ 缩放:让超参数与秩解耦

直接训 $BA$ 时,最优学习率与 r 强相关(r 变了,$BAx$ 的方差变)。
LoRA 引入固定缩放 $\alpha/r$,使得**换 r 时不用重调学习率**。
常见约定:$\alpha = 2r$(如 r=16, α=32)。

变体 rsLoRA 指出 $\alpha/r$ 在大 r 时梯度不稳定,主张 $\alpha/\sqrt{r}$;
日常训练用标准约定即可。

## 4. 三个关键选择

**秩 r**:常见 8~64。

- 任务越"表面"(格式、风格),r 越小够用(r=8 常已饱和);
- 需要学较多新东西(新领域术语、代码新框架),r=32~64;
- r 继续增大收益递减,且参数/显存线性涨、过拟合风险涨;
- LoRA 论文的消融:即使 r=1~4 也能学到不少东西——再次印证低秩假设。

**target modules(插在哪)**:论文最初只插 attention 的 $W_q, W_v$;
后来的共识是**所有线性层**(q/k/v/o + gate/up/down,即"all-linear")
效果明显更好,代价只是参数量从 ~0.3% 涨到 ~1%,显存仍很小。
embedding 和 lm_head 一般不插(词表大,LoRA 收益比低;需要学新 token
时例外,那时要全量训这两个)。

**dropout / bias**:LoRA 分支 dropout 常设 0(参数太少,不太过拟合);
bias 默认不训。

## 5. merge:推理时零开销

训练完可以把旁路并回主干:

$$W' = W_0 + \frac{\alpha}{r} BA$$

得到一个**结构与基座完全相同的普通模型**,推理延迟、显存、部署流程
与基座一模一样(08 篇走的就是这条路)。 $BA$ 是 $r$ 秩矩阵,
乘出来再相加即可,没有精度陷阱(在 bf16/fp32 下做一次矩阵乘)。

不 merge 的玩法:基座常驻,adapter 按需加载——vLLM 支持多 LoRA
热切换,适合"一个基座 + N 个任务/租户"的 serving 场景。代价是
每次前向多走 LoRA 分支(少量计算)与引擎复杂度。

**多 LoRA 的组合性**:同一基座的多个 adapter 各自独立,可以换着挂;
但直接合并多个 adapter(权重相加)效果无保证,需要专门方法
(如 LoRAhub),属研究前沿。

## 6. LoRA 的局限

- **容量上限**:低秩是假设也是天花板。学全新知识(新语言、大段私有
  语料)时 LoRA 明显弱于全量微调;
- **merge 精度**:量化基座(如 QLoRA 的 4 bit 主干)上训出的 LoRA
  merge 回 bf16 主干时,理论上存在"训练时看到的 $W_0$ 与 merge 时的
  $W_0$ 不一致"的缝隙,实测影响小,但值得知道(05 篇 §3);
- **不是免遗忘金牌**:主干不动所以通用能力保住了,但 LoRA 层本身
  照样会在多任务间互相干扰。

## 7. 主要变体一句话地图

| 变体 | 改动 | 何时考虑 |
|---|---|---|
| LoRA+ | $A$、 $B$ 用不同学习率($B$ 更大) | 大模型上想再榨点效果 |
| PiSSA | 用 $W_0$ 的 SVD 主成分初始化 $A$、 $B$ | 收敛更快、效果略好 |
| DoRA | 把权重分解为方向×幅度,LoRA 只调方向 | 效果接近全量微调,开销略增 |
| VeRA / AdaLoRA 等 | 更极端的参数压缩 / 按重要性分配秩 | 研究向,工程慎用 |

## 8. 小结

$$\text{全量微调:学 } \Delta W \in \mathbb{R}^{d \times d}
\qquad\xrightarrow{\text{低秩假设}}\qquad
\text{LoRA:学 } BA,\ B \in \mathbb{R}^{d \times r},\ A \in \mathbb{R}^{r \times d}$$

- LoRA = 冻结主干 + 低秩旁路 + 零初始化,可训练参数降两个数量级;
- 默认配置:r=16~32、α=2r、target=all-linear、dropout=0;
- merge 后推理零开销,是它击败 Adapter 等方案成为事实标准的关键;
- 下一篇:主干那 15 GB(bf16)在 16 GB 卡上依然放不下——QLoRA
  把冻结主干压到 4 bit,补齐最后一块拼图。
