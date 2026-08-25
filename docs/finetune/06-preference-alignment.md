# 偏好对齐:从 RLHF 到 DPO 再到 GRPO

前置:[03-full-finetuning.md](03-full-finetuning.md)(SFT)。
目标:理解为什么 SFT 之后还需要偏好对齐、RLHF 三步各自在做什么、
PPO 为什么又贵又脆、**DPO 如何用一个闭式解把 RL 问题变成二分类**,
以及 GRPO/RLVR 为什么在推理模型时代复兴了 RL。

## 1. SFT 的两个天花板

SFT 是"模仿标准答案",有两类信号它天然学不到:

1. **相对比较**:两个回答都对,但一个更好(更清晰、更安全)。SFT 的
   数据里没有"差回答",模型无从得知梯度应该往哪偏;
2. **负向约束**:"不该说什么"(有害内容、幻觉、车轱辘话)。模仿学习
   只能向正样本靠近,不能显式远离负样本。

偏好对齐的数据形态从 (指令, 答案) 变为 (指令, 好答案 $y_w$, 差答案
$y_l$),目标从"拟合答案"变为"把好坏答案的概率拉开"。

## 2. RLHF 三步曲

经典管线(InstructGPT, 2022):

**第 1 步:SFT**。得 $\pi_{\text{SFT}}$,作为后续的策略起点与参考模型。

**第 2 步:训奖励模型(reward model, RM)**。收集人类偏好:标注员对
同一 prompt 的多个回答排序,构成 $(x, y_w, y_l)$ 数据集。RM 是一个
(通常由 SFT 模型改造的)打分器 $r_\phi(x, y) \in \mathbb{R}$,用
**Bradley-Terry 模型**把排序转成概率:

$$P(y_w \succ y_l \mid x) = \frac{e^{r(x,y_w)}}{e^{r(x,y_w)} + e^{r(x,y_l)}}
= \sigma\big( r(x, y_w) - r(x, y_l) \big)$$

训练目标就是 pairwise 交叉熵:

$$\mathcal{L}_{\text{RM}} = -\mathbb{E}_{(x, y_w, y_l) \sim D}
\log \sigma\big( r_\phi(x, y_w) - r_\phi(x, y_l) \big)$$

注意:**可学习的只是分差**,分数的绝对零点无意义——这个冗余自由度
正是后面 DPO 推导里 $Z(x)$ 能被消掉的原因。

**第 3 步:用 RL 优化策略**。以 RM 的打分当奖励,同时用 KL 散度拴住
策略不要跑偏太远(防 reward hacking 与语言崩坏):

$$\max_{\pi_\theta}\ \mathbb{E}_{x \sim D}\,
\mathbb{E}_{y \sim \pi_\theta(\cdot|x)}\big[ r_\phi(x, y) \big]
\;-\; \beta\, \mathrm{KL}\big( \pi_\theta(\cdot|x) \,\|\, \pi_{\text{ref}}(\cdot|x) \big)$$

其中 $\pi_{\text{ref}} = \pi_{\text{SFT}}$ 冻结。这个式子是本篇的题眼:
它是所有偏好对齐算法(RLHF-PPO、DPO、GRPO)共同的优化目标,区别只在
**怎么解它**。

## 3. PPO 的代价:四个模型一台戏

上面的期望没法直接对 $\theta$ 求导(采样操作不可导),标准解法是用
PPO 这类策略梯度算法。工程上要同时维护:

| 模型 | 角色 | 是否需要梯度 |
|---|---|---|
| policy $\pi_\theta$ | 生成回答,被优化 | 是 |
| reference $\pi_{\text{ref}}$ | 算 KL,防跑偏 | 否(但每次前向) |
| reward model $r_\phi$ | 给回答打分 | 否(但每次前向) |
| critic $V_\psi$ | 估计价值函数,降方差 | **是,且与 policy 同规模** |

四个模型同时驻留显存,其中两个要做完整训练——这就是 RLHF "又贵又
脆"的来源:显存翻倍不说,PPO 的超参数(clip 范围、critic 学习率、
KL 系数 $\beta$、advantage 归一化)任何一组没调好,轻则训不动,重则
模型学会刷分(reward hacking:输出 RM 偏爱但人类讨厌的文本)。

## 4. DPO:把 RL 问题闭式解成二分类

DPO(Rafailov et al., 2023)的关键观察:§2 末尾的 KL 约束目标
**有闭式最优解**,根本不需要在线 RL。

### 4.1 第一步:解出最优策略

固定 prompt $x$,把目标写成(略去条件 $x$ 的记号):

$$\max_{\pi}\ \mathbb{E}_{y \sim \pi}\Big[ r(y) - \beta \log \frac{\pi(y)}{\pi_{\text{ref}}(y)} \Big]$$

定义配分函数 $Z(x) = \sum_y \pi_{\text{ref}}(y|x)\, e^{r(x,y)/\beta}$
和分布 $\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x)\, e^{r(x,y)/\beta}$。
把目标改写:

$$\mathbb{E}_{y\sim\pi}\Big[ \beta \log \frac{\pi_{\text{ref}}(y)\, e^{r(y)/\beta}}{\pi(y)} \Big]
= \beta\, \log Z(x) - \beta\, \mathrm{KL}(\pi \| \pi^*)$$

$\log Z(x)$ 与 $\pi$ 无关,$\mathrm{KL} \ge 0$ 当且仅当 $\pi = \pi^*$ 时
取零——所以最优策略就是:

$$\boxed{\ \pi^*(y|x) = \frac{1}{Z(x)}\, \pi_{\text{ref}}(y|x)\,
\exp\!\big( r(x, y)/\beta \big)\ }$$

读法:最优策略 = 参考策略按奖励指数加权。 $\beta$ 越小,越贪婪地倒向
高奖励;$\beta \to \infty$ 时退化为 $\pi_{\text{ref}}$。

### 4.2 第二步:反解奖励

上式两边取对数整理,奖励可用策略表示:

$$r(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)$$

### 4.3 第三步:代入 Bradley-Terry,消掉 $Z(x)$

把 $r$ 的这个参数化代入 RM 的 pairwise 损失(§2),$y_w$ 与 $y_l$ 的
$Z(x)$ 项**精确抵消**:

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(x, y_w, y_l) \sim D}
\log \sigma\Big(
\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)}
- \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}
\Big)$$

RM 没了,PPO 没了,采样没了——剩下的是对偏好数据做**带隐式奖励的
二分类**。定义隐式奖励:

$$\hat{r}_\theta(x, y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$$

DPO 就是在直接拉大 $\hat{r}(x, y_w)$ 与 $\hat{r}(x, y_l)$ 的差。

### 4.4 梯度直觉

$$\nabla_\theta \mathcal{L}_{\text{DPO}} = -\beta\, \mathbb{E}\Big[
\underbrace{\sigma\big(\hat{r}_\theta(x, y_l) - \hat{r}_\theta(x, y_w)\big)}_{\text{权重:分错/不自信时大}}
\Big( \nabla_\theta \log \pi_\theta(y_w|x) - \nabla_\theta \log \pi_\theta(y_l|x) \Big)\Big]$$

两项各是什么意思:提高好答案的似然、压低坏答案的似然;前面的
$\sigma$ 项是自适应权重——已经分对且自信的样本梯度趋零,分错的
样本梯度最大。这与 SFT 的"只拉高正样本"形成对照:DPO 显式地推远
负样本。

工程上 $\log \pi(y|x) = \sum_t \log \pi_\theta(y_t \mid x, y_{<t})$,
即对整条回答的 logprob 求和(序列级而非 token 级)。

### 4.5 为什么 DPO 便宜且稳

- 只需两个模型:训练中的 $\pi_\theta$ + 冻结的 $\pi_{\text{ref}}$
  (而且 ref 的 logprob 可以离线预计算);
- 无采样、无 critic、超参数基本只有 $\beta$(常用 0.1~0.5)和学习率
  (~5e-7 全量 / 1e-6~5e-6 LoRA);
- **与 QLoRA 完美叠加**:ref 就是"冻结主干不加 LoRA 的前向",
  policy 就是"同一主干加 LoRA 的前向"——一份 4 bit 主干服务两个角色,
  16 GB 卡上 7B 级 DPO 完全可行。

DPO 的标准前置是先做 SFT($\pi_{\text{ref}} = \pi_{\text{SFT}}$):
直接在 base model 上 DPO,ref 分布与偏好数据差太远,效果差。

## 5. DPO 家族变体一句话地图

| 变体 | 动机 | 改动 |
|---|---|---|
| IPO | DPO 在确定性偏好上会过拟合(σ 饱和) | 换成平方损失,正则化隐式奖励差 |
| KTO | 偏好对难收集,单条"好/坏"标签易得 | 用 Kahneman-Tversky 前景理论改损失,无需成对数据 |
| ORPO | 省掉 SFT→DPO 两阶段 | SFT loss + 偏好项合并为一个目标 |
| SimPO | 去掉 ref 模型更省 | 用长度归一化 logprob 当隐式奖励,无 ref |

实践默认仍是 DPO:理论最干净、复现最稳、框架支持最好。

## 6. GRPO 与 RLVR:RL 的复兴

DPO 是离线方法:数据固定,无法探索。当任务有**可程序化验证的正确
答案**(数学、代码)时,在线 RL 重新变得划算——这就是 DeepSeekMath /
DeepSeek-R1 带火的路线:

- **RLVR(可验证奖励 RL)**:奖励不来自 RM,来自规则——数学答案等于
  标准答案给 1,否则 0;代码通过测试给 1。**彻底绕开 reward hacking**;
- **GRPO(组相对策略优化)**:PPO 的 critic 太贵,GRPO 用"组内
  baseline"替代——同一 prompt 采 $G$ 个回答 $\{y_1, \dots, y_G\}$,
  各自得奖励 $r_i$,优势函数直接用组内标准化:

$$A_i = \frac{r_i - \mathrm{mean}(r_1..r_G)}{\mathrm{std}(r_1..r_G)}$$

再套 PPO 式 clip 目标 + KL 项。省掉 critic 模型,显存与调参负担减半;
代价是要在线采样 $G$ 条回答,推理吞吐成为训练瓶颈(需要 vLLM 这类
高速推理引擎陪跑——又一个推理与训练的交叉点)。

## 7. 选型法则

```
有偏好/负向信号要教?
├─ 成对偏好数据、对话/安全场景      → DPO(先 SFT)
├─ 只有单条好/坏标签               → KTO
├─ 数学/代码等有 verifier 的推理任务 → GRPO/RLVR
├─ 追求极限效果且有 RL 工程团队      → RLHF-PPO(完整版)
└─ 个人/小团队默认                 → SFT + DPO,QLoRA 执行
```

## 8. 小结

$$\text{RLHF: RM 打分 + PPO 在线优化(4 模型)}
\;\xrightarrow{\text{闭式解}}\;
\text{DPO: 隐式奖励 } \hat{r}_\theta = \beta \log \tfrac{\pi_\theta}{\pi_{\text{ref}}} \text{ 的二分类(2 模型)}$$

- 偏好对齐补的是 SFT 学不到的信号:相对比较与负向约束;
- DPO 的精髓是 KL 约束目标的闭式解让 RM 与 RL 同时消失,$Z(x)$ 在
  分差中抵消是整条推导的关键一步;
- DPO 与 QLoRA 叠加后,偏好对齐第一次进入消费卡;
- 有 verifier 的场景,GRPO/RLVR 是当下推理模型的主流路线。
