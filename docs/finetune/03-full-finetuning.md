# 全量微调机制:SFT 的标准动作

前置:[02-training-memory.md](02-training-memory.md)(显存账)、
`docs/inference/01-inference-pipeline.md`(前向计算图)。
目标:理解一次 SFT 训练的完整机制——数据怎么组织、loss 怎么算、
超参数怎么定、什么时候停。全量微调是所有微调算法的基准形态,
LoRA 只是改了"更新哪些参数"(04 篇),其余机制原封不动。

## 1. 数据:从 (指令, 回答) 到 token 序列

一条 SFT 样本:

```json
{"instruction": "把下面的话翻译成英文", "input": "今天天气不错", "output": "The weather is nice today."}
```

进入模型前,要按基座的 **chat template** 拼成一条 token 序列
(以 Llama-3 风格为例):

```
<|begin_of_text|><|start_header_id|>user<|end_header_id|>
把下面的话翻译成英文
今天天气不错<|eot_id|><|start_header_id|>assistant<|end_header_id|>
The weather is nice today.<|eot_id|>
```

**template 一致性是 SFT 的第一大坑**:训练时的特殊 token、角色标记、
换行必须与推理部署时完全一致,否则等于白训(07 篇 §4 展开)。

## 2. loss mask:只对回答部分算 loss

整段序列都会过模型,但 loss 只应该衡量"模型生成回答的能力",
不该惩罚它对指令部分的预测。做法是给 label 加掩码:

$$\mathcal{L} = -\sum_{t=1}^{N} m_t \log p_\theta(x_t \mid x_{<t}),
\qquad m_t = \begin{cases} 1 & x_t \in \text{回答部分} \\ 0 & x_t \in \text{指令/模板部分} \end{cases}$$

工程实现(PyTorch 惯例):labels 中被掩盖的位置填 `-100`
(`CrossEntropyLoss` 的 `ignore_index`),移位交给框架:

```python
input_ids = tok(full_text)              # 完整序列
labels    = input_ids.clone()
labels[:answer_start] = -100            # 指令部分不参与 loss
# 框架内部:logits[..., :-1, :] 对 labels[..., 1:] 算交叉熵
```

两个细节:

- 回答**结尾的 EOS**(`<|eot_id|>`)必须计入 loss——这是教模型
  "该停就停"的唯一信号,漏掉会导致生成不停;
- 是否对指令部分也算 loss(全序列 loss)有少量实验支持(防遗忘),
  主流仍是只算回答。

## 3. packing:把算力用满

naive 做法每条样本 pad 到 max_len,短样本浪费一半算力。**packing**
把多条样本首尾相接塞进定长序列:

```
[样本1 tokens][样本2 tokens][样本3 tokens ......][pad]
```

配合 FlashAttention 的 varlen/position_ids 机制,让 attention 不跨样本
(否则样本互相"偷看",轻微掉点)。packing 通常带来 2~5 倍吞吐提升,
生产 SFT 默认开启;调试期建议关掉,便于逐样本查问题。

## 4. 训练循环与超参数直觉

循环本身与预训练无异:前向 → loss → 反向 → optimizer.step(),
只是规模小。关键超参数:

| 超参 | 全量微调典型值 | 直觉 |
|---|---|---|
| 学习率 | 1e-5 ~ 2e-5 | 预训练的 1/10;太大→遗忘,太小→学不动 |
| epoch | 1 ~ 3 | SFT 数据小,>3 基本必过拟合 |
| global batch | 64 ~ 256 条样本 | 梯度累积凑,等价大 batch 更稳 |
| warmup | 总步数 3% 左右 | 防止开头大梯度冲坏预训练权重 |
| schedule | cosine 降到 ~0 | 小训练的标准选择 |
| weight decay | 0 ~ 0.1 | 数据小时给一点正则 |
| max_len | 覆盖 99% 样本即可 | 显存随长度线性(激活),别盲目开 8K |

监控三件套:

- **train loss**:应平滑下降;剧烈震荡→ lr 太大或数据有问题;
- **eval loss**(held-out):**回升即停**——这是过拟合的直接信号,
  SFT 经常 1 个 epoch 内就到最优点;
- **生成抽查**:loss 看不出格式错误,每几百 step 手动看几条生成结果。

## 5. 灾难性遗忘:微调的头号副作用

定义:在新任务上学得越多,把预训练的通用能力忘得越多。机制很直接——
梯度把权重往新数据分布拉,远离原来精心找到的"通用解"。

缓解手段(按性价比排序):

1. **小学习率 + 少 epoch**:遗忘量与"权重移动距离"正相关;
2. **混入 5%~10% 通用指令数据**(replay):保留通用分布的梯度信号;
3. **冻结大部分参数**(LoRA 的天然副作用:主干不动,遗忘被物理限制,
   这是 PEFT 常被忽视的优点);
4. 早停:eval loss 最低点处收手。

评估遗忘:在通用 benchmark(MMLU、ARC 等)上对比微调前后,
而不只是看目标任务指标。

## 6. 一个最小全流程清单

```
1. 定任务 → 2. 收集/构造数据(07 篇)→ 3. 按基座 template 格式化
→ 4. 切 held-out 集(几百条)→ 5. 设定超参(上表)→ 6. 训练,监控三件套
→ 7. eval loss 最低点选 checkpoint → 8. 人工抽查生成 → 9. 通用能力回归测试
```

每一步都有独立的失败模式;SFT 的实际工作量 70% 在数据(第 2、3 步),
训练本身(第 5、6 步)在 2025 年的框架(TRL、LLaMA-Factory、unsloth)
里已是几十行配置的事。

## 7. 小结

- SFT = 按 template 拼序列 + 只对回答算交叉熵,机制上没有更多魔法;
- 超参数的核心矛盾是"学进去 vs 忘光光",lr 与 epoch 是旋钮,
  eval loss 是刹车;
- 全量微调机制上是基准,硬件上是奢侈品(02 篇)——下一篇的 LoRA
  保持本篇全部机制,只把"更新哪些参数"换掉。
