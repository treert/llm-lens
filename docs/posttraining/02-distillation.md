# 蒸馏:个人获取推理能力的最短路径

前置:[01-posttraining-overview.md](01-posttraining-overview.md)、
[../finetune/03-full-finetuning.md](../finetune/03-full-finetuning.md)(SFT 机制)、
[../finetune/08-qlora-hands-on.md](../finetune/08-qlora-hands-on.md)(训练流程,
本篇动手直接复用)。
目标:理解蒸馏的三种形态、为什么对小模型"蒸馏 > 直接 RL",并完成
一次真实的小规模蒸馏:API 生成思维链数据 → QLoRA 灌给本地小模型。

## 1. 蒸馏的三种形态

**序列级蒸馏(sequence-level)**——本目录主线。老师生成完整回答
(含思维链),学生当普通 SFT 数据学:

$$\mathcal{L} = -\sum_t \log p_{\text{学生}}(y_t^{\text{老师}} \mid x, y_{<t}^{\text{老师}})$$

就是 SFT,唯一特殊的是**数据由模型生成**。R1-Distill 系列的做法:
80 万条 R1 输出直接 SFT Qwen/Llama 各尺寸模型。

**logits 蒸馏(经典 KD)**。学生匹配老师在每个位置输出的完整
概率分布(软标签),而不只是最终 token:

$$\mathcal{L}_{\text{KD}} = T^2 \cdot \mathrm{KL}\big(
\mathrm{softmax}(z_{\text{老师}}/T) \,\|\, \mathrm{softmax}(z_{\text{学生}}/T) \big)$$

信息量更大(分布里的"次优选项"暗含老师的犹豫),但有两个硬门槛:
需要老师的 **logits**(API 不开放,只有白盒可用)且**词表必须一致**。
个人场景基本不用,知道概念即可。

**on-policy 蒸馏(GKD 等)**。学生自己生成,老师对**学生的**输出
打分/纠正,数据分布始终贴着学生的当前水平。介于前两者之间,
工程上需要老师在线陪跑。进阶玩法,了解即可。

| 形态 | 需要老师什么 | 个人可行性 |
|---|---|---|
| 序列级 | 只要最终输出(API 即可) | ★★★ |
| logits | 完整 logprobs + 同词表 | ☆ |
| on-policy | 在线评分 | ★ |

## 2. 为什么"蒸馏 > 直接 RL 小模型"

DeepSeek-R1 论文做了关键对照:同为 Qwen 小模型,**蒸馏 R1 输出
vs 自己跑大规模 RL**,蒸馏的推理成绩大约翻倍(AIME 上
R1-Distill-Qwen-7B ~55%,论文指出直接 RL 同规模模型远达不到)。

机制解释:

- RL 是**探索**:模型从自己的分布出发,靠奖励信号慢慢发现长思维链
  这种有效模式。小模型探索效率低——它的策略空间里"推理链"的先验
  概率本来就小,稀疏奖励下可能要探索极久;
- 蒸馏是**继承**:强模型已经把"怎么搜索、怎么反思"的轨迹摆在数据里,
  学生直接模仿这些轨迹的**模式**,一步到位;
- 更本质地(01 篇):RL 放大 base 中已有的模式,而蒸馏可以**注入**
  base 中几乎没有的模式。

推论(对个人很重要):**RL 的上限更高(不依赖老师),但蒸馏的性价比
碾压**。没有千卡集群时,先蒸馏;蒸馏到顶了再考虑 RL 锦上添花。

## 3. 蒸馏的边界

- **天花板**:学生超不过老师(在同分布任务上);蒸馏是压缩不是创造;
- **风格继承**:老师的口癖、思维链格式、甚至错误模式都会被继承;
- **多样性税**:蒸馏后学生的输出分布变窄(模式坍缩到老师的典型
  轨迹),开放式创作类任务慎用;
- **合规**:OpenAI 等明确禁止用输出训练竞品模型;DeepSeek-R1 输出
  按 MIT 协议可用。**用哪家 API 蒸馏,先读哪家 ToS**。

## 4. 动手:蒸馏一个会"思考"的本地小模型

目标:让 Qwen3-1.7B(或同级)学会"先想再答"的推理格式,
训练流程完全复用 finetune 08 篇(unsloth QLoRA),差异只在数据
来源与评估。

### 4.1 数据:API 生成思维链

选 200~500 道**有标准答案**的题(GSM8K 训练集、小学奥数题),
调强模型 API(如 deepseek-reasoner)生成带思维链的解答,
**程序化校验答案正确**,只留对的:

```python
"""生成蒸馏数据:调 API 解题,校验答案,存 JSONL。
用法:python make_distill_data.py --in data/gsm8k_train.jsonl --out output/distill.jsonl
"""
# 伪代码骨架(真实实现按所用 API SDK 调整):
for item in load("data/gsm8k_train.jsonl"):          # 题目 + 标准答案
    resp = api.chat(model="deepseek-reasoner",
                    messages=[{"role": "user", "content": item["question"]}])
    if extract_number(resp) == item["answer"]:        # 程序化校验
        save({"messages": [
            {"role": "user", "content": item["question"]},
            {"role": "assistant", "content": resp},   # 完整思维链
        ]})
```

要点:

- **只留答对的**:蒸馏数据的质量过滤(错误思维链会被忠实模仿,
  finetune 07 篇);
- 成本:每条约 1K~3K tokens 输出,500 条约 1M tokens,按 API 定价
  通常**几元~十几元人民币**;
- 题材可混合:数学题为主,掺少量通用问答防止行为畸变
  (finetune 03 篇 §5 的 replay 思想)。

### 4.2 训练与评估

- 训练:与 finetune 08 篇完全相同的 unsloth QLoRA 配置
  (r=16, lr=2e-4, 2 epoch),500 条约十几分钟;
- 评估(程序化,对应 finetune 07 篇 §5):held-out 题目上对比蒸馏
  前后——**答案正确率**(正则提取最终数字)与**思维链出现率**;
- 预期:小模型正确率提升几个~十几个百分点,且输出带上"先分析
  再给答案"的结构。**预期管理**:1.7B 模型的天花板低,别指望
  质变;这个实验的价值在于亲手验证 §2 的论断。

## 5. 小结

- 序列级蒸馏 = 数据由强模型生成的 SFT,工程零新成本,是个人
  后训练的第一武器;
- 蒸馏 > 直接 RL 小模型:继承强于探索,这是 R1 论文最重要的
  实践结论;
- 边界:天花板、风格继承、多样性税、ToS;
- 动手环节验证了核心论断,也为下一篇(推理 RL)提供对照基线:
  蒸馏出来的模型,正是 RL 继续提升的起点。
