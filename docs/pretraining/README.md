# 预训练(Pre-training)笔记

目标:从原理层面理解大模型预训练——数据怎么炼、Scaling Law 怎么用、
千卡集群怎么训稳、base model 怎么评估。

本系列是**纯原理**:个人无法实操预训练(成本与消费级差 5 个数量级),
但预训练是决定模型一切的上游——开源模型的每个配置(参数量、训练
token 数、词表大小、上下文长度)背后都是本系列讲的账。理解预训练,
才能读懂技术报告,才能理解微调与推理里各种"为什么是这样"。

与仓库其他文档的分工:

- `docs/inference/`:权重冻结之后怎么高效跑(系统/工程视角);
- `docs/finetune/`:权重怎么被**小步更新**(后训练:微调、对齐);
- 本目录:权重**从哪来**(预训练)——回答"开源模型为什么是这个配置";
- `docs/kimi-k3.md` 等模型结构笔记:具体模型的架构参数,是预训练
  决策的产物。

## 笔记索引

建议按序:

1. [01-pretraining-overview.md](01-pretraining-overview.md):预训练全景——
   next-token prediction 为什么能造出能力、一次 run 的完整时间线、
   $6ND$ 成本账与实例、base model 的能与不能。
2. [02-data-pipeline.md](02-data-pipeline.md):数据工程——语料来源与构成、
   清洗流水线六步、去重的数学与收益、数据配比这门艺术、BPE 与词表
   大小的权衡。
3. [03-scaling-laws.md](03-scaling-laws.md):Scaling Law——幂律现象、
   Kaplan vs Chinchilla(计算最优 20 tokens/参数)、用小实验外推大模型、
   局限与修正、推理时 scaling 的新范式。
4. [04-large-scale-training.md](04-large-scale-training.md):大规模训练
   工程——并行策略的训练视角(ZeRO/FSDP)、混合精度细节、loss spike
   与稳定性、lr 调度与 batch size、千卡集群的容错与监控。
5. [05-annealing-and-mid-training.md](05-annealing-and-mid-training.md):
   退火与 mid-training——数据退火为什么有效、长上下文扩展(RoPE
   外推失效、PI/NTK/YaRN)、课程学习。
6. [06-base-model-eval.md](06-base-model-eval.md):基座模型评估——
   perplexity 的能与不能、benchmark 体系与 few-shot 评测、训练过程
   监控、数据污染、怎么读开源模型的技术报告。

原则与 inference / finetune 系列一致:先讲清"数学/机制上为什么",
再落到"工程上怎么做"。本系列没有动手篇,但每篇尽量回答一个问题:
**这对个人学习者意味着什么**。
