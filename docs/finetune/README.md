# 模型微调(Fine-tuning)笔记

目标:从原理层面理解大模型微调——训练的显存账怎么算、LoRA 为什么有效、
DPO 的数学怎么推,最终在单张 16 GB 消费卡上完成一次 QLoRA 微调,
跑通"数据 → 训练 → 合并 → 部署"全链路。

与仓库其他文档的分工:

- `docs/inference/`:权重冻结之后怎么高效跑(系统/工程视角);
- 本目录:权重怎么被**小步更新**——微调的机制、数学与单卡实操;
- `docs/kimi-k3.md` 等模型结构笔记:回答"哪些张量可以改"的地图。

定位问题(常被混淆):微调在机制上**是训练**(前向 + 反向传播 + 优化器
更新权重),与预训练的差别是规模与配置,不是种类;但从知识体系上它与
预训练几乎不相交(千卡并行、数据配比、训练稳定性都用不上),可以也应该
当作与"推理"平行的独立领域来学。学习路径上它卡在中间:
推理(消费权重)→ 微调(小改权重)→ 预训练(从头造权重)。

## 笔记索引

**主线**(建议按序):

1. [01-finetuning-overview.md](01-finetuning-overview.md):微调的定位——
   预训练/后训练/推理分期,微调家族谱系(全量 vs PEFT、SFT vs 偏好对齐),
   微调能改变什么、不能改变什么,prompt/RAG/微调的选型。
2. [02-training-memory.md](02-training-memory.md):训练的显存账——
   权重+梯度+优化器状态+激活四项的定量推导,混合精度为什么 16 字节/参数,
   省显存的四大杠杆。这是"为什么需要 LoRA"的动机篇。
3. [03-full-finetuning.md](03-full-finetuning.md):全量微调机制——SFT 数据
   格式与 loss mask(只对 answer 算 loss)、训练循环、超参数直觉、
   packing、灾难性遗忘。
4. [04-lora.md](04-lora.md):LoRA 原理——低秩假设、 $W + BA$ 的数学、
   参数量账、 $\alpha/r$ 缩放、秩与 target modules 的选择、merge 与变体。
5. [05-qlora-and-peft.md](05-qlora-and-peft.md):QLoRA 与其他 PEFT——NF4
   分位数量化、双重量化、paged optimizer 三大件,训练态量化与推理态量化
   的异同,Adapter/Prompt/Prefix tuning 对比与选型法则。
6. [06-preference-alignment.md](06-preference-alignment.md):偏好对齐——
   RLHF 三步与 reward model(Bradley-Terry)、PPO 的代价、**DPO 完整数学
   推导**(KL 约束目标 → 闭式最优解 → 消掉 reward model)、GRPO/RLVR。
7. [07-data-and-eval.md](07-data-and-eval.md):数据工程与评估——"数据是
   上限,算法只是逼近上限"、LIMA 的表面对齐假说、指令数据构造路线、
   去污染、chat template 一致性、微调效果的评估方法。
8. [08-qlora-hands-on.md](08-qlora-hands-on.md):动手篇——单卡 16 GB
   (RTX 5070 Ti)用 unsloth 完成一次 QLoRA 微调:环境、数据、训练、
   显存对账(验证 02 的账)、merge 导出、转 GGUF 用推理工具链部署验证。

原则与 inference 系列一致:每个主题先讲清"数学/机制上为什么",
再落到"工程上怎么做";能实测的用本机实测验证(08 篇即 02 篇显存账的
实验验证)。
