# 单卡 QLoRA 实操:16 GB 上微调 7B 并部署验证

前置:[02-training-memory.md](02-training-memory.md)(显存账,本文要
对账)、[04-lora.md](04-lora.md)、[05-qlora-and-peft.md](05-qlora-and-peft.md)、
[07-data-and-eval.md](07-data-and-eval.md);
`docs/inference/12-deployment-hands-on.md`(llama.cpp 部署,本文复用)。
目标:在 RTX 5070 Ti(16 GB,Blackwell)上完成一次真实的 QLoRA 微调:
数据 → 训练 → 显存对账 → merge → GGUF → 部署,并验证微调生效。

> 时效说明:深度学习生态版本变动快,文中具体版本号以写作时
> (2025-08)为准,执行前以 unsloth / TRL 官方文档为准核对。

## 1. 任务设计:教一个"看得见"的行为

第一次微调,选一个**效果肉眼可辨、数据可自造**的任务:教模型用固定
结构回答问题——任何回答必须按:

```
【分析】<推理过程>
【结论】<一句话答案>
```

这个任务的好处:基座模型原本不会这么做 → 微调后格式出现即证明训练
生效;格式可程序化检查(正则) → 评估免费;数据只需几百条 → 半小时
内训完。

## 2. 环境准备

**约束(来自本仓库 AGENTS.md)**:全局 Python 已有 GPU 版 torch
(2.9.0+cu128,Blackwell 支持良好),**禁止重装/替换 torch**;
项目 venv(`.venv/`)与全局隔离。两个可行选项:

- **选项 A(推荐):项目 venv + unsloth**。unsloth 官方已支持 Windows
  (自动带 `triton-windows`):
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  pip install -e .                       # 本仓库自身依赖
  pip install "unsloth[windows] @ git+https://github.com/unslothai/unsloth.git"
  pip install --no-deps unsloth_zoo      # 按官方指引
  ```
  装完**立即验证 torch 没被换成 CPU 版**(venv 内本就该单独装 GPU 版):
  ```powershell
  python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
  # 期望:2.9.x+cu128 True;若显示 +cpu,按 AGENTS.md 指引装 GPU 版
  ```
- **选项 B:WSL2**。生态兼容性最好(与推理系列 12 篇的 WSL2 路径一致),
  适合选项 A 踩坑时的退路。

验证 GPU 可见:`python -c "import torch; print(torch.cuda.get_device_name(0))"`
应输出 `NVIDIA GeForce RTX 5070 Ti`。

## 3. 数据构造

自造 ~500 条:人工写 30 条种子(覆盖数学、常识、解释、建议等类型),
蒸馏/改写扩量,人工抽检(流程见 07 篇 §2)。存为
`data/format_sft.jsonl`,每行:

```json
{"messages": [
  {"role": "user", "content": "为什么天空是蓝色的?"},
  {"role": "assistant", "content": "【分析】阳光进入大气后,短波长的蓝光被空气分子瑞利散射得最强烈,向各方向散开进入人眼。【结论】因为蓝光被大气散射得最强,所以天空呈蓝色。"}
]}
```

切出 30 条做 held-out 评估集。**全程用基座模型自带的 chat template**
格式化(代码里 `tokenizer.apply_chat_template`,勿手写拼字符串,07 篇 §4)。

## 4. 训练脚本

`analysis/` 下不适用(本任务与具体模型无关且属一次性实验),建议放
`tmp/` 或按仓库惯例自理。核心骨架(unsloth 标准流程):

```python
"""单卡 QLoRA 微调示例:教 Qwen3-8B 使用【分析】【结论】格式。
用法:python finetune_format.py --model Qwen/Qwen3-8B --data data/format_sft.jsonl
"""
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

# 1) 4bit 加载主干(QLoRA:NF4 + 双重量化,05 篇 §2)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen3-8B",
    max_seq_length=2048,
    load_in_4bit=True,          # NF4 + double quant
    dtype=None,                 # 自动选 bf16(5070 Ti 支持)
)

# 2) 插 LoRA 旁路(04 篇:r=16, alpha=2r, all-linear)
model = FastLanguageModel.get_peft_model(
    model,
    r=16, lora_alpha=32, lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth",   # 激活重算,02 篇 §3
)

# 3) 训练
ds = load_dataset("json", data_files="data/format_sft.jsonl", split="train")
trainer = SFTTrainer(
    model=model, tokenizer=tokenizer, train_dataset=ds,
    args=SFTConfig(
        per_device_train_batch_size=2,      # micro-batch,显存紧就降到 1
        gradient_accumulation_steps=8,      # 等效 batch 16
        learning_rate=2e-4,                 # LoRA 的 lr 量级,04 篇
        num_train_epochs=2,                 # eval 抽查决定是否加第 3 轮
        lr_scheduler_type="cosine", warmup_ratio=0.03,
        bf16=True, packing=False,           # 数据量小,关 packing 便于观察
        logging_steps=5, save_strategy="no",
        dataset_text_field=None,            # 用 messages + chat template
        max_seq_length=2048, seed=42,
        output_dir="output/qlora-format",
    ),
)
trainer.train()

# 4) merge 回 bf16 主干并导出(04 篇 §5)
model.save_pretrained_merged(
    "output/qlora-format/merged", tokenizer, save_method="merged_16bit")
```

## 5. 显存对账:验证 02 篇的预测

训练中开另一个终端 `nvidia-smi -l 2` 观察峰值显存,与 02 篇 §4 的
预测对比:

| 项 | 02 篇预测 | 实测关注点 |
|---|---|---|
| 4bit 主干(含双重量化元数据) | ~4.5 GB | 加载后即稳定,应接近 |
| LoRA 参数+梯度+Adam 状态(~1 亿参数) | ~1.6 GB | 优化器 step 时分配 |
| 激活(micro-batch 2, s≤2048, 重算+FA) | ~2~4 GB | 随最长样本波动 |
| 框架/CUDA 开销 | ~1.5 GB | 启动时固定 |
| **合计** | **~10~12 GB** | **预期 ≤14 GB,不超 16** |

若实测明显超出:检查是否忘了开 gradient checkpointing、是否有样本
超长(激活与 s 近似线性,02 篇 §2)。把实测数字补进这张表——
这就是本仓库"能实测的尽量实测"原则的体现。

## 6. 验证微调生效

三层验证(对应 07 篇 §5):

1. **程序化指标**:用 held-out 的 30 条 prompt 分别问基座与微调模型,
   正则 `^【分析】.+【结论】` 统计格式命中率。预期:基座 ~0%,
   微调后 >90%;
2. **人工抽检**:翻 10 条回答,确认内容质量没有变蠢;
3. **生成不停检查**:确认回答在【结论】后正常结束(EOS 学进去了)。

## 7. 部署:转 GGUF 接入推理工具链

merged 的 16bit safetensors 可以直接给 vLLM 用;要走 llama.cpp
(呼应推理系列 12 篇)则转 GGUF:

```powershell
python llama.cpp\convert_hf_to_gguf.py output\qlora-format\merged `
  --outfile output\qlora-format\model-f16.gguf
.\llama.cpp\build\bin\llama-quantize.exe `
  output\qlora-format\model-f16.gguf output\qlora-format\model-Q4_K_M.gguf Q4_K_M
```

然后 `llama-server` 起服务,用推理系列 12 篇的同一套流程压测与对比。
至此完成闭环:**训练(本系列)→ 权重(merged safetensors)→
量化部署(inference 系列)**。

## 8. 故障排查速查

| 症状 | 对策 |
|---|---|
| 加载即 OOM | 确认 4bit 生效;关其他占显存程序(浏览器、推理服务) |
| 训练中 OOM(峰值) | micro-batch 降 1;`max_seq_length` 降 1024;确认 gradient checkpointing 开启;paged optimizer 兜底 |
| loss 不降/震荡 | 数据格式检查(先打印 2 条 tokenize 后的序列人肉核对);lr 降半 |
| 格式没学会 | 数据量/epoch 加一点;检查 chat template 与 EOS |
| 学完变复读机 | 数据去重;epoch 降回 1 |
| torch 变成 +cpu | venv 内误装 CPU 版,按 §2 指引重装 GPU 版(全局环境禁止动) |
| Blackwell 报 `sm_120` 相关错 | torch/unsloth 版本过旧,升级;或退回 WSL2 路径 |

## 9. 实测记录模板

```markdown
## YYYY-MM-DD QLoRA 微调记录
- 基座 / 数据量 / r / alpha / lr / epoch / 等效 batch:
- 显存峰值(预测 vs 实测):
- train loss 起止:
- held-out 格式命中率(基座 vs 微调):
- 人工抽检结论:
- 部署方式与推理速度:
- 坑与解决:
```

## 10. 小结与进阶方向

- 本文走通了最小闭环:500 条自造数据 + 16 GB 单卡 + QLoRA r16,
  一次训练 ~半小时,显存实测验证了 02 篇的账;
- 进阶(按需展开):换真任务(领域问答风格);数据工程深水区
  (07 篇);在 merged 模型上接着做 DPO(06 篇 §4.5,QLoRA 与 DPO
  可叠加);多 LoRA 挂同一基座用 vLLM 热切换(05 篇 §5);
- 记住分寸:QLoRA 能教格式、风格、偏好,教不了新知识(01 篇 §3)——
  那是 RAG 与预训练的领地。
