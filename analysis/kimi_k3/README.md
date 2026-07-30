# analysis/kimi_k3 — Kimi-K3 专属分析脚本

本目录的脚本依赖 Kimi-K3 的具体结构,包括:

- 权重命名规则(`language_model.model.layers.N.*`、`vision_tower.*`、`mm_projector.*`)
- 混合注意力布局:24 个 MLA 全注意力层 + 69 个 KDA 线性注意力层(见 `docs/kimi-k3.md`)
- MoE 结构:896 个 mxfp4 量化路由专家 + 2 个共享专家 + latent 投影
- mxfp4 反量化格式(`weight_packed` / `weight_scale`,group_size=32)

模型结构笔记见 [`docs/kimi-k3.md`](../../docs/kimi-k3.md)。

## 脚本一览

| 脚本 | 功能 |
| --- | --- |
| `inspect_weights.py` | 只读解析 `model.safetensors.index.json`,输出权重分组统计总览 |

前置条件:已在仓库根目录执行 `pip install -e .`(见根目录 README 的快速开始)。

所有脚本默认从 `config/models.local.yaml` 读取模型路径,可用 `--model-dir` 覆盖:

```powershell
python analysis/kimi_k3/inspect_weights.py
python analysis/kimi_k3/inspect_weights.py --model-dir G:/llm-models/Kimi-K3
```
