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
| `probe_structure.py` | 探测完整结构(config + index + 各分片头部),输出 JSON 描述文件到 `output/kimi_k3/`:`structure_overview.json`(整体总览)、`layers.json`(逐层结构与权重形状)、`weight_patterns.json`(命名模式统计) |
| `extract_qk_spectrum.py` | 提取 24 个 MLA 层逐头的 QK 谱(M=W_Q^T W_K 的奇异值,潜空间 Gram 复用),输出 `qk_spectrum.npz`(原始谱数据)与 `qk_spectrum_summary.json`(逐层聚合);支持 `--layers` 跑子集调试 |
| `plot_qk_spectrum.py` | 可视化谱数据:指标层×头热力图、代表层谱曲线 vs 随机基线、潜空间谱,图进 `output/kimi_k3/figures/` |
| `analyze_head_types.py` | 基于 sym/u·v 的头分型(相似性 vs 序列匹配)与极端头个案,输出 `head_types.json` + 散点图 |
| `analyze_shared_latent.py` | MLA 共享 K 潜库指纹:96 头 top-k 右奇异子空间两两重叠 vs 随机基线,输出 `shared_latent_overlap.json` + 分布图 |
| `mla_common.py` | 公共模块:MLA 层权重加载、潜空间 Gram、逐头谱计算(被 extract/analyze 脚本复用) |

前置条件:已在仓库根目录执行 `pip install -e .`(见根目录 README 的快速开始)。

所有脚本默认从 `config/models.local.yaml` 读取模型路径,可用 `--model-dir` 覆盖:

```powershell
python analysis/kimi_k3/inspect_weights.py
python analysis/kimi_k3/inspect_weights.py --model-dir G:/llm-models/Kimi-K3
```
