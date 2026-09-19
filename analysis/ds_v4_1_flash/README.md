# analysis/ds_v4_1_flash — DeepSeek-V4.1-Flash 专属分析脚本

本目录的脚本依赖 DeepSeek-V4.1-Flash 的具体结构,包括:

- 权重命名规则(`layers.N.*`、`mtp.N.*`、`vision.*`、`aligner.*`,无 `model.` 前缀)
- 注意力结构:latent attention(单 KV 头 512 维、Q 低秩 1280、分组低秩输出投影)、
  滑窗 + 压缩稀疏注意力(compressor/indexer 只存在于部分层)
- MoE 结构:384 个 FP4 量化路由专家 + 1 个 FP8 共享专家
- 量化格式(见下,反量化由 `llm_lens` 提供并经 `tmp` 对拍验证):
  - FP8 E4M3 权重 + UE8M0 块缩放(32×32):注意力投影、共享专家、engram 投影等
  - FP4 E2M1(每字节打包 2 个值,低半字节在前)+ UE8M0 行块缩放(1×32):路由专家

模型结构笔记见 [`docs/ds_v4_1_flash.md`](../../docs/ds_v4_1_flash.md)。

## 脚本一览

| 脚本 | 功能 |
| --- | --- |
| `inspect_weights.py` | 只读解析索引 + 各分片头部,输出权重分组统计与 dtype 分布总览 |
| `ds_common.py` | 公共模块:结构常量、weight_map 加载、FP8 块/FP4 专家反量化读取 |
| `analyze_weight_moments.py` | 训练后权重的逐向量均值/方差/范数统计,对比初始化基线:嵌入/LM Head 按行(token 向量),FFN 的 w1/w3 按行(key)、w2 按列(value);覆盖共享专家与抽样路由专家;输出 `weight_moments/`(npz + summary.json)与直方图 |
| `analyze_mlp_scale_depth.py` | 逐层扫描全部 40 层 MoE 共享专家的权重尺度与死行指标(p5/中位数),输出 `mlp_scale_depth.{npz,json}` + 深度曲线 |
| `analyze_embed_lmhead.py` | 嵌入与 LM Head 配对分析:同 token 配对余弦 vs 随机基线、各向异性(均值方向)、范数极端 token(经 tokenizer.json 解码),输出 `embed_lmhead.{npz,json}` + 四联图 |
| `analyze_norm_gains.py` | 全部 RMSNorm 增益向量的分布分析:逐层 mean/std/CV/极端通道占比、相邻层余弦(**原始 + 去均值两口径**)、attn vs ffn 分支一致性、画像集中度(有效维数 PR / top-k 方差占比 / 跨层 PCA / 常驻突出维)、与 FFN 尺度联动,输出 `norm_gains.{npz,json}` + 九联图 |
| `analyze_norm_distribution.py` | 增益向量的**单层分布形状**与数值分辨率:δ = g/median(|g|)−1 逐层 IQR 标准化后的主体偏度/峰度、超 kσ 尾部占比 vs 高斯理论 (检验"高斯主体 + 稀疏尖峰")、单层唯一值与众数占比、主体宽度相当于几个 bf16 步,输出 `norm_distribution.json` + 六联图 |
| `analyze_residual_equalization.py` | 入口增益的「均衡器」机制检验(只用权重代理量):跟踪维的写回(`wo_b` 行范)/读取(`wq_a` 列范)响度 ρ + 随机通道对照、因果配对的滞后相关(δ=±2,含共享/路由专家 `w2` 画像)、累积写回衰减扫描,输出 `gain_equalization.json` + 四联图 |
| `analyze_cosine_dist.py` | 行向量两两余弦分布(采样 8192 行):embed×embed、lm_head×lm_head、跨表异 token 对照,叠加 N(0,1/d) 理论曲线,输出 `cosine_dist.{json,npz}` + 双联图 |

前置条件:已在仓库根目录执行 `pip install -e .`(见根目录 README 的快速开始)。

所有脚本默认从 `config/models.local.yaml` 读取模型路径(当前 default_model 即
`ds_v4_1_flash`),可用 `--model-dir` 覆盖:

```powershell
python analysis/ds_v4_1_flash/inspect_weights.py
python analysis/ds_v4_1_flash/inspect_weights.py --model-dir D:/llm-models/DeepSeek-V4.1-Flash
```
