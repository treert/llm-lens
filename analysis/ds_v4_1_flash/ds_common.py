"""DeepSeek-V4.1-Flash 专属的权重加载公共模块。

权重命名与量化格式见 docs/ds_v4_1_flash.md:
- 大部分矩阵是 FP8 E4M3 + UE8M0 块缩放(32x32);
- 路由专家是 FP4 E2M1(每字节打包 2 个,低半字节在前)+ UE8M0 行块缩放(1x32);
- 词嵌入/LM Head/norm/路由器/视觉塔为 bf16 或 fp32,直接可读。
"""

import json
from pathlib import Path

import numpy as np

from llm_lens import dequant_block, read_tensor, unpack_fp4_e2m1

# 结构常量(来自 config.json,见 docs/ds_v4_1_flash.md)
VOCAB = 129280
DIM = 5120
N_LAYERS = 40
N_MTP_LAYERS = 3
MOE_INTER = 2304
N_ROUTED_EXPERTS = 384
INIT_RANGE = 0.02  # config.json: text_config.initializer_range

EMBED_NAME = "embed.weight"
HEAD_NAME = "head.weight"
FINAL_NORM_NAME = "norm.weight"


def load_weight_map(model_dir: str | Path) -> dict[str, str]:
    """读取 model.safetensors.index.json 的 weight_map(张量名 -> 分片文件名)。"""
    with open(Path(model_dir) / "model.safetensors.index.json", encoding="utf-8") as f:
        return json.load(f)["weight_map"]


def read(model_dir: str | Path, weight_map: dict[str, str], name: str,
         dtype=np.float64) -> np.ndarray:
    """按张量名读取(自动定位分片),不做反量化。"""
    return read_tensor(Path(model_dir) / weight_map[name], name, dtype=dtype)


def read_fp8_block(model_dir: str | Path, weight_map: dict[str, str], base: str) -> np.ndarray:
    """读取 FP8 E4M3 块量化矩阵并反量化为 float32。

    Args:
        base: 不带后缀的权重名,如 "layers.0.attn.wq_a"(实际读取 base.weight 与 base.scale)。
    """
    w = read(model_dir, weight_map, base + ".weight", dtype=np.float32)
    s = read(model_dir, weight_map, base + ".scale", dtype=np.float32)
    return dequant_block(w, s)


def read_fp4_expert(model_dir: str | Path, weight_map: dict[str, str], base: str) -> np.ndarray:
    """读取 FP4 E2M1 路由专家矩阵并反量化为 float32(1x32 行块缩放)。"""
    w_packed = read(model_dir, weight_map, base + ".weight", dtype=np.int8)
    w = unpack_fp4_e2m1(w_packed)
    s = read(model_dir, weight_map, base + ".scale", dtype=np.float32)
    return w * np.kron(s, np.ones((1, 32), dtype=np.float32))
