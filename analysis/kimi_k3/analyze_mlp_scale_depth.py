"""逐层扫描 K3 的 MLP 权重尺度(dense L0 + 92 层 MoE 共享专家),看训练后尺度随深度的演化。

每层统计 gate/up_proj(按行=key 检测器)与 down_proj(按列=value 方向):
元素 std、逐向量 std 的中位/p5/p95、逐向量范数中位。
回答 analyze_weight_moments.py 留下的两个问题:
1. L91 共享专家 up/down 缩小是孤例还是深层趋势;
2. dense L0 那种近零"死行"(p5 远低于中位数)在 MoE 共享专家中是否存在。

输出:
- output/kimi_k3/weight_moments/mlp_scale_depth.npz:逐层逐矩阵的统计数组;
- output/kimi_k3/weight_moments/mlp_scale_depth.json:同上,便于查看;
- output/kimi_k3/figures/mlp_scale_depth.png:尺度-深度曲线(带初始化基线)。

用法(在仓库根目录下):
    python analysis/kimi_k3/analyze_mlp_scale_depth.py
    python analysis/kimi_k3/analyze_mlp_scale_depth.py --layers 0 1 45 91
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_weight_moments import OUT_DIR, FIG_DIR, accumulate
from llm_lens import get_model_dir
from llm_lens.cli import add_model_args

KINDS = ["gate_proj", "up_proj", "down_proj"]
HIDDEN = 7168


def layer_prefix(layer: int) -> str:
    """dense L0 用 mlp.*,其余 MoE 层用共享专家。"""
    if layer == 0:
        return f"language_model.model.layers.{layer}.mlp."
    return f"language_model.model.layers.{layer}.block_sparse_moe.shared_experts."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_args(parser)
    parser.add_argument("--layers", type=int, nargs="*", default=None,
                        help="只扫指定层(0 基,默认全部 93 层)")
    parser.add_argument("--chunk-rows", type=int, default=4096)
    args = parser.parse_args()

    model_dir = get_model_dir(args.model, args.model_dir, args.config)
    print(f"模型目录: {model_dir}")
    with open(model_dir / "model.safetensors.index.json", encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]

    layers = args.layers if args.layers else list(range(93))
    rng = np.random.default_rng(0)
    records: dict[str, dict[int, dict]] = {k: {} for k in KINDS}

    for layer in layers:
        prefix = layer_prefix(layer)
        for kind in KINDS:
            name = prefix + kind + ".weight"
            axis = "col" if kind == "down_proj" else "row"
            stats = accumulate(model_dir / weight_map[name], name, axis, args.chunk_rows, rng)
            vs = stats["vec_std"]
            records[kind][layer] = {
                "elem_std": float(stats["elem_std"]),
                "vec_std_p5": float(np.percentile(vs, 5)),
                "vec_std_median": float(np.median(vs)),
                "vec_std_p95": float(np.percentile(vs, 95)),
                "vec_norm_median": float(np.median(stats["vec_norm"])),
                "fan_in": int(stats["shape"][1]),
            }
        g = records["gate_proj"][layer]
        print(f"L{layer:>2}: gate elem_std={g['elem_std']:.4e} "
              f"up={records['up_proj'][layer]['elem_std']:.4e} "
              f"down={records['down_proj'][layer]['elem_std']:.4e} "
              f"key_p5/med={g['vec_std_p5']/g['vec_std_median']:.3f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    ls = np.array(sorted(records["gate_proj"].keys()))
    data = {k: {m: np.array([records[k][l][m] for l in ls])
                for m in ("elem_std", "vec_std_p5", "vec_std_median", "vec_std_p95")}
            for k in KINDS}
    np.savez(OUT_DIR / "mlp_scale_depth.npz", layers=ls,
             **{f"{k}_{m}": data[k][m] for k in KINDS for m in data[k]})
    with open(OUT_DIR / "mlp_scale_depth.json", "w", encoding="utf-8") as f:
        json.dump({k: {str(l): records[k][l] for l in ls} for k in KINDS},
                  f, ensure_ascii=False, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for kind, marker in zip(KINDS, ("o", "s", "^")):
        axes[0].plot(ls, data[kind]["elem_std"], marker=marker, ms=3, label=kind)
    axes[0].axhline(0.02, color="r", ls="--", lw=1, label="fixed init 0.02")
    axes[0].axhline(1 / math.sqrt(HIDDEN), color="gray", ls=":", lw=1,
                    label=f"fan-in 1/sqrt({HIDDEN})")
    axes[0].set_xlabel("layer")
    axes[0].set_title("element std vs depth (dense L0 + MoE shared experts)")
    axes[0].legend()

    for kind, marker in zip(KINDS, ("o", "s", "^")):
        ratio = data[kind]["vec_std_p5"] / data[kind]["vec_std_median"]
        axes[1].plot(ls, ratio, marker=marker, ms=3, label=kind)
    axes[1].axhline(1.0, color="gray", ls=":", lw=1)
    axes[1].set_xlabel("layer")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("dead-row indicator: vec_std p5 / median")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(FIG_DIR / "mlp_scale_depth.png", dpi=120)
    plt.close(fig)
    print(f"\n输出: {OUT_DIR / 'mlp_scale_depth.json'}  {FIG_DIR / 'mlp_scale_depth.png'}")


if __name__ == "__main__":
    main()
