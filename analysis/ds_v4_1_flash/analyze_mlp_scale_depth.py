"""逐层扫描 DeepSeek-V4.1-Flash 的 40 层 MoE 共享专家(FP8 反量化)权重尺度,看训练后尺度随深度的演化。

每层统计 w1/w3(按行=key 检测器)与 w2(按列=value 方向):
元素 std、逐向量 std 的中位/p5/p95、逐向量范数中位。

输出:
- output/ds_v4_1_flash/weight_moments/mlp_scale_depth.npz:逐层逐矩阵的统计数组;
- output/ds_v4_1_flash/weight_moments/mlp_scale_depth.json:同上,便于查看;
- output/ds_v4_1_flash/figures/mlp_scale_depth.png:尺度-深度曲线(带初始化基线)。

用法(在仓库根目录下):
    python analysis/ds_v4_1_flash/analyze_mlp_scale_depth.py
    python analysis/ds_v4_1_flash/analyze_mlp_scale_depth.py --layers 0 1 20 39
"""

import argparse
import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_weight_moments import FIG_DIR, OUT_DIR, accumulate_array
from ds_common import DIM, INIT_RANGE, N_LAYERS, load_weight_map, read_fp8_block
from llm_lens import get_model_dir
from llm_lens.cli import add_model_args

KINDS = ["w1", "w3", "w2"]  # w1=gate, w3=up, w2=down


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_args(parser)
    parser.add_argument("--layers", type=int, nargs="*", default=None,
                        help="只扫指定层(0 基,默认全部 40 层)")
    args = parser.parse_args()

    model_dir = get_model_dir(args.model, args.model_dir, args.config)
    print(f"模型目录: {model_dir}")
    weight_map = load_weight_map(model_dir)

    layers = args.layers if args.layers else list(range(N_LAYERS))
    rng = np.random.default_rng(0)
    records: dict[str, dict[int, dict]] = {k: {} for k in KINDS}

    for layer in layers:
        prefix = f"layers.{layer}.ffn.shared_experts"
        for kind in KINDS:
            arr = read_fp8_block(model_dir, weight_map, f"{prefix}.{kind}")
            axis = "col" if kind == "w2" else "row"
            stats = accumulate_array(arr, axis, rng)
            vs = stats["vec_std"]
            records[kind][layer] = {
                "elem_std": float(stats["elem_std"]),
                "vec_std_p5": float(np.percentile(vs, 5)),
                "vec_std_median": float(np.median(vs)),
                "vec_std_p95": float(np.percentile(vs, 95)),
                "vec_norm_median": float(np.median(stats["vec_norm"])),
                "fan_in": int(arr.shape[1]),
            }
        g = records["w1"][layer]
        print(f"L{layer:>2}: w1 elem_std={g['elem_std']:.4e} "
              f"w3={records['w3'][layer]['elem_std']:.4e} "
              f"w2={records['w2'][layer]['elem_std']:.4e} "
              f"key_p5/med={g['vec_std_p5']/g['vec_std_median']:.3f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    ls = np.array(sorted(records["w1"].keys()))
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
    axes[0].axhline(INIT_RANGE, color="r", ls="--", lw=1, label=f"fixed init {INIT_RANGE}")
    axes[0].axhline(1 / math.sqrt(DIM), color="gray", ls=":", lw=1,
                    label=f"fan-in 1/sqrt({DIM})")
    axes[0].set_xlabel("layer")
    axes[0].set_title("element std vs depth (MoE shared expert, fp8 dequantized)")
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
