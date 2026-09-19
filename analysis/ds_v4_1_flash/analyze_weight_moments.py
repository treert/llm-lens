"""计算 DeepSeek-V4.1-Flash 训练后权重的逐向量均值/方差/范数,对比初始化基线。

视角:嵌入/LM Head 的每行是一个 token 向量(分量视为 D 次采样);
FFN 看作 key-value 表——w1/w3 的每行是 key(特征检测器),w2 的每列是 value
(神经元激活时写回残差流的方向)。

初始化基线(见 docs/rmsnorm.md §3/§4 的方差链):
- embed/head:分量 std = config.json 的 initializer_range(0.02);
- FFN 矩阵:分量 std = 1/sqrt(fan_in),fan_in = 矩阵列数(另报相对固定值 0.02 的比值);
- 向量范数基线 = 分量 std x sqrt(向量维度)。

覆盖:embed、head、若干层的共享专家(FP8 反量化,--moe-layers,默认 0 20 39)、
抽样路由专家(FP4 反量化,--routed-layers/--routed-per-layer)。

输出:
- output/ds_v4_1_flash/weight_moments/<tag>.npz:逐向量的 mean/std/norm 数组;
- output/ds_v4_1_flash/weight_moments/summary.json:各矩阵汇总(分位数 + 基线比值);
- output/ds_v4_1_flash/figures/weight_moments_<tag>.png:分布直方图(带基线标记)。

用法(在仓库根目录下):
    python analysis/ds_v4_1_flash/analyze_weight_moments.py
    python analysis/ds_v4_1_flash/analyze_weight_moments.py --moe-layers 0 10 20 30 39
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ds_common import (
    DIM,
    EMBED_NAME,
    HEAD_NAME,
    INIT_RANGE,
    N_ROUTED_EXPERTS,
    load_weight_map,
    read_fp4_expert,
    read_fp8_block,
)
from llm_lens import get_model_dir, iter_row_chunks
from llm_lens.cli import add_model_args

OUT_DIR = Path("output/ds_v4_1_flash/weight_moments")
FIG_DIR = Path("output/ds_v4_1_flash/figures")
ELEM_SAMPLE_PER_MATRIX = 200_000  # 每矩阵抽样的元素数(画元素分布直方图用)
QUANTILES = [0, 5, 25, 50, 75, 95, 100]


def accumulate(path: Path, name: str, axis: str, chunk_rows: int, rng: np.random.Generator):
    """分块扫描 bf16 矩阵,累加逐向量和全局元素的一阶/二阶统计(不整表载入内存)。

    axis="row": 每行是一个向量(嵌入行、key 行);
    axis="col": 每列是一个向量(value 列)。
    """
    vec_sums = None  # 逐向量元素和
    vec_sqs = None   # 逐向量元素平方和
    elem_s = elem_sq = 0.0
    elem_n = 0
    samples = []
    n_rows_seen = 0
    n_cols = None

    for chunk, _r0 in iter_row_chunks(path, name, chunk_rows):
        n_rows_seen += chunk.shape[0]
        n_cols = chunk.shape[1]
        c = chunk.astype(np.float64)
        elem_s += c.sum()
        elem_sq += (c * c).sum()
        elem_n += c.size
        if axis == "row":
            s = c.sum(axis=1)
            sq = np.einsum("ij,ij->i", c, c)
            if vec_sums is None:
                vec_sums, vec_sqs = s, sq
            else:
                vec_sums = np.concatenate([vec_sums, s])
                vec_sqs = np.concatenate([vec_sqs, sq])
        else:
            s = c.sum(axis=0)
            sq = np.einsum("ij,ij->j", c, c)
            if vec_sums is None:
                vec_sums, vec_sqs = s, sq
            else:
                vec_sums += s
                vec_sqs += sq
        flat = chunk.ravel()
        take = min(ELEM_SAMPLE_PER_MATRIX // 8, flat.size)
        samples.append(flat[rng.integers(0, flat.size, size=take)])

    shape = (n_rows_seen, n_cols)
    vec_dim = n_cols if axis == "row" else n_rows_seen
    vec_mean = vec_sums / vec_dim
    vec_var = np.maximum(vec_sqs / vec_dim - vec_mean**2, 0.0)
    return {
        "shape": shape,
        "vec_mean": vec_mean,
        "vec_std": np.sqrt(vec_var),
        "vec_norm": np.sqrt(vec_sqs),
        "elem_mean": elem_s / elem_n,
        "elem_std": math.sqrt(max(elem_sq / elem_n - (elem_s / elem_n) ** 2, 0.0)),
        "elem_samples": np.concatenate(samples),
    }


def accumulate_array(arr: np.ndarray, axis: str, rng: np.random.Generator):
    """对已在内存中的矩阵(反量化后)做与 accumulate 相同的统计。"""
    c = arr.astype(np.float64)
    elem_s, elem_sq = c.sum(), float((c * c).sum())
    if axis == "row":
        vec_sums = c.sum(axis=1)
        vec_sqs = np.einsum("ij,ij->i", c, c)
    else:
        vec_sums = c.sum(axis=0)
        vec_sqs = np.einsum("ij,ij->j", c, c)
    vec_dim = c.shape[1] if axis == "row" else c.shape[0]
    vec_mean = vec_sums / vec_dim
    vec_var = np.maximum(vec_sqs / vec_dim - vec_mean**2, 0.0)
    flat = arr.ravel()
    take = min(ELEM_SAMPLE_PER_MATRIX, flat.size)
    return {
        "shape": arr.shape,
        "vec_mean": vec_mean,
        "vec_std": np.sqrt(vec_var),
        "vec_norm": np.sqrt(vec_sqs),
        "elem_mean": elem_s / c.size,
        "elem_std": math.sqrt(max(elem_sq / c.size - (elem_s / c.size) ** 2, 0.0)),
        "elem_samples": flat[rng.integers(0, flat.size, size=take)],
    }


def _vec_dim(stats: dict) -> int:
    """向量维度 = 另一个轴的长度。"""
    n_vec = stats["vec_mean"].size
    r, c = stats["shape"]
    return c if r == n_vec else r


def summarize(stats: dict, baseline_std: float) -> dict:
    """把逐向量统计压成分位数摘要,并算与初始化基线的比值。"""
    q = lambda a: {f"p{p}": float(np.percentile(a, p)) for p in QUANTILES}
    vec_dim = _vec_dim(stats)
    return {
        "shape": list(stats["shape"]),
        "n_vectors": int(stats["vec_mean"].size),
        "vec_dim": vec_dim,
        "elem_mean": float(stats["elem_mean"]),
        "elem_std": float(stats["elem_std"]),
        "baseline_elem_std": baseline_std,
        "elem_std_over_baseline": float(stats["elem_std"] / baseline_std),
        "elem_std_over_fixed_002": float(stats["elem_std"] / INIT_RANGE),
        "vec_mean_quantiles": q(stats["vec_mean"]),
        "vec_std_quantiles": q(stats["vec_std"]),
        "vec_norm_quantiles": q(stats["vec_norm"]),
        "baseline_vec_norm": baseline_std * math.sqrt(vec_dim),
        "vec_norm_median_over_baseline": float(
            np.median(stats["vec_norm"]) / (baseline_std * math.sqrt(vec_dim))
        ),
    }


def plot_matrix(stats: dict, tag: str, baseline_std: float) -> Path:
    """画三联直方图:逐向量 std、逐向量范数、元素分布(均带初始化基线标记)。"""
    vec_dim = _vec_dim(stats)
    baseline_norm = baseline_std * math.sqrt(vec_dim)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(stats["vec_std"], bins=200, density=True)
    axes[0].axvline(baseline_std, color="r", ls="--", label=f"init {baseline_std:.4g}")
    axes[0].set_title(f"{tag}: per-vector std (dim={vec_dim})")
    axes[0].legend()

    axes[1].hist(stats["vec_norm"], bins=200, density=True)
    axes[1].axvline(baseline_norm, color="r", ls="--", label=f"init {baseline_norm:.4g}")
    axes[1].set_title(f"{tag}: per-vector norm")
    axes[1].legend()

    xs = np.linspace(*np.percentile(stats["elem_samples"], [0.05, 99.95]), 400)
    axes[2].hist(stats["elem_samples"], bins=200, density=True)
    pdf = np.exp(-0.5 * (xs / baseline_std) ** 2) / (baseline_std * math.sqrt(2 * math.pi))
    axes[2].plot(xs, pdf, "r--", label=f"N(0, {baseline_std:.4g}^2)")
    axes[2].set_title(f"{tag}: elements (sampled)")
    axes[2].legend()

    fig.tight_layout()
    out = FIG_DIR / f"weight_moments_{tag}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_args(parser)
    parser.add_argument("--moe-layers", type=int, nargs="*", default=[0, 20, 39],
                        help="统计共享专家的层号(0 基,默认 0 20 39)")
    parser.add_argument("--routed-layers", type=int, nargs="*", default=[0, 20, 39],
                        help="抽样路由专家的层号(0 基,默认 0 20 39)")
    parser.add_argument("--routed-per-layer", type=int, default=4,
                        help="每层抽样的路由专家数(固定种子,默认 4)")
    parser.add_argument("--chunk-rows", type=int, default=4096, help="分块行数(默认 4096)")
    args = parser.parse_args()

    model_dir = get_model_dir(args.model, args.model_dir, args.config)
    print(f"模型目录: {model_dir}")
    weight_map = load_weight_map(model_dir)
    print(f"initializer_range(嵌入/LM Head 基线 std): {INIT_RANGE}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    summary = {"initializer_range": INIT_RANGE, "matrices": {}}

    def report(tag: str, name: str, axis: str, stats: dict, baseline_std: float) -> None:
        r, c = stats["shape"]
        vec_dim = _vec_dim(stats)
        print(f"\n[{tag}] {name}  shape={r}x{c}  axis={axis}  向量数={stats['vec_mean'].size} 维数={vec_dim}")
        print(f"  元素: mean={stats['elem_mean']:.4e}  std={stats['elem_std']:.4e}"
              f"  (基线 std={baseline_std:.4e}, 比值={stats['elem_std']/baseline_std:.3f},"
              f"  vs 固定0.02={stats['elem_std']/INIT_RANGE:.3f})")
        print(f"  逐向量 std:   中位={np.median(stats['vec_std']):.4e}"
              f"  [p5={np.percentile(stats['vec_std'],5):.4e}, p95={np.percentile(stats['vec_std'],95):.4e}]")
        print(f"  逐向量范数:   中位={np.median(stats['vec_norm']):.4e}"
              f"  (基线={baseline_std*math.sqrt(vec_dim):.4e})")
        print(f"  逐向量均值:   中位={np.median(stats['vec_mean']):.4e}"
              f"  |mean|p95={np.percentile(np.abs(stats['vec_mean']),95):.4e}")
        np.savez(OUT_DIR / f"{tag}.npz",
                 vec_mean=stats["vec_mean"], vec_std=stats["vec_std"], vec_norm=stats["vec_norm"])
        fig_path = plot_matrix(stats, tag, baseline_std)
        summary["matrices"][tag] = {
            "name": name, "axis": axis, **summarize(stats, baseline_std),
            "figure": str(fig_path),
        }

    # 1) 词嵌入 / LM Head(bf16 直接读)
    for tag, name in (("embed", EMBED_NAME), ("head", HEAD_NAME)):
        stats = accumulate(model_dir / weight_map[name], name, "row", args.chunk_rows, rng)
        report(tag, name, "row", stats, INIT_RANGE)

    # 2) 共享专家(FP8 反量化)
    for layer in args.moe_layers:
        prefix = f"layers.{layer}.ffn.shared_experts"
        for kind, axis in (("w1", "row"), ("w3", "row"), ("w2", "col")):
            arr = read_fp8_block(model_dir, weight_map, f"{prefix}.{kind}")
            fan_in = arr.shape[1]
            report(f"moeL{layer}_{kind}", f"{prefix}.{kind}.weight", axis,
                   accumulate_array(arr, axis, rng), 1.0 / math.sqrt(fan_in))

    # 3) 抽样路由专家(FP4 反量化)
    for layer in args.routed_layers:
        picks = np.random.default_rng(layer).choice(N_ROUTED_EXPERTS, args.routed_per_layer,
                                                    replace=False)
        for e in sorted(picks):
            prefix = f"layers.{layer}.ffn.experts.{e}"
            for kind, axis in (("w1", "row"), ("w3", "row"), ("w2", "col")):
                arr = read_fp4_expert(model_dir, weight_map, f"{prefix}.{kind}")
                fan_in = arr.shape[1]
                report(f"routedL{layer}E{e}_{kind}", f"{prefix}.{kind}.weight", axis,
                       accumulate_array(arr, axis, rng), 1.0 / math.sqrt(fan_in))

    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n汇总: {OUT_DIR / 'summary.json'}")
    print(f"逐向量数据与图: {OUT_DIR}/  {FIG_DIR}/weight_moments_*.png")


if __name__ == "__main__":
    main()
