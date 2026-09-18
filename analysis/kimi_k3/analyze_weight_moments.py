"""计算 Kimi-K3 训练后权重的逐向量均值/方差/范数,对比初始化基线。

视角:嵌入/LM Head 的每行是一个 token 向量(分量视为 D 次采样);
FFN 看作 key-value 表——gate/up_proj 的每行是 key(特征检测器向量),
down_proj 的每列是 value(神经元激活时写回残差流的方向)。

初始化基线(见 docs/rmsnorm.md §3/§4 的方差链):
- embed/lm_head:分量 std = config.json 的 initializer_range(默认 0.02);
- FFN 矩阵:分量 std = 1/sqrt(fan_in),fan_in = 矩阵列数;
- 向量范数基线 = 分量 std x sqrt(向量维度)。

覆盖:embed_tokens、lm_head、1 个 dense MLP 层(--dense-layer,默认 layers.0)、
若干 MoE 层的共享专家(--moe-layers,默认 10 45 91)。
路由专家是 mxfp4 量化权重,本脚本跳过。

输出:
- output/kimi_k3/weight_moments/<tag>.npz:逐向量的 mean/std/norm 数组;
- output/kimi_k3/weight_moments/summary.json:各矩阵汇总(分位数 + 基线比值);
- output/kimi_k3/figures/weight_moments_<tag>.png:分布直方图(带基线标记)。

用法(在仓库根目录下):
    python analysis/kimi_k3/analyze_weight_moments.py
    python analysis/kimi_k3/analyze_weight_moments.py --moe-layers 10 45 91
    python analysis/kimi_k3/analyze_weight_moments.py --model-dir G:/llm-models/Kimi-K3
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from llm_lens import get_model_dir, iter_row_chunks
from llm_lens.cli import add_model_args

OUT_DIR = Path("output/kimi_k3/weight_moments")
FIG_DIR = Path("output/kimi_k3/figures")
ELEM_SAMPLE_PER_CHUNK = 16384  # 每块抽样的元素数(画元素分布直方图用)
QUANTILES = [0, 5, 25, 50, 75, 95, 100]

LM_PREFIX = "language_model.model."


def matrix_specs(dense_layer: int, moe_layers: list[int]) -> list[dict]:
    """生成待分析矩阵清单:(tag, 张量名, 向量轴向, 基线类型)。"""
    specs = [
        ("embed_tokens", LM_PREFIX + "embed_tokens.weight", "row", "init_range"),
        ("lm_head", "language_model.lm_head.weight", "row", "init_range"),
    ]
    p = f"{LM_PREFIX}layers.{dense_layer}.mlp."
    specs += [
        (f"denseL{dense_layer}_gate_proj", p + "gate_proj.weight", "row", "fan_in"),
        (f"denseL{dense_layer}_up_proj", p + "up_proj.weight", "row", "fan_in"),
        (f"denseL{dense_layer}_down_proj", p + "down_proj.weight", "col", "fan_in"),
    ]
    for L in moe_layers:
        p = f"{LM_PREFIX}layers.{L}.block_sparse_moe.shared_experts."
        specs += [
            (f"moeL{L}_gate_proj", p + "gate_proj.weight", "row", "fan_in"),
            (f"moeL{L}_up_proj", p + "up_proj.weight", "row", "fan_in"),
            (f"moeL{L}_down_proj", p + "down_proj.weight", "col", "fan_in"),
        ]
    return [
        {"tag": t, "name": n, "axis": a, "baseline": b} for t, n, a, b in specs
    ]


def accumulate(path: Path, name: str, axis: str, chunk_rows: int, rng: np.random.Generator):
    """分块扫描矩阵,累加逐向量和全局元素的一阶/二阶统计(不整表载入内存)。

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
        take = min(ELEM_SAMPLE_PER_CHUNK, flat.size)
        samples.append(flat[rng.integers(0, flat.size, size=take)])

    shape = (n_rows_seen, n_cols)
    vec_dim = n_cols if axis == "row" else n_rows_seen
    vec_mean = vec_sums / vec_dim
    vec_var = np.maximum(vec_sqs / vec_dim - vec_mean**2, 0.0)
    stats = {
        "shape": shape,
        "vec_mean": vec_mean,
        "vec_std": np.sqrt(vec_var),
        "vec_norm": np.sqrt(vec_sqs),
        "elem_mean": elem_s / elem_n,
        "elem_std": math.sqrt(max(elem_sq / elem_n - (elem_s / elem_n) ** 2, 0.0)),
        "elem_samples": np.concatenate(samples),
    }
    return stats


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
        "vec_mean_quantiles": q(stats["vec_mean"]),
        "vec_std_quantiles": q(stats["vec_std"]),
        "vec_norm_quantiles": q(stats["vec_norm"]),
        "baseline_vec_norm": baseline_std * math.sqrt(vec_dim),
        "vec_norm_median_over_baseline": float(
            np.median(stats["vec_norm"]) / (baseline_std * math.sqrt(vec_dim))
        ),
    }


def _vec_dim(stats: dict) -> int:
    """向量维度 = 另一个轴的长度。"""
    n_vec = stats["vec_mean"].size
    r, c = stats["shape"]
    return c if r == n_vec else r


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
    parser.add_argument("--dense-layer", type=int, default=0,
                        help="dense MLP 层号(0 基,默认 0,即 config 第 1 层)")
    parser.add_argument("--moe-layers", type=int, nargs="*", default=[10, 45, 91],
                        help="统计共享专家的 MoE 层号(0 基,默认 10 45 91)")
    parser.add_argument("--chunk-rows", type=int, default=4096, help="分块行数(默认 4096)")
    args = parser.parse_args()

    model_dir = get_model_dir(args.model, args.model_dir, args.config)
    print(f"模型目录: {model_dir}")
    with open(model_dir / "model.safetensors.index.json", encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]

    init_range = 0.02
    try:
        with open(model_dir / "config.json", encoding="utf-8") as f:
            cfg = json.load(f)
        init_range = cfg.get("initializer_range")
        if init_range is None:
            init_range = cfg.get("text_config", {}).get("initializer_range", 0.02)
    except FileNotFoundError:
        print("未找到 config.json,initializer_range 按 0.02 处理")
    print(f"initializer_range(嵌入/LM Head 基线 std): {init_range}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    summary = {"initializer_range": init_range, "matrices": {}}

    for spec in matrix_specs(args.dense_layer, args.moe_layers):
        tag, name, axis = spec["tag"], spec["name"], spec["axis"]
        stats = accumulate(model_dir / weight_map[name], name, axis, args.chunk_rows, rng)
        r, c = stats["shape"]
        fan_in = c  # 矩阵列数 = 扇入
        baseline_std = init_range if spec["baseline"] == "init_range" else 1.0 / math.sqrt(fan_in)

        vec_dim = _vec_dim(stats)
        print(f"\n[{tag}] {name}  shape={r}x{c}  axis={axis}  向量数={stats['vec_mean'].size} 维数={vec_dim}")
        print(f"  元素: mean={stats['elem_mean']:.4e}  std={stats['elem_std']:.4e}"
              f"  (基线 std={baseline_std:.4e}, 比值={stats['elem_std']/baseline_std:.3f})")
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

    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n汇总: {OUT_DIR / 'summary.json'}")
    print(f"逐向量数据与图: {OUT_DIR}/  {FIG_DIR}/weight_moments_*.png")


if __name__ == "__main__":
    main()
