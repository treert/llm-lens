"""词嵌入 / LM Head 行向量的两两余弦分布,对比"高维随机向量近正交"理论曲线。

背景:d 维球面上两个均匀随机单位向量的余弦近似 N(0, 1/d)
(精确密度正比于 (1-c^2)^((d-3)/2))。K3 的 d=7168,理论 std 约 0.0118。

全表 163840 行的 Gram 需 2·V²·d ≈ 3.9e17 FLOP 与 107 GB 内存,不可行;
本脚本从每张表随机抽 m 行(默认 8192,m(m-1)/2 ≈ 3.4e7 对),计算:
1. 表内两两余弦:embed×embed、lm_head×lm_head;
2. 跨表异 token 余弦(embed 行 i × lm_head 行 j,i≠j)作对照;
与同 token 配对余弦(analyze_embed_lmhead.py)互补。

输出:
- output/kimi_k3/weight_moments/cosine_dist.json:分位数/均值/std/|cos|>0.1 占比;
- output/kimi_k3/weight_moments/cosine_dist.npz:直方图计数(供重绘);
- output/kimi_k3/figures/cosine_dist.png:线性/对数双联图 + 理论曲线。

用法(在仓库根目录下):
    python analysis/kimi_k3/analyze_cosine_dist.py
    python analysis/kimi_k3/analyze_cosine_dist.py --sample-rows 4096
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_weight_moments import FIG_DIR, OUT_DIR
from llm_lens import get_model_dir, iter_row_chunks
from llm_lens.cli import add_model_args

VOCAB = 163840
DIM = 7168
QUANTILES = [0, 1, 5, 25, 50, 75, 95, 99, 100]
HIST_RANGE = (-0.5, 0.5)
HIST_BINS = 400


def collect_rows(path: Path, name: str, chunk_rows: int, sample_idx: np.ndarray) -> np.ndarray:
    """顺序扫一遍张量,收集 sample_idx(升序)指定的行为 float32 矩阵。"""
    rows = []
    si = 0
    for chunk, r0 in iter_row_chunks(path, name, chunk_rows):
        while si < len(sample_idx) and sample_idx[si] < r0 + len(chunk):
            rows.append(chunk[sample_idx[si] - r0])
            si += 1
        if si >= len(sample_idx):
            break
    return np.stack(rows)


def normalize_rows(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-30)


def pair_cosines(xn: np.ndarray, yn: np.ndarray | None = None) -> np.ndarray:
    """上三角(k=1)两两余弦;yn 为 None 时取 xn 自身。"""
    if yn is None:
        g = xn @ xn.T
    else:
        g = xn @ yn.T
    return g[np.triu_indices(len(xn), k=1)]


def summarize(c: np.ndarray, theory_std: float) -> dict:
    return {
        "n_pairs": int(c.size),
        "mean": float(c.mean()),
        "std": float(c.std()),
        "std_over_theory": float(c.std() / theory_std),
        "quantiles": {f"p{p}": float(np.percentile(c, p)) for p in QUANTILES},
        "frac_abs_gt_0.1": float(np.mean(np.abs(c) > 0.1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_args(parser)
    parser.add_argument("--sample-rows", type=int, default=8192)
    parser.add_argument("--chunk-rows", type=int, default=8192)
    args = parser.parse_args()

    model_dir = get_model_dir(args.model, args.model_dir, args.config)
    print(f"模型目录: {model_dir}")
    with open(model_dir / "model.safetensors.index.json", encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]
    ne, nh = "language_model.model.embed_tokens.weight", "language_model.lm_head.weight"
    pe, ph = model_dir / weight_map[ne], model_dir / weight_map[nh]

    rng = np.random.default_rng(0)
    sample_idx = np.sort(rng.choice(VOCAB, args.sample_rows, replace=False))
    print(f"采样 {args.sample_rows} 行(固定种子),收集 embed 行 ...")
    xe = collect_rows(pe, ne, args.chunk_rows, sample_idx)
    print("收集 lm_head 行 ...")
    xh = collect_rows(ph, nh, args.chunk_rows, sample_idx)

    xen, xhn = normalize_rows(xe), normalize_rows(xh)
    print("计算两两余弦(Gram)...")
    c_ee = pair_cosines(xen)
    c_hh = pair_cosines(xhn)
    c_eh = pair_cosines(xen, xhn)

    theory_std = 1 / math.sqrt(DIM)
    summary = {
        "sample_rows": int(args.sample_rows),
        "theory": {"dist": "approx N(0, 1/d)", "d": DIM, "std": theory_std},
        "embed_x_embed": summarize(c_ee, theory_std),
        "lmhead_x_lmhead": summarize(c_hh, theory_std),
        "embed_x_lmhead_offdiag": summarize(c_eh, theory_std),
    }
    for tag, c in (("embed×embed", c_ee), ("lm_head×lm_head", c_hh), ("embed×lm_head", c_eh)):
        print(f"{tag:<16} mean={c.mean():+.5f} std={c.std():.5f}"
              f"(理论 {theory_std:.5f},{c.std()/theory_std:.2f}x) |cos|>0.1 占比 {np.mean(np.abs(c)>0.1):.4%}"
              f" 极值 [{c.min():.3f}, {c.max():.3f}]")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "cosine_dist.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    edges = np.linspace(*HIST_RANGE, HIST_BINS + 1)
    hists = {tag: np.histogram(c, bins=edges)[0]
             for tag, c in (("ee", c_ee), ("hh", c_hh), ("eh", c_eh))}
    np.savez(OUT_DIR / "cosine_dist.npz", bin_edges=edges, **hists)

    xs = np.linspace(*HIST_RANGE, 2000)
    pdf = np.exp(-0.5 * (xs / theory_std) ** 2) / (theory_std * math.sqrt(2 * math.pi))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    for ax, logy in zip(axes, (False, True)):
        ax.hist(c_ee, bins=edges, density=True, alpha=0.55, label="embed × embed")
        ax.hist(c_hh, bins=edges, density=True, alpha=0.55, label="lm_head × lm_head")
        ax.hist(c_eh, bins=edges, density=True, alpha=0.45, label="embed × lm_head (i≠j)")
        ax.plot(xs, pdf, "k--", lw=1.5, label=f"theory N(0, 1/{DIM})")
        if logy:
            ax.set_yscale("log")
            ax.set_ylim(1e-1, 1e3)
        ax.set_title("pairwise cosine (log y)" if logy else "pairwise cosine")
        ax.legend(fontsize=8)
    fig.suptitle(f"Kimi-K3 row cosine distribution ({args.sample_rows} sampled rows each)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "cosine_dist.png", dpi=120)
    plt.close(fig)
    print(f"\n输出: {OUT_DIR / 'cosine_dist.json'}  {FIG_DIR / 'cosine_dist.png'}")


if __name__ == "__main__":
    main()
