"""词嵌入与 LM Head 的配对分析:不共享权重后,同一 token 的两个向量还像不像。

DeepSeek-V4.1-Flash 的 tie_word_embeddings=false,embed 与 head 各自训练。本脚本回答:
1. 配对余弦:同一 token 的嵌入行与 LM Head 行的余弦相似度分布
   (基线:随机两向量的余弦 ~ N(0, 1/5120),std 约 0.0140);
2. 各向异性:词嵌入的均值向量是否显著(余弦到均值方向的分布);
3. 范数极端 token:嵌入范数最小/最大的 token 是谁(经 tokenizer.json 解码);
4. 两个表的范数是否联动:e_norm 与 h_norm 的逐 token 相关。

输出:
- output/ds_v4_1_flash/weight_moments/embed_lmhead.npz:逐 token 的 e_norm/h_norm/pair_cos/cos_to_mean;
- output/ds_v4_1_flash/weight_moments/embed_lmhead.json:分布分位数、极端 token 清单;
- output/ds_v4_1_flash/figures/embed_lmhead.png:四联图。

用法(在仓库根目录下):
    python analysis/ds_v4_1_flash/analyze_embed_lmhead.py
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
from ds_common import DIM, EMBED_NAME, HEAD_NAME, VOCAB, load_weight_map
from llm_lens import get_model_dir, iter_row_chunks
from llm_lens.cli import add_model_args

N_SAMPLE_ROWS = 1024  # 随机两两余弦基线的采样行数
QUANTILES = [0, 5, 25, 50, 75, 95, 100]


def load_vocab(model_dir: Path) -> dict[int, str]:
    """解析 HF tokenizer.json(model.vocab 为 token->id),失败时回退为 <id N>。"""
    path = model_dir / "tokenizer.json"
    try:
        with open(path, "r", encoding="utf-8") as f:
            vocab = json.load(f)["model"]["vocab"]
        return {int(i): t for t, i in vocab.items()}
    except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
        print(f"未能解析 tokenizer.json({e}),token 只显示 id")
        return {}


def tok_label(vocab: dict[int, str], idx: int) -> str:
    return ascii(vocab.get(idx, f"<id {idx}>"))


def pass1(pe: Path, ne: str, ph: Path, nh: str, chunk_rows: int, sample_idx: np.ndarray):
    """双表同步分块扫描:逐 token 的 e_norm²/h_norm²/点积 + 两表列和 + 采样嵌入行。"""
    e2, h2, dots = [], [], []
    col_e = col_h = None
    rows_e = []
    si = 0
    gen_e = iter_row_chunks(pe, ne, chunk_rows)
    gen_h = iter_row_chunks(ph, nh, chunk_rows)
    for (ce, r0), (ch, _) in zip(gen_e, gen_h):
        a = ce.astype(np.float64)
        b = ch.astype(np.float64)
        e2.append(np.einsum("ij,ij->i", a, a))
        h2.append(np.einsum("ij,ij->i", b, b))
        dots.append(np.einsum("ij,ij->i", a, b))
        col_e = a.sum(axis=0) if col_e is None else col_e + a.sum(axis=0)
        col_h = b.sum(axis=0) if col_h is None else col_h + b.sum(axis=0)
        while si < len(sample_idx) and sample_idx[si] < r0 + len(a):
            rows_e.append(a[sample_idx[si] - r0].astype(np.float32))
            si += 1
    return {
        "e2": np.concatenate(e2), "h2": np.concatenate(h2), "dot": np.concatenate(dots),
        "col_e": col_e, "col_h": col_h, "rows_e": np.array(rows_e),
    }


def cos_to_direction(path: Path, name: str, chunk_rows: int, direction: np.ndarray) -> np.ndarray:
    """逐 token 向量与指定单位方向的余弦。"""
    out = []
    for chunk, _ in iter_row_chunks(path, name, chunk_rows):
        a = chunk.astype(np.float64)
        n = np.sqrt(np.einsum("ij,ij->i", a, a))
        out.append((a @ direction) / np.maximum(n, 1e-30))
    return np.concatenate(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_args(parser)
    parser.add_argument("--chunk-rows", type=int, default=4096)
    args = parser.parse_args()

    model_dir = get_model_dir(args.model, args.model_dir, args.config)
    print(f"模型目录: {model_dir}")
    weight_map = load_weight_map(model_dir)
    pe, ph = model_dir / weight_map[EMBED_NAME], model_dir / weight_map[HEAD_NAME]

    rng = np.random.default_rng(0)
    sample_idx = np.sort(rng.choice(VOCAB, N_SAMPLE_ROWS, replace=False))
    print("pass 1: 逐 token 范数/点积/列和 ...")
    r = pass1(pe, EMBED_NAME, ph, HEAD_NAME, args.chunk_rows, sample_idx)

    e_norm = np.sqrt(r["e2"])
    h_norm = np.sqrt(r["h2"])
    pair_cos = r["dot"] / np.maximum(e_norm * h_norm, 1e-30)
    mean_e = r["col_e"] / VOCAB
    mean_h = r["col_h"] / VOCAB
    mean_e_dir = mean_e / np.linalg.norm(mean_e)
    mean_h_dir = mean_h / np.linalg.norm(mean_h)

    print("pass 2: 逐 token 与均值方向的余弦 ...")
    cos_me = cos_to_direction(pe, EMBED_NAME, args.chunk_rows, mean_e_dir)
    cos_mh = cos_to_direction(ph, HEAD_NAME, args.chunk_rows, mean_h_dir)

    # 随机两两余弦基线(采样行)
    X = r["rows_e"]
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    G = X @ X.T
    triu = G[np.triu_indices(len(X), k=1)]

    q = lambda a: {f"p{p}": float(np.percentile(a, p)) for p in QUANTILES}
    vocab = load_vocab(model_dir)

    def extremes(arr: np.ndarray, k: int = 10):
        order = np.argsort(arr)
        return ([(int(i), tok_label(vocab, int(i)), float(arr[i])) for i in order[:k]],
                [(int(i), tok_label(vocab, int(i)), float(arr[i])) for i in order[-k:][::-1]])

    norm_lo, norm_hi = extremes(e_norm)
    cos_lo, cos_hi = extremes(pair_cos)

    summary = {
        "pair_cos": {"quantiles": q(pair_cos), "mean": float(pair_cos.mean())},
        "random_pair_cos_baseline": {
            "quantiles": q(triu), "mean": float(triu.mean()), "std": float(triu.std()),
            "theory_std": 1 / math.sqrt(DIM),
        },
        "anisotropy": {
            "embed_mean_vec_norm": float(np.linalg.norm(mean_e)),
            "embed_mean_norm_over_median_token_norm": float(np.linalg.norm(mean_e) / np.median(e_norm)),
            "lm_head_mean_vec_norm": float(np.linalg.norm(mean_h)),
            "cos_to_mean_embed": q(cos_me),
            "cos_to_mean_lm_head": q(cos_mh),
        },
        "norm_correlation": float(np.corrcoef(e_norm, h_norm)[0, 1]),
        "embed_norm_extremes": {"smallest": norm_lo, "largest": norm_hi},
        "pair_cos_extremes": {"most_negative": cos_lo, "most_positive": cos_hi},
    }

    print(f"\n配对余弦: mean={pair_cos.mean():.4f} 中位={np.median(pair_cos):.4f}"
          f"  [p5={np.percentile(pair_cos,5):.4f}, p95={np.percentile(pair_cos,95):.4f}]")
    print(f"随机基线: mean={triu.mean():.5f} std={triu.std():.5f}(理论 {1/math.sqrt(DIM):.5f})")
    print(f"各向异性: ||mean_e||={np.linalg.norm(mean_e):.4f}"
          f"(中位 token 范数 {np.median(e_norm):.4f} 的 {np.linalg.norm(mean_e)/np.median(e_norm):.1%})"
          f",cos_to_mean 中位={np.median(cos_me):.4f}")
    print(f"范数相关: corr(e_norm, h_norm)={summary['norm_correlation']:.3f}")
    print("\n嵌入范数最小 token:")
    for i, t, v in norm_lo:
        print(f"  {i:>7} {t:<32} norm={v:.4f}")
    print("嵌入范数最大 token:")
    for i, t, v in norm_hi:
        print(f"  {i:>7} {t:<32} norm={v:.4f}")
    print("\n配对余弦最低 token:")
    for i, t, v in cos_lo:
        print(f"  {i:>7} {t:<32} cos={v:.4f}")
    print("配对余弦最高 token:")
    for i, t, v in cos_hi:
        print(f"  {i:>7} {t:<32} cos={v:.4f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_DIR / "embed_lmhead.npz",
             e_norm=e_norm, h_norm=h_norm, pair_cos=pair_cos,
             cos_to_mean_embed=cos_me, cos_to_mean_lm_head=cos_mh)
    with open(OUT_DIR / "embed_lmhead.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes[0, 0].hist(pair_cos, bins=200, density=True, alpha=0.7, label="paired (same token)")
    axes[0, 0].hist(triu, bins=200, density=True, alpha=0.7, label="random pairs")
    axes[0, 0].set_title("cosine: embed row vs lm_head row")
    axes[0, 0].legend()
    axes[0, 1].hist(cos_me, bins=200, density=True, alpha=0.7, label="embed")
    axes[0, 1].hist(cos_mh, bins=200, density=True, alpha=0.7, label="lm_head")
    axes[0, 1].set_title("cosine to mean vector (anisotropy)")
    axes[0, 1].legend()
    sel = rng.choice(VOCAB, 5000, replace=False)
    axes[1, 0].scatter(e_norm[sel], h_norm[sel], s=1, alpha=0.3)
    axes[1, 0].set_xlabel("embed norm")
    axes[1, 0].set_ylabel("lm_head norm")
    axes[1, 0].set_title(f"norm correlation r={summary['norm_correlation']:.3f}")
    axes[1, 1].plot(np.sort(e_norm))
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlabel("token rank")
    axes[1, 1].set_title("embed norm, sorted (log y)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "embed_lmhead.png", dpi=120)
    plt.close(fig)
    print(f"\n输出: {OUT_DIR / 'embed_lmhead.json'}  {FIG_DIR / 'embed_lmhead.png'}")


if __name__ == "__main__":
    main()
