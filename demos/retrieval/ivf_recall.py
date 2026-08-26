"""自写 k-means IVF 索引,实测"召回率 vs 扫描比例"曲线。

以暴力检索结果为 ground truth,在 docs/ 语料的 TF-IDF 向量上验证
docs/retrieval/02-vector-index.md 的核心权衡:nprobe 越大召回越高、扫描越多。
输出 recall 表格,并把曲线画到 output/ivf_recall.png。

用法(在仓库根目录下):
    python demos/retrieval/ivf_recall.py
    python demos/retrieval/ivf_recall.py --clusters 16 --nprobes 1 2 4 8 16
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tfidf_retrieval import build_tfidf, load_chunks

ROOT = Path(__file__).resolve().parents[2]


def kmeans(x: np.ndarray, n_clusters: int, n_iter: int = 20, seed: int = 0):
    """k-means++ 初始化 + Lloyd 迭代(余弦版,输入需已 L2 归一)。返回 (质心, 簇分配)。"""
    rng = np.random.default_rng(seed)
    centroids = [x[rng.integers(len(x))]]
    for _ in range(n_clusters - 1):
        sims = x @ np.array(centroids).T
        d2 = (1.0 - sims.max(axis=1)) ** 2  # 距离 = 1 - 与最近质心的余弦
        d2 /= d2.sum() + 1e-12
        centroids.append(x[rng.choice(len(x), p=d2)])
    centroids = np.array(centroids)

    assign = np.zeros(len(x), dtype=np.int64)
    for _ in range(n_iter):
        assign = (x @ centroids.T).argmax(axis=1)
        for c in range(n_clusters):
            members = x[assign == c]
            if len(members):
                centroids[c] = members.mean(axis=0)
                centroids[c] /= np.linalg.norm(centroids[c]) + 1e-12
    return centroids, assign


def ivf_topk(q, mat, centroids, lists, k: int, nprobe: int):
    """先比质心选出 nprobe 个簇,再只在簇内暴力取 top-k。返回 (top 索引, 扫描数)。"""
    order = (centroids @ q).argsort()[::-1][:nprobe]
    cand = np.concatenate([lists[c] for c in order if len(lists[c])])
    sims = mat[cand] @ q
    top = cand[sims.argsort()[::-1][:k]]
    return set(top.tolist()), len(cand)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clusters", type=int, default=16, help="IVF 簇数(默认 16)")
    parser.add_argument("--nprobes", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--topk", type=int, default=10, help="recall@k 的 k(默认 10)")
    parser.add_argument("--n-queries", type=int, default=64, help="抽样查询数(默认 64)")
    args = parser.parse_args()

    chunks = load_chunks()
    mat, _, _, _ = build_tfidf(chunks)
    n = len(mat)
    print(f"语料: {n} 个 chunk;簇数 {args.clusters}")

    centroids, assign = kmeans(mat, args.clusters)
    lists = [np.where(assign == c)[0] for c in range(args.clusters)]

    # 用语料 chunk 自身当查询(自检索):最近邻必含自己,
    # recall@k 的区分度体现在其余 k-1 个近邻上
    rng = np.random.default_rng(0)
    q_idx = rng.choice(n, size=min(args.n_queries, n), replace=False)
    brute = [set((mat @ mat[i]).argsort()[::-1][: args.topk].tolist()) for i in q_idx]

    rows = []
    for nprobe in args.nprobes:
        recalls, scanned = [], []
        for qi, i in enumerate(q_idx):
            top, n_scan = ivf_topk(mat[i], mat, centroids, lists, args.topk, nprobe)
            recalls.append(len(top & brute[qi]) / args.topk)
            scanned.append(n_scan / n)
        rows.append((nprobe, float(np.mean(recalls)), float(np.mean(scanned))))
        print(f"nprobe={nprobe:3d}  recall@{args.topk}={rows[-1][1]:.3f}  "
              f"扫描比例={rows[-1][2] * 100:.1f}%")

    out_dir = ROOT / "output"
    out_dir.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    xs = [r[2] * 100 for r in rows]
    ys = [r[1] for r in rows]
    ax.plot(xs, ys, "o-")
    for x, y, r in zip(xs, ys, rows):
        ax.annotate(f"nprobe={r[0]}", (x, y), textcoords="offset points", xytext=(6, 6))
    ax.set_xlabel("Scanned fraction (%)")
    ax.set_ylabel(f"Recall@{args.topk}")
    ax.set_title(f"IVF recall vs cost ({args.clusters} clusters, {n} docs)")
    ax.grid(alpha=0.3)
    out_path = out_dir / "ivf_recall.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"曲线已保存: {out_path}")


if __name__ == "__main__":
    main()
