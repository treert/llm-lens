"""MLA 共享 K 潜库的指纹分析(docs/qk-spectrum.md §6.3 的 MLA 版)。

GQA 中同组 Q 头共享一个 K 头;MLA 中 96 头共享同一个 512+64 维 K 潜空间
(kv_a 压缩段 + rot 段)。可检验指纹:同一层内各头 M 的右奇异子空间
(被迫住在同一潜空间内)的两两重叠是否显著高于随机。

做法:逐头取 M 的 top-k 右奇异向量,用潜空间的正交基坐标表示
(V 的潜空间坐标 = R_A @ B_h^T @ U_K @ Sigma_K^-1 @ svd_Q,推导见代码注释),
层内 4560 对头两两算主视角余弦;与随机 k 维子空间的基线对比。

产出(output/kimi_k3/):
- shared_latent_overlap.json            : 逐层重叠分布统计、最重叠头对、潜空间占用有效秩
- figures/shared_latent_overlap.png     : 重叠分布直方图(代表层)+ 层均值曲线 vs 随机基线

用法(在仓库根目录下):
    python analysis/kimi_k3/analyze_shared_latent.py            # 全量 24 层
    python analysis/kimi_k3/analyze_shared_latent.py --layers 3 75 --topk 32
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from llm_lens import get_model_dir
from llm_lens.cli import add_model_args
from mla_common import (
    KV_LORA_RANK, NUM_HEADS, QK_ROT_DIM,
    head_grams, head_qk_slices, head_spectrum, load_mla_layer, prepare_latents,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

LATENT_DIM = KV_LORA_RANK + QK_ROT_DIM  # 576


def head_right_subspace(lat, w, h, R_A, topk):
    """第 h 头 M 的 top-k 右奇异向量在潜空间正交基下的坐标 (576×k,列正交)。

    W_Kh = B_h @ A,其中 A = [kv_a; a_rot] (576×7168),B_h = [[kb, 0],[0, I_64]]。
    对 A^T = Q_A R_A (QR),有 W_Kh = B_h R_A^T Q_A^T,故 W_Kh 的右奇异向量
    V_K = Q_A (R_A B_h^T U_K Sigma_K^-1),括号内即潜空间坐标。
    M 的右奇异向量 = V_K @ svd_Q(docs §2:M = V_Q S V_K^T,S = P Sigma Q^T)。
    """
    qb, kb = head_qk_slices(w, h)
    G_Q, G_K, G_QK = head_grams(lat, qb, kb)
    res = head_spectrum(G_Q, G_K, G_QK)
    B_T = np.zeros((LATENT_DIM, kb.shape[0] + QK_ROT_DIM))
    B_T[:KV_LORA_RANK, :kb.shape[0]] = kb.T
    B_T[KV_LORA_RANK:, kb.shape[0]:] = np.eye(QK_ROT_DIM)
    C_K = R_A @ B_T @ (res["U_K"] / res["s_K"])  # 576×192
    V_M = C_K @ res["svd_Q"]                     # 576×192,列正交
    return V_M[:, :topk], res["sv"], float(
        np.abs(V_M.T @ V_M - np.eye(V_M.shape[1])).max())


def pairwise_overlaps(subs: list[np.ndarray]):
    """返回层内所有头对的 (平均平方余弦/top-k, 最大主视角余弦, 头对列表)。"""
    k = subs[0].shape[1]
    n = len(subs)
    mean_ov, max_cos, pairs = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            cos = np.linalg.svd(subs[i].T @ subs[j], compute_uv=False)
            mean_ov.append(float(cos @ cos / k))
            max_cos.append(float(cos[0]))
            pairs.append((i, j))
    return np.array(mean_ov), np.array(max_cos), pairs


def random_baseline(n_dim: int, k: int, n_pairs: int, seed: int = 0):
    """随机 k 维子空间两两重叠的 Monte Carlo 基线。"""
    rng = np.random.default_rng(seed)
    mean_ov = np.zeros(n_pairs)
    for i in range(n_pairs):
        A = np.linalg.qr(rng.standard_normal((n_dim, k)))[0]
        B = np.linalg.qr(rng.standard_normal((n_dim, k)))[0]
        cos = np.linalg.svd(A.T @ B, compute_uv=False)
        mean_ov[i] = cos @ cos / k
    return mean_ov


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_args(parser)
    parser.add_argument("--layers", type=int, nargs="*", default=None,
                        help="只处理指定的 0-based 层号(默认全部 MLA 层)")
    parser.add_argument("--topk", type=int, default=32,
                        help="每头取 M 的前 k 个右奇异方向(默认 32)")
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "output" / "kimi_k3"),
                        help="输出目录(默认 output/kimi_k3)")
    args = parser.parse_args()

    model_dir = get_model_dir(args.model, args.model_dir, args.config)
    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "layers.json", "r", encoding="utf-8") as f:
        layers_doc = json.load(f)
    mla_layers = [l["index"] for l in layers_doc["layers"] if l["attn_type"] == "mla"]
    if args.layers is not None:
        bad = sorted(set(args.layers) - set(mla_layers))
        if bad:
            raise ValueError(f"指定的层不是 MLA 层: {bad};MLA 层为 {mla_layers}")
        mla_layers = sorted(args.layers)
    print(f"MLA 层(0-based): {mla_layers}, topk={args.topk}")

    with open(model_dir / "model.safetensors.index.json", "r", encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]

    k = args.topk
    baseline = random_baseline(LATENT_DIM, k, 2000)
    print(f"随机基线(期望 k/n={k / LATENT_DIM:.4f}): "
          f"中位 {np.median(baseline):.4f}, p99 {np.quantile(baseline, 0.99):.4f}")

    per_layer = []
    curve_x, curve_mean, curve_p90 = [], [], []
    hist_data = {}
    t0 = time.time()
    for idx in mla_layers:
        w = load_mla_layer(model_dir, weight_map, idx)
        lat = prepare_latents(w)
        # 潜空间正交化:A^T = Q_A R_A,取 576×576 上三角 R_A
        _, R_A = np.linalg.qr(
            np.vstack([lat["kv_a"], lat["a_rot"]]).T, mode="reduced")

        subs, svs = [], []
        for h in range(NUM_HEADS):
            U_h, sv_h, orth_err = head_right_subspace(lat, w, h, R_A, k)
            if h == 0:
                print(f"[验证] 层 {idx} 头 0: 右奇异向量列正交性误差 = {orth_err:.3e}")
            subs.append(U_h)
            svs.append(sv_h)
        mean_ov, max_cos, pairs = pairwise_overlaps(subs)

        # 潜空间占用:所有头 top-k 方向堆叠后的有效秩
        stacked = np.hstack(subs)  # 576×(96k)
        stack_sv = np.linalg.svd(stacked, compute_uv=False)
        usage_r_eff = float(stack_sv.sum() ** 2 / (stack_sv**2).sum())

        # 最重叠的 3 对头
        top_pairs = sorted(
            ((float(mean_ov[p]), pairs[p][0], pairs[p][1])
             for p in range(len(pairs))), reverse=True)[:3]

        rec = {
            "layer": idx,
            "overlap_mean": round(float(mean_ov.mean()), 4),
            "overlap_p10": round(float(np.quantile(mean_ov, 0.1)), 4),
            "overlap_p50": round(float(np.quantile(mean_ov, 0.5)), 4),
            "overlap_p90": round(float(np.quantile(mean_ov, 0.9)), 4),
            "overlap_max": round(float(mean_ov.max()), 4),
            "max_cos_median": round(float(np.median(max_cos)), 4),
            "overlap_over_baseline": round(float(mean_ov.mean() / np.median(baseline)), 2),
            "usage_r_eff": round(usage_r_eff, 1),
            "usage_r_eff_over_dim": round(usage_r_eff / LATENT_DIM, 3),
            "top_overlapping_pairs": [
                {"head_i": i, "head_j": j, "overlap": round(o, 4)}
                for o, i, j in top_pairs],
        }
        per_layer.append(rec)
        curve_x.append(idx)
        curve_mean.append(rec["overlap_mean"])
        curve_p90.append(rec["overlap_p90"])
        if idx in (3, 43, 75, 92):
            hist_data[idx] = mean_ov
        print(f"层 {idx} 完成 ({time.time() - t0:.1f}s): "
              f"重叠均值 {rec['overlap_mean']:.4f} (基线 {np.median(baseline):.4f}, "
              f"{rec['overlap_over_baseline']}x), 占用有效秩 {rec['usage_r_eff']}")

    doc = {
        "note": ("层内 96 头两两(C(96,2)=4560 对)top-k 右奇异子空间的平均平方余弦/k;"
                 "随机基线 = 576 维空间中随机 k 维子空间;usage_r_eff = 全部头的 top-k "
                 "方向堆叠后的有效秩(上限 576)"),
        "topk": k,
        "latent_dim": LATENT_DIM,
        "random_baseline": {
            "expected_k_over_n": round(k / LATENT_DIM, 4),
            "median": round(float(np.median(baseline)), 4),
            "p99": round(float(np.quantile(baseline, 0.99)), 4),
        },
        "per_layer": per_layer,
    }
    json_path = out_dir / "shared_latent_overlap.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    print(f"\n已写出: {json_path}")

    # ---- 图:左 = 代表层重叠分布直方图;右 = 层均值曲线 vs 基线 ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    bins = np.linspace(0, max(curve_p90) * 1.2, 60)
    axes[0].hist(baseline, bins=bins, density=True, alpha=0.5, color="gray",
                 label="随机基线")
    for layer, ov in hist_data.items():
        axes[0].hist(ov, bins=bins, density=True, histtype="step",
                     linewidth=1.5, label=f"层 {layer}")
    axes[0].set_xlabel("头对重叠(平均平方余弦 / k)")
    axes[0].set_ylabel("密度")
    axes[0].set_title("层内头对重叠分布 vs 随机基线")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(curve_x, curve_mean, "o-", label="层内重叠均值")
    axes[1].plot(curve_x, curve_p90, "s-", markersize=4, alpha=0.7, label="p90")
    axes[1].axhline(np.median(baseline), color="gray", linestyle="--",
                    label="随机基线中位")
    axes[1].set_xlabel("层")
    axes[1].set_ylabel("重叠")
    axes[1].set_title("MLA 共享潜库重叠强度随层变化")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig_path = fig_dir / "shared_latent_overlap.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"已写出: {fig_path}")


if __name__ == "__main__":
    main()
