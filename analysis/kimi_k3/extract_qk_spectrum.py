"""提取 Kimi-K3 全部 MLA 层的 QK 谱数据(静态权重分析,不跑模型)。

对 24 个 MLA 层逐头计算 M_h = W_Qh^T W_Kh 的奇异值谱与派生指标。
方法见 docs/qk-spectrum.md §2/§6.1 与本目录 mla_common.py 的 docstring。

输出(output/kimi_k3/):
- qk_spectrum.npz         : 逐层逐头奇异值谱 {sigma_i}、{u_i·v_i}、标量指标、潜空间谱
- qk_spectrum_summary.json: 逐层跨头聚合统计,便于快速查看

用法(在仓库根目录下):
    python analysis/kimi_k3/extract_qk_spectrum.py              # 全量 24 层
    python analysis/kimi_k3/extract_qk_spectrum.py --layers 3 7 # 只跑指定层(0-based,调试用)
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from llm_lens import get_model_dir
from llm_lens.cli import add_model_args
from mla_common import (
    KV_LORA_RANK, NUM_HEADS, Q_HEAD_DIM, Q_LORA_RANK, QK_NOPE_DIM,
    head_grams, head_qk_slices, head_spectrum, load_mla_layer, prepare_latents,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def verify_first_head(w: dict, sv_gram: np.ndarray) -> float:
    """小步验证:物化第 0 头的 W_Q/W_K,用 docs §2 直接法重算谱,返回与 Gram 法的最大偏差。"""
    q_a_g = w["q_a_norm"][:, None] * w["q_a"]
    kv_a_g = w["kv_a_norm"][:, None] * w["kv_a"][:KV_LORA_RANK]
    a_rot = w["kv_a"][KV_LORA_RANK:]
    W_Q = w["q_b"][:Q_HEAD_DIM] @ q_a_g                         # 192×7168
    W_K = np.vstack([w["kv_b"][:QK_NOPE_DIM] @ kv_a_g, a_rot])  # 192×7168
    Uq, sq, _ = np.linalg.svd(W_Q, full_matrices=False)
    Uk, sk, _ = np.linalg.svd(W_K, full_matrices=False)
    sv_direct = np.linalg.svd(
        (sq[:, None] * (Uq.T @ Uk)) * sk[None, :], compute_uv=False
    )
    return float(np.abs(sv_gram - sv_direct[: len(sv_gram)]).max())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_args(parser)
    parser.add_argument("--layers", type=int, nargs="*", default=None,
                        help="只处理指定的 0-based 层号(默认全部 MLA 层)")
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "output" / "kimi_k3"),
                        help="输出目录(默认 output/kimi_k3)")
    args = parser.parse_args()

    model_dir = get_model_dir(args.model, args.model_dir, args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "layers.json", "r", encoding="utf-8") as f:
        layers_doc = json.load(f)
    mla_layers = [l["index"] for l in layers_doc["layers"] if l["attn_type"] == "mla"]
    if args.layers is not None:
        bad = sorted(set(args.layers) - set(mla_layers))
        if bad:
            raise ValueError(f"指定的层不是 MLA 层: {bad};MLA 层为 {mla_layers}")
        mla_layers = sorted(args.layers)
    print(f"MLA 层(0-based): {mla_layers}")

    with open(model_dir / "model.safetensors.index.json", "r", encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]

    L, H = len(mla_layers), NUM_HEADS
    sv_all = np.zeros((L, H, Q_HEAD_DIM), dtype=np.float32)
    uv_all = np.zeros((L, H, Q_HEAD_DIM), dtype=np.float32)
    metric_names = ["sigma1", "fro2", "r_eff", "sym", "align_max",
                    "align_frac_05", "align_frac_01", "align_r_eff", "nope_energy_frac"]
    metrics = {k: np.zeros((L, H), dtype=np.float64) for k in metric_names}
    latent_sv_q = np.zeros((L, Q_LORA_RANK), dtype=np.float32)
    latent_sv_k = np.zeros((L, KV_LORA_RANK), dtype=np.float32)

    t0 = time.time()
    for li, idx in enumerate(mla_layers):
        w = load_mla_layer(model_dir, weight_map, idx)
        lat = prepare_latents(w)
        latent_sv_q[li] = np.sqrt(np.linalg.eigvalsh(lat["G_qq"])[::-1])
        latent_sv_k[li] = np.sqrt(np.linalg.eigvalsh(lat["G_kk"])[::-1])

        for h in range(NUM_HEADS):
            qb, kb = head_qk_slices(w, h)
            G_Q, G_K, G_QK = head_grams(lat, qb, kb)
            res = head_spectrum(G_Q, G_K, G_QK)
            sv_all[li, h, : len(res["sv"])] = res["sv"]
            uv_all[li, h, : len(res["uv"])] = res["uv"]
            for k in metric_names[:-1]:
                metrics[k][li, h] = res[k]
            # nope-only M 的能量占比: tr(G_Q_nn G_K_nn) / tr(G_Q G_K)
            qn = qb[:QK_NOPE_DIM]
            G_Q_nn = qn @ lat["G_qq"] @ qn.T
            metrics["nope_energy_frac"][li, h] = (
                np.trace(G_Q_nn @ (kb @ lat["G_kk"] @ kb.T)) / np.trace(G_Q @ G_K)
            )

            if li == 0 and h == 0:  # 小步验证:直接法交叉核对
                diff = verify_first_head(w, res["sv"])
                print(f"[验证] 层 {idx} 头 0: Gram 法 vs 直接法 奇异值最大偏差 = {diff:.3e}")
                print(f"[验证] 头 0 谱: top5={np.round(res['sv'][:5], 3)}, "
                      f"fro2={res['fro2']:.2f}, r_eff={res['r_eff']:.1f}, sym={res['sym']:.3f}")

        print(f"层 {idx} 完成 ({time.time() - t0:.1f}s): "
              f"sigma1 中位 {np.median(metrics['sigma1'][li]):.2f}, "
              f"r_eff 中位 {np.median(metrics['r_eff'][li]):.1f}")

    npz_path = out_dir / "qk_spectrum.npz"
    np.savez(
        npz_path,
        layer_indices=np.array(mla_layers, dtype=np.int32),
        singular_values=sv_all, uv_dots=uv_all,
        latent_sv_q=latent_sv_q, latent_sv_k=latent_sv_k,
        **{f"metric_{k}": v for k, v in metrics.items()},
    )
    print(f"\n已写出: {npz_path} ({npz_path.stat().st_size / 2**20:.1f} MiB)")

    summary = {
        "method": (
            "M_h = W_Qh^T W_Kh 的奇异值谱;潜空间 Gram 复用,不构造 D×D 矩阵;"
            "q_a/kv_a 的 RMSNorm 增益已折入;NoPE(rot 段无旋转,K 侧 rot 全头共享)"
        ),
        "mla_layers": mla_layers,
        "num_heads": NUM_HEADS,
        "head_dim": Q_HEAD_DIM,
        "per_layer": [
            {
                "layer": idx,
                **{f"{k}_{stat}": round(float(v), 4)
                   for k in metric_names
                   for stat, v in {
                       "median": np.median(metrics[k][li]),
                       "mean": metrics[k][li].mean(),
                       "min": metrics[k][li].min(),
                       "max": metrics[k][li].max(),
                   }.items()},
                "latent_q_r_eff": round(float(
                    latent_sv_q[li].sum() ** 2 / (latent_sv_q[li] ** 2).sum()), 2),
                "latent_k_r_eff": round(float(
                    latent_sv_k[li].sum() ** 2 / (latent_sv_k[li] ** 2).sum()), 2),
            }
            for li, idx in enumerate(mla_layers)
        ],
    }
    json_path = out_dir / "qk_spectrum_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"已写出: {json_path} ({json_path.stat().st_size / 1024:.1f} KiB)")
    print(f"总耗时: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
