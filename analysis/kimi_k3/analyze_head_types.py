"""基于 qk_spectrum.npz 的 MLA 头分型与个案分析(纯数据分析,不读权重)。

分型依据(docs/qk-spectrum.md §2 对称性探针):
- sym(M) ≈ 0     -> 相似性匹配头(W_Q ≈ W_K,M 近对称半正定)
- sym(M) ≈ sqrt(2) -> 序列匹配头(M 强非对称,如 induction 型)
- sigma 加权的 mean(u_i·v_i) 为辅助:越接近 +1 越偏相似性

产出(output/kimi_k3/):
- head_types.json            : 分型计数(绝对阈值 + 分位阈值)与极端头个案数据
- figures/head_type_scatter.png: sym vs 加权 u·v 散点(按层着色)

用法(在仓库根目录下):
    python analysis/kimi_k3/analyze_head_types.py
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

# docs 理论锚点:相似性头 sym≈0,序列头 sym≈sqrt(2)
ABS_THRESHOLDS = {"偏相似(sym<0.5)": (0.0, 0.5),
                  "混合(0.5~1.2)": (0.5, 1.2),
                  "偏序列(sym>1.2)": (1.2, 2.0)}
TOPK_CASE = 32  # 个案中保留的谱/uv 长度


def case_entry(d: dict, li: int, h: int, why: str) -> dict:
    layers = list(d["layer_indices"])
    return {
        "layer": int(layers[li]), "head": int(h), "why": why,
        "sigma1": round(float(d["metric_sigma1"][li, h]), 4),
        "fro2": round(float(d["metric_fro2"][li, h]), 4),
        "r_eff": round(float(d["metric_r_eff"][li, h]), 4),
        "sym": round(float(d["metric_sym"][li, h]), 4),
        "align_max": round(float(d["metric_align_max"][li, h]), 4),
        "nope_energy_frac": round(float(d["metric_nope_energy_frac"][li, h]), 4),
        "sv_top": [round(float(x), 4) for x in d["singular_values"][li, h, :TOPK_CASE]],
        "uv_top": [round(float(x), 4) for x in d["uv_dots"][li, h, :TOPK_CASE]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "output" / "kimi_k3"),
                        help="qk_spectrum.npz 所在目录(默认 output/kimi_k3)")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    d = dict(np.load(out_dir / "qk_spectrum.npz"))
    layers = list(d["layer_indices"])
    sym, sv, uv = d["metric_sym"], d["singular_values"], d["uv_dots"]
    w_uv = (sv * uv).sum(-1) / sv.sum(-1)  # sigma 加权 mean(u·v)
    L, H = sym.shape

    # ---- 分型计数 ----
    abs_counts = {name: int(((sym >= lo) & (sym < hi)).sum())
                  for name, (lo, hi) in ABS_THRESHOLDS.items()}
    q10, q90 = np.quantile(sym, [0.1, 0.9])
    rel_counts = {
        f"相对相似(sym<p10={q10:.3f})": int((sym < q10).sum()),
        f"相对序列(sym>p90={q90:.3f})": int((sym > q90).sum()),
    }
    per_layer_type = {
        int(layers[li]): {
            "sym_median": round(float(np.median(sym[li])), 4),
            "sym_min": round(float(sym[li].min()), 4),
            "w_uv_median": round(float(np.median(w_uv[li])), 4),
            "most_similar_head": int(sym[li].argmin()),
            "most_sequence_head": int(sym[li].argmax()),
        }
        for li in range(L)
    }

    # ---- 个案:最相似 3 头、最序列 3 头、sigma1 最大、r_eff 最小、w_uv 最大 ----
    flat_sym = sym.ravel()
    cases = []
    for rank in flat_sym.argsort()[:3]:
        cases.append(case_entry(d, *np.unravel_index(rank, sym.shape),
                                why="sym 最低(最接近相似性头)"))
    for rank in flat_sym.argsort()[-3:][::-1]:
        cases.append(case_entry(d, *np.unravel_index(rank, sym.shape),
                                why="sym 最高(最偏序列头)"))
    cases.append(case_entry(d, *np.unravel_index(d["metric_sigma1"].argmax(), sym.shape),
                            why="sigma1 全局最大"))
    cases.append(case_entry(d, *np.unravel_index(d["metric_r_eff"].argmin(), sym.shape),
                            why="r_eff 全局最小(谱最集中)"))
    cases.append(case_entry(d, *np.unravel_index(w_uv.argmax(), sym.shape),
                            why="加权 u·v 最大(方向对齐最强)"))

    doc = {
        "note": "分型锚点见 docs/qk-spectrum.md §2:相似性头 sym≈0,序列头 sym≈sqrt(2)≈1.414",
        "total_heads": int(L * H),
        "sym_quantiles": {str(q): round(float(v), 4) for q, v in
                          zip([0, 0.1, 0.5, 0.9, 1.0], np.quantile(sym, [0, .1, .5, .9, 1]))},
        "w_uv_quantiles": {str(q): round(float(v), 4) for q, v in
                           zip([0, 0.1, 0.5, 0.9, 1.0], np.quantile(w_uv, [0, .1, .5, .9, 1]))},
        "abs_type_counts": abs_counts,
        "rel_type_counts": rel_counts,
        "per_layer": per_layer_type,
        "case_studies": cases,
    }
    json_path = out_dir / "head_types.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    print(f"已写出: {json_path}")

    # ---- 散点图 ----
    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0, 1, L))
    for li in range(L):
        ax.scatter(sym[li], w_uv[li], s=10, alpha=0.7, color=colors[li],
                   label=f"{layers[li]}")
    ax.axvline(np.sqrt(2), color="gray", linestyle="--", linewidth=1,
               label="sqrt(2)(完全非对称)")
    ax.set_xlabel("sym(M)")
    ax.set_ylabel(r"$\sigma$ 加权 mean($u_i \cdot v_i$)")
    ax.set_title("MLA 头分型散点(颜色=层,图例为层号)")
    ax.legend(fontsize=6, ncol=4, markerscale=1.5)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig_path = fig_dir / "head_type_scatter.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"已写出: {fig_path}")

    print("\n== 分型结论 ==")
    print(f"sym 分位数: {doc['sym_quantiles']}")
    print(f"绝对阈值计数: {abs_counts}")
    print(f"相对阈值计数: {rel_counts}")


if __name__ == "__main__":
    main()
