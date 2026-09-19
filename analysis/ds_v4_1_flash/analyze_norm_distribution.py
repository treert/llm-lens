"""分析 DeepSeek-V4.1-Flash 各 norm 增益向量的**单层分布形状**(概率密度角度)。

增益向量初始化全 1,训练后主体"极平"、只有少数通道突出。本脚本不看均值大小,
而看**相对偏差的分布形态**:delta = g/median(|g|) - 1(无量纲),逐层用 IQR 尺子
σ_l = IQR/1.349 标准化(逐层是为了剔除层间主体宽度差;IQR 比 MAD 更耐受 bf16 并列值)。

回答三个问题:
1. 形状:主体是高斯吗?偏度/超额峰度多少?——用截尾(|z|<4)统计量,避免尖峰污染矩;
2. 尾部:超出 k 倍 σ 的实际占比 vs 高斯理论值——检验"高斯主体 + 稀疏尖峰"是否成立;
3. 分辨率:单层向量实际用到多少个 bf16 可表示值、主体宽度相当于几个量化步
   (bf16 一个 binade 内相对步长 ≈ 2^-8,即 0.39%),这决定了通道画像的数值分辨率。

输出:
- output/ds_v4_1_flash/weight_moments/norm_distribution.json:逐层逐类型指标;
- output/ds_v4_1_flash/figures/norm_distribution.png:六联图。

用法(在仓库根目录下):
    python analysis/ds_v4_1_flash/analyze_norm_distribution.py
"""

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_weight_moments import FIG_DIR, OUT_DIR  # noqa: E402
from ds_common import load_weight_map  # noqa: E402
from llm_lens import get_model_dir, read_tensor  # noqa: E402
from llm_lens.cli import add_model_args  # noqa: E402

LS = list(range(40))
BF16_STEP = 2.0**-8  # bf16(7 位尾数)一个 binade 内的相对步长;高端取 2^-8

TYPES = {
    "attn_norm": (f"layers.{{}}.attn_norm.weight", "#4f6ef7"),
    "ffn_norm": (f"layers.{{}}.ffn_norm.weight", "#0ea47a"),
    "q_norm": (f"layers.{{}}.attn.q_norm.weight", "#e08a1e"),
    "kv_norm": (f"layers.{{}}.attn.kv_norm.weight", "#db2777"),
}
DIMS = {"attn_norm": 5120, "ffn_norm": 5120, "q_norm": 1280, "kv_norm": 512}


def robust_sigma(d: np.ndarray) -> float:
    """IQR/1.349;并列值过多导致为 0 时退回 MAD,再退回 bf16 步长。"""
    q1, q3 = np.percentile(d, [25, 75])
    s = (q3 - q1) / 1.349
    if s <= 0:
        s = 1.4826 * np.median(np.abs(d - np.median(d)))
    return max(float(s), BF16_STEP / 4)


def pr_of(v: np.ndarray) -> float:
    """有效维数 PR =(Σz²)²/Σz⁴(z 为去均值标准化后的向量)= N²/Σz⁴。"""
    z = v - v.mean()
    z = z / z.std()
    return float((z**2).sum() ** 2 / (z**4).sum())


def round_mantissa(x: np.ndarray, keep: int) -> np.ndarray:
    """把浮点值截到 keep 位尾数(bf16 原生 7 位;keep=7 即原值不变)。"""
    e = np.floor(np.log2(np.abs(x)))
    step = np.exp2(e - keep)
    return np.round(x / step) * step


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def occupancy_vs_gaussian(g: np.ndarray, sigma_abs: float, n_sigma: float = 3.0) -> tuple:
    """主体(±n_sigma)内每个 bf16 能级的实际占用 vs 高斯期望,返回 (占用, 期望, z)。

    用来检验"主体是否只是被离散化的平滑密度":若中位 实际/期望 ≈ 1,则没有隐藏的离散结构。
    """
    med = np.median(np.abs(g))
    d = g / med - 1.0
    u, c = np.unique(g, return_counts=True)
    z = (u / med - 1.0) / sigma_abs
    keep = np.abs(z) < n_sigma
    u, c, z = u[keep], c[keep], z[keep]
    dz = np.gradient(z)
    exp = np.exp(-z**2 / 2) * np.abs(dz)
    exp *= c.sum() / exp.sum()
    return c.astype(float), exp, u


def pr_sweep(g: np.ndarray, bits: tuple[int, ...] = (7, 6, 5, 4, 3)) -> list[float]:
    return [pr_of(round_mantissa(g, p)) for p in bits]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_args(parser)
    args = parser.parse_args()
    model_dir = get_model_dir(args.model, args.model_dir, args.config)
    print(f"模型目录: {model_dir}")
    wm = load_weight_map(model_dir)

    stats: dict[str, dict] = {}
    zs: dict[str, np.ndarray] = {}
    per_layer: dict[str, dict[int, dict]] = {}
    for tag, (fmt, _) in TYPES.items():
        Z, rows = [], {}
        for l in LS:
            n = fmt.format(l)
            g = np.asarray(read_tensor(model_dir / wm[n], n, dtype=np.float64)).reshape(-1)
            d = g / np.median(np.abs(g)) - 1.0
            s = robust_sigma(d)
            Z.append(d / s)
            _, cnt = np.unique(g, return_counts=True)
            med = np.median(g)
            rows[l] = {
                "sigma": s,
                "sigma_over_bf16_step": s / BF16_STEP,
                "n_unique": int(np.unique(g).size),
                "mode_share": float(cnt.max() / g.size),
                "pr": pr_of(g),
                "n_below_half": int((g < 0.5 * med).sum()),
                "n_below_tenth": int((g < 0.1 * med).sum()),
                "n_above_2x": int((g > 2.0 * med).sum()),
                "rounding_noise_over_sigma": float((BF16_STEP / np.sqrt(12)) / s),
                "raw_skew": float(((g - g.mean()) ** 3).mean() / g.std() ** 3),
                "raw_kurt": float(((g - g.mean()) ** 4).mean() / g.std() ** 4 - 3),
            }
        Z = np.concatenate(Z)
        zs[tag] = Z
        per_layer[tag] = rows
        t = Z[np.abs(Z) < 4]
        stats[tag] = {
            "n_tensors": len(LS),
            "n_values": int(Z.size),
            "sigma_median": float(np.median([r["sigma"] for r in rows.values()])),
            "sigma_over_step_median": float(np.median([r["sigma_over_bf16_step"] for r in rows.values()])),
            "sigma_over_step_min": float(min(r["sigma_over_bf16_step"] for r in rows.values())),
            "sigma_over_step_max": float(max(r["sigma_over_bf16_step"] for r in rows.values())),
            "bulk_skew": float(((t - t.mean()) ** 3).mean() / t.std() ** 3),
            "bulk_excess_kurt": float(((t - t.mean()) ** 4).mean() / t.var() ** 2 - 3),
            "tail_gt3": float((np.abs(Z) > 3).mean()),
            "tail_gt5": float((np.abs(Z) > 5).mean()),
            "tail_gt12": float((np.abs(Z) > 12).mean()),
            "max_abs_z": float(np.abs(Z).max()),
            "n_unique_median": int(np.median([r["n_unique"] for r in rows.values()])),
            "n_unique_min": int(min(r["n_unique"] for r in rows.values())),
            "mode_share_median": float(np.median([r["mode_share"] for r in rows.values()])),
            "mode_share_max": float(max(r["mode_share"] for r in rows.values())),
        }

    print("\n[形状]逐层 IQR 标准化后的主体统计(截尾 |z|<4;高斯 = 0)")
    print(f"  {'类型':10s} {'σ_l 中位':>9s} {'σ/bf16步':>9s} {'主体偏度':>9s} {'主体超额峰度':>12s}")
    for tag, s in stats.items():
        print(f"  {tag:10s} {s['sigma_median']:9.4f} {s['sigma_over_step_median']:9.1f} "
              f"{s['bulk_skew']:9.2f} {s['bulk_excess_kurt']:12.2f}")

    print("\n[尾部]超出 k 倍 σ_l 的占比(高斯:|z|>3 = 2.70e-3、>5 = 5.7e-7、>12 = 2e-33)")
    print(f"  {'类型':10s} {'|z|>3':>10s} {'|z|>5':>10s} {'|z|>12':>10s} {'最大|z|':>9s}")
    for tag, s in stats.items():
        print(f"  {tag:10s} {s['tail_gt3']:10.2e} {s['tail_gt5']:10.2e} "
              f"{s['tail_gt12']:10.2e} {s['max_abs_z']:9.1f}")

    print("\n[分辨率]单层向量用到的 bf16 可表示值")
    print(f"  {'类型':10s} {'唯一值中位':>10s} {'最少':>6s} {'众数占比中位':>12s} {'最大':>8s}")
    for tag, s in stats.items():
        print(f"  {tag:10s} {s['n_unique_median']:10d} {s['n_unique_min']:6d} "
              f"{s['mode_share_median']:12.2%} {s['mode_share_max']:8.2%}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------- 分辨率检验 ----------------
    # 问题:bf16 的量化粒度(主体只有 3.6~16.7 个量化步宽)是否限制了"看到结构"的能力?
    # 三道检验:跨类型对比、类型内秩相关、精度敏感性(把 bf16 再截粗,看指标变不变)。
    coef = {}
    print("\n[分辨率 vs 结构]跨类型(若分辨率受限,分辨率高的类型应显出更多结构)")
    print(f"  {'类型':10s} {'σ/bf16步':>9s} {'PR 中位':>9s} {'ρ(σ/步, PR)':>13s}")
    for tag, s in stats.items():
        pr = np.array([per_layer[tag][l]["pr"] for l in LS])
        sg = np.array([per_layer[tag][l]["sigma_over_bf16_step"] for l in LS])
        coef[tag] = spearman(sg, pr)
        print(f"  {tag:10s} {s['sigma_over_step_median']:9.1f} {np.median(pr):9.1f} "
              f"{coef[tag]:13.3f}")

    print("\n[精度敏感性]把 bf16(7 位尾数)继续截粗,PR 怎么变(每类型取 σ/步 最小/最大的层)")
    sweep: dict[str, list[float]] = {}
    print(f"  {'层':28s} " + " ".join(f"{'p=%d' % p:>8s}" for p in (7, 6, 5, 4, 3))
          + f" {'σ/步':>7s} {'@4/@7':>7s}")
    for tag, (fmt, _) in TYPES.items():
        sg = {l: per_layer[tag][l]["sigma_over_bf16_step"] for l in LS}
        for l in (min(sg, key=sg.get), max(sg, key=sg.get)):
            g = np.asarray(read_tensor(model_dir / wm[fmt.format(l)], fmt.format(l),
                                       dtype=np.float64)).reshape(-1)
            vals = pr_sweep(g)
            key = f"{tag} L{l}"
            sweep[key] = vals
            print(f"  {key + f'(σ/步 {sg[l]:.1f})':28s} " + " ".join(f"{v:8.1f}" for v in vals)
                  + f" {sg[l]:7.1f} {vals[3] / vals[0]:7.2f}")

    print("\n[能级占用 vs 高斯期望]主体是否只是被离散化的平滑密度(中位≈1 即无隐藏离散结构)")
    occ: dict[str, tuple] = {}
    for tag, l in (("q_norm", 20), ("attn_norm", 0), ("ffn_norm", 20)):
        fmt = TYPES[tag][0]
        g = np.asarray(read_tensor(model_dir / wm[fmt.format(l)], fmt.format(l),
                                   dtype=np.float64)).reshape(-1)
        c, exp, _ = occupancy_vs_gaussian(g, per_layer[tag][l]["sigma"])
        occ[f"{tag} L{l}"] = (c, exp)
        ratio = c / np.maximum(exp, 1e-9)
        print(f"  {tag} L{l:2d}: 主体能级={len(c):3d} 维数={int(c.sum()):5d} "
              f"实际/期望 中位={np.median(ratio):.3f} p95={np.percentile(ratio, 95):.2f} "
              f"最大={ratio.max():.2f}")

    print("\n[两侧极端维]以同层中位为基准(40 层合计;1280/512 维类型只看自身口径)")
    for tag in TYPES:
        lo = sum(per_layer[tag][l]["n_below_half"] for l in LS)
        lo1 = sum(per_layer[tag][l]["n_below_tenth"] for l in LS)
        hi = sum(per_layer[tag][l]["n_above_2x"] for l in LS)
        prs = {l: per_layer[tag][l]["pr"] for l in LS}
        lmin = min(prs, key=prs.get)
        print(f"  {tag:10s} <0.5×中位={lo:4d}  <0.1×中位={lo1:3d}  >2×中位={hi:4d}   "
              f"PR 最小 L{lmin}={prs[lmin]:8.1f},最大={max(prs.values()):7.0f}"
              f"(白噪声基线 {DIMS[tag] / 3:.0f})")

    print("\n[PR 逐层]attn_norm(白噪声基线 1707;10 层一行)")
    for i in range(0, 40, 10):
        print("  " + " ".join(f"L{l:<2}:{per_layer['attn_norm'][l]['pr']:>6.0f}" for l in LS[i:i + 10]))

    # 合成标定:固定 bf16 网格(7 位尾数),扫描"真值 σ/步",量出 PR 被网格抬高的倍率。
    # 这是"分辨率检验"的定量骨架:σ/步 ≳ 5 时偏差 ≤1%,≲1 时才出现并列值造成的抬升。
    rng = np.random.default_rng(0)
    calib = {}
    print("\n[合成标定]固定 bf16 网格,PR 被网格抬高的倍率(纯高斯 slab / 外加 2 个 30σ 尖峰)")
    print(f"  {'真值 σ/步':>9s} {'PR_true':>9s} {'PR_bf16':>9s} {'倍率':>7s} {'带尖峰倍率':>10s}")
    for target in (1.3, 2.6, 4.3, 5.1, 10.0, 20.0, 41.0):
        pt, pm, pt2, pm2 = [], [], [], []
        for _ in range(5):
            tau = rng.standard_normal(5120)
            sig = target * BF16_STEP
            gt = 0.02 * (1 + sig * tau)
            pt.append(pr_of(gt))
            pm.append(pr_of(round_mantissa(gt, 7)))
            t2 = tau.copy()
            t2[rng.choice(5120, 2, replace=False)] += 30.0
            g2 = 0.02 * (1 + sig * t2)
            pt2.append(pr_of(g2))
            pm2.append(pr_of(round_mantissa(g2, 7)))
        calib[f"{target}"] = {"pr_true": float(np.mean(pt)), "pr_bf16": float(np.mean(pm)),
                              "ratio": float(np.mean(pm) / np.mean(pt)),
                              "ratio_with_spikes": float(np.mean(pm2) / np.mean(pt2))}
        print(f"  {target:9.1f} {np.mean(pt):9.0f} {np.mean(pm):9.0f} "
              f"{calib[f'{target}']['ratio']:7.3f} {calib[f'{target}']['ratio_with_spikes']:10.3f}")

    with open(OUT_DIR / "norm_distribution.json", "w", encoding="utf-8") as f:
        json.dump({"stats": stats,
                   "per_layer": {t: {str(l): r for l, r in d.items()} for t, d in per_layer.items()},
                   "resolution_spearman_sigma_vs_pr": coef,
                   "precision_sweep_bits_7_to_3": sweep,
                   "synthetic_calibration_pr_bias": calib,
                   "occupancy_default": {k: {"level": int(v[0].size),
                                             "observed": v[0].tolist()
                                             if v[0].size <= 200 else None,
                                             "expected": v[1].tolist()
                                             if v[1].size <= 200 else None} for k, v in occ.items()}},
                  f, ensure_ascii=False)

    # ---------------- 图 ----------------
    fig, axes = plt.subplots(3, 3, figsize=(18, 14.5))
    grid = np.linspace(-6, 6, 241)

    for tag, (_, color) in TYPES.items():
        axes[0, 0].hist(zs[tag][np.abs(zs[tag]) < 6], bins=120, range=(-6, 6), density=True,
                        histtype="step", lw=1.4, color=color, label=tag)
        ks = np.arange(1, 16)
        surv = np.maximum([(np.abs(zs[tag]) > k).mean() for k in ks], 1e-8)
        axes[0, 1].semilogy(ks, surv, marker="o", ms=3, color=color, label=tag)
        axes[0, 2].semilogy(LS, [per_layer[tag][l]["sigma_over_bf16_step"] for l in LS],
                            marker="o", ms=3, color=color, label=tag)
        axes[1, 0].semilogy(LS, [per_layer[tag][l]["n_unique"] for l in LS],
                            marker="o", ms=3, color=color, label=tag)
        axes[1, 1].plot(LS, [per_layer[tag][l]["mode_share"] for l in LS],
                        marker="o", ms=3, color=color, label=tag)
        axes[1, 2].scatter([per_layer[tag][l]["sigma_over_bf16_step"] for l in LS],
                           [per_layer[tag][l]["raw_kurt"] for l in LS],
                           s=16, color=color, label=tag, alpha=0.85)

    axes[0, 0].plot(grid, np.exp(-grid**2 / 2) / np.sqrt(2 * np.pi), "k--", lw=1.2, label="N(0,1)")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_ylim(1e-6, 1)
    axes[0, 0].set_title("density of per-layer relative deviation (log y)")
    axes[0, 0].set_xlabel("z = delta / sigma_layer")
    axes[0, 0].legend(fontsize=8)

    ks = np.arange(1, 16)
    axes[0, 1].semilogy(ks, [math.erfc(k / math.sqrt(2)) for k in ks], "k--", lw=1.2,
                        label="Gaussian")
    axes[0, 1].set_ylim(1e-8, 1)
    axes[0, 1].set_title("tail: P(|z| > k)")
    axes[0, 1].set_xlabel("k")
    axes[0, 1].legend(fontsize=8)

    axes[0, 2].axhline(1.0, color="k", ls=":", lw=1, label="1 bf16 step")
    axes[0, 2].set_title("bulk width / bf16 step (= resolution of the profile)")
    axes[0, 2].set_xlabel("layer")
    axes[0, 2].legend(fontsize=8)

    axes[1, 0].set_title("distinct bf16 values per layer gain vector")
    axes[1, 0].set_xlabel("layer")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].set_title("share of dims tied at the mode value")
    axes[1, 1].set_xlabel("layer")
    axes[1, 1].legend(fontsize=8)

    axes[1, 2].set_yscale("log")
    axes[1, 2].set_xlabel("sigma / bf16 step")
    axes[1, 2].set_ylabel("raw excess kurtosis (spike-driven, log)")
    axes[1, 2].set_title("narrower bulk -> more extreme spikes")
    axes[1, 2].legend(fontsize=8)

    for tag, (_, color) in TYPES.items():
        axes[2, 0].scatter([per_layer[tag][l]["sigma_over_bf16_step"] for l in LS],
                           [per_layer[tag][l]["pr"] for l in LS],
                           s=18, color=color, alpha=0.85, label=f"{tag} (rho={coef[tag]:+.2f})")
    axes[2, 0].set_yscale("log")
    axes[2, 0].set_xlabel("sigma / bf16 step (numerical resolution)")
    axes[2, 0].set_ylabel("PR (effective dims, log)")
    axes[2, 0].set_title("resolution vs measured structure (within type)")
    axes[2, 0].legend(fontsize=7)

    for key, vals in sweep.items():
        tag = key.split(" L")[0]
        axes[2, 1].plot([7, 6, 5, 4, 3], [v / vals[0] for v in vals], marker="o", ms=4,
                        color=TYPES[tag][1], alpha=0.85, label=key)
    axes[2, 1].axhspan(0.95, 1.05, color="0.85", zorder=0, label="+-5%")
    axes[2, 1].set_xlabel("mantissa bits (7 = bf16 native)")
    axes[2, 1].set_ylabel("PR(p) / PR(7)")
    axes[2, 1].set_title("precision sensitivity: coarsening the grid")
    axes[2, 1].legend(fontsize=7)

    for key, (c, exp) in occ.items():
        m = (c > 0) & (exp > 0.2)
        axes[2, 2].scatter(exp[m], c[m], s=14, alpha=0.75, label=key)
    lim = [0.2, max(max(v[1].max() for v in occ.values()), max(v[0].max() for v in occ.values())) * 1.5]
    axes[2, 2].plot(lim, lim, "k--", lw=1, label="Gaussian expectation")
    axes[2, 2].set_xscale("log")
    axes[2, 2].set_yscale("log")
    axes[2, 2].set_xlim(*lim)
    axes[2, 2].set_xlabel("expected dims per bf16 level (Gaussian)")
    axes[2, 2].set_ylabel("observed dims on that level")
    axes[2, 2].set_title("level occupancy vs smooth-Gaussian expectation")
    axes[2, 2].legend(fontsize=7)

    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "norm_distribution.png", dpi=120)
    plt.close(fig)
    print(f"\n输出: {OUT_DIR / 'norm_distribution.json'}  {FIG_DIR / 'norm_distribution.png'}")


if __name__ == "__main__":
    main()
