"""分析 DeepSeek-V4.1-Flash 全部 RMSNorm 增益向量的分布:训练对"全 1"初始化改写了多少。

增益是投影前的逐通道闸门(可折入相邻线性层:W·diag(g)),尺度信息的主要载体。
本脚本枚举所有增益类 1D 权重(attn_norm/ffn_norm/q_norm/kv_norm/indexer.k_norm/
compressor.norm/engram.q_weight/k_weight/最终 norm/mtp/vision),按命名模式分组,逐层提取:

- 集中与分化:mean/std/min/p5/p50/p95/max;
- 极端通道占比:g<0.5(压制)与 g>2(放大)的维数比例;
- 画像集中度(z-score 口径):有效维数 PR、top-k 维方差占比、常驻突出维(跨层 top-k 重合);
- 跨层稳定性:相邻层同类型增益的余弦(原始 + 去均值两个口径);
- 分支一致性:同层 attn_norm 与 ffn_norm 的余弦(同两个口径);
- 跨层共同画像:各层 z-score 画像矩阵的 SVD 前 5 个主成分方差占比;
- 与 FFN 尺度的联动:ffn_norm 增益指标 vs mlp_scale_depth.json 的逐层 elem_std。

输出:
- output/ds_v4_1_flash/weight_moments/norm_gains.npz:全部增益向量(按模式分组);
- output/ds_v4_1_flash/weight_moments/norm_gains.json:逐层逐类指标;
- output/ds_v4_1_flash/figures/norm_gains.png:六联图。

用法(在仓库根目录下):
    python analysis/ds_v4_1_flash/analyze_norm_gains.py
"""

import argparse
import json
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_weight_moments import FIG_DIR, OUT_DIR
from ds_common import load_weight_map
from llm_lens import get_model_dir, read_tensor
from llm_lens.cli import add_model_args

MAIN_PATTERNS = [  # 深度曲线重点展示的 4 类(主干每层都有)
    "layers.*.attn_norm.weight",
    "layers.*.ffn_norm.weight",
    "layers.*.attn.q_norm.weight",
    "layers.*.attn.kv_norm.weight",
]


def is_gain_weight(name: str) -> bool:
    """增益类权重:倒数第二段名以 norm 结尾,或是 norm1/norm2/主 norm,
    或 engram 的逐通道门 q_weight/k_weight(初始化全 1,命名不带 .weight 后缀)。"""
    if ".engram." in name and name.rsplit(".", 1)[-1] in ("q_weight", "k_weight"):
        return True
    if not name.endswith(".weight"):
        return False
    seg = name.split(".")[-2]
    return seg.endswith("norm") or seg in ("norm1", "norm2")


def gain_metrics(g: np.ndarray, topk: int = 10) -> dict:
    """单个增益向量的分布指标(初始化 = 全 1)。

    mean 可能被正负抵消压低,故同时给 |g| 口径:
    abs_mean/abs_p50 是"典型通道力度",topk_mean 是最大 k 个 |g| 的均值。

    集中度口径(z-score 画像,即去均值再除 std,只看"形状"):
    - pr:参与度 $(\sum p^2)^2/\sum p^4$,有效维数;白噪声期望 $d/3$,故 pr_ratio<1 表示有结构;
    - topk_share:最大 k 维占中心化方差的比例,白噪声基线 $k/d$;
    - skew:偏度(>0 长尾在右 = 少数维被放大;<0 长尾在左 = 少数维被压制)。
    """
    mean = float(g.mean())
    ag = np.abs(g)
    k = min(topk, g.size)
    sd = float(g.std())
    z = (g - mean) / sd if sd > 0 else np.zeros_like(g)
    z2 = z**2
    pr = float(z2.sum() ** 2 / (z**4).sum())
    return {
        "dim": int(g.size),
        "mean": mean,
        "std": sd,
        "cv": float(g.std() / mean) if mean > 0 else float("nan"),  # 变异系数:相对分化
        "min": float(g.min()),
        "p5": float(np.percentile(g, 5)),
        "p50": float(np.median(g)),
        "p95": float(np.percentile(g, 95)),
        "max": float(g.max()),
        "abs_mean": float(ag.mean()),
        "abs_p50": float(np.median(ag)),
        "abs_p95": float(np.percentile(ag, 95)),
        "topk_mean": float(np.sort(ag)[-k:].mean()),
        "frac_neg": float((g < 0).mean()),  # 负增益(翻转通道符号)占比
        "frac_low": float((g < 0.5).mean()),
        "frac_high": float((g > 2.0).mean()),
        "pr": pr,
        "pr_ratio": pr / (g.size / 3.0),  # <1 才是有结构(=1 等价白噪声)
        "topk_share": float(np.sort(z2)[-k:].sum() / z2.sum()),
        "skew": float((z**3).mean()),
    }


def zscore(g: np.ndarray) -> np.ndarray:
    """去均值再除 std 的画像向量(只看"哪些通道相对被放大/压制")。"""
    sd = float(g.std())
    return (g - g.mean()) / sd if sd > 0 else np.zeros_like(g)


def cross_layer_pca(rows: list[np.ndarray]) -> dict:
    """把各层的 z-score 画像堆成矩阵做 SVD,返回前几个主成分解释的方差占比。

    各层画像若互不相关,占比 ≈ 1/n_layers;明显高于它说明存在跨层共同画像。
    """
    z = np.stack([zscore(g) for g in rows])
    s = np.linalg.svd(z, compute_uv=False)
    ev = (s**2) / (s**2).sum()
    return {f"PC{i + 1}": float(ev[i]) for i in range(min(5, len(ev)))}


def persistent_dims(rows: list[np.ndarray], topk: int = 10, min_layers: int = 3) -> dict:
    """在多少层里挤进 top-k |z| 的维度(按层数降序);用于找"常驻突出通道"。"""
    n = len(rows)
    counts: dict[int, int] = {}
    for g in rows:
        z = np.abs(zscore(g))
        for j in np.argsort(z)[-topk:][::-1]:
            counts[int(j)] = counts.get(int(j), 0) + 1
    return {str(j): c for j, c in sorted(counts.items(), key=lambda t: -t[1]) if c >= min_layers}


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """原始余弦。注意:增益向量几乎是平的(见下 CV),这个值会被共同均值撑到 ≈0.999,
    不能当作"通道画像一致"的证据,需与 cosine_centered 一起看。"""
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def cosine_centered(a: np.ndarray, b: np.ndarray) -> float:
    """去均值(皮尔逊)余弦,即真正的"通道画像"相似度。

    两条独立、同 CV 的平正向量原始余弦基线 ≈ 1/(1+CV²):CV=0.02 时基线就有 0.9992,
    所以原始余弦在低 CV 层毫无判别力;去均值后基线为 0。
    """
    ac, bc = a - a.mean(), b - b.mean()
    return float(ac @ bc / (np.linalg.norm(ac) * np.linalg.norm(bc)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_args(parser)
    args = parser.parse_args()

    model_dir = get_model_dir(args.model, args.model_dir, args.config)
    print(f"模型目录: {model_dir}")
    weight_map = load_weight_map(model_dir)

    # 枚举增益类权重,按模式(层号归一)分组;只取小向量(1D 或 [4,5120] 的 engram 门)
    groups: dict[str, list[tuple[str, str]]] = {}
    globals_: dict[str, str] = {}
    for name in weight_map:
        if not is_gain_weight(name):
            continue
        m = re.search(r"(?:layers|blocks|mtp)\.(\d+)\.", name)
        if m:
            pat = re.sub(r"\.\d+\.", ".*.", name)
            groups.setdefault(pat, []).append((f"{m.group(1)}", name))
        else:
            globals_[name] = name
    print(f"增益模式 {len(groups)} 类,全局 {len(globals_)} 个")

    # 读取全部向量 + 计算指标(engram 的 [4,5120] 门按整体展平统计)
    gains: dict[str, dict[str, np.ndarray]] = {}
    metrics: dict[str, dict[str, dict]] = {}
    for pat, items in sorted(groups.items()):
        gains[pat], metrics[pat] = {}, {}
        for layer, name in sorted(items, key=lambda t: int(t[0])):
            g = read_tensor(model_dir / weight_map[name], name).reshape(-1)
            gains[pat][layer] = g
            metrics[pat][layer] = gain_metrics(g)
    for key, name in globals_.items():
        g = read_tensor(model_dir / weight_map[name], name).reshape(-1)
        gains[f"GLOBAL:{key}"] = {"-": g}
        metrics[f"GLOBAL:{key}"] = {"-": gain_metrics(g)}

    # 跨层/跨分支关系指标
    relations = {}
    persistent: dict[str, dict[str, int]] = {}
    for pat in MAIN_PATTERNS:
        if pat not in gains:
            continue
        ls = sorted(gains[pat], key=int)
        relations[f"{pat}|adjacent_cos"] = {
            ls[i + 1]: cosine(gains[pat][ls[i]], gains[pat][ls[i + 1]]) for i in range(len(ls) - 1)
        }
        relations[f"{pat}|adjacent_cos_centered"] = {
            ls[i + 1]: cosine_centered(gains[pat][ls[i]], gains[pat][ls[i + 1]])
            for i in range(len(ls) - 1)
        }
        rows = [gains[pat][l] for l in ls]
        relations[f"{pat}|cross_layer_pca"] = cross_layer_pca(rows)
        persistent[pat] = persistent_dims(rows)
    pa, pf = "layers.*.attn_norm.weight", "layers.*.ffn_norm.weight"
    relations["attn_vs_ffn_norm|same_layer_cos"] = {
        l: cosine(gains[pa][l], gains[pf][l]) for l in sorted(set(gains[pa]) & set(gains[pf]), key=int)
    }
    relations["attn_vs_ffn_norm|same_layer_cos_centered"] = {
        l: cosine_centered(gains[pa][l], gains[pf][l])
        for l in sorted(set(gains[pa]) & set(gains[pf]), key=int)
    }

    # 与 FFN 尺度的联动
    linkage = {}
    ffn_path = OUT_DIR / "mlp_scale_depth.json"
    if ffn_path.exists():
        ffn = json.loads(ffn_path.read_text(encoding="utf-8"))
        post = metrics[pf]
        ls = sorted(set(post) & set(ffn["w3"]), key=int)
        for stat in ("mean", "std", "cv", "frac_neg"):
            xs = np.array([post[l][stat] for l in ls])
            for kind in ("w1", "w3", "w2"):
                ys = np.array([ffn[kind][l]["elem_std"] for l in ls])
                linkage[f"ffn_gain_{stat}__vs__{kind}_elem_std"] = {
                    "layers": [int(l) for l in ls], "x": xs.tolist(), "y": ys.tolist(),
                    "corr": float(np.corrcoef(xs, ys)[0, 1]),
                }

    # 汇总打印:结构边界层 + 全局
    focus = ["0", "1", "2", "14", "19", "20", "38", "39"]
    for pat in MAIN_PATTERNS:
        print(f"\n[{pat}]")
        for l in focus:
            if l in metrics[pat]:
                m = metrics[pat][l]
                print(f"  L{l:>2}: mean={m['mean']:.3f} |g|中位={m['abs_p50']:.3f}"
                      f"  |g|p95={m['abs_p95']:.3f} top10|g|={m['topk_mean']:.3f}"
                      f"  min={m['min']:.3f} max={m['max']:.3f}  neg:{m['frac_neg']:.1%}")
    print("\n[全局增益]")
    for key, mm in metrics.items():
        if key.startswith("GLOBAL:"):
            m = mm["-"]
            print(f"  {key[7:]}: dim={m['dim']} mean={m['mean']:.3f} std={m['std']:.3f}"
                  f"  [min={m['min']:.3f}, max={m['max']:.3f}]")
    print("\n[其他模式概览]")
    for pat, mm in metrics.items():
        if pat in MAIN_PATTERNS or pat.startswith("GLOBAL:"):
            continue
        means = [m["mean"] for m in mm.values()]
        dims = {m["dim"] for m in mm.values()}
        print(f"  {pat:50s} n={len(mm):>3} dim={sorted(dims)} mean 范围 [{min(means):.3f}, {max(means):.3f}]")
    print("\n[余弦口径]原始余弦 vs 去均值余弦(中位);增益越平,原始余弦越接近 1 的假象")
    for pat in MAIN_PATTERNS + ["attn_vs_ffn_norm"]:
        wall = f"{pat}|adjacent_cos" if pat in MAIN_PATTERNS else f"{pat}|same_layer_cos"
        wcent = (f"{pat}|adjacent_cos_centered" if pat in MAIN_PATTERNS
                 else f"{pat}|same_layer_cos_centered")
        if wall in relations:
            raw = np.median(list(relations[wall].values()))
            cen = np.median(list(relations[wcent].values()))
            cvs = [m["cv"] for m in metrics[pat].values()] if pat in metrics else []
            cvtxt = f" CV 中位={np.median(cvs):.3f}" if cvs else ""
            print(f"  {pat:26s} 原始={raw:.4f}  去均值={cen:.4f} (最低 {min(relations[wcent].values()):.3f}){cvtxt}")

    print("\n[画像集中度]z-score 口径:PR = 有效维数(白噪声 = d/3),top10 占比(基线 10/d)")
    for pat in MAIN_PATTERNS:
        if pat not in metrics:
            continue
        ms = list(metrics[pat].values())
        d = ms[0]["dim"]
        pca = relations.get(f"{pat}|cross_layer_pca", {})
        pcatxt = " ".join(f"{k}={v:.3f}" for k, v in list(pca.items())[:3])
        print(f"  {pat:26s} d={d:5d}  PR 中位={np.median([m['pr'] for m in ms]):8.1f}"
              f" (基线 {d / 3:.0f}, ={np.median([m['pr_ratio'] for m in ms]):.3f})"
              f"  top10={np.median([m['topk_share'] for m in ms]):.3f} (基线 {10 / d:.4f})"
              f"  偏度={np.median([m['skew'] for m in ms]):+.2f}  跨层 {pcatxt}"
              f" (独立基线 {1 / len(ms):.3f})")

    print("\n[常驻突出通道]在 ≥3 层挤进 top-10 |z| 的维度(维:层数)")
    for pat, dims in persistent.items():
        if dims:
            top = list(dims.items())[:8]
            print(f"  {pat:26s} " + " ".join(f"d{j}x{c}" for j, c in top))

    print("\n[联动相关系数](ffn_norm 增益 vs 共享专家元素 std)")
    for k, v in linkage.items():
        print(f"  {k}: r={v['corr']:.3f}")

    # 保存
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_DIR / "norm_gains.npz",
             **{f"{pat}__{l}": g for pat, d in gains.items() for l, g in d.items()})
    with open(OUT_DIR / "norm_gains.json", "w", encoding="utf-8") as f:
        json.dump({"metrics": {p: {str(l): m for l, m in d.items()} for p, d in metrics.items()},
                   "relations": {k: {str(l): v for l, v in d.items()} for k, d in relations.items()},
                   "persistent_dims": persistent,
                   "linkage": linkage}, f, ensure_ascii=False)

    # 画图
    short = {"layers.*.attn_norm.weight": "attn_norm",
             "layers.*.ffn_norm.weight": "ffn_norm",
             "layers.*.attn.q_norm.weight": "q_norm",
             "layers.*.attn.kv_norm.weight": "kv_norm"}
    fig, axes = plt.subplots(3, 3, figsize=(18, 14.5))
    for ax, stat, title in ((axes[0, 0], "mean", "gain mean vs depth"),
                            (axes[0, 1], "std", "gain std vs depth (channel differentiation)"),
                            (axes[1, 0], "frac_high", "frac of dims with g>2")):
        for pat in MAIN_PATTERNS:
            if pat in metrics:
                ls = sorted(metrics[pat], key=int)
                ax.plot([int(l) for l in ls], [metrics[pat][l][stat] for l in ls],
                        ms=3, marker="o", label=short[pat])
        ax.set_title(title)
        ax.set_xlabel("layer")
        ax.legend(fontsize=8)

    for pat in MAIN_PATTERNS:
        for suffix, ls_style, tag in (("|adjacent_cos", "-", " (raw)"),
                                      ("|adjacent_cos_centered", "--", " (centered)")):
            key = f"{pat}{suffix}"
            if key in relations:
                ls = sorted(relations[key], key=int)
                axes[0, 2].plot([int(l) for l in ls], [relations[key][l] for l in ls],
                                ls=ls_style, alpha=0.55 if "centered" in suffix else 0.95,
                                ms=3, marker="o", label=f"{short[pat]}{tag}")
    axes[0, 2].axhline(0.0, color="k", lw=0.6, ls=":")
    axes[0, 2].set_title("adjacent-layer cosine: raw (solid) vs centered (dashed)")
    axes[0, 2].set_xlabel("layer")
    axes[0, 2].legend(fontsize=8)

    for pat, style in ((pa, "-"), (pf, "--")):
        for l, alpha in (("0", 0.6), ("20", 0.75), ("39", 1.0)):
            if l in gains[pat]:
                axes[1, 1].hist(gains[pat][l], bins=150, density=True, alpha=alpha * 0.6,
                                ls=style, histtype="step", lw=1.5,
                                label=f"{short[pat]} L{l}")
    axes[1, 1].axvline(1.0, color="r", ls=":", lw=1, label="init")
    axes[1, 1].set_title("gain histograms (L0/L20/L39)")
    axes[1, 1].legend(fontsize=7)

    lk = linkage.get("ffn_gain_std__vs__w3_elem_std")
    if lk:
        sc = axes[1, 2].scatter(lk["x"], lk["y"], c=lk["layers"], cmap="viridis", s=18)
        for l in (0, 1, 38, 39):
            if l in lk["layers"]:
                i = lk["layers"].index(l)
                axes[1, 2].annotate(f"L{l}", (lk["x"][i], lk["y"][i]), fontsize=8)
        axes[1, 2].set_xlabel("ffn_norm gain std")
        axes[1, 2].set_ylabel("w3 elem std")
        axes[1, 2].set_title(f"linkage (r={lk['corr']:.3f})")
        fig.colorbar(sc, ax=axes[1, 2], label="layer")

    # 集中度:PR(有效维数)与 top10 方差占比 —— 平坦的向量 PR 接近 d/3、top10 接近 10/d
    for pat in MAIN_PATTERNS:
        if pat not in metrics:
            continue
        ls = sorted(metrics[pat], key=int)
        d = metrics[pat][ls[0]]["dim"]
        x = [int(l) for l in ls]
        axes[2, 0].plot(x, [metrics[pat][l]["pr"] for l in ls], ms=3, marker="o", label=short[pat])
        axes[2, 1].plot(x, [metrics[pat][l]["topk_share"] for l in ls],
                        ms=3, marker="o", label=short[pat])
        axes[2, 0].axhline(d / 3, color="k", lw=0.6, ls=":", label=f"white-noise d/3 ({d // 3})")
        axes[2, 1].axhline(10 / d, color="k", lw=0.6, ls=":", label=f"white-noise 10/d ({10 / d:.4f})")
    axes[2, 0].set_yscale("log")
    axes[2, 0].set_title("effective dims PR = (sum z^2)^2 / sum z^4")
    axes[2, 0].set_ylabel("PR (log)")
    axes[2, 1].set_title("top-10 dims share of centered variance")
    for ax in (axes[2, 0], axes[2, 1]):
        ax.set_xlabel("layer")
        ax.legend(fontsize=7)

    # 跨层共同画像:各类型 z-score 画像矩阵的前 5 个主成分
    width = 0.2
    for i, pat in enumerate(MAIN_PATTERNS):
        pca = relations.get(f"{pat}|cross_layer_pca")
        if not pca:
            continue
        axes[2, 2].bar(np.arange(len(pca)) + i * width, list(pca.values()), width=width,
                       label=short[pat])
    axes[2, 2].axhline(1 / 40, color="k", lw=0.6, ls=":", label="independent layers (1/40)")
    axes[2, 2].set_xticks(np.arange(5) + 1.5 * width - width / 2)
    axes[2, 2].set_xticklabels([f"PC{i + 1}" for i in range(5)])
    axes[2, 2].set_title("cross-layer PCA: shared channel profile?")
    axes[2, 2].legend(fontsize=7)

    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "norm_gains.png", dpi=120)
    plt.close(fig)
    print(f"\n输出: {OUT_DIR / 'norm_gains.json'}  {FIG_DIR / 'norm_gains.png'}")


if __name__ == "__main__":
    main()
