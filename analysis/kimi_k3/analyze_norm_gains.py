"""分析 Kimi-K3 全部 RMSNorm 增益向量的分布:训练对"全 1"初始化改写了多少。

增益是投影前的逐通道闸门(可折入相邻线性层:W·diag(g)),尺度信息的主要载体。
本脚本枚举 language_model 下所有增益类 1D 权重(layernorm/res_norm/res_proj/o_norm/
routed_expert_norm/final norm),按命名模式分组,逐层提取:

- 集中与分化:mean/std/min/p5/p50/p95/max;
- 极端通道占比:g<0.5(压制)与 g>2(放大)的维数比例;
- 跨层稳定性:相邻层同类型增益的余弦;
- 分支一致性:同层 input_layernorm 与 post_attention_layernorm 的余弦;
- 与 FFN 尺度的联动:post_attention 增益指标 vs mlp_scale_depth.json 的逐层 elem_std
  (重点对照末段塌缩层 L88-92 与死行层 L12/L22)。

输出:
- output/kimi_k3/weight_moments/norm_gains.npz:全部增益向量(按模式分组);
- output/kimi_k3/weight_moments/norm_gains.json:逐层逐类指标;
- output/kimi_k3/figures/norm_gains.png:六联图。

用法(在仓库根目录下):
    python analysis/kimi_k3/analyze_norm_gains.py
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_weight_moments import FIG_DIR, OUT_DIR
from llm_lens import get_model_dir, read_tensor
from llm_lens.cli import add_model_args

def is_gain_weight(name: str) -> bool:
    """增益类权重:倒数第二段名以 norm 结尾(含 layernorm/o_norm/res_norm/最终 norm)
    或为 res_proj(AttnRes 的读出向量)。"""
    if not name.endswith(".weight"):
        return False
    seg = name.split(".")[-2]
    return seg.endswith("norm") or seg.endswith("res_proj")
MAIN_PATTERNS = [  # 深度曲线重点展示的 4 类(每层都有、7168 维)
    "layers.*.input_layernorm.weight",
    "layers.*.post_attention_layernorm.weight",
    "layers.*.self_attention_res_norm.weight",
    "layers.*.mlp_res_norm.weight",
]


def gain_metrics(g: np.ndarray, topk: int = 10) -> dict:
    """单个增益向量的分布指标(初始化 = 全 1;实测训练后整体衰减到 0.1~0.4)。

    mean 可能被正负抵消压低(res_norm 类 30~45% 为负),故同时给 |g| 口径:
    abs_mean/abs_p50 是"典型通道力度",topk_mean 是最大 k 个 |g| 的均值。
    """
    mean = float(g.mean())
    ag = np.abs(g)
    k = min(topk, g.size)
    return {
        "dim": int(g.size),
        "mean": mean,
        "std": float(g.std()),
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
    }


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_args(parser)
    args = parser.parse_args()

    model_dir = get_model_dir(args.model, args.model_dir, args.config)
    print(f"模型目录: {model_dir}")
    with open(model_dir / "model.safetensors.index.json", encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]

    # 枚举增益类权重,按模式(层号归一)分组;只取 1D 或 [1,N] 的小向量
    groups: dict[str, list[tuple[int, str]]] = {}
    globals_: dict[str, str] = {}
    for name in weight_map:
        if not name.startswith("language_model.") or not is_gain_weight(name):
            continue
        m = re.search(r"layers\.(\d+)\.", name)
        if m:
            pat = re.sub(r"\.\d+\.", ".*.", name)
            groups.setdefault(pat.removeprefix("language_model.model."), []).append((int(m.group(1)), name))
        else:
            globals_[name.removeprefix("language_model.")] = name
    print(f"增益模式 {len(groups)} 类,全局 {len(globals_)} 个")

    # 读取全部向量 + 计算指标
    gains: dict[str, dict[int, np.ndarray]] = {}
    metrics: dict[str, dict[int, dict]] = {}
    for pat, items in sorted(groups.items()):
        gains[pat], metrics[pat] = {}, {}
        for layer, name in sorted(items):
            g = read_tensor(model_dir / weight_map[name], name).reshape(-1)
            gains[pat][layer] = g
            metrics[pat][layer] = gain_metrics(g)
    for key, name in globals_.items():
        g = read_tensor(model_dir / weight_map[name], name).reshape(-1)
        gains[f"GLOBAL:{key}"] = {0: g}
        metrics[f"GLOBAL:{key}"] = {0: gain_metrics(g)}

    # 跨层/跨分支关系指标
    relations = {}
    for pat in MAIN_PATTERNS:
        if pat not in gains:
            continue
        ls = sorted(gains[pat])
        relations[f"{pat}|adjacent_cos"] = {
            ls[i + 1]: cosine(gains[pat][ls[i]], gains[pat][ls[i + 1]]) for i in range(len(ls) - 1)
        }
    pi, pp = "layers.*.input_layernorm.weight", "layers.*.post_attention_layernorm.weight"
    relations["input_vs_post_attention|same_layer_cos"] = {
        l: cosine(gains[pi][l], gains[pp][l]) for l in sorted(set(gains[pi]) & set(gains[pp]))
    }

    # 与 FFN 尺度的联动
    linkage = {}
    ffn_path = OUT_DIR / "mlp_scale_depth.json"
    if ffn_path.exists():
        ffn = json.loads(ffn_path.read_text(encoding="utf-8"))
        post = metrics[pp]
        ls = sorted(set(post) & {int(l) for l in ffn["up_proj"]})
        for stat in ("mean", "std", "cv", "frac_neg"):
            xs = np.array([post[l][stat] for l in ls])
            for kind in ("gate_proj", "up_proj", "down_proj"):
                ys = np.array([ffn[kind][str(l)]["elem_std"] for l in ls])
                linkage[f"post_gain_{stat}__vs__{kind}_elem_std"] = {
                    "layers": ls, "x": xs.tolist(), "y": ys.tolist(),
                    "corr": float(np.corrcoef(xs, ys)[0, 1]),
                }

    # 汇总打印:重点层 + 全局
    focus = [0, 12, 22, 45, 88, 89, 90, 91, 92]
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
            m = mm[0]
            print(f"  {key[7:]}: dim={m['dim']} mean={m['mean']:.3f} std={m['std']:.3f}"
                  f"  [min={m['min']:.3f}, max={m['max']:.3f}]")
    print("\n[联动相关系数](post_attention 增益 vs FFN 元素 std)")
    for k, v in linkage.items():
        print(f"  {k}: r={v['corr']:.3f}")

    # 保存
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_DIR / "norm_gains.npz",
             **{f"{pat}__{l}": g for pat, d in gains.items() for l, g in d.items()})
    with open(OUT_DIR / "norm_gains.json", "w", encoding="utf-8") as f:
        json.dump({"metrics": {p: {str(l): m for l, m in d.items()} for p, d in metrics.items()},
                   "relations": {k: {str(l): v for l, v in d.items()} for k, d in relations.items()},
                   "linkage": linkage}, f, ensure_ascii=False)

    # 画图
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    short = {"layers.*.input_layernorm.weight": "input_ln",
             "layers.*.post_attention_layernorm.weight": "post_attn_ln",
             "layers.*.self_attention_res_norm.weight": "attn_res_norm",
             "layers.*.mlp_res_norm.weight": "mlp_res_norm"}
    for ax, stat, title in ((axes[0, 0], "mean", "gain mean vs depth"),
                            (axes[0, 1], "std", "gain std vs depth (channel differentiation)"),
                            (axes[1, 0], "frac_high", "frac of dims with g>2")):
        for pat in MAIN_PATTERNS:
            if pat in metrics:
                ls = sorted(metrics[pat])
                ax.plot(ls, [metrics[pat][l][stat] for l in ls], ms=3, marker="o", label=short[pat])
        ax.set_title(title)
        ax.set_xlabel("layer")
        ax.legend(fontsize=8)

    for pat in MAIN_PATTERNS:
        key = f"{pat}|adjacent_cos"
        if key in relations:
            ls = sorted(relations[key])
            axes[0, 2].plot(ls, [relations[key][l] for l in ls], ms=3, marker="o", label=short[pat])
    axes[0, 2].set_title("adjacent-layer cosine (stability of gain profile)")
    axes[0, 2].set_xlabel("layer")
    axes[0, 2].legend(fontsize=8)

    for pat, style in ((pi, "-"), (pp, "--")):
        for l, alpha in ((0, 0.6), (45, 0.75), (92, 1.0)):
            if l in gains[pat]:
                axes[1, 1].hist(gains[pat][l], bins=150, density=True, alpha=alpha * 0.6,
                                ls=style, histtype="step", lw=1.5,
                                label=f"{short[pat]} L{l}")
    axes[1, 1].axvline(1.0, color="r", ls=":", lw=1, label="init")
    axes[1, 1].set_title("gain histograms (L0/L45/L92)")
    axes[1, 1].legend(fontsize=7)

    lk = linkage.get("post_gain_std__vs__up_proj_elem_std")
    if lk:
        sc = axes[1, 2].scatter(lk["x"], lk["y"], c=lk["layers"], cmap="viridis", s=18)
        for l in (12, 22, 88, 89, 90, 91, 92):
            if l in lk["layers"]:
                i = lk["layers"].index(l)
                axes[1, 2].annotate(f"L{l}", (lk["x"][i], lk["y"][i]), fontsize=8)
        axes[1, 2].set_xlabel("post_attn_ln gain std")
        axes[1, 2].set_ylabel("up_proj elem std")
        axes[1, 2].set_title(f"linkage (r={lk['corr']:.3f})")
        fig.colorbar(sc, ax=axes[1, 2], label="layer")

    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "norm_gains.png", dpi=120)
    plt.close(fig)
    print(f"\n输出: {OUT_DIR / 'norm_gains.json'}  {FIG_DIR / 'norm_gains.png'}")


if __name__ == "__main__":
    main()
