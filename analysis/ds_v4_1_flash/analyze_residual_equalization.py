"""分析入口增益的「均衡器」行为:增益是否在压那些把残差写得很响的通道。

背景:2.5 发现 attn_norm 的极端维分两侧,且压制侧各自绑定一段深度。本脚本检验机制假设——
**残差流里被写得越响的通道,下一层入口增益把它压得越狠**(RMSNorm 的分母是全通道 RMS,
单个"响"通道会挤压其余通道的相对尺度,增益正好可以按通道把它抵消)。

不需要激活,只用权重做代理量:
- 写响度:注意力写回 `wo_b` 的逐行范数(5120 维,对应写进残差的通道);
- 读响度:`wq_a` 的逐列范数(5120 维,对应被读取的输入通道);
- 相对增益:该层增益 / 该层中位(消除层间整体幅度差)。

对每个被跟踪的通道,跨 40 层算 Spearman ρ(相对增益, log 代理量);ρ 显著为负 = 越响越压。
**对照组**是随机抽的 60 个通道(若只有已知通道显著,才算真机制)。

进一步做两件事(见 docs/ds_v4_1_flash/norm-gains.md §2.6.1):
- **滞后相关**:改用因果配对 —— 层 l 的增益 vs 层 l+δ 的写回(δ=−2…+2),写回画像扩到共享专家与
  抽样 8 个路由专家的 `w2` 行范数;峰值落在 δ=0 的 `g_ffn`↔`wo_b`(中位 ρ=−0.411)、
  且 `g_attn` 各滞后都无信号 → 均衡器只作用于 FFN 入口、且只对"刚收到的那次写回";
- **累积衰减扫描**:R_l = embed + Σ_{j<l} e^{−(l−1−j)/τ}·(W_attn_j + W_ffn_j),扫 τ 检验"多层累积
  响度"能否比单层配对解释得更好(实测最好也只有 −0.46 → 不是多层积分)。

输出:
- output/ds_v4_1_flash/weight_moments/gain_equalization.json:逐通道 ρ、逐层曲线、滞后表、衰减扫描;
- output/ds_v4_1_flash/figures/gain_equalization.png:四联图。

用法(在仓库根目录下):
    python analysis/ds_v4_1_flash/analyze_residual_equalization.py
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_weight_moments import FIG_DIR, OUT_DIR  # noqa: E402
from ds_common import load_weight_map, read_fp4_expert, read_fp8_block  # noqa: E402
from llm_lens import get_model_dir, iter_row_chunks, read_tensor  # noqa: E402
from llm_lens.cli import add_model_args  # noqa: E402

LS = list(range(40))
# 被跟踪的通道:2.5 里的压制维 + ffn 的全局放大维 + 一个 attn 局部放大维
TRACK = {
    3136: ("suppressed L14-20", "#3559d6"),
    1474: ("suppressed L14-20", "#6f8ff5"),
    1233: ("suppressed L0-3", "#0ea47a"),
    529: ("suppressed L34-38", "#e08a1e"),
    1726: ("suppressed L4-11", "#db2777"),
    2455: ("amplified ffn global", "#7c3aed"),
    3909: ("amplified attn L0", "#a78bfa"),
}
N_CONTROL = 60
N_EXPERT_SAMPLE = 8  # 每层抽样的路由专家数(按固定步长铺开,用于 FFN 写回画像)
SEED = 0


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_args(parser)
    args = parser.parse_args()
    model_dir = get_model_dir(args.model, args.model_dir, args.config)
    print(f"模型目录: {model_dir}")
    wm = load_weight_map(model_dir)

    def read(name: str) -> np.ndarray:
        return np.asarray(read_tensor(model_dir / wm[name], name, dtype=np.float64)).reshape(-1)

    print("读 40 层的增益与读写范数代理(wo_b 行范 / wq_a 列范)…")
    g = {}
    for tag, f in (("attn", "layers.{}.attn_norm.weight"), ("ffn", "layers.{}.ffn_norm.weight")):
        arr = np.array([read(f.format(l)) for l in LS])
        g[tag] = arr / np.median(arr, axis=1, keepdims=True)  # 相对增益
    row_wo = np.array([np.linalg.norm(read_fp8_block(model_dir, wm, f"layers.{l}.attn.wo_b"), axis=1)
                       for l in LS])
    col_q = np.array([np.linalg.norm(read_fp8_block(model_dir, wm, f"layers.{l}.attn.wq_a"), axis=0)
                      for l in LS])
    pct_wo = np.array([(row_wo[l][:, None] < row_wo[l]).mean(axis=0) for l in LS])  # 逐层百分位
    pct_q = np.array([(col_q[l][:, None] < col_q[l]).mean(axis=0) for l in LS])

    print("\n[被跟踪通道]跨 40 层 ρ(相对增益, log 代理量);负值 = 越响越被压")
    print(f"  {'通道':>18s} {'ρ(attn, 读)':>12s} {'ρ(attn, 写)':>12s} {'ρ(ffn, 写)':>11s} "
          f"{'被压最狠 6 层的写百分位':>22s}")
    out = {}
    for d, (label, _) in TRACK.items():
        r_aq = spearman(g["attn"][:, d], np.log(col_q[:, d]))
        r_aw = spearman(g["attn"][:, d], np.log(row_wo[:, d]))
        r_fw = spearman(g["ffn"][:, d], np.log(row_wo[:, d]))
        worst = np.argsort(g["attn"][:, d])[:6]
        med_pct = float(np.median(pct_wo[worst, d]))
        out[str(d)] = {"label": label, "rho_attn_read": r_aq, "rho_attn_write": r_aw,
                       "rho_ffn_write": r_fw, "write_pct_median_at_worst6": med_pct}
        print(f"  d{d}({label})".ljust(20) + f"{r_aq:12.3f}{r_aw:12.3f}{r_fw:11.3f}"
              f"{med_pct:22.0%}")

    rng = np.random.default_rng(SEED)
    ctrl = rng.choice(5120, N_CONTROL, replace=False)
    ctrl_fw = np.array([spearman(g["ffn"][:, d], np.log(row_wo[:, d])) for d in ctrl])
    ctrl_aq = np.array([spearman(g["attn"][:, d], np.log(col_q[:, d])) for d in ctrl])

    print(f"\n[对照]随机 {N_CONTROL} 维:ρ(ffn, 写) 中位={np.median(ctrl_fw):+.3f} "
          f"(p5={np.percentile(ctrl_fw, 5):+.2f}, p95={np.percentile(ctrl_fw, 95):+.2f}); "
          f"ρ(attn, 读) 中位={np.median(ctrl_aq):+.3f}")

    # ---- 因果配对 / 滞后 / 累积 ----
    # 数据流:层 l 的 attention 入口看到"上一层 FFN 写完之后"的残差,FFN 入口才看到
    # "本层 attention 刚写完"的残差;HC 的 post/comb 是逐 token 逐副本标量,只调相对权重、
    # 不改变单个写回的通道图案。因此正确的强配对应是 g_ffn[l] ↔ W_attn[l]。
    print("\n读 FFN 写回画像(共享专家 + 抽样路由专家)…")
    row_w2_sh = np.array([
        np.linalg.norm(read_fp8_block(model_dir, wm, f"layers.{l}.ffn.shared_experts.w2"), axis=1)
        for l in LS])
    stride = 384 // N_EXPERT_SAMPLE
    row_w2_rt = np.zeros((len(LS), 5120))
    for l in LS:
        acc = np.zeros(5120)
        for k in range(N_EXPERT_SAMPLE):
            acc += np.linalg.norm(
                read_fp4_expert(model_dir, wm, f"layers.{l}.ffn.experts.{k * stride}.w2"), axis=1)
        row_w2_rt[l] = acc / N_EXPERT_SAMPLE
    row_w2 = row_w2_sh + row_w2_rt

    print("\n[滞后相关] ρ(层 l 的增益画像, 层 l+δ 的写回画像),40 层取中位")
    print(f"  {'配对':30s} " + "".join(f"{'d=%+d' % d:>9s}" for d in (-2, -1, 0, 1, 2)))
    lag: dict[str, dict[str, float]] = {}
    for gtag in ("attn", "ffn"):
        for wtag, wv in (("W_attn(wo_b)", row_wo), ("W_ffn(shared)", row_w2_sh),
                         ("W_ffn(routed)", row_w2_rt), ("W_ffn(total)", row_w2)):
            cells = {}
            for d in (-2, -1, 0, 1, 2):
                rs = [spearman(g[gtag][l], wv[l + d]) for l in LS if 0 <= l + d < len(LS)]
                cells[f"{d:+d}"] = float(np.median(rs))
            lag[f"{gtag}|{wtag}"] = cells
            print(f"  {gtag + ' vs ' + wtag:30s} " + "".join(f"{cells[f'{d:+d}']:9.3f}"
                                                             for d in (-2, -1, 0, 1, 2)))

    print("\n[累积写回 + 衰减] R_l = embed + Σ_{j<l} e^{-(l-1-j)/τ} · (W_attn_j + W_ffn_j)")
    print(f"  {'τ (inf = 不衰减)':>16s} {'g_attn':>9s} {'g_ffn':>9s}"
          f"   | 每层先 z-score 再累加: {'g_attn':>8s} {'g_ffn':>8s}")
    emb_sq = np.zeros(5120)
    for arr_, _ in iter_row_chunks(model_dir / wm["embed.weight"], "embed.weight", chunk_rows=16384):
        emb_sq += (arr_.astype(np.float32) ** 2).sum(axis=0)
    row_embed = np.sqrt(emb_sq)
    writes_raw = [row_wo[l] + row_w2[l] for l in LS]
    writes_z = [(w - w.mean()) / w.std() for w in writes_raw]
    emb_z = (row_embed - row_embed.mean()) / row_embed.std()
    decay_scan: dict[str, dict[str, float]] = {}
    for tau in (0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 16.0, 40.0, 999.0):
        decay = 1.0 if tau >= 999 else float(np.exp(-1.0 / tau))
        acc_r, acc_z = row_embed.copy(), emb_z.copy()
        rr, zz = {"attn": [], "ffn": []}, {"attn": [], "ffn": []}
        for l in LS:
            for t in ("attn", "ffn"):
                rr[t].append(spearman(g[t][l], acc_r))
                zz[t].append(spearman(g[t][l], acc_z))
            acc_r = acc_r * decay + writes_raw[l]
            acc_z = acc_z * decay + writes_z[l]
        lab = "inf" if tau >= 999 else f"{tau:g}"
        decay_scan[lab] = {f"raw_{t}": float(np.median(rr[t])) for t in rr}
        decay_scan[lab].update({f"z_{t}": float(np.median(zz[t])) for t in zz})
        print(f"  {lab:>16s} {np.median(rr['attn']):9.3f} {np.median(rr['ffn']):9.3f}"
              f"{'':16s}{np.median(zz['attn']):8.3f} {np.median(zz['ffn']):8.3f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "gain_equalization.json", "w", encoding="utf-8") as f:
        json.dump({"tracked": out,
                   "control_rho_ffn_write": ctrl_fw.tolist(),
                   "control_rho_attn_read": ctrl_aq.tolist(),
                   "lag_median_rho": lag,
                   "decay_scan": decay_scan,
                   "per_layer": {str(d): {"rel_gain_attn": g["attn"][:, d].tolist(),
                                          "rel_gain_ffn": g["ffn"][:, d].tolist(),
                                          "write_pct": pct_wo[:, d].tolist(),
                                          "read_pct": pct_q[:, d].tolist()}
                                 for d in TRACK}}, f, ensure_ascii=False)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for d, (label, color) in TRACK.items():
        axes[0, 0].plot(LS, g["ffn"][:, d], marker="o", ms=2.5, lw=1.1, color=color,
                        label=f"d{d} ({label})")
        axes[0, 1].plot(LS, pct_wo[:, d], marker="o", ms=2.5, lw=1.1, color=color, label=f"d{d}")
    axes[0, 0].axhline(1.0, color="k", ls=":", lw=1)
    axes[0, 0].set_ylim(0.6, 1.6)
    axes[0, 0].set_title("ffn_norm relative gain (x layer median)")
    axes[0, 0].set_xlabel("layer")
    axes[0, 0].legend(fontsize=7)
    axes[0, 1].set_title("attention write-back loudness of the same channel\n"
                         "(wo_b row-norm percentile)")
    axes[0, 1].set_xlabel("layer")
    axes[0, 1].legend(fontsize=7)

    axes[1, 0].hist(ctrl_fw, bins=18, color="0.75", label=f"random dims (n={N_CONTROL})")
    for d, (label, color) in TRACK.items():
        axes[1, 0].axvline(out[str(d)]["rho_ffn_write"], color=color, lw=1.4, label=f"d{d}")
    axes[1, 0].set_xlabel("rho (relative ffn gain, log wo_b row-norm) over 40 layers")
    axes[1, 0].set_ylabel("count")
    axes[1, 0].set_title("suppressed dims sit at the negative tail of the control")

    for key, cells in lag.items():
        gtag, wtag = key.split("|")
        axes[1, 1].plot([int(k) for k in cells], list(cells.values()), marker="o", ms=4,
                        lw=1.3, alpha=0.85 if wtag == "W_attn(wo_b)" else 0.5,
                        label=f"g_{gtag} vs {wtag}")
    axes[1, 1].axhline(0.0, color="k", lw=0.6, ls=":")
    axes[1, 1].axvline(0, color="k", lw=0.6, ls=":")
    axes[1, 1].set_xlabel("lag delta (gain layer l vs write layer l+delta)")
    axes[1, 1].set_ylabel("median rho over layers")
    axes[1, 1].set_title("causal pairing: g_ffn peaks at delta=0 with the attention write")
    axes[1, 1].legend(fontsize=6)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "gain_equalization.png", dpi=120)
    plt.close(fig)
    print(f"\n输出: {OUT_DIR / 'gain_equalization.json'}  {FIG_DIR / 'gain_equalization.png'}")


if __name__ == "__main__":
    main()
