"""可视化 qk_spectrum.npz(需先运行 extract_qk_spectrum.py)。

产出三张图到 output/kimi_k3/figures/:
- qk_metric_heatmaps.png : sigma1 / r_eff / sym / nope_energy_frac 的层×头热力图
- qk_spectrum_curves.png : 代表层的逐头谱曲线 + 随机高斯基线(Fuss-Catalan  Monte Carlo)
- qk_latent_spectra.png  : q_a / kv_a 潜空间谱曲线与有效秩随层变化

随机基线(docs/qk-spectrum.md §3):W_Q、W_K 取同形状高斯矩阵(sigma_w^2 = 1/D),
用 §2 捷径求 M 的奇异值——初始化时 E||M||_F^2 = H,谱服从 Fuss-Catalan 分布。

用法(在仓库根目录下):
    python analysis/kimi_k3/plot_qk_spectrum.py
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

HIDDEN = 7168
HEAD_DIM = 192
CURVE_LAYERS = [3, 19, 43, 63, 75, 92]  # 早/中/晚/峰值代表层


def random_baseline(n_samples: int = 8, seed: int = 0) -> np.ndarray:
    """Monte Carlo 生成初始化基线的 M 谱,返回 (n_samples, HEAD_DIM) 奇异值。"""
    rng = np.random.default_rng(seed)
    svs = np.zeros((n_samples, HEAD_DIM))
    for i in range(n_samples):
        W_Q = rng.standard_normal((HEAD_DIM, HIDDEN)) / np.sqrt(HIDDEN)
        W_K = rng.standard_normal((HEAD_DIM, HIDDEN)) / np.sqrt(HIDDEN)
        Uq, sq, _ = np.linalg.svd(W_Q, full_matrices=False)
        Uk, sk, _ = np.linalg.svd(W_K, full_matrices=False)
        svs[i] = np.linalg.svd(
            (sq[:, None] * (Uq.T @ Uk)) * sk[None, :], compute_uv=False)
    return svs


def plot_heatmaps(d: dict, fig_dir: Path) -> None:
    items = [("sigma1", r"$\sigma_1$"), ("r_eff", "有效秩 $r_{eff}$"),
             ("sym", "对称性 sym(M)"), ("nope_energy_frac", "nope 能量占比")]
    layers = d["layer_indices"]
    fig, axes = plt.subplots(4, 1, figsize=(14, 13), sharex=True)
    for ax, (key, label) in zip(axes, items):
        im = ax.imshow(d[f"metric_{key}"], aspect="auto", origin="lower",
                       cmap="viridis",
                       extent=[-0.5, d["metric_sigma1"].shape[1] - 0.5,
                               -0.5, len(layers) - 0.5])
        ax.set_ylabel("层")
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels(layers, fontsize=7)
        ax.set_title(label)
        fig.colorbar(im, ax=ax, pad=0.01)
    axes[-1].set_xlabel("头编号")
    fig.suptitle("K3 MLA 逐层逐头 QK 谱指标", y=0.995)
    fig.tight_layout()
    path = fig_dir / "qk_metric_heatmaps.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"已写出: {path}")


def plot_spectrum_curves(d: dict, fig_dir: Path) -> None:
    sv = d["singular_values"]
    layers = list(d["layer_indices"])
    base = random_baseline()
    x = np.arange(1, sv.shape[2] + 1)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for ax, layer in zip(axes.flat, CURVE_LAYERS):
        li = layers.index(layer)
        for h in range(sv.shape[1]):
            ax.plot(x, sv[li, h], color="tab:blue", alpha=0.12, linewidth=0.6)
        ax.plot(x, np.median(sv[li], axis=0), color="tab:blue", linewidth=1.8,
                label="训练后(96 头中位)")
        ax.fill_between(x, base.min(axis=0), base.max(axis=0), color="gray",
                        alpha=0.4, label="随机基线(min~max)")
        ax.plot(x, np.median(base, axis=0), color="gray", linewidth=1.2,
                linestyle="--")
        ax.set_yscale("log")
        ax.set_title(f"层 {layer}")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8, loc="lower left")
    for ax in axes[-1]:
        ax.set_xlabel("奇异值序号 i")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\sigma_i$")
    fig.suptitle("M 奇异值谱:训练后 vs 初始化随机基线(Fuss-Catalan Monte Carlo)")
    fig.tight_layout()
    path = fig_dir / "qk_spectrum_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"已写出: {path}")


def plot_latent(d: dict, fig_dir: Path) -> None:
    layers = d["layer_indices"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0, 1, len(layers)))
    for c, li in zip(colors, range(len(layers))):
        axes[0].plot(np.sqrt(np.maximum(d["latent_sv_q"][li], 0)),
                     color=c, alpha=0.8, linewidth=1)
        axes[0].plot(np.sqrt(np.maximum(d["latent_sv_k"][li], 0)),
                     color=c, alpha=0.8, linewidth=1, linestyle="--")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("奇异值序号")
    axes[0].set_ylabel("奇异值")
    axes[0].set_title("潜空间谱(实线 q_a / 虚线 kv_a,颜色由浅到深 = 层由浅到深)")
    axes[0].grid(alpha=0.3)

    def r_eff(sv):
        return sv.sum(axis=1) ** 2 / (sv**2).sum(axis=1)

    axes[1].plot(layers, r_eff(d["latent_sv_q"]), "o-", label="q_a (上限 1536)")
    axes[1].plot(layers, r_eff(d["latent_sv_k"]), "s-", label="kv_a (上限 512)")
    axes[1].set_xlabel("层")
    axes[1].set_ylabel("有效秩")
    axes[1].set_title("潜空间有效秩随层变化")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    path = fig_dir / "qk_latent_spectra.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"已写出: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "output" / "kimi_k3"),
                        help="qk_spectrum.npz 所在目录(默认 output/kimi_k3)")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    d = dict(np.load(out_dir / "qk_spectrum.npz"))
    plot_heatmaps(d, fig_dir)
    plot_spectrum_curves(d, fig_dir)
    plot_latent(d, fig_dir)


if __name__ == "__main__":
    main()
