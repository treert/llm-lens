"""画 Fuss-Catalan 分布 FC_2 的密度曲线,并用随机矩阵蒙特卡洛验证。

FC_2 = 两个 Marchenko-Pastur 律的自由乘性卷积 MP_c ⊠ MP_c (c = H/D),
即 M = W_Q^T W_K (独立高斯, 元素方差 1/D) 的非零平方奇异值的渐近密度。

理论密度由 S-变换推出: 矩生成函数 M(w) 满足三次方程
  w c^2 M^3 + 2w(1-c)c M^2 + (w(1-c)^2 - 1) M + 1 = 0,  w = 1/x
密度 p(x) = -Im M(1/x) / (π x) (取 Im M < 0 的根)。
c = 1 时退化为 w M^3 = M - 1, 即 Fuss-Catalan 生成函数方程, 支撑 [0, 27/4]。

用法:
  python web-tools/spectrum-demo/test/plot_fuss_catalan.py
输出: <仓库根>/output/fuss-catalan.png
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(0)


def squared_singular_values(D: int, H: int, n_trials: int) -> np.ndarray:
    """Monte Carlo: M = W_Q^T W_K 的平方奇异值 (元素方差 1/D, 与初始化一致)。"""
    out = []
    for _ in range(n_trials):
        Wq = rng.standard_normal((H, D)) / np.sqrt(D)
        Wk = rng.standard_normal((H, D)) / np.sqrt(D)
        M = Wq.T @ Wk
        sv2 = np.linalg.svd(M, compute_uv=False) ** 2
        out.append(sv2[sv2 > 1e-10])  # H<D 时 D-H 个奇异值精确为 0, 只看非零谱
    return np.concatenate(out)


def product_free_poisson_density(xs: np.ndarray, c: float) -> np.ndarray:
    """MP_c 与 MP_c 自由乘积 (M = W_Q^T W_K, H/D -> c) 的渐近密度。c=1 即 FC_2。"""
    out = np.zeros_like(xs)
    a = 1.0 - c
    for i, x in enumerate(xs):
        w = 1.0 / x
        roots = np.roots([w * c * c, 2 * w * a * c, w * a * a - 1.0, 1.0])
        cplx = roots[np.abs(roots.imag) > 1e-9]
        if len(cplx) == 0:
            continue  # 支撑之外, 密度为 0
        out[i] = -cplx[cplx.imag < 0][0].imag / (np.pi * x)
    return out


# --- 理论曲线的自洽性: 数值积分应 = 1 ---
x_fine = np.linspace(1e-3, 7.0, 20000)
p_fine = product_free_poisson_density(x_fine, c=1.0)
print(f"[检查] c=1 理论密度数值积分 = {np.trapezoid(p_fine, x_fine):.4f} (应为 1)")
print(f"[检查] c=1 理论密度一阶矩 = {np.trapezoid(x_fine * p_fine, x_fine):.4f} (应为 1)")

# --- 蒙特卡洛 ---
sq_square = squared_singular_values(D=256, H=256, n_trials=60)   # H = D, 对应 FC_2
sq_tall = squared_singular_values(D=2048, H=128, n_trials=60)    # H << D, c = 1/16

print(f"[H=D]  均值 = {sq_square.mean():.4f} (理论 1), 方差 = {sq_square.var():.4f} (理论 2)")
print(f"[H=D]  最大值 = {sq_square.max():.3f} (理论支撑右端 27/4 = 6.75)")
print(f"[H<<D] 均值 = {sq_tall.mean():.4f} (理论 1), 方差 = {sq_tall.var():.4f} (理论 2c = {2/16:.4f})")
print(f"[H<<D] 最大值 = {sq_tall.max():.3f} (理论支撑右端约 1.87)")

# --- 画图 ---
# 输出到仓库根的 output/（向上找到含 pyproject.toml 的目录, 不依赖脚本摆放位置）
_repo_root = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
out_path = _repo_root / "output" / "fuss-catalan.png"
out_path.parent.mkdir(exist_ok=True)

fig, ax = plt.subplots(figsize=(8, 5))
bins = np.linspace(0, 7, 141)
ax.hist(sq_square, bins=bins, density=True, alpha=0.45, color="tab:blue",
        label="蒙特卡洛 $M=W_Q^\\top W_K$ ($H=D=256$)")
ax.hist(sq_tall, bins=bins, density=True, histtype="step", lw=1.5, color="tab:green",
        label="蒙特卡洛 ($H=128, D=2048$, $c=1/16$)")
x_theory = np.linspace(1e-3, 6.749, 1500)
ax.plot(x_theory, product_free_poisson_density(x_theory, c=1.0), "r-", lw=2,
        label="理论: Fuss-Catalan $FC_2$ ($c=1$)")
x_gen = np.linspace(1e-3, 2.5, 1500)
ax.plot(x_gen, product_free_poisson_density(x_gen, c=1 / 16), "k--", lw=1.5,
        label="理论: $MP_c \\boxtimes MP_c$ ($c=1/16$)")
ax.axvline(27 / 4, color="r", ls=":", lw=1)
ax.text(27 / 4 - 0.1, 0.02, "$27/4$", color="r", ha="right")
ax.set_xlabel("$x$ = 平方奇异值")
ax.set_ylabel("密度")
ax.set_title("初始化时 $M=W_Q^\\top W_K$ 的平方奇异值分布 vs 理论")
ax.set_ylim(0, 3)
ax.legend()
fig.tight_layout()
fig.savefig(out_path, dpi=150)
print(f"图已保存: {out_path}")
