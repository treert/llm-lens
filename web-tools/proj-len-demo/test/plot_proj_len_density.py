"""随机投影链长度平方 s = ‖B···BA X‖² 的密度：蒙特卡洛 vs 理论（对照 web 演示）。

四个子图：
(a) L=1：χ²_H/D 精确伽马密度，与 X 方向无关（两个不同 X 的直方图重合）；
(b) L=2：独立卡方乘积，K₀ 闭式密度；对数纵轴可见右尾 e^(−2√(s/θ₁θ₂)) 拉伸指数；
(c) L=8 深链：对数正态近似，均值 c 不变而中位数 ≈ c·e^(−L/H) 萎缩（均值-中位数分裂）；
(d) quenched（固定链、随机 X 球面）：三个种子的直方图形状一致、贴合实例高斯
    （self-averaging）——实例内方差 ≈ annealed 方差的大部分，中心 trW/D 的种子间
    跳动 ~√(L(L−1+2c))/D 随维度消失（trW 是平方和；对照点积问题 trM 符号和的
    涨落 √(H/D) 不消失）。

用法:
  python web-tools/proj-len-demo/test/plot_proj_len_density.py
输出: <仓库根>/output/proj-len-density.png
"""

import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(0)


# ---------- ψ / ψ′（与 js/theory.js 同款渐近展开） ----------

def digamma(z: float) -> float:
    acc = 0.0
    while z < 8:
        acc -= 1 / z
        z += 1
    iz, iz2 = 1 / z, 1 / z ** 2
    return acc + math.log(z) - iz / 2 - iz2 * (1 / 12 - iz2 * (1 / 120 - iz2 * (1 / 252 - iz2 / 240)))


def trigamma(z: float) -> float:
    acc = 0.0
    while z < 8:
        acc += 1 / z ** 2
        z += 1
    iz, iz2 = 1 / z, 1 / z ** 2
    return acc + iz * (1 + iz * (0.5 + iz * (1 / 6 - iz2 * (1 / 30 - iz2 / 42))))


# ---------- K₀（Numerical Recipes 系数，与 js 同源） ----------

def bessk0(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    m = x <= 2.0
    y = x[m] ** 2 / 4
    yi = (x[m] / 3.75) ** 2
    i0 = 1 + yi * (3.5156229 + yi * (3.0899424 + yi * (1.2067492 +
        yi * (0.2659732 + yi * (0.0360768 + yi * 0.0045813)))))
    out[m] = -np.log(x[m] / 2) * i0 + (-0.57721566 + y * (0.42278420 +
        y * (0.23069756 + y * (0.03488590 + y * (0.00262698 + y * (0.00010750 +
        y * 0.0000074))))))
    y2 = 2 / x[~m]
    out[~m] = (np.exp(-x[~m]) / np.sqrt(x[~m])) * (1.25331414 +
        y2 * (-0.07832358 + y2 * (0.02189568 + y2 * (-0.01062446 +
        y2 * (0.00587872 + y2 * (-0.00251540 + y2 * 0.00053208))))))
    return out


# ---------- 理论密度 ----------

def gamma_density(s, H, D):
    """L=1：χ²_H/D = Gamma(k=H/2, θ=2/D)。"""
    k, th = H / 2, 2 / D
    s = np.asarray(s, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_f = (k - 1) * np.log(np.maximum(s, 1e-300)) - s / th - math.lgamma(k) - k * math.log(th)
    return np.exp(log_f)


def prod_gamma_density(s, H, D):
    """L=2：f(s) = 2s^(k−1)K₀(2√(s/θ₁θ₂)) / (Γ(k)²(θ₁θ₂)^k)，θ₁=2/D，θ₂=2/H。"""
    k, t1, t2 = H / 2, 2 / D, 2 / H
    s = np.asarray(s, dtype=float)
    z = 2 * np.sqrt(np.maximum(s, 1e-300) / (t1 * t2))
    with np.errstate(divide="ignore"):
        log_f = (math.log(2) + (k - 1) * np.log(np.maximum(s, 1e-300))
                 - 2 * math.lgamma(k) - k * math.log(t1 * t2) + np.log(bessk0(z)))
    return np.exp(log_f)


def lognormal_density(s, mu, var):
    s = np.asarray(s, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.exp(-((np.log(np.maximum(s, 1e-300)) - mu) ** 2) / (2 * var)) / \
            (np.maximum(s, 1e-300) * np.sqrt(2 * np.pi * var))
    return out


def chain_log_params(H, D, L):
    mu = -math.log(D) + L * (math.log(2) + digamma(H / 2)) - (L - 1) * math.log(H)
    var = L * trigamma(H / 2)
    return mu, var


# ---------- 蒙特卡洛 ----------

def mc_annealed(H, D, L, n):
    """方案一：卡方因子化 s = (1/D)·χ²_H·∏(χ²_H/H)（精确分布，与 X 无关）。"""
    s = np.full(n, 1.0 / D)
    s *= np.sum(rng.standard_normal((n, H)) ** 2, axis=1)
    for _ in range(L - 1):
        s *= np.sum(rng.standard_normal((n, H)) ** 2, axis=1) / H
    return s


def gen_chain(H, D, L):
    """合成 M = B_{L−1}···B_1 A，返回 (M, trW, trW2)，W = MᵀM 免构造。"""
    M = rng.standard_normal((H, D)) / np.sqrt(D)
    for _ in range(L - 1):
        M = (rng.standard_normal((H, H)) / np.sqrt(H)) @ M
    trW = float(np.sum(M ** 2))
    G = M @ M.T
    trW2 = float(np.sum(G ** 2))  # G 对称 ⇒ tr(G²) = Σ G_kl²
    return M, trW, trW2


def mc_quenched(M, n):
    """方案二：固定 M（H×D），X 球面均匀，s = ‖MX‖²。"""
    H, D = M.shape
    X = rng.standard_normal((n, D))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return np.sum((X @ M.T) ** 2, axis=1)


H, D = 64, 256
c = H / D

# (a) L=1
s_a = mc_annealed(H, D, 1, 200000)
print(f"[(a) L=1] 均值 {s_a.mean():.4f}（理论 c={c}），std {s_a.std():.5f}"
      f"（理论 √(2H)/D = {np.sqrt(2*H)/D:.5f}）")

# (b) L=2
s_b = mc_annealed(H, D, 2, 200000)
var_b = c ** 2 * ((1 + 2 / H) ** 2 - 1)
print(f"[(b) L=2] 均值 {s_b.mean():.4f}（c={c}），std {s_b.std():.5f}（理论 {np.sqrt(var_b):.5f}）")

# (c) L=8
s_c = mc_annealed(H, D, 8, 200000)
mu8, var8 = chain_log_params(H, D, 8)
med8 = math.exp(mu8)
print(f"[(c) L=8] 均值 {s_c.mean():.4f}（c={c}），样本中位数 {np.median(s_c):.4f}"
      f"（LN 近似 {med8:.4f} ≈ c·e^(−L/H) = {c*math.exp(-8/H):.4f}），"
      f"均值/中位数 {s_c.mean()/np.median(s_c):.3f}（≈ e^(L/H) = {math.exp(8/H):.3f}）")

# (d) quenched：三个种子的链（验证 self-averaging：形状不动、中心微跳）
center_std = np.sqrt(2 * (1 + 2 * c)) / D  # √(L(L−1+2c))/D，L=2
print("[(d) quenched] L=2 固定链、随机 X（对照 annealed std = "
      f"{np.sqrt(var_b):.5f}，中心跳动理论 {center_std:.5f}）：")
quench_samples = []
quench_stats = []
for seed in range(3):
    M, trW, trW2 = gen_chain(H, D, 2)
    sq = mc_quenched(M, 40000)
    inst_mean = trW / D
    inst_var = 2 * (trW2 - trW ** 2 / D) / (D * (D + 2))
    quench_samples.append(sq)
    quench_stats.append((inst_mean, inst_var))
    print(f"  种子 {seed+1}: 样本均值 {sq.mean():.4f}（理论 trW/D = {inst_mean:.4f}），"
          f"样本 std {sq.std():.5f}（理论 {np.sqrt(inst_var):.5f}）")

# ---------- 画图 ----------
_repo_root = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
out_path = _repo_root / "output" / "proj-len-density.png"
out_path.parent.mkdir(exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
t = np.linspace(0, 5, 800)  # 归一坐标 t = s/c

# (a)
ax = axes[0, 0]
bins = np.linspace(0, 5, 161)
ax.hist(s_a / c, bins=bins, density=True, alpha=0.45, color="tab:blue", label="蒙特卡洛")
ax.plot(t[1:], c * gamma_density(t[1:] * c, H, D), "k-", lw=2, label="理论：伽马（χ²_H/D）")
ax.axvline(1, color="gray", ls="--", lw=1, label="均值 c（归一 = 1）")
ax.set_title(f"(a) L=1：卡方精确（H={H}, D={D}, c={c}）")
ax.set_xlabel("t = s / c")
ax.legend(fontsize=9)

# (b)
ax = axes[0, 1]
ax.hist(s_b / c, bins=bins, density=True, alpha=0.45, color="tab:blue", label="蒙特卡洛")
ax.plot(t[1:], c * prod_gamma_density(t[1:] * c, H, D), "r-", lw=2, label="理论：$K_0$ 乘积伽马")
ax.set_yscale("log")
ax.set_ylim(1e-4, 10)
ax.set_title("(b) L=2：乘积卡方（对数纵轴：拉伸指数尾）")
ax.set_xlabel("t = s / c")
ax.legend(fontsize=9)

# (c)
ax = axes[1, 0]
bins_c = np.linspace(0, 5, 201)
ax.hist(s_c / c, bins=bins_c, density=True, alpha=0.45, color="tab:green", label="蒙特卡洛（L=8）")
ax.plot(t[1:], c * lognormal_density(t[1:] * c, mu8, var8), "-", color="purple",
        lw=2, label="对数正态近似")
ax.axvline(1, color="gray", ls="--", lw=1.2, label="均值 c（不变）")
ax.axvline(med8 / c, color="darkorange", ls=":", lw=1.5,
           label=f"中位数 ≈ e^(−L/H) = {med8/c:.3f}")
ax.set_title("(c) L=8 深链：均值-中位数分裂")
ax.set_xlabel("t = s / c")
ax.legend(fontsize=9)

# (d)
ax = axes[1, 1]
bins_d = np.linspace(0, 2.5, 161)
colors = ["tab:blue", "tab:red", "tab:olive"]
for i, (sq, (im, iv)) in enumerate(zip(quench_samples, quench_stats)):
    ax.hist(sq / c, bins=bins_d, density=True, histtype="step", lw=1.3, color=colors[i],
            label=f"种子 {i+1}（trW/D = {im:.3f}）")
ax.plot(t, c * prod_gamma_density(t * c, H, D), "k--", lw=1.5,
        label="annealed 密度（方案一，L=2）")
ax.axvline(1, color="gray", ls="--", lw=1)
ax.set_title("(d) quenched ≈ annealed：self-averaging（中心仅微跳）")
ax.set_xlabel("t = s / c")
ax.legend(fontsize=9)

for ax in axes.flat:
    ax.set_ylabel("密度")
fig.tight_layout()
fig.savefig(out_path, dpi=150)
print(f"图已保存: {out_path}")
