"""双随机投影点积 s = (Ma A)·(Mb B) 的密度：蒙特卡洛 vs 理论（对照 web 演示）。

四个子图：
(a) 方案一（固定 A、B，随机 Ma、Mb）：ρ = 0 与 ρ = 0.8 的直方图重合，
    且都落在对称 Variance-Gamma（Bessel-K）密度上——分布与夹角严格无关；
(b) 方案一的形状随 H 变化：H=1 原点对数发散、H=2 即 Laplace、大 H 高斯化
    （超额峰度 6/H，归一坐标 t = s/(√H/D) 下只剩峰度差异）；
(c) 方案二（固定无关矩阵，随机 A、B）：同一批矩阵下 ρ 从 0 调到 0.8，
    均值偏移 ρ·trM/D 由该批矩阵的 trM 决定（换种子方向随机翻转）；
(d) 方案三（Mb = αMa + √(1-α²)G）：trM ≈ αH 成为确定性信号，ρ 效应稳定显现。

用法:
  python web-tools/proj-dot-demo/test/plot_proj_dot_density.py
输出: <仓库根>/output/proj-dot-density.png
"""

import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(0)


# ---------- 理论：对称 Variance-Gamma 密度（与 js/theory.js 同款逼近） ----------

def _bessk0(x: np.ndarray) -> np.ndarray:
    """K_0(x)，x > 0，Numerical Recipes 系数。"""
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


def vg_density(s: np.ndarray, H: int, beta: float) -> np.ndarray:
    """f(s) = 2|s|^ν K_ν(|s|/β) / [(2β)^(ν+1) √π Γ(H/2)]，ν = (H-1)/2。

    用对数形式 ln f = -ν ln2 - ln β - ½lnπ - lnΓ(H/2) + ln(z^ν K_ν(z)) 避免 0·∞。
    本脚本只需 H=1（K_0）与偶数 H（半整数阶 K_{n+1/2} 的初等和）。
    """
    s = np.asarray(s, dtype=float)
    z = np.abs(s) / beta
    if H == 1:
        return _bessk0(np.maximum(z, 1e-300)) / (np.pi * beta)
    assert H % 2 == 0, "本对照脚本只实现了 H=1 与偶数 H"
    n = H // 2 - 1
    nu = n + 0.5
    # ln(z^ν K_ν(z)) = ½ln(π/2) - z + ln(Σ_{k=0..n} a_k z^{n-k})，a_k/a_{k-1} = (n+k)(n-k+1)/(2k)
    a = 1.0
    poly = np.where(z > 0, z, 1.0) ** n
    poly = np.where(z > 0, poly, 0.0)
    for k in range(1, n + 1):
        a *= (n + k) * (n - k + 1) / (2 * k)
        if k < n:
            poly = poly + a * np.where(z > 0, z, 1.0) ** (n - k)
        else:
            poly = poly + a  # k = n 为常数项，z=0 时唯一的幸存项
    log_scaled = 0.5 * math.log(math.pi / 2) - z + np.log(poly)
    log_f = (-nu * math.log(2) - math.log(beta) - 0.5 * math.log(math.pi)
             - math.lgamma(H / 2) + log_scaled)
    return np.exp(log_f)


# ---------- 蒙特卡洛 ----------

def mc_scheme1(H: int, D: int, rho: float, n: int) -> np.ndarray:
    """方案一（显式两列版）：A = e1，B = ρe1 + √(1-ρ²)e2。
    u = σg1，v = σ(ρg2 + √(1-ρ²)g3)——ρ 在实现里出现，在分布里消失。"""
    sig = 1 / np.sqrt(D)
    sr = np.sqrt(1 - rho ** 2)
    u = sig * rng.standard_normal((n, H))
    v = sig * (rho * rng.standard_normal((n, H)) + sr * rng.standard_normal((n, H)))
    return np.sum(u * v, axis=1)


def gen_matrices(H: int, D: int, alpha: float):
    """Ma、Mb（元素 N(0, 1/D)）；alpha > 0 时 Mb = αMa + √(1-α²)G（方案三）。
    返回 (Ma, Mb, (trM, ‖M‖²_F, ‖M_s‖²_F))，不变量免构造 D×D 的 M。"""
    sig = 1 / np.sqrt(D)
    Ma = sig * rng.standard_normal((H, D))
    if alpha > 0:
        Mb = alpha * Ma + np.sqrt(1 - alpha ** 2) * sig * rng.standard_normal((H, D))
    else:
        Mb = sig * rng.standard_normal((H, D))
    trM = float(np.sum(Ma * Mb))
    Ga, Gb, Gx = Ma @ Ma.T, Mb @ Mb.T, Ma @ Mb.T
    normF2 = float(np.sum(Ga * Gb))
    trM2 = float(np.sum(Gx * Gx.T))
    return Ma, Mb, (trM, normF2, (normF2 + trM2) / 2)


def mc_fixed_m(Ma, Mb, rho: float, n: int) -> np.ndarray:
    """方案二/三：固定矩阵，球面构造精确 cos = ρ 的单位向量对（A 均匀，B = ρA + √(1-ρ²)Ẑ⊥）。"""
    H, D = Ma.shape
    A = rng.standard_normal((n, D))
    A /= np.linalg.norm(A, axis=1, keepdims=True)
    Z = rng.standard_normal((n, D))
    Z -= np.sum(Z * A, axis=1, keepdims=True) * A
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    B = rho * A + np.sqrt(1 - rho ** 2) * Z
    return np.sum((A @ Ma.T) * (B @ Mb.T), axis=1)


def exkurt(s: np.ndarray) -> float:
    d = s - s.mean()
    return float(np.mean(d ** 4) / d.var() ** 2 - 3)


H, D = 64, 256
beta = 1 / D
sig0 = np.sqrt(H) / D  # VG 标准差 √Hβ

# (a) 方案一：ρ = 0 vs ρ = 0.8
s1_r0 = mc_scheme1(H, D, 0.0, 200000)
s1_r8 = mc_scheme1(H, D, 0.8, 200000)
print(f"[(a) 方案一] ρ=0:   均值 {s1_r0.mean():+.2e}, std {s1_r0.std():.5f}, 超额峰度 {exkurt(s1_r0):.4f}")
print(f"[(a) 方案一] ρ=0.8: 均值 {s1_r8.mean():+.2e}, std {s1_r8.std():.5f}, 超额峰度 {exkurt(s1_r8):.4f}")
print(f"          理论: std = √H/D = {sig0:.5f}, 超额峰度 6/H = {6/H:.4f}——两组样本统计一致")

# (c) 方案二：同一批无关矩阵下 ρ = 0 vs ρ = 0.8
Ma2, Mb2, st2 = gen_matrices(H, D, alpha=0.0)
s2_r0 = mc_fixed_m(Ma2, Mb2, 0.0, 40000)
s2_r8 = mc_fixed_m(Ma2, Mb2, 0.8, 40000)
trM2, normF2, normMs2 = st2
mean2 = 0.8 * trM2 / D
var2 = (2 * 0.64 * normMs2 + 0.36 * normF2) / D ** 2
print(f"[(c) 方案二] 同一批矩阵（trM = {trM2:+.3f}）：")
print(f"  ρ=0:   样本均值 {s2_r0.mean():+.5f}（理论 0）")
print(f"  ρ=0.8: 样本均值 {s2_r8.mean():+.5f}，理论 ρ·trM/D = {mean2:+.5f}")

# (d) 方案三：α = 0.7 相关矩阵
Ma3, Mb3, st3 = gen_matrices(H, D, alpha=0.7)
s3 = mc_fixed_m(Ma3, Mb3, 0.8, 40000)
trM3, normF3, normMs3 = st3
mean3 = 0.8 * trM3 / D
var3 = (2 * 0.64 * normMs3 + 0.36 * normF3) / D ** 2
print(f"[(d) 方案三] α=0.7, ρ=0.8: 样本均值 {s3.mean():+.5f}，理论 ρ·trM/D = {mean3:+.5f} "
      f"(trM = {trM3:+.2f} ≈ αH = {0.7 * H:.1f})")

# ---------- 画图 ----------
_repo_root = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
out_path = _repo_root / "output" / "proj-dot-density.png"
out_path.parent.mkdir(exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
t = np.linspace(-5, 5, 800)  # 归一坐标 t = s/σ0
bins = np.linspace(-5, 5, 161)


def gauss_pdf(s, mu, var):
    return np.exp(-((s - mu) ** 2) / (2 * var)) / np.sqrt(2 * np.pi * var)


# (a)
ax = axes[0, 0]
ax.hist(s1_r0 / sig0, bins=bins, density=True, alpha=0.4, color="tab:blue", label="蒙特卡洛 ρ = 0")
ax.hist(s1_r8 / sig0, bins=bins, density=True, histtype="step", lw=1.2,
        color="tab:orange", label="蒙特卡洛 ρ = 0.8")
ax.plot(t, sig0 * vg_density(t * sig0, H, beta), "k-", lw=2, label="理论：Variance-Gamma")
ax.set_title(f"(a) 方案一：分布与 ρ 严格无关（H={H}, D={D}）")
ax.set_xlabel("t = s / (√H/D)")
ax.legend(fontsize=9)

# (b)
ax = axes[0, 1]
for h in [1, 2, 8, 64]:
    s0 = np.sqrt(h) / D
    dens = s0 * vg_density(np.abs(t) * s0 * np.sign(t), h, beta)
    ax.plot(t, dens, lw=1.5, label=f"H={h}（峰度 6/H = {6/h:.2f}）")
ax.plot(t, gauss_pdf(t, 0, 1), "k--", lw=1, label="高斯（H→∞）")
ax.set_yscale("log")
ax.set_ylim(1e-3, 30)
ax.set_title("(b) 方案一的形状随 H：超额峰度 6/H")
ax.set_xlabel("t = s / (√H/D)")
ax.legend(fontsize=8)

# (c)
ax = axes[1, 0]
ax.hist(s2_r0 / sig0, bins=bins, density=True, alpha=0.4, color="tab:blue", label="蒙特卡洛 ρ = 0")
ax.hist(s2_r8 / sig0, bins=bins, density=True, alpha=0.35, color="tab:red", label="蒙特卡洛 ρ = 0.8")
ax.plot(t, sig0 * gauss_pdf(t * sig0, mean2, var2), "r-", lw=2,
        label=f"实例高斯（ρ=0.8，trM = {trM2:+.2f}）")
ax.plot(t, sig0 * vg_density(t * sig0, H, beta), "k--", lw=1.5, label="VG 参考（方案一）")
ax.axvline(mean2 / sig0, color="r", ls=":", lw=1)
ax.set_title("(c) 方案二：同一批无关矩阵，均值偏移 ρ·trM/D")
ax.set_xlabel("t = s / (√H/D)")
ax.legend(fontsize=9)

# (d)
ax = axes[1, 1]
ax.hist(s3 / sig0, bins=bins, density=True, alpha=0.45, color="tab:green",
        label="蒙特卡洛（α=0.7, ρ=0.8）")
ax.plot(t, sig0 * gauss_pdf(t * sig0, mean3, var3), "r-", lw=2,
        label=f"实例高斯（trM = {trM3:.1f} ≈ αH）")
ax.plot(t, sig0 * vg_density(t * sig0, H, beta), "k--", lw=1.5, label="VG 参考")
ax.axvline(mean3 / sig0, color="r", ls=":", lw=1)
ax.set_title("(d) 方案三：相关矩阵，ρ 效应成为确定性信号")
ax.set_xlabel("t = s / (√H/D)")
ax.legend(fontsize=9)

for ax in axes.flat:
    ax.set_ylabel("密度")
fig.tight_layout()
fig.savefig(out_path, dpi=150)
print(f"图已保存: {out_path}")
