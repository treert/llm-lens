"""瓶颈投影链长度平方 s = ‖(AB)^L x‖² 的密度：蒙特卡洛 vs 理论（对照 web 演示）。

B 为 M×D（方差 1/D 下投）、A 为 D×M（方差 1/M 上投），配对 fan-in 使
E‖ABx‖² = ‖x‖²（均值恒 1）。四个子图：
(a) 中间层 z=Bx：χ²_M/D 精确伽马密度，与 x 方向无关（两个不同 x 的直方图重合）；
(b) 单块 AB：不同形状卡方乘积，广义 Bessel-K_ν 闭式（ν=(M−D)/2）；对数纵轴可见
    右尾 e^(−2√(s/θ₁θ₂)) 拉伸指数；
(c) (AB)^8 深链：对数正态近似，均值 1 不变而中位数 ≈ e^(−L(1/M+1/D)) 萎缩；
(d) quenched（固定链、随机 x 球面）：三个种子的直方图形状一致、贴合实例高斯
    （self-averaging）——实例内方差 ≈ annealed 方差的大部分，中心 trW/D 的种子间
    跳动 ~√((2M+4D+2)/(MD²)) 随维度消失（trW 是平方和；对照点积问题 trM 符号和
    的涨落不消失）。

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


# ---------- 任意阶 Bessel K_ν（log 域：积分表示 + 正向递推，与 js 同款） ----------
# K_ν(x) = ∫₀^∞ e^{−x cosh t} cosh(νt) dt（DLMF 10.32.6）

def log_bessknu(x: float, nu: float) -> float:
    n = int(math.floor(nu))
    mu = nu - n

    def base(nn: float) -> float:  # log K_nn(x)，nn ∈ [0, 1]
        h = 0.1 / max(1.0, math.sqrt(x))
        vals = []
        gmax = -math.inf
        t = 0.0
        while True:
            g = -x * math.cosh(t) + (math.log(math.cosh(nn * t)) if nn > 0 else 0.0)
            vals.append(g)
            if g > gmax:
                gmax = g
            if t > 0 and g < gmax - 42 and len(vals) > 8:
                break
            t += h
        a = np.asarray(vals)
        w = np.ones(len(a))
        w[0] = w[-1] = 0.5
        return gmax + math.log(h * float(np.sum(w * np.exp(a - gmax))))

    if n == 0:
        return base(mu)
    lkm, lk1 = base(mu), base(mu + 1.0)
    for j in range(1, n):  # K_{μ+j+1} = K_{μ+j−1} + 2(μ+j)/x · K_{μ+j}
        lkm, lk1 = lk1, np.logaddexp(lkm, math.log(2 * (mu + j) / x) + lk1)
    return lk1


# ---------- 理论密度 ----------

def gamma_density(s, M, var_b):
    """中间层：varB·χ²_M = Gamma(k=M/2, θ=2·varB)。varB = 1/D（均值 M/D）或 1/M（均值 1）。"""
    k, th = M / 2, 2 * var_b
    s = np.asarray(s, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_f = (k - 1) * np.log(np.maximum(s, 1e-300)) - s / th - math.lgamma(k) - k * math.log(th)
    return np.exp(log_f)


def prod_gamma_density_ab(s, M, D):
    """单块 AB：f(s) = 2s^(k̄−1)K_{k1−k2}(2√(s/θ₁θ₂)) / (Γ(k₁)Γ(k₂)(θ₁θ₂)^k̄)，
    k₁=M/2, θ₁=2/D, k₂=D/2, θ₂=2/M，k̄=(k₁+k₂)/2。"""
    k1, t1, k2, t2 = M / 2, 2 / D, D / 2, 2 / M
    nu = abs(k1 - k2)
    ka = (k1 + k2) / 2
    s = np.asarray(s, dtype=float).ravel()
    log_norm = (math.log(2) - math.lgamma(k1) - math.lgamma(k2) - ka * math.log(t1 * t2))
    out = np.empty_like(s)
    for i, si in enumerate(s):
        si = max(float(si), 1e-300)
        z = 2 * math.sqrt(si / (t1 * t2))
        out[i] = math.exp(log_norm + (ka - 1) * math.log(si) + log_bessknu(z, nu))
    return out


def lognormal_density(s, mu, var):
    s = np.asarray(s, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.exp(-((np.log(np.maximum(s, 1e-300)) - mu) ** 2) / (2 * var)) / \
            (np.maximum(s, 1e-300) * np.sqrt(2 * np.pi * var))
    return out


def chain_log_params(M, D, L):
    mu = L * (digamma(M / 2) - math.log(M / 2) + digamma(D / 2) - math.log(D / 2))
    var = L * (trigamma(M / 2) + trigamma(D / 2))
    return mu, var


# ---------- 蒙特卡洛 ----------

def mc_mid(M, D, n, var_b=None):
    """中间层：s = varB·χ²_M（精确分布，与 x 无关）。varB 默认 1/D。"""
    var_b = var_b if var_b is not None else 1 / D
    return rng.chisquare(M, n) * var_b


def mc_chain(M, D, L, n):
    """完整链：s = ∏_{l=1..L} χ²_M·χ²_D/(DM)（2L 个独立卡方因子）。"""
    s = np.ones(n)
    for _ in range(L):
        s *= rng.chisquare(M, n) * rng.chisquare(D, n) / (D * M)
    return s


def gen_chain(M, D, L):
    """合成 P = A(BA)^{L−1}B（结合律，BA 仅 M×M），返回 (P, trW, trW2)，W = PᵀP 免构造。"""
    B = rng.standard_normal((M, D)) / np.sqrt(D)
    A = rng.standard_normal((D, M)) / np.sqrt(M)
    R = B @ A
    P = A @ np.linalg.matrix_power(R, L - 1) @ B if L > 1 else A @ B
    trW = float(np.sum(P ** 2))
    G = P @ P.T
    trW2 = float(np.sum(G ** 2))  # G 对称 ⇒ tr(G²) = Σ G_kl²
    return P, trW, trW2


def mc_quenched(P, n):
    """方案二：固定 P（p×D），x 球面均匀，s = ‖Px‖²。"""
    p, D = P.shape
    X = rng.standard_normal((n, D))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return np.sum((X @ P.T) ** 2, axis=1)


M, D = 64, 256
c_mid = M / D

# (a) 中间层 z=Bx（varB = 1/D；另验证 varB = 1/M 的均值归一情形）
s_a = mc_mid(M, D, 200000)
print(f"[(a) 中间层, varB=1/D] 均值 {s_a.mean():.4f}（理论 M/D={c_mid}），std {s_a.std():.5f}"
      f"（理论 √(2M)/D = {np.sqrt(2*M)/D:.5f}）")
s_a_m = mc_mid(M, D, 200000, var_b=1 / M)
print(f"[(a) 中间层, varB=1/M] 均值 {s_a_m.mean():.4f}（理论恒 1），std {s_a_m.std():.5f}"
      f"（理论 √(2/M) = {np.sqrt(2/M):.5f}）")

# (b) 单块 AB
s_b = mc_chain(M, D, 1, 200000)
var_b = (1 + 2 / M) * (1 + 2 / D) - 1
print(f"[(b) 单块 AB] 均值 {s_b.mean():.4f}（理论恒 1），std {s_b.std():.5f}"
      f"（理论 {np.sqrt(var_b):.5f}）")

# (c) (AB)^8 深链
L8 = 8
s_c = mc_chain(M, D, L8, 200000)
mu8, var8 = chain_log_params(M, D, L8)
med8 = math.exp(mu8)
print(f"[(c) (AB)^{L8}] 均值 {s_c.mean():.4f}（恒 1），样本中位数 {np.median(s_c):.4f}"
      f"（LN 近似 {med8:.4f} ≈ e^(−L(1/M+1/D)) = {math.exp(-L8*(1/M+1/D)):.4f}），"
      f"均值/中位数 {s_c.mean()/np.median(s_c):.3f}"
      f"（≈ e^(L(1/M+1/D)) = {math.exp(L8*(1/M+1/D)):.3f}）")

# (d) quenched：三个种子的单块链（验证 self-averaging：形状不动、中心微跳）
center_std = math.sqrt((2 * M + 4 * D + 2) / (M * D ** 2))  # 单块 AB
print("[(d) quenched] 单块固定链、随机 x（对照 annealed std = "
      f"{np.sqrt(var_b):.5f}，中心跳动理论 {center_std:.5f}）：")
quench_samples = []
quench_stats = []
for seed in range(3):
    P, trW, trW2 = gen_chain(M, D, 1)
    sq = mc_quenched(P, 40000)
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
t = np.linspace(1e-4, 5, 800)  # 归一坐标 t = s/c

# (a)
ax = axes[0, 0]
bins = np.linspace(0, 5, 161)
ax.hist(s_a / c_mid, bins=bins, density=True, alpha=0.45, color="tab:blue", label="蒙特卡洛")
ax.plot(t, c_mid * gamma_density(t * c_mid, M, 1 / D), "k-", lw=2, label="理论：伽马（χ²_M/D）")
ax.axvline(1, color="gray", ls="--", lw=1, label="均值 M/D（归一 = 1）")
ax.set_title(f"(a) 中间层 z=Bx：卡方精确（M={M}, D={D}, M/D={c_mid}）")
ax.set_xlabel("t = s / (M/D)")
ax.legend(fontsize=9)

# (b)
ax = axes[0, 1]
ax.hist(s_b, bins=bins, density=True, alpha=0.45, color="tab:blue", label="蒙特卡洛")
ax.plot(t, prod_gamma_density_ab(t, M, D), "r-", lw=2, label="理论：$K_\\nu$ 乘积伽马")
ax.set_yscale("log")
ax.set_ylim(1e-4, 10)
ax.set_title(f"(b) 单块 AB：乘积卡方，ν=(M−D)/2={-96}（对数纵轴：拉伸指数尾）")
ax.set_xlabel("t = s（均值恒 1）")
ax.legend(fontsize=9)

# (c)
ax = axes[1, 0]
bins_c = np.linspace(0, 5, 201)
ax.hist(s_c, bins=bins_c, density=True, alpha=0.45, color="tab:green",
        label=f"蒙特卡洛（(AB)^{L8}）")
ax.plot(t, lognormal_density(t, mu8, var8), "-", color="purple", lw=2, label="对数正态近似")
ax.axvline(1, color="gray", ls="--", lw=1.2, label="均值 1（不变）")
ax.axvline(med8, color="darkorange", ls=":", lw=1.5,
           label=f"中位数 ≈ e^(−L(1/M+1/D)) = {med8:.3f}")
ax.set_title(f"(c) (AB)^{L8} 深链：均值-中位数分裂")
ax.set_xlabel("t = s（均值恒 1）")
ax.legend(fontsize=9)

# (d)
ax = axes[1, 1]
bins_d = np.linspace(0, 2.5, 161)
colors = ["tab:blue", "tab:red", "tab:olive"]
for i, (sq, (im, iv)) in enumerate(zip(quench_samples, quench_stats)):
    ax.hist(sq, bins=bins_d, density=True, histtype="step", lw=1.3, color=colors[i],
            label=f"种子 {i+1}（trW/D = {im:.3f}）")
ax.plot(t, prod_gamma_density_ab(t, M, D), "k--", lw=1.5,
        label="annealed 密度（方案一，单块）")
ax.axvline(1, color="gray", ls="--", lw=1)
ax.set_title("(d) quenched ≈ annealed：self-averaging（中心仅微跳）")
ax.set_xlabel("t = s（均值恒 1）")
ax.legend(fontsize=9)

for ax in axes.flat:
    ax.set_ylabel("密度")
fig.tight_layout()
fig.savefig(out_path, dpi=150)
print(f"图已保存: {out_path}")
