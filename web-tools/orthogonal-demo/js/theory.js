/**
 * 高维随机向量最大点乘（最小夹角）的理论公式。
 *
 * 理论来源：Cai, Fan, Jiang, "Distributions of Angles in Random Packing on Spheres",
 * JMLR 14 (2013), arXiv:1306.0256。
 *
 * 记号：N = 维数，K = 向量个数（论文中分别为 p、n）。
 * 所有向量独立同分布，均匀取自单位球面 S^{N-1}。
 */
(function (global) {
  'use strict';

  // 极值分布常数 kappa：单侧 max ρ 为 1/(4√(2π))；双侧 max|ρ| 为其两倍
  function kappa(twoSided) {
    return twoSided
      ? 1 / (2 * Math.sqrt(2 * Math.PI))
      : 1 / (4 * Math.sqrt(2 * Math.PI));
  }

  /**
   * Lanczos 近似的 log Γ(x)，双精度下足够精确。
   */
  function lgamma(x) {
    const c = [
      0.99999999999980993, 676.5203681218851, -1259.1392167224028,
      771.32342877765313, -176.61502916214059, 12.507343278686905,
      -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
    ];
    if (x < 0.5) {
      return Math.log(Math.PI / Math.sin(Math.PI * x)) - lgamma(1 - x);
    }
    x -= 1;
    let a = c[0];
    const t = x + 7.5;
    for (let i = 1; i < 9; i++) a += c[i] / (x + i);
    return (
      0.5 * Math.log(2 * Math.PI) +
      (x + 0.5) * Math.log(t) -
      t +
      Math.log(a)
    );
  }

  /**
   * 单对向量点乘 ρ = cosθ 的精确密度（引理 6.1）：
   * g(ρ) = Γ(N/2) / (√π Γ((N-1)/2)) · (1-ρ²)^{(N-3)/2}, |ρ| < 1
   */
  function pairDotDensity(r, N) {
    if (Math.abs(r) >= 1) return 0;
    const logC =
      lgamma(N / 2) - lgamma((N - 1) / 2) - 0.5 * Math.log(Math.PI);
    return Math.exp(logC + ((N - 3) / 2) * Math.log(1 - r * r));
  }

  /**
   * 极值规范化常数 a(N,K) = 4 lnK − ln lnK。
   * 渐近结果（Cai–Fan–Jiang 定理 5）：W = N·(max ρ)² − a 依分布收敛到
   * 标准 Gumbel：P(W ≤ y) = exp(−κ e^{−y/2})。
   * 注意：论文原文给出的是 −W 的分布 1 − exp(−κ e^{y/2})
   * （由 2N log sinΘ_min ≈ −N(max ρ)² 而来），此处统一用 W 的形式。
   */
  function centering(K) {
    return 4 * Math.log(K) - Math.log(Math.log(K));
  }

  /**
   * max ρ 的近似密度（Gumbel 渐近）。R = sqrt((W+a)/N) 的变量替换：
   * f_R(r) = 2N·r · (κ/2)·e^{−y/2}·exp(−κ e^{−y/2})，其中 y = N r² − a。
   */
  function maxDotDensity(r, N, K, twoSided) {
    if (r <= 0 || r >= 1) return 0;
    const kp = kappa(twoSided);
    const y = N * r * r - centering(K);
    const ey = Math.exp(-y / 2);
    const logf = Math.log(kp / 2) - y / 2 - kp * ey;
    if (!isFinite(logf)) return 0;
    return 2 * N * r * Math.exp(logf);
  }

  /**
   * max ρ 的近似分位数：给定 p∈(0,1)，返回 r 使 P(max ρ ≤ r) ≈ p。
   * 由 F = exp(−κ e^{−y/2}) 反解 y = −2 ln(−ln(p)/κ)。
   */
  function maxDotQuantile(p, N, K, twoSided) {
    const kp = kappa(twoSided);
    const y = -2 * Math.log(-Math.log(p) / kp);
    const v = (centering(K) + y) / N;
    return v > 0 ? Math.sqrt(v) : 0;
  }

  /**
   * 一阶近似：E[max ρ] ≈ 2√(lnK / N)。
   * 由 M≈K²/2 对近似独立 N(0,1/N) 取最大值的经典极值启发式。
   */
  function firstOrderMean(N, K) {
    return 2 * Math.sqrt(Math.log(K) / N);
  }

  // ================= 方案 A：F_beta^M 近似（全 K 范围） =================

  /** 不完全 beta 的连分式部分（Numerical Recipes betacf） */
  function betaCF(a, b, x) {
    const MAXIT = 200;
    const EPS = 3e-14;
    const FPMIN = 1e-300;
    const qab = a + b;
    const qap = a + 1;
    const qam = a - 1;
    let c = 1;
    let d = 1 - (qab * x) / qap;
    if (Math.abs(d) < FPMIN) d = FPMIN;
    d = 1 / d;
    let h = d;
    for (let m = 1; m <= MAXIT; m++) {
      const m2 = 2 * m;
      let aa = (m * (b - m) * x) / ((qam + m2) * (a + m2));
      d = 1 + aa * d;
      if (Math.abs(d) < FPMIN) d = FPMIN;
      c = 1 + aa / c;
      if (Math.abs(c) < FPMIN) c = FPMIN;
      d = 1 / d;
      h *= d * c;
      aa = (-(a + m) * (qab + m) * x) / ((a + m2) * (qap + m2));
      d = 1 + aa * d;
      if (Math.abs(d) < FPMIN) d = FPMIN;
      c = 1 + aa / c;
      if (Math.abs(c) < FPMIN) c = FPMIN;
      d = 1 / d;
      const del = d * c;
      h *= del;
      if (Math.abs(del - 1) < EPS) break;
    }
    return h;
  }

  /** 正则化不完全 beta 函数 I_x(a,b) */
  function regIncBeta(x, a, b) {
    if (x <= 0) return 0;
    if (x >= 1) return 1;
    const bt = Math.exp(
      lgamma(a + b) - lgamma(a) - lgamma(b) +
        a * Math.log(x) + b * Math.log(1 - x)
    );
    if (x < (a + 1) / (a + b + 2)) return (bt * betaCF(a, b, x)) / a;
    return 1 - (bt * betaCF(b, a, 1 - x)) / b;
  }

  /**
   * 单对点乘的精确 CDF：F(t) = P(ρ ≤ t)。
   * 由 ρ² ~ Beta(1/2, (N-1)/2) 及对称性：
   *   t ≥ 0: F(t) = 1 − ½·I_{1−t²}((N−1)/2, 1/2)
   *   t < 0: F(t) = ½·I_{1−t²}((N−1)/2, 1/2)
   */
  function pairDotCDF(t, N) {
    if (t <= -1) return 0;
    if (t >= 1) return 1;
    const half = 0.5 * regIncBeta(1 - t * t, (N - 1) / 2, 0.5);
    return t >= 0 ? 1 - half : half;
  }

  /**
   * max ρ 的 CDF（独立性近似）：P(max ≤ t) ≈ F(t)^M。
   * 双侧：P(max|ρ| ≤ t) ≈ (2F(t) − 1)^M（t ≥ 0）。
   * K=2 时精确；大 K 时渐近等价于 Gumbel 形式。
   */
  function maxDotCDFBeta(t, N, K, twoSided) {
    const M = (K * (K - 1)) / 2;
    const F = pairDotCDF(t, N);
    if (twoSided) {
      if (t <= 0) return 0;
      const v = 2 * F - 1;
      return v <= 0 ? 0 : Math.exp(M * Math.log(v));
    }
    return F <= 0 ? 0 : Math.exp(M * Math.log(F));
  }

  /**
   * max ρ 的密度（F^M 求导）：单侧 M·F^{M−1}·g；双侧 2M·(2F−1)^{M−1}·g。
   */
  function maxDotDensityBeta(t, N, K, twoSided) {
    const M = (K * (K - 1)) / 2;
    const g = pairDotDensity(t, N);
    if (g === 0) return 0;
    const F = pairDotCDF(t, N);
    let logv;
    if (twoSided) {
      if (t <= 0) return 0;
      const v = 2 * F - 1;
      if (v <= 0) return 0;
      logv = Math.log(2 * M) + (M - 1) * Math.log(v) + Math.log(g);
    } else {
      if (F <= 0) return 0;
      logv = Math.log(M) + (M - 1) * Math.log(F) + Math.log(g);
    }
    if (!isFinite(logv)) return 0;
    return Math.exp(logv);
  }

  /** max ρ 的分位数（F^M 形式，二分求逆） */
  function maxDotQuantileBeta(p, N, K, twoSided) {
    let lo = twoSided ? 0 : -1;
    let hi = 1;
    for (let i = 0; i < 80; i++) {
      const mid = (lo + hi) / 2;
      if (maxDotCDFBeta(mid, N, K, twoSided) < p) lo = mid;
      else hi = mid;
    }
    return (lo + hi) / 2;
  }

  global.Theory = {
    kappa,
    lgamma,
    pairDotDensity,
    maxDotDensity,
    maxDotQuantile,
    firstOrderMean,
    centering,
    regIncBeta,
    pairDotCDF,
    maxDotCDFBeta,
    maxDotDensityBeta,
    maxDotQuantileBeta,
  };
})(window);
