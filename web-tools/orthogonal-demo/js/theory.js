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

  const EULER_GAMMA = 0.5772156649015329; // Euler–Mascheroni 常数

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

  /**
   * Gumbel 修正的近似均值：E[max ρ] ≈ sqrt((a + 2(γ + ln κ)) / N)。
   * 标准 Gumbel（尺度 2）的均值为 E[W] = 2(γ + ln κ)。
   * 注意 E[max ρ] ≠ sqrt(E[(max ρ)²])，此处取后者作为近似，仅作参考。
   */
  function gumbelMeanApprox(N, K, twoSided) {
    const kp = kappa(twoSided);
    const v = (centering(K) + 2 * (EULER_GAMMA + Math.log(kp))) / N;
    return v > 0 ? Math.sqrt(v) : 0;
  }

  global.Theory = {
    EULER_GAMMA,
    kappa,
    lgamma,
    pairDotDensity,
    maxDotDensity,
    maxDotQuantile,
    firstOrderMean,
    gumbelMeanApprox,
    centering,
  };
})(window);
