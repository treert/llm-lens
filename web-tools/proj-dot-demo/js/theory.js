/**
 * 双随机投影点积 s = (Ma A)·(Mb B) = Aᵀ M B（M = MaᵀMb）的理论密度（纯数学层，无 DOM 依赖）。
 *
 * 设定：Ma、Mb 为 H×D 独立高斯矩阵（元素方差 1/D，初始化缩放），A、B ∈ R^D 为单位向量，
 * ρ = cos(A, B) 精确受控（B = ρA + √(1−ρ²)·Z⊥，Z⊥ 为 A 的正交补方向）。
 *
 * 方案一（固定 A、B，随机 Ma、Mb）——精确结果：
 *   u = Ma A ~ N(0, β I_H)，v = Mb B ~ N(0, β I_H)，u ⊥ v，β = (1/D)·‖A‖‖B‖ = 1/D。
 *   s = Σ_{k=1..H} u_k v_k：H 个独立高斯乘积之和，特征函数 (1+β²t²)^(−H/2)，
 *   即对称 Variance-Gamma（Bessel-K）分布：
 *     f(s) = 2|s|^ν K_ν(|s|/β) / [(2β)^(ν+1) √π Γ(H/2)]，  ν = (H−1)/2
 *   只依赖 ‖A‖、‖B‖、H、β——与夹角 ρ 严格无关（M 双各向同性，A、B 可被各自独立旋转）。
 *   特例：H=1 时 f(s) = K_0(|s|/β)/(πβ)（原点对数发散）；H=2 时退化为 Laplace
 *   e^(−|s|/β)/(2β)；H→∞ 高斯化。方差 Hβ²，超额峰度 6/H。
 *
 * 方案二/三（固定 Ma、Mb，随机 A、B）——大维 concentrate 近似：
 *   s = ρ·AᵀMA + √(1−ρ²)·(MᵀA)·Z⊥，条件于 A 为高斯混合，矩为
 *     E[s]   = ρ·tr(M)/D
 *     Var(s) = [2ρ²‖M_s‖²_F + (1−ρ²)‖M‖²_F] / D²      （M_s = M 的对称部分）
 *   对随机无关的 Ma、Mb：tr(M) 是均值 0、std √(H/D) 的随机量（换种子变号），
 *   ‖M‖²_F ≈ H concentrate，故 ρ 的均值效应相对涨落被压 ~ ρ/√D。
 *
 * 参考：docs/qk-spectrum.md（QK 谱的姊妹问题：那里看奇异值，这里看 score 标量）。
 */
(function (global) {
  'use strict';

  var LN2 = Math.LN2;

  // ---------- Γ 函数（Lanczos，g = 7，全正半轴 ~1e-15 相对精度） ----------

  var LG_COEF = [
    0.99999999999980993, 676.5203681218851, -1259.1392167224028,
    771.32342877765313, -176.61502916214059, 12.507343278686905,
    -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
  ];
  function logGamma(z) {
    if (z < 0.5) {
      return Math.log(Math.PI / Math.sin(Math.PI * z)) - logGamma(1 - z);
    }
    z -= 1;
    var x = LG_COEF[0];
    for (var i = 1; i < 9; i++) x += LG_COEF[i] / (z + i);
    var t = z + 7.5;
    return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x);
  }

  // ---------- 零阶/一阶 Bessel I、K（Numerical Recipes 系数，全域 ~1e-7） ----------

  function bessi0(x) {
    var ax = Math.abs(x);
    if (ax < 3.75) {
      var y = (x / 3.75) * (x / 3.75);
      return 1 + y * (3.5156229 + y * (3.0899424 + y * (1.2067492 +
        y * (0.2659732 + y * (0.0360768 + y * 0.0045813)))));
    }
    var y2 = 3.75 / ax;
    return (Math.exp(ax) / Math.sqrt(ax)) * (0.39894228 + y2 * (0.01328592 +
      y2 * (0.00225319 + y2 * (-0.00157565 + y2 * (0.00916281 + y2 * (-0.02057706 +
      y2 * (0.02635537 + y2 * (-0.01647633 + y2 * 0.00392377))))))));
  }

  function bessi1(x) {
    var ax = Math.abs(x);
    var ans;
    if (ax < 3.75) {
      var y = (x / 3.75) * (x / 3.75);
      ans = ax * (0.5 + y * (0.87890594 + y * (0.51498869 + y * (0.15084934 +
        y * (0.02658733 + y * (0.00301532 + y * 0.00032411))))));
    } else {
      var y2 = 3.75 / ax;
      ans = (Math.exp(ax) / Math.sqrt(ax)) * (0.39894228 + y2 * (-0.03988024 +
        y2 * (-0.00362018 + y2 * (0.01638001 + y2 * (-0.01031555 + y2 * (0.02282967 +
        y2 * (-0.02895312 + y2 * (0.01787654 - y2 * 0.00420059))))))));
    }
    return x < 0 ? -ans : ans;
  }

  function bessk0(x) { // x > 0
    if (x <= 2) {
      var y = (x * x) / 4;
      return -Math.log(x / 2) * bessi0(x) + (-0.57721566 + y * (0.42278420 +
        y * (0.23069756 + y * (0.03488590 + y * (0.00262698 + y * (0.00010750 +
        y * 0.0000074))))));
    }
    var y2 = 2 / x;
    return (Math.exp(-x) / Math.sqrt(x)) * (1.25331414 + y2 * (-0.07832358 +
      y2 * (0.02189568 + y2 * (-0.01062446 + y2 * (0.00587872 + y2 * (-0.00251540 +
      y2 * 0.00053208))))));
  }

  function bessk1(x) { // x > 0
    if (x <= 2) {
      var y = (x * x) / 4;
      return Math.log(x / 2) * bessi1(x) + (1 / x) * (1 + y * (0.15443144 +
        y * (-0.67278579 + y * (-0.18156897 + y * (-0.01919402 + y * (-0.00110404 +
        y * (-0.00004686)))))));
    }
    var y2 = 2 / x;
    return (Math.exp(-x) / Math.sqrt(x)) * (1.25331414 + y2 * (0.23498619 +
      y2 * (-0.03655620 + y2 * (0.01504268 + y2 * (-0.00780353 + y2 * (0.00325614 +
      y2 * (-0.00068245)))))));
  }

  // ---------- 缩放 Bessel：ln(z^ν K_ν(z))，ν 为整数或半整数（2ν 为整数） ----------
  // z^ν K_ν(z) 在 z → 0 时收敛到 2^(ν−1) Γ(ν)，全对数域计算，避免 0·∞ 与溢出。
  // 适用 z ≲ 700（更大 z 时 e^(−z) 下溢；本工具 z = |s|/β ≲ 几十）。

  /** 整数阶 ν = n：P_n(z) = z^n K_n(z)，递推 P_{k+1} = z²P_{k−1} + 2k·P_k（稳定） */
  function logScaledKInt(n, z) {
    if (z === 0) return (n - 1) * LN2 + logGamma(n); // n ≥ 1 的极限
    if (n === 0) return Math.log(bessk0(z));
    if (n === 1) return Math.log(z) + Math.log(bessk1(z));
    var p0 = bessk0(z);       // P_0 = K_0
    var p1 = z * bessk1(z);   // P_1 = z K_1
    var shift = 0;
    var z2 = z * z;
    for (var k = 1; k < n; k++) {
      var p2 = z2 * p0 + 2 * k * p1;
      p0 = p1;
      p1 = p2;
      // P_n ~ 2^(n−1)(n−1)! 增长，周期性缩量防溢出（均为正数相加，递推稳定）
      if (p1 > 1e100) {
        p0 *= 1e-100;
        p1 *= 1e-100;
        shift += 100 * Math.LN10;
      }
    }
    return Math.log(p1) + shift;
  }

  /**
   * 半整数阶 ν = n + 1/2（H 为偶数时用到）：初等有限和
   *   K_{n+1/2}(z) = √(π/(2z)) e^(−z) Σ_{k=0..n} a_k z^(−k)，
   *   a_0 = 1，a_k/a_{k−1} = (n+k)(n−k+1)/(2k)
   * z^(n+1/2) K_{n+1/2}(z) = √(π/2) e^(−z) Σ a_k z^(n−k)：logsumexp 求和。
   */
  function logScaledKHalf(n, z) {
    if (z === 0) return (n - 0.5) * LN2 + logGamma(n + 0.5); // ν = n+1/2 的极限
    var terms = new Float64Array(n + 1);
    var lz = Math.log(z);
    var la = 0;
    terms[0] = n * lz;
    var maxL = terms[0];
    for (var k = 1; k <= n; k++) {
      la += Math.log(n + k) + Math.log(n - k + 1) - Math.log(2 * k);
      terms[k] = la + (n - k) * lz;
      if (terms[k] > maxL) maxL = terms[k];
    }
    var sum = 0;
    for (var j = 0; j <= n; j++) sum += Math.exp(terms[j] - maxL);
    return 0.5 * Math.log(Math.PI / 2) - z + maxL + Math.log(sum);
  }

  /** ln(z^ν K_ν(z))：ν 为整数或半整数，z ≥ 0 */
  function logScaledBesselK(nu, z) {
    var twoNu = Math.round(2 * nu);
    if (twoNu % 2 === 0) return logScaledKInt(twoNu / 2, z);
    return logScaledKHalf((twoNu - 1) / 2, z);
  }

  /** K_ν(z)，ν 为整数或半整数，z > 0 */
  function besselK(nu, z) {
    return Math.exp(logScaledBesselK(nu, z) - nu * Math.log(z));
  }

  // ---------- 方案一：对称 Variance-Gamma 密度 ----------

  /**
   * ln f(s)：f(s) = 2|s|^ν K_ν(|s|/β) / [(2β)^(ν+1) √π Γ(H/2)]，ν = (H−1)/2。
   * 化简（用 z^ν K_ν(z) 有界改写，z = |s|/β）：
   *   ln f = −ν ln2 − ln β − ½lnπ − lnΓ(H/2) + ln(z^ν K_ν(z))
   */
  function logVgDensity(s, H, beta) {
    var z = Math.abs(s) / beta;
    if (H === 1) { // ν = 0：f(s) = K_0(|s|/β)/(πβ)，s=0 对数发散
      if (z === 0) return Infinity;
      return Math.log(bessk0(z)) - Math.log(Math.PI * beta);
    }
    var nu = (H - 1) / 2;
    return -nu * LN2 - Math.log(beta) - 0.5 * Math.log(Math.PI) -
      logGamma(H / 2) + logScaledBesselK(nu, z);
  }

  function vgDensity(s, H, beta) {
    return Math.exp(logVgDensity(s, H, beta));
  }

  /** 峰值 f(0⁺) = Γ((H−1)/2) / (2β√π Γ(H/2))（H ≥ 2；H = 1 时发散返回 Infinity） */
  function vgPeak(H, beta) {
    if (H === 1) return Infinity;
    return Math.exp(logGamma((H - 1) / 2) - logGamma(H / 2)) /
      (2 * beta * Math.sqrt(Math.PI));
  }

  /** 方差 = Hβ²（E[s²] = Σ E[u_k²]E[v_k²] = Hβ²） */
  function vgVariance(H, beta) { return H * beta * beta; }

  /** 超额峰度 = 6/H（单个高斯乘积的 6 被 H 重独立和稀释；μ4 = 3H(H+2)β⁴） */
  function vgExcessKurtosis(H) { return 6 / H; }

  // ---------- 方案二/三：固定 M = MaᵀMb 的大维近似 ----------

  /** E[s] = ρ·tr(M)/D（单位向量口径：E[AᵀMA] = tr(M)/D） */
  function scheme2Mean(rho, trM, D) { return rho * trM / D; }

  /**
   * Var(s) ≈ [2ρ²‖M_s‖²_F + (1−ρ²)‖M‖²_F] / D²
   * 来源：AᵀMA 的涨落 2‖M_s‖²_F/D²；交叉项条件于 A 为高斯，方差 ‖MᵀA‖² 经
   * E‖MᵀA‖² = ‖M‖²_F/D 再除以 Z⊥ 的维数 D。
   */
  function scheme2Var(rho, normF2, normMs2, D) {
    return (2 * rho * rho * normMs2 + (1 - rho * rho) * normF2) / (D * D);
  }

  function gaussDensity(s, mu, var0) {
    var d = s - mu;
    return Math.exp(-d * d / (2 * var0)) / Math.sqrt(2 * Math.PI * var0);
  }

  global.ProjDotTheory = {
    logGamma: logGamma,
    bessi0: bessi0,
    bessi1: bessi1,
    bessk0: bessk0,
    bessk1: bessk1,
    besselK: besselK,
    logScaledBesselK: logScaledBesselK,
    logVgDensity: logVgDensity,
    vgDensity: vgDensity,
    vgPeak: vgPeak,
    vgVariance: vgVariance,
    vgExcessKurtosis: vgExcessKurtosis,
    scheme2Mean: scheme2Mean,
    scheme2Var: scheme2Var,
    gaussDensity: gaussDensity,
  };
})(typeof window !== 'undefined' ? window : globalThis);
