/**
 * 瓶颈投影链 y = (AB)^L x 长度平方 s = ‖y‖² 的理论密度（纯数学层，无 DOM 依赖）。
 *
 * 设定：x ∈ R^D 为单位向量，B 为 M×D 高斯矩阵（元素方差 1/D），A 为 D×M 高斯
 * 矩阵（元素方差 1/M）。方差配对（fan-in 1/D 下投、1/M 上投）使单块保长度：
 * E‖ABx‖² = ‖x‖²。M ∈ [1, 4D]（瓶颈维，可小于也可大于 D）。
 *
 * 方案一（固定 x，随机矩阵）——卡方因子化（精确）：
 *   中间层 z = Bx：‖z‖² = varB·χ²_M ~ Gamma(M/2, 2·varB)，均值 M·varB
 *   （B 的元素方差 varB 可选 1/D——配对下投、均值 M/D，或 1/M——均值归一 = 1）。
 *   单块 y = Az：条件于 z，‖Az‖² = (‖z‖²/M)·χ²_D，卡方与 ‖z‖² 独立，递推得
 *     s_L = ‖(AB)^L x‖² = ∏_{l=1..L} [χ²_M · χ²_D / (DM)]      （2L 个独立卡方因子）
 *   - 均值恒 1（任意 M、L：长度期望保值）
 *   - 方差 [(1+2/M)(1+2/D)]^L − 1（相对方差层层累积）
 *   - L=1：两个不同形状伽马的乘积 = 广义 Bessel-K_ν 闭式（ν = (M−D)/2）：
 *       f(s) = 2 s^{k̄−1} K_{k1−k2}(2√(s/(θ1θ2))) / [Γ(k1)Γ(k2)(θ1θ2)^{k̄}]
 *       k1 = M/2, θ1 = 2/D, k2 = D/2, θ2 = 2/M, k̄ = (k1+k2)/2
 *   - L≥2：无初等闭式，对数域渐近：
 *       ln s ≈ N(μ, σ²)，μ = L[ψ(M/2)−ln(M/2) + ψ(D/2)−ln(D/2)] ≈ −L(1/M+1/D)，
 *       σ² = L[ψ′(M/2)+ψ′(D/2)]
 *   - 均值被重尾撑着恒为 1，中位数 ≈ e^μ ≈ e^{−L(1/M+1/D)} 指数萎缩——
 *     "均值-中位数分裂"，LayerNorm 存在意义的随机矩阵视角（瓶颈版）。
 *
 * 方案二（固定矩阵，随机 x 球面均匀，quenched）：
 *   s = xᵀWx，W = PᵀP（P = 合成链）。球面二次型精确矩：
 *     E_x[s] = trW/D，Var_x(s) = 2[tr(W²) − (trW)²/D] / (D(D+2))
 *   合成用结合律 (AB)^L = A(BA)^{L−1}B：BA 仅 M×M，避免 O(LD³)。
 *   self-averaging：实例内方差 ≈ annealed 方差的大部分，换一批矩阵直方图形状
 *   不动、仅中心平移。实例中心 trW/D 的跳动（Wick 全方差律）：
 *     中间层（P=B）：Var(trW/D) = 2M·varB²/D（varB = 1/D 时即 2M/D³）
 *     单块 AB（L=1）：Var(trW/D) = (2M+4D+2)/(MD²)      （M=D 时 ≈ 6/D²）
 *   随维度消失（trW 是平方和的自平均）；L≥2 无简单闭式，量级 O(L²/D²)。
 *
 * 参考：docs/qk-spectrum.md、web-tools/spectrum-demo（谱）、proj-dot-demo（点积）。
 */
(function (global) {
  'use strict';

  var LN2 = Math.LN2;

  // ---------- Γ 函数（Lanczos，g = 7） ----------

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

  // ---------- ψ (digamma) 与 ψ′ (trigamma)：递推提升到 z ≥ 8 后渐近展开 ----------

  /** ψ(z) = ln z − 1/(2z) − 1/(12z²) + 1/(120z⁴) − 1/(252z⁶) + 1/(240z⁸) + O(z⁻¹⁰)，z > 0 */
  function digamma(z) {
    var acc = 0;
    while (z < 8) { acc -= 1 / z; z += 1; }
    var iz = 1 / z, iz2 = iz * iz;
    return acc + Math.log(z) - iz / 2 -
      iz2 * (1 / 12 - iz2 * (1 / 120 - iz2 * (1 / 252 - iz2 / 240)));
  }

  /** ψ′(z) = 1/z + 1/(2z²) + 1/(6z³) − 1/(30z⁵) + 1/(42z⁷) + O(z⁻⁹)，z > 0 */
  function trigamma(z) {
    var acc = 0;
    while (z < 8) { acc += 1 / (z * z); z += 1; }
    var iz = 1 / z, iz2 = iz * iz;
    return acc + iz * (1 + iz * (0.5 + iz * (1 / 6 - iz2 * (1 / 30 - iz2 / 42))));
  }

  // ---------- 零阶/一阶 Bessel K（Numerical Recipes 系数；整数阶对照用） ----------

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

  // ---------- 任意阶 Bessel K_ν（log 域；积分表示 + 正向递推） ----------
  // K_ν(x) = ∫₀^∞ e^{−x cosh t} cosh(νt) dt（DLMF 10.32.6），ν ≥ 0。
  // 起始阶 μ ∈ [0,1) 的 K_μ、K_{μ+1} 用被积函数的 log-sum-exp 梯形积分；
  // 之后正向递推 K_{ν+1} = K_{ν−1} + (2ν/x)K_ν（DLMF 10.29.1，两项同号，
  // log 域 logaddexp 无条件稳定，且 K_ν 大 ν 时的巨量量级全程不溢出）。

  function logAddExp(a, b) {
    if (a === -Infinity) return b;
    if (b === -Infinity) return a;
    var m = Math.max(a, b);
    return m + Math.log(Math.exp(a - m) + Math.exp(b - m));
  }

  /** log K_ν(x)，ν ∈ [0, 1]，x > 0：g(t) = −x cosh t + log cosh(νt) 的梯形积分 */
  function logBessKnuBase(x, nu) {
    var h = 0.1 / Math.max(1, Math.sqrt(x)); // 峰宽 ~max(1, 1/√x)，约 10 点/宽度
    var gmax = -Infinity;
    var vals = [];
    for (var t = 0; ; t += h) {
      var g = -x * Math.cosh(t) + (nu > 0 ? Math.log(Math.cosh(nu * t)) : 0);
      vals.push(g);
      if (g > gmax) gmax = g;
      if (t > 0 && g < gmax - 42 && vals.length > 8) break; // 尾部贡献 < e^{−42}
      if (vals.length > 50000) break; // 保险
    }
    var acc = 0, n = vals.length;
    for (var i = 0; i < n; i++) {
      var w = (i === 0 || i === n - 1) ? 0.5 : 1;
      acc += w * Math.exp(vals[i] - gmax);
    }
    return gmax + Math.log(h * acc);
  }

  /** log K_ν(x)，任意 ν ≥ 0，x > 0 */
  function logBessKnu(x, nu) {
    var n = Math.floor(nu);
    var mu = nu - n;
    var logKm = logBessKnuBase(x, mu);      // log K_μ
    if (n === 0) return logKm;
    var logK1 = logBessKnuBase(x, mu + 1);  // log K_{μ+1}
    for (var j = 1; j < n; j++) {
      // K_{μ+j+1} = K_{μ+j−1} + 2(μ+j)/x · K_{μ+j}
      var t = logAddExp(logKm, Math.log(2 * (mu + j) / x) + logK1);
      logKm = logK1;
      logK1 = t;
    }
    return logK1;
  }

  // ---------- 伽马密度（中间层 z=Bx：χ²_M/D = Gamma(k = M/2, scale θ = 2/D)） ----------

  function gammaDensity(s, k, theta) {
    if (s <= 0) {
      if (k < 1) return Infinity;  // M = 1：s^(−1/2) 发散
      if (k === 1) return 1 / theta; // M = 2：指数分布
      return 0;
    }
    return Math.exp((k - 1) * Math.log(s) - s / theta - logGamma(k) - k * Math.log(theta));
  }

  // ---------- 单块 AB：两个不同形状伽马的乘积密度（广义 Bessel-K_ν 闭式） ----------
  // P = G1·G2，G1 ~ Gamma(k1, θ1)（来自 χ²_M/D），G2 ~ Gamma(k2, θ2)（来自 χ²_D/M）：
  //   f(s) = 2 s^{k̄−1} K_{k1−k2}(2√(s/(θ1θ2))) / [Γ(k1)Γ(k2)(θ1θ2)^{k̄}]，k̄ = (k1+k2)/2
  // k1 = k2 时退化为等形状乘积的 K₀ 公式。s→0 时 f ~ s^{min(k1,k2)−1}（K_ν = K_{−ν}）。

  function prodGammaDensityAB(s, k1, theta1, k2, theta2) {
    if (s <= 0) {
      var kmin = Math.min(k1, k2);
      if (kmin < 1) return Infinity;          // min(M,D) = 1：s^{kmin−1} 发散
      if (kmin === 1 && k1 === k2) return Infinity; // M = D = 2：K₀ 对数发散
      return 0;
    }
    var nu = Math.abs(k1 - k2);
    var z = 2 * Math.sqrt(s / (theta1 * theta2));
    var ka = (k1 + k2) / 2;
    var logF = LN2 + (ka - 1) * Math.log(s) - logGamma(k1) - logGamma(k2) -
      ka * Math.log(theta1 * theta2) + logBessKnu(z, nu);
    return Math.exp(logF);
  }

  // ---------- 链的矩与对数域参数 ----------

  /** 完整链均值 = 1（任意 M、L：配对 fan-in 保长度） */
  function abMean() { return 1; }

  /** 完整链方差 = [(1+2/M)(1+2/D)]^L − 1 */
  function abVar(M, D, L) {
    return Math.pow((1 + 2 / M) * (1 + 2 / D), L) - 1;
  }

  /** 完整链 E[ln s] = L[ψ(M/2)−ln(M/2) + ψ(D/2)−ln(D/2)] ≈ −L(1/M+1/D) */
  function abLogMean(M, D, L) {
    return L * (digamma(M / 2) - Math.log(M / 2) + digamma(D / 2) - Math.log(D / 2));
  }

  /** 完整链 Var(ln s) = L[ψ′(M/2) + ψ′(D/2)] */
  function abLogVar(M, D, L) {
    return L * (trigamma(M / 2) + trigamma(D / 2));
  }

  /** 完整链中位数 ≈ exp(E[ln s]) ≈ e^{−L(1/M+1/D)} */
  function abMedianApprox(M, D, L) {
    return Math.exp(abLogMean(M, D, L));
  }

  // 中间层 z = Bx：s = varB·χ²_M ~ Gamma(M/2, 2·varB)，varB 为 B 的元素方差。
  // varB = 1/D（配对下投，均值 M/D）；varB = 1/M（均值归一 = 1）。

  /** 中间层均值 = M·varB */
  function midMean(M, varB) { return M * varB; }

  /** 中间层方差 = 2M·varB² */
  function midVar(M, varB) { return 2 * M * varB * varB; }

  /** 中间层 E[ln s] = ln2 + ψ(M/2) + ln varB */
  function midLogMean(M, varB) {
    return LN2 + digamma(M / 2) + Math.log(varB);
  }

  /** 中间层 Var(ln s) = ψ′(M/2) */
  function midLogVar(M) {
    return trigamma(M / 2);
  }

  /** 对数正态密度（L ≥ 2 完整链的近似曲线；其余情形也可叠加对照） */
  function lognormalDensity(s, mu, var0) {
    if (s <= 0) return 0;
    var d = Math.log(s) - mu;
    return Math.exp(-d * d / (2 * var0)) / (s * Math.sqrt(2 * Math.PI * var0));
  }

  function gaussDensity(s, mu, var0) {
    var d = s - mu;
    return Math.exp(-d * d / (2 * var0)) / Math.sqrt(2 * Math.PI * var0);
  }

  // ---------- 方案二（quenched）：固定链 P，x 球面均匀的精确矩 ----------

  /** E_x[s] = trW/D（W = PᵀP；trW = ‖P‖²_F） */
  function quenchMean(trW, D) { return trW / D; }

  /** Var_x(s) = 2[tr(W²) − (trW)²/D] / (D(D+2))（球面二次型精确公式） */
  function quenchVar(trW, trW2, D) {
    return 2 * (trW2 - (trW * trW) / D) / (D * (D + 2));
  }

  /** 中间层实例中心跳动：trW = ‖B‖²_F = varB·χ²_{MD} 精确 ⇒ Var(trW/D) = 2M·varB²/D */
  function quenchCenterSdMid(M, D, varB) {
    return varB * Math.sqrt(2 * M / D);
  }

  /** 单块 AB 实例中心跳动：Var(trW/D) = (2M+4D+2)/(MD²)（Wick 全方差律；M=D 时 ≈ 6/D²） */
  function quenchCenterSdAB1(M, D) {
    return Math.sqrt((2 * M + 4 * D + 2) / (M * D * D));
  }

  // ---------- 伽马/卡方采样（Marsaglia–Tsang，方案一采样的提速核心） ----------
  // 逐高斯平方和抽 χ²_M 每层要 O(M)；M 可达 4D = 4096，必须 O(1) 的精确伽马采样。

  /** Gamma(k, 1) 采样，k > 0；rand01 返回 [0,1) 均匀，gauss 返回标准正态 */
  function gammaSampleMT(k, rand01, gauss) {
    if (k < 1) { // Gamma(k) = Gamma(k+1)·U^{1/k}
      return gammaSampleMT(k + 1, rand01, gauss) * Math.pow(rand01(), 1 / k);
    }
    var d = k - 1 / 3, c = 1 / Math.sqrt(9 * d);
    for (;;) {
      var x = gauss();
      var v = 1 + c * x;
      if (v <= 0) continue;
      v = v * v * v;
      if (Math.log(rand01()) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
    }
  }

  /** χ²_ν 采样 = 2·Gamma(ν/2, 1) */
  function chi2Sample(nu, rand01, gauss) {
    return 2 * gammaSampleMT(nu / 2, rand01, gauss);
  }

  global.ProjLenTheory = {
    logGamma: logGamma,
    digamma: digamma,
    trigamma: trigamma,
    bessk0: bessk0,
    bessk1: bessk1,
    logBessKnu: logBessKnu,
    gammaDensity: gammaDensity,
    prodGammaDensityAB: prodGammaDensityAB,
    abMean: abMean,
    abVar: abVar,
    abLogMean: abLogMean,
    abLogVar: abLogVar,
    abMedianApprox: abMedianApprox,
    midMean: midMean,
    midVar: midVar,
    midLogMean: midLogMean,
    midLogVar: midLogVar,
    lognormalDensity: lognormalDensity,
    gaussDensity: gaussDensity,
    quenchMean: quenchMean,
    quenchVar: quenchVar,
    quenchCenterSdMid: quenchCenterSdMid,
    quenchCenterSdAB1: quenchCenterSdAB1,
    gammaSampleMT: gammaSampleMT,
    chi2Sample: chi2Sample,
  };
})(typeof window !== 'undefined' ? window : globalThis);
