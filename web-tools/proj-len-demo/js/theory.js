/**
 * 随机投影链长度平方 s = ‖B_{L−1}···B_1 A X‖² 的理论密度（纯数学层，无 DOM 依赖）。
 *
 * 设定：A 为 H×D 高斯矩阵（元素方差 1/D，初始化缩放），B_l 为 H×H 高斯矩阵
 * （元素方差 1/H，fan-in），X ∈ R^D 为单位向量，L = 矩阵总数（L=1 即只有 A）。
 *
 * 方案一（固定 X，随机矩阵）——卡方因子化（精确）：
 *   条件于上一层输出 Y，下一层 ‖BY‖² = (‖Y‖²/H)·χ²_H，卡方与 ‖Y‖² 独立，递推得
 *     s_L = (1/D)·χ²_H · ∏_{l=1..L−1} (χ²_H / H)     （L 个独立卡方因子）
 *   - 均值 H/D = c（任意 L 不变：fan-in 每层保持 E‖·‖²）
 *   - 相对方差 (1+2/H)^L − 1（层层累积）
 *   - L=1：伽马密度（χ²_H/D）；L=2：独立伽马乘积 = Bessel-K₀ 闭式；
 *     L≥3：无初等闭式（Meijer G），对数域渐近：
 *     ln s ≈ N(μ_ln, σ²_ln)，μ_ln = −lnD + L(ln2 + ψ(H/2)) − (L−1)lnH，σ²_ln = L·ψ′(H/2)
 *   - 均值被重尾撑着不变，中位数 ≈ exp(μ_ln) ≈ c·e^{−L/H} 指数萎缩——
 *     “均值-中位数分裂”，LayerNorm 存在意义的随机矩阵视角。
 *
 * 方案二（固定矩阵链，随机 X 球面均匀，quenched）：
 *   s = X^T W X，W = MᵀM（M = 合成链 H×D）。球面二次型精确矩：
 *     E_X[s] = trW/D，Var_X(s) = 2[tr(W²) − (trW)²/D] / (D(D+2))
 *   self-averaging：实例内方差 ≈ annealed 方差的大部分（球面约束下谱加权的
 *   (1+c) 修正被 −(trW)²/D 项抵消），换一批矩阵直方图形状不动、仅中心平移；
 *   实例中心 trW/D 的跳动（逐层全方差律递推 + 自由概率二阶矩 m₂ = L+c）
 *     Var(trW/D) = L(L−1+2c)/D²   （L=1 即 2H/D³；大 H 渐近，MC 已验证）
 *   随维度消失（trW 是平方和——对照点积问题的 trM 是符号和、涨落 √(H/D)
 *   不消失，见 proj-dot-demo）。
 *
 * 参考：docs/qk-spectrum.md、web-tools/spectrum-demo（谱）、proj-dot-demo（点积）的姊妹篇。
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

  // ---------- 零阶/一阶 Bessel I、K（Numerical Recipes 系数，与 proj-dot-demo 同源） ----------

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

  // ---------- L=1：伽马密度（χ²_H/D = Gamma(k = H/2, scale θ = 2/D)） ----------

  function gammaDensity(s, k, theta) {
    if (s <= 0) {
      if (k < 1) return Infinity;  // H = 1：s^(−1/2) 发散
      if (k === 1) return 1 / theta; // H = 2：指数分布
      return 0;
    }
    return Math.exp((k - 1) * Math.log(s) - s / theta - logGamma(k) - k * Math.log(theta));
  }

  // ---------- L=2：独立伽马乘积密度（Bessel-K₀ 闭式） ----------
  // P = G1·G2，Gi ~ Gamma(k, θi)：
  //   f(s) = 2 s^(k−1) K₀(2√(s/(θ1θ2))) / [Γ(k)² (θ1θ2)^k]
  // s→0 时 K₀ 对数发散（k=1 即 H=2 时 f 本身对数发散；k≥2 时 s^(k−1) 压回 0）。

  function prodGammaDensity(s, k, theta1, theta2) {
    if (s <= 0) return k === 1 ? Infinity : 0;
    var z = 2 * Math.sqrt(s / (theta1 * theta2));
    if (z > 600) return 0; // K₀(z) ~ e^(−z) 早已数值为 0
    var logK0 = Math.log(bessk0(z));
    return Math.exp(LN2 + (k - 1) * Math.log(s) - 2 * logGamma(k) -
      k * Math.log(theta1 * theta2) + logK0);
  }

  // ---------- L 层链的矩与对数域参数 ----------

  /** 均值 = H/D = c（任意 L 不变） */
  function chainMean(H, D) { return H / D; }

  /** 方差 = c²[(1+2/H)^L − 1]（相对方差 (1+2/H)^L − 1 ≈ e^(2L/H) − 1） */
  function chainVar(H, D, L) {
    const c = H / D;
    return c * c * (Math.pow(1 + 2 / H, L) - 1);
  }

  /** E[ln s] = −lnD + L(ln2 + ψ(H/2)) − (L−1)lnH */
  function chainLogMean(H, D, L) {
    return -Math.log(D) + L * (LN2 + digamma(H / 2)) - (L - 1) * Math.log(H);
  }

  /** Var(ln s) = L·ψ′(H/2) */
  function chainLogVar(H, L) {
    return L * trigamma(H / 2);
  }

  /** 中位数 ≈ exp(E[ln s])（对数正态近似；L=1 时即 χ² 中位数的 Wilson-Hilferty 级别近似） */
  function chainMedianApprox(H, D, L) {
    return Math.exp(chainLogMean(H, D, L));
  }

  /** 对数正态密度（L ≥ 3 的近似曲线；L ≤ 2 也可叠加对照） */
  function lognormalDensity(s, mu, var0) {
    if (s <= 0) return 0;
    var d = Math.log(s) - mu;
    return Math.exp(-d * d / (2 * var0)) / (s * Math.sqrt(2 * Math.PI * var0));
  }

  function gaussDensity(s, mu, var0) {
    var d = s - mu;
    return Math.exp(-d * d / (2 * var0)) / Math.sqrt(2 * Math.PI * var0);
  }

  // ---------- 方案二（quenched）：固定链 M，X 球面均匀的精确矩 ----------

  /** E_X[s] = trW/D（W = MᵀM；trW = ‖M‖²_F） */
  function quenchMean(trW, D) { return trW / D; }

  /** Var_X(s) = 2[tr(W²) − (trW)²/D] / (D(D+2))（球面二次型精确公式） */
  function quenchVar(trW, trW2, D) {
    return 2 * (trW2 - (trW * trW) / D) / (D * (D + 2));
  }

  global.ProjLenTheory = {
    logGamma: logGamma,
    digamma: digamma,
    trigamma: trigamma,
    bessk0: bessk0,
    bessk1: bessk1,
    gammaDensity: gammaDensity,
    prodGammaDensity: prodGammaDensity,
    chainMean: chainMean,
    chainVar: chainVar,
    chainLogMean: chainLogMean,
    chainLogVar: chainLogVar,
    chainMedianApprox: chainMedianApprox,
    lognormalDensity: lognormalDensity,
    gaussDensity: gaussDensity,
    quenchMean: quenchMean,
    quenchVar: quenchVar,
  };
})(typeof window !== 'undefined' ? window : globalThis);
