/**
 * 噪音向量基本运算的理论分布层：概率密度、均值、方差。
 *
 * 设定：x、y 为 D 维向量，各分量独立，x_i ~ N(0, σ²)，y_i ~ N(0, σ²)。
 * 覆盖运算：
 *   按元素：x（原噪音）、x+y、x∘y（乘积正态）、x²（卡方）
 *   求和类：点积 x·y（双方随机 / 一方固定）、长度平方 ‖x‖²
 *
 * 关键结论（README.md 有完整推导）：
 *   - 相加保持正态：方差相加
 *   - 按元素乘是"乘积正态"：f(z) = K_0(|z|/σ²)/(πσ²)，方差 = σx²σy²，尖峰重尾
 *   - 平方是缩放卡方 σ²χ²₁：均值 σ²（漂离 0），方差 2σ⁴
 *   - 双方随机点积：Var = D·σx²σy²，密度含 Bessel K_{(D-1)/2}，
 *     D=1 即乘积正态，D=2 恰为 Laplace，D 大由 CLT 趋于正态
 *   - 一方固定（‖v‖²=D）点积：精确正态 N(0, Dσ²)
 *
 * 无 DOM 依赖，node 可直接 require 跑 test/theory-selftest.js。
 */
(function (global) {
  'use strict';

  // ---------- 基础数值 ----------

  /** Lanczos 近似的 log Γ(x)，双精度下足够精确 */
  function lgamma(x) {
    var c = [
      0.99999999999980993, 676.5203681218851, -1259.1392167224028,
      771.32342877765313, -176.61502916214059, 12.507343278686905,
      -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
    ];
    if (x < 0.5) {
      return Math.log(Math.PI / Math.sin(Math.PI * x)) - lgamma(1 - x);
    }
    x -= 1;
    var a = c[0];
    var t = x + 7.5;
    for (var i = 1; i < 9; i++) a += c[i] / (x + i);
    return (
      0.5 * Math.log(2 * Math.PI) +
      (x + 0.5) * Math.log(t) -
      t +
      Math.log(a)
    );
  }

  /** log(e^a + e^b)，避免溢出 */
  function logaddexp(a, b) {
    var m = Math.max(a, b);
    if (m === -Infinity) return m;
    return m + Math.log1p(Math.exp(-Math.abs(a - b)));
  }

  // ---------- log 域第二类修正 Bessel 函数 K_ν(x) ----------
  // K_ν 对 ν 递增方向前向递推稳定：K_{ν+1} = K_{ν-1} + (2ν/x)·K_ν。
  // 半整数阶有初等起点，整数阶用 Abramowitz & Stegun 9.8 有理逼近。

  /** log K_0(x)，x > 0。A&S 9.8.5（x≤2）与 9.8.6（x>2） */
  function logK0(x) {
    if (x <= 2) {
      var t2 = (x / 2) * (x / 2);
      var u = (x / 3.75) * (x / 3.75);
      var i0 =
        1 +
        u * (3.5156229 +
        u * (3.0899424 +
        u * (1.2067492 +
        u * (0.2659732 +
        u * (0.0360768 +
        u * 0.0045813)))));
      var poly =
        -0.57721566 +
        t2 * (0.4227842 +
        t2 * (0.23069756 +
        t2 * (0.0348859 +
        t2 * (0.00262698 +
        t2 * (0.0001075 +
        t2 * 0.0000074)))));
      return Math.log(-Math.log(x / 2) * i0 + poly);
    }
    var v = 2 / x;
    var p =
      1.25331414 +
      v * (-0.07832358 +
      v * (0.02189568 +
      v * (-0.01062446 +
      v * (0.00587872 +
      v * (-0.0025154 +
      v * 0.00053208)))));
    // K_0 ≈ x^{-1/2} e^{-x} · p(2/x)，直接取 log 避免 e^{-x} 下溢
    return Math.log(p) - 0.5 * Math.log(x) - x;
  }

  /** log K_1(x)，x > 0。A&S 9.8.7（x≤2）与 9.8.8（x>2） */
  function logK1(x) {
    if (x <= 2) {
      var t2 = (x / 2) * (x / 2);
      var u = (x / 3.75) * (x / 3.75);
      var i1 =
        x *
        (0.5 +
          u * (0.87890594 +
          u * (0.51498869 +
          u * (0.15084934 +
          u * (0.02658733 +
          u * (0.00301532 +
          u * 0.00032411))))));
      var poly =
        1 +
        t2 * (0.15443144 +
        t2 * (-0.67278579 +
        t2 * (-0.18156897 +
        t2 * (-0.01919402 +
        t2 * (-0.00110404 +
        t2 * (-0.00004686))))));
      return Math.log(Math.log(x / 2) * i1 + poly / x);
    }
    var v = 2 / x;
    var p =
      1.25331414 +
      v * (0.23498619 +
      v * (-0.0365562 +
      v * (0.01504268 +
      v * (-0.00780353 +
      v * (0.00325614 +
      v * (-0.00068245))))));
    return Math.log(p) - 0.5 * Math.log(x) - x;
  }

  /**
   * log K_ν(x)，ν ≥ 0 且为整数或半整数（本工具只需要这两类），x > 0。
   * 前向递推：logK_{n+1} = logaddexp(logK_{n-1}, log(2n/x) + logK_n)。
   */
  function logBesselK(nu, x) {
    if (x <= 0) return Infinity; // K_ν(0+) 发散
    var isHalf = Math.abs(nu - Math.round(nu)) > 1e-9;
    var logKm1, logK, n;
    if (isHalf) {
      // K_{1/2}(x) = √(π/(2x)) e^{-x}；K_{3/2} = K_{1/2}·(1 + 1/x)
      var logKhalf = 0.5 * Math.log(Math.PI / (2 * x)) - x;
      if (nu < 0.75) return logKhalf;
      logKm1 = logKhalf;
      logK = logKhalf + Math.log1p(1 / x);
      n = 0.5;
    } else {
      if (nu < 0.5) return logK0(x);
      logKm1 = logK0(x);
      logK = logK1(x);
      n = 0;
    }
    while (n + 1 < nu - 1e-9) {
      var nxt = logaddexp(logKm1, Math.log(2 * (n + 1) / x) + logK);
      logKm1 = logK;
      logK = nxt;
      n += 1;
    }
    return logK;
  }

  // ---------- 各分布的密度（log 域计算，返回普通密度；下溢为 0 无妨） ----------

  /** 正态 N(mean, var) 的密度 */
  function normalPDF(z, mean, variance) {
    var d = (z - mean) / Math.sqrt(variance);
    return Math.exp(-0.5 * d * d) / Math.sqrt(2 * Math.PI * variance);
  }

  /**
   * 乘积正态：z = x·y，x、y 独立 N(0, σ²)。
   * f(z) = K_0(|z|/a) / (πa)，a = σxσy = σ²；均值 0，方差 a²。
   * z=0 处对数发散（可积）——返回 Infinity，绘图层转为断点。
   */
  function productPDF(z, sigma) {
    if (z === 0) return Infinity;
    var a = sigma * sigma;
    var az = Math.max(Math.abs(z), 1e-300);
    return Math.exp(logK0(az / a)) / (Math.PI * a);
  }

  /**
   * 平方：z = x²，x ~ N(0, σ²)。即 σ²·χ²₁；均值 σ²，方差 2σ⁴。
   * z<0 密度为 0；z=0 处 z^{-1/2} 奇异——返回 Infinity，绘图层转为断点。
   */
  function squarePDF(z, sigma) {
    if (z < 0) return 0;
    if (z === 0) return Infinity;
    var s2 = sigma * sigma;
    // f(z) = z^{-1/2} e^{-z/(2σ²)} / (σ√(2π))
    return Math.exp(-0.5 * Math.log(z) - z / (2 * s2)) / (sigma * Math.sqrt(2 * Math.PI));
  }

  /**
   * 双方随机点积：z = Σ_{i=1}^D x_i y_i，各分量独立 N(0, σ²)。
   * 特征函数 (1 + a²t²)^{-D/2}（a = σ²），反演得
   *   f(z) = (|z|/(2a))^ν · K_ν(|z|/a) / (√π Γ(D/2) a)，ν = (D-1)/2
   * 验证：D=1 退化为乘积正态；D=2 时 K_{1/2} 初等，退化为 Laplace(0, a)。
   * 均值 0，方差 D·a²。
   * z=0 边界：D=1 对数发散（Infinity，绘图层转断点）；
   * D≥2 时极限有限——由 K_ν(u) ~ 2^{ν-1}Γ(ν)u^{-ν} 得
   * f(0) = Γ(ν) / (2√π·Γ(p)·a)（D=2 时为 1/(2a)，即 Laplace 起点）。
   */
  function dotRandomPDF(z, D, sigma) {
    var a = sigma * sigma;
    var p = D / 2;
    var nu = (D - 1) / 2;
    if (z === 0) {
      if (D === 1) return Infinity;
      return Math.exp(lgamma(nu) - lgamma(p)) / (2 * Math.sqrt(Math.PI) * a);
    }
    var az = Math.abs(z);
    var logf =
      nu * (Math.log(az) - Math.log(2 * a)) +
      logBesselK(nu, az / a) -
      0.5 * Math.log(Math.PI) -
      lgamma(p) -
      Math.log(a);
    return Math.exp(logf);
  }

  /** 一方固定点积：z = Σ v_i x_i，v 固定且 ‖v‖² = D。精确正态 N(0, Dσ²) */
  function dotFixedPDF(z, D, sigma) {
    return normalPDF(z, 0, D * sigma * sigma);
  }

  /**
   * 长度平方：z = ‖x‖² = Σ x_i²，即 σ²·χ²_D。均值 Dσ²，方差 2Dσ⁴。
   * z=0 边界按自由度区分：D=1 时 z^{-1/2} 奇异（Infinity，绘图层转断点）；
   * D=2 时为指数分布，f(0) = 1/(2σ²)；D≥3 时 f(0) = 0。
   */
  function norm2PDF(z, D, sigma) {
    if (z < 0) return 0;
    var s2 = sigma * sigma;
    if (z === 0) {
      if (D === 1) return Infinity;
      if (D === 2) return 1 / (2 * s2);
      return 0;
    }
    var k = D;
    var logf =
      (k / 2 - 1) * Math.log(z / s2) -
      z / (2 * s2) -
      Math.log(s2) -
      (k / 2) * Math.log(2) -
      lgamma(k / 2);
    return Math.exp(logf);
  }

  // ---------- 运算描述表：label、理论矩、密度、建议画图范围（mean±6std 口径） ----------

  /**
   * 按元素运算。sigma 为参数。
   * range 返回 [lo, hi]：覆盖均值 ±6 倍标准差（平方运算下界截到 0）。
   */
  var ELEMENT_OPS = [
    {
      id: 'x',
      label: 'x（原噪音）',
      color: '#6b7280',
      mean: function () { return 0; },
      variance: function (sigma) { return sigma * sigma; },
      pdf: function (z, sigma) { return normalPDF(z, 0, sigma * sigma); },
      range: function (sigma) { return [-6 * sigma, 6 * sigma]; },
    },
    {
      id: 'add',
      label: 'x + y',
      color: '#2563eb',
      mean: function () { return 0; },
      variance: function (sigma) { return 2 * sigma * sigma; },
      pdf: function (z, sigma) { return normalPDF(z, 0, 2 * sigma * sigma); },
      range: function (sigma) { return [-6 * Math.SQRT2 * sigma, 6 * Math.SQRT2 * sigma]; },
    },
    {
      id: 'product',
      label: 'x ∘ y（按元素乘）',
      color: '#c2410c',
      mean: function () { return 0; },
      variance: function (sigma) { return Math.pow(sigma, 4); },
      pdf: function (z, sigma) { return productPDF(z, sigma); },
      range: function (sigma) { var a = sigma * sigma; return [-6 * a, 6 * a]; },
    },
    {
      id: 'square',
      label: 'x ∘ x（平方）',
      color: '#b91c1c',
      mean: function (sigma) { return sigma * sigma; },
      variance: function (sigma) { return 2 * Math.pow(sigma, 4); },
      pdf: function (z, sigma) { return squarePDF(z, sigma); },
      range: function (sigma) {
        var s2 = sigma * sigma;
        return [0, s2 + 6 * Math.SQRT2 * s2];
      },
    },
  ];

  /**
   * 求和类运算。D、sigma 为参数；一方固定模式取 v_i ∈ {±1}（‖v‖² = D）。
   * projDot（投影点积 / attention 分数）额外需要头维 H（第三个参数）：
   * σ_w² = 1/D 时投影保持分量方差，分布退化为 H 维双方随机点积，
   * 推导见 docs/attention-score-distribution.md。
   */
  var SUM_MODES = [
    {
      id: 'dotRandom',
      label: '点积 x·y（双方随机）',
      color: '#c2410c',
      mean: function () { return 0; },
      variance: function (D, sigma) { return D * Math.pow(sigma, 4); },
      pdf: function (z, D, sigma) { return dotRandomPDF(z, D, sigma); },
      range: function (D, sigma) {
        var s = sigma * sigma * Math.sqrt(D);
        return [-6 * s, 6 * s];
      },
    },
    {
      id: 'dotFixed',
      label: '点积 x·v（一方固定 ±1）',
      color: '#2563eb',
      mean: function () { return 0; },
      variance: function (D, sigma) { return D * sigma * sigma; },
      pdf: function (z, D, sigma) { return dotFixedPDF(z, D, sigma); },
      range: function (D, sigma) {
        var s = sigma * Math.sqrt(D);
        return [-6 * s, 6 * s];
      },
    },
    {
      id: 'norm2',
      label: '长度平方 ‖x‖²',
      color: '#b91c1c',
      mean: function (D, sigma) { return D * sigma * sigma; },
      variance: function (D, sigma) { return 2 * D * Math.pow(sigma, 4); },
      pdf: function (z, D, sigma) { return norm2PDF(z, D, sigma); },
      range: function (D, sigma) {
        var s2 = sigma * sigma;
        var m = D * s2;
        var s = Math.sqrt(2 * D) * s2;
        return [Math.max(0, m - 6 * s), m + 6 * s];
      },
    },
    {
      id: 'projDot',
      label: '投影点积 (W_QA)·(W_KB)/√H',
      color: '#7c3aed',
      mean: function () { return 0; },
      // ÷√H 后 Var = σ⁴，与 D、H 都无关（scaled dot-product attention 的做法）
      variance: function (D, sigma) { return Math.pow(sigma, 4); },
      pdf: function (z, D, sigma, H) {
        // s/√H 的密度：√H·f(√H·z)，f 为 H 维双方随机点积（分量方差 σ²）
        var r = Math.sqrt(H);
        return r * dotRandomPDF(r * z, H, sigma);
      },
      range: function (D, sigma) {
        var s = sigma * sigma;
        return [-6 * s, 6 * s];
      },
    },
  ];

  var NoiseTheory = {
    lgamma: lgamma,
    logBesselK: logBesselK,
    normalPDF: normalPDF,
    productPDF: productPDF,
    squarePDF: squarePDF,
    dotRandomPDF: dotRandomPDF,
    dotFixedPDF: dotFixedPDF,
    norm2PDF: norm2PDF,
    ELEMENT_OPS: ELEMENT_OPS,
    SUM_MODES: SUM_MODES,
  };

  global.NoiseTheory = NoiseTheory;
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = NoiseTheory;
  }
})(typeof window !== 'undefined' ? window : globalThis);
