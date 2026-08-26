/**
 * LLM 常见激活函数注册表（activation-demo 的唯一事实源）。
 * 条目：id、中文名、公式文本、逐元素函数 fn、解析导数 dfn、
 * 参数规格 params、y=0 点质量标记 atom、面板二说明 distNote。
 * 无 DOM 依赖，node 可直接 require。
 */
(function (global) {
  'use strict';

  /** 数值稳定的 sigmoid */
  function sigmoid(x) {
    if (x >= 0) return 1 / (1 + Math.exp(-x));
    var e = Math.exp(x);
    return e / (1 + e);
  }

  /** 数值稳定的 softplus：ln(1+e^x) */
  function softplus(x) {
    return Math.max(x, 0) + Math.log1p(Math.exp(-Math.abs(x)));
  }

  /** Abramowitz–Stegun 7.1.26 误差函数逼近，|ε| ≤ 1.5e-7 */
  function erf(x) {
    var sign = x < 0 ? -1 : 1;
    var ax = Math.abs(x);
    var t = 1 / (1 + 0.3275911 * ax);
    var poly = (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
      - 0.284496736) * t + 0.254829592) * t;
    return sign * (1 - poly * Math.exp(-ax * ax));
  }

  /** N(0, σ²) 的 pdf 与 cdf */
  function normPdf(x, sigma) {
    var z = x / sigma;
    return Math.exp(-0.5 * z * z) / (sigma * Math.sqrt(2 * Math.PI));
  }
  function normCdf(x, sigma) {
    return 0.5 * (1 + erf(x / (sigma * Math.SQRT2)));
  }

  var K = Math.sqrt(2 / Math.PI); // GELU tanh 近似系数

  var list = [
    {
      id: 'sigmoid', name: 'Sigmoid', formula: 'σ(x) = 1/(1+e^(−x))',
      params: [], atom: false,
      fn: function (x) { return sigmoid(x); },
      dfn: function (x) { var s = sigmoid(x); return s * (1 - s); },
      distNote: '把质量压到 (0,1)，两端饱和区堆积、密度在边界发散（可积）；均值≈0.5 不再为零，饱和区梯度→0。',
    },
    {
      id: 'tanh', name: 'Tanh', formula: 'tanh(x)',
      params: [], atom: false,
      fn: function (x) { return Math.tanh(x); },
      dfn: function (x) { var t = Math.tanh(x); return 1 - t * t; },
      distNote: '压到 (−1,1) 且为奇函数、保持零均值；两端仍有饱和堆积。RNN 时代的标配。',
    },
    {
      id: 'relu', name: 'ReLU', formula: 'max(0, x)',
      params: [], atom: true,
      fn: function (x) { return x > 0 ? x : 0; },
      dfn: function (x) { return x > 0 ? 1 : 0; },
      distNote: '一半质量塌缩成 y=0 的点质量（精确稀疏！），正半轴密度原样保留——均值 0 → σ/√(2π)。',
    },
    {
      id: 'leaky-relu', name: 'Leaky ReLU', formula: 'x>0 ? x : αx',
      params: [{ key: 'alpha', label: 'α', min: 0.01, max: 0.3, step: 0.01, def: 0.01 }],
      atom: false,
      fn: function (x, p) { return x > 0 ? x : p.alpha * x; },
      dfn: function (x, p) { return x > 0 ? 1 : p.alpha; },
      distNote: '负半轴斜率 α 把左半质量压向原点（y<0 支密度放大 1/α），无点质量，缓解 ReLU 死区。',
    },
    {
      id: 'elu', name: 'ELU', formula: 'x>0 ? x : α(e^x−1)',
      params: [{ key: 'alpha', label: 'α', min: 0.5, max: 2, step: 0.1, def: 1 }],
      atom: false,
      fn: function (x, p) { return x > 0 ? x : p.alpha * (Math.exp(x) - 1); },
      dfn: function (x, p) { return x > 0 ? 1 : p.alpha * Math.exp(x); },
      distNote: '负半轴指数饱和到 −α，左尾质量堆积在 −α 附近；负值把均值拉回，有自归一化倾向。',
    },
    {
      id: 'softplus', name: 'Softplus', formula: 'ln(1+e^x)',
      params: [], atom: false,
      fn: function (x) { return softplus(x); },
      dfn: function (x) { return sigmoid(x); },
      distNote: '平滑版 ReLU：无点质量，y<0 支密度几乎为零，质量集中在 0 附近的指数薄尾。',
    },
    {
      id: 'gelu-exact', name: 'GELU（精确）', formula: 'x·Φ(x)',
      params: [], atom: false,
      fn: function (x) { return x * normCdf(x, 1); },
      dfn: function (x) { return normCdf(x, 1) + x * normPdf(x, 1); },
      distNote: '非单调：x≈−0.75 极小值处密度有可积奇点；均值有闭式 σ²/√(2π(1+σ²))，σ=1 时为 1/(2√π)≈0.282。',
    },
    {
      id: 'gelu-tanh', name: 'GELU（tanh 近似）', formula: '½x(1+tanh(√(2/π)(x+0.044715x³)))',
      params: [], atom: false,
      fn: function (x) {
        var u = K * (x + 0.044715 * x * x * x);
        return 0.5 * x * (1 + Math.tanh(u));
      },
      dfn: function (x) {
        var u = K * (x + 0.044715 * x * x * x);
        var t = Math.tanh(u);
        return 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * K * (1 + 3 * 0.044715 * x * x);
      },
      distNote: '与精确版几乎重合（面板一可叠加对比）；GPT-2 实际使用的就是这个近似。',
    },
    {
      id: 'silu', name: 'SiLU / Swish', formula: 'x·σ(βx)',
      params: [{ key: 'beta', label: 'β', min: 0.1, max: 5, step: 0.1, def: 1 }],
      atom: false,
      fn: function (x, p) { return x * sigmoid(p.beta * x); },
      dfn: function (x, p) {
        var s = sigmoid(p.beta * x);
        return s * (1 + p.beta * x * (1 - s));
      },
      distNote: 'β 控制软化程度：β→∞ 趋 ReLU，β→0 趋 x/2；β=1 时在 x≈−1.28 有极小值与密度奇点。SwiGLU 的门支就是它。',
    },
    {
      id: 'mish', name: 'Mish', formula: 'x·tanh(softplus(x))',
      params: [], atom: false,
      fn: function (x) { return x * Math.tanh(softplus(x)); },
      dfn: function (x) {
        var t = Math.tanh(softplus(x));
        return t + x * sigmoid(x) * (1 - t * t);
      },
      distNote: '形状与 SiLU 相似，极小值更浅（x≈−1.19，f≈−0.31），负半支更平滑。',
    },
    {
      id: 'relu2', name: 'ReLU²', formula: 'max(0, x)²',
      params: [], atom: true,
      fn: function (x) { return x > 0 ? x * x : 0; },
      dfn: function (x) { return x > 0 ? 2 * x : 0; },
      distNote: '点质量同 ReLU（0.5），正半轴平方把密度拉向 0：均值=σ²/2，尖峰重尾。',
    },
  ];

  function byId(id) {
    for (var i = 0; i < list.length; i++) if (list[i].id === id) return list[i];
    return undefined;
  }

  function defaultParams(act) {
    var p = {};
    act.params.forEach(function (spec) { p[spec.key] = spec.def; });
    return p;
  }

  var ActFns = {
    list: list,
    byId: byId,
    defaultParams: defaultParams,
    helpers: { sigmoid: sigmoid, softplus: softplus, erf: erf, normPdf: normPdf, normCdf: normCdf },
  };

  global.ActFns = ActFns;
  if (typeof module !== 'undefined' && module.exports) module.exports = ActFns;
})(typeof window !== 'undefined' ? window : globalThis);
