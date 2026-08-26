# activation-demo 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `web-tools/activation-demo/` 实现激活函数可视化双面板工具（函数/导数对比 + 噪音过激活的分布对照），并登记到根目录工具列表。

**Architecture:** 纯前端静态页（ECharts CDN，无构建）。`functions.js` 注册表为唯一事实源，`theory.js`（临界点/求逆/密度/Gauss–Hermite 矩）与 `sampler.js`（RNG/直方图）无 DOM 依赖，`app.js` 负责 UI 与渲染。node 自检覆盖全部数值层。

**Tech Stack:** 原生 JS（IIFE + var 风格，浏览器 global / node module.exports 双兼容）、ECharts 5.5.0 CDN、node（自检）。

**设计文档:** `docs/superpowers/specs/2026-08-26-activation-demo-design.md`

## Global Constraints

- 提交信息用中文，格式 `<type>: <摘要>`（仓库 AGENTS.md 约定）。
- 文档与代码注释一律用中文。
- markdown 行内公式：`$` 前若是文字或全角标点必须加空格；写完 README 后跑
  `python tools/fix_md_math_spacing.py --apply web-tools/activation-demo/README.md`。
- js 数值层（functions/theory/sampler）禁止出现 DOM 依赖，保持 node 可直接 require。
- ECharts 走 CDN（`https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js`），与现有 demo 一致。
- 自检命令：`node web-tools/activation-demo/test/selftest.js`，失败必须非零退出。

---

### Task 1: 激活函数注册表 `js/functions.js` + 自检骨架

**Files:**
- Create: `web-tools/activation-demo/js/functions.js`
- Test: `web-tools/activation-demo/test/selftest.js`

**Interfaces:**
- Produces（后续所有任务依赖）:
  - `ActFns.list`：条目数组，形状 `{ id, name, formula, atom, params: [{key,label,min,max,step,def}], fn(x,p), dfn(x,p), distNote }`（`p` 为参数对象；`atom=true` 表示 y=0 处有点质量）
  - `ActFns.byId(id)` → 条目或 `undefined`
  - `ActFns.defaultParams(act)` → `{ key: def, ... }`
  - `ActFns.helpers` → `{ sigmoid(x), softplus(x), erf(x), normPdf(x,sigma), normCdf(x,sigma) }`

- [ ] **Step 1: 写自检骨架（此时模块不存在，必然失败）**

`web-tools/activation-demo/test/selftest.js`：

```js
/**
 * activation-demo 数值层 node 自检。
 * 用法：node web-tools/activation-demo/test/selftest.js
 * 任一断言失败以非零码退出。
 */
'use strict';

var failures = 0;
function check(name, cond) {
  if (cond) { console.log('  ok  ' + name); }
  else { failures++; console.error('FAIL  ' + name); }
}
function approx(name, a, b, tol) {
  check(name + '（|' + a + ' - ' + b + '| <= ' + tol + '）', Math.abs(a - b) <= tol);
}
function approxRel(name, a, b, relTol) {
  var scale = Math.max(Math.abs(a), Math.abs(b), 1e-12);
  check(name + '（相对误差 <= ' + relTol + '）', Math.abs(a - b) / scale <= relTol);
}

var ActFns = require('../js/functions.js');

console.log('== functions.js ==');
check('注册 11 个函数', ActFns.list.length === 11);
var ids = ActFns.list.map(function (a) { return a.id; });
['sigmoid', 'tanh', 'relu', 'leaky-relu', 'elu', 'softplus',
 'gelu-exact', 'gelu-tanh', 'silu', 'mish', 'relu2'].forEach(function (id) {
  check('含 ' + id, ids.indexOf(id) >= 0);
  check('byId(' + id + ')', ActFns.byId(id).id === id);
});
approx('sigmoid(0)=0.5', ActFns.byId('sigmoid').fn(0, {}), 0.5, 1e-15);
approx('tanh(0)=0', ActFns.byId('tanh').fn(0, {}), 0, 1e-15);
approx('softplus(0)=ln2', ActFns.byId('softplus').fn(0, {}), Math.LN2, 1e-12);
approx('gelu(0)=0', ActFns.byId('gelu-exact').fn(0, {}), 0, 1e-15);
approx('relu2(2)=4', ActFns.byId('relu2').fn(2, {}), 4, 1e-15);
approx('normCdf(0)=0.5', ActFns.helpers.normCdf(0, 1), 0.5, 1e-7);

// 解析导数 vs 中心差分（ReLU 族折点附近跳过）
ActFns.list.forEach(function (act) {
  var p = ActFns.defaultParams(act);
  var bad = 0;
  for (var x = -6; x <= 6; x += 0.013) {
    if ((act.id === 'relu' || act.id === 'relu2' || act.id === 'leaky-relu')
        && Math.abs(x) < 0.05) continue;
    var h = 1e-5;
    var num = (act.fn(x + h, p) - act.fn(x - h, p)) / (2 * h);
    var ana = act.dfn(x, p);
    var scale = Math.max(1, Math.abs(ana));
    if (Math.abs(num - ana) / scale > 1e-5) bad++;
  }
  check('dfn 与数值差分一致: ' + act.id, bad === 0);
});

console.log(failures === 0 ? '\n全部通过' : '\n失败 ' + failures + ' 项');
process.exit(failures === 0 ? 0 : 1);
```

- [ ] **Step 2: 跑自检确认失败**

Run: `node web-tools/activation-demo/test/selftest.js`
Expected: 报错 `Cannot find module '../js/functions.js'`

- [ ] **Step 3: 实现 `js/functions.js`**

文件头部 docstring：「LLM 常见激活函数注册表（activation-demo 的唯一事实源）。条目：id、中文名、公式文本、fn、dfn、参数规格 params、y=0 点质量标记 atom、面板二说明 distNote。无 DOM 依赖，node 可直接 require。」完整实现如下：

```js
(function (global) {
  'use strict';

  function sigmoid(x) {                    // 数值稳定
    if (x >= 0) return 1 / (1 + Math.exp(-x));
    var e = Math.exp(x);
    return e / (1 + e);
  }
  function softplus(x) {                   // 数值稳定：ln(1+e^x)
    return Math.max(x, 0) + Math.log1p(Math.exp(-Math.abs(x)));
  }
  function erf(x) {                        // A&S 7.1.26，|ε|≤1.5e-7
    var sign = x < 0 ? -1 : 1;
    var ax = Math.abs(x);
    var t = 1 / (1 + 0.3275911 * ax);
    var poly = (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
      - 0.284496736) * t + 0.254829592) * t;
    return sign * (1 - poly * Math.exp(-ax * ax));
  }
  function normPdf(x, sigma) {
    var z = x / sigma;
    return Math.exp(-0.5 * z * z) / (sigma * Math.sqrt(2 * Math.PI));
  }
  function normCdf(x, sigma) {
    return 0.5 * (1 + erf(x / (sigma * Math.SQRT2)));
  }

  var K = Math.sqrt(2 / Math.PI);          // GELU tanh 近似系数

  var list = [
    { id: 'sigmoid', name: 'Sigmoid', formula: 'σ(x) = 1/(1+e^(−x))',
      params: [], atom: false,
      fn: function (x) { return sigmoid(x); },
      dfn: function (x) { var s = sigmoid(x); return s * (1 - s); },
      distNote: '把质量压到 (0,1)，两端饱和区堆积、密度在边界发散（可积）；均值≈0.5 不再为零，饱和区梯度→0。' },
    { id: 'tanh', name: 'Tanh', formula: 'tanh(x)',
      params: [], atom: false,
      fn: function (x) { return Math.tanh(x); },
      dfn: function (x) { var t = Math.tanh(x); return 1 - t * t; },
      distNote: '压到 (−1,1) 且为奇函数、保持零均值；两端仍有饱和堆积。RNN 时代的标配。' },
    { id: 'relu', name: 'ReLU', formula: 'max(0, x)',
      params: [], atom: true,
      fn: function (x) { return x > 0 ? x : 0; },
      dfn: function (x) { return x > 0 ? 1 : 0; },
      distNote: '一半质量塌缩成 y=0 的点质量（精确稀疏！），正半轴密度原样保留——均值 0 → σ/√(2π)。' },
    { id: 'leaky-relu', name: 'Leaky ReLU', formula: 'x>0 ? x : αx',
      params: [{ key: 'alpha', label: 'α', min: 0.01, max: 0.3, step: 0.01, def: 0.01 }],
      atom: false,
      fn: function (x, p) { return x > 0 ? x : p.alpha * x; },
      dfn: function (x, p) { return x > 0 ? 1 : p.alpha; },
      distNote: '负半轴斜率 α 把左半质量压向原点（y<0 支密度放大 1/α），无点质量，缓解 ReLU 死区。' },
    { id: 'elu', name: 'ELU', formula: 'x>0 ? x : α(e^x−1)',
      params: [{ key: 'alpha', label: 'α', min: 0.5, max: 2, step: 0.1, def: 1 }],
      atom: false,
      fn: function (x, p) { return x > 0 ? x : p.alpha * (Math.exp(x) - 1); },
      dfn: function (x, p) { return x > 0 ? 1 : p.alpha * Math.exp(x); },
      distNote: '负半轴指数饱和到 −α，左尾质量堆积在 −α 附近；负值把均值拉回，有自归一化倾向。' },
    { id: 'softplus', name: 'Softplus', formula: 'ln(1+e^x)',
      params: [], atom: false,
      fn: function (x) { return softplus(x); },
      dfn: function (x) { return sigmoid(x); },
      distNote: '平滑版 ReLU：无点质量，y<0 支密度几乎为零，质量集中在 0 附近的指数薄尾。' },
    { id: 'gelu-exact', name: 'GELU（精确）', formula: 'x·Φ(x)',
      params: [], atom: false,
      fn: function (x) { return x * normCdf(x, 1); },
      dfn: function (x) { return normCdf(x, 1) + x * normPdf(x, 1); },
      distNote: '非单调：x≈−0.75 极小值处密度有可积奇点；均值有闭式 σ²/√(2π(1+σ²))，σ=1 时为 1/(2√π)≈0.282。' },
    { id: 'gelu-tanh', name: 'GELU（tanh 近似）', formula: '½x(1+tanh(√(2/π)(x+0.044715x³)))',
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
      distNote: '与精确版几乎重合（面板一可叠加对比）；GPT-2 实际使用的就是这个近似。' },
    { id: 'silu', name: 'SiLU / Swish', formula: 'x·σ(βx)',
      params: [{ key: 'beta', label: 'β', min: 0.1, max: 5, step: 0.1, def: 1 }],
      atom: false,
      fn: function (x, p) { return x * sigmoid(p.beta * x); },
      dfn: function (x, p) {
        var s = sigmoid(p.beta * x);
        return s * (1 + p.beta * x * (1 - s));
      },
      distNote: 'β 控制软化程度：β→∞ 趋 ReLU，β→0 趋 x/2；β=1 时在 x≈−1.28 有极小值与密度奇点。SwiGLU 的门支就是它。' },
    { id: 'mish', name: 'Mish', formula: 'x·tanh(softplus(x))',
      params: [], atom: false,
      fn: function (x) { return x * Math.tanh(softplus(x)); },
      dfn: function (x) {
        var t = Math.tanh(softplus(x));
        return t + x * sigmoid(x) * (1 - t * t);
      },
      distNote: '形状与 SiLU 相似，极小值更浅（x≈−1.19，f≈−0.31），负半支更平滑。' },
    { id: 'relu2', name: 'ReLU²', formula: 'max(0, x)²',
      params: [], atom: true,
      fn: function (x) { return x > 0 ? x * x : 0; },
      dfn: function (x) { return x > 0 ? 2 * x : 0; },
      distNote: '点质量同 ReLU（0.5），正半轴平方把密度拉向 0：均值=σ²/2，尖峰重尾。' },
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
    list: list, byId: byId, defaultParams: defaultParams,
    helpers: { sigmoid: sigmoid, softplus: softplus, erf: erf,
               normPdf: normPdf, normCdf: normCdf },
  };
  global.ActFns = ActFns;
  if (typeof module !== 'undefined' && module.exports) module.exports = ActFns;
})(typeof window !== 'undefined' ? window : globalThis);
```

- [ ] **Step 4: 跑自检确认通过**

Run: `node web-tools/activation-demo/test/selftest.js`
Expected: 全部 `ok`，退出码 0

- [ ] **Step 5: Commit**

```bash
git add web-tools/activation-demo/js/functions.js web-tools/activation-demo/test/selftest.js
git commit -m "feat: activation-demo 激活函数注册表与导数自检"
```

---

### Task 2: 采样层 `js/sampler.js`

**Files:**
- Create: `web-tools/activation-demo/js/sampler.js`
- Test: `web-tools/activation-demo/test/selftest.js`（追加）

**Interfaces:**
- Consumes: `ActFns`（条目 `fn(x,p)`）
- Produces: `ActSampler`：
  - `makeRng(seed)` → 标准正态采样函数 `gauss()`
  - `sampleInput(gauss, N, sigma)` → `Float64Array(N)`
  - `applyActivation(act, p, xs)` → `Float64Array`
  - `histogram(samples, lo, hi, nBins)` → `{ centers, density, under, over }`
  - `sampleMeanVar(samples)` → `{ mean, variance }`（无偏）
  - `countExactZeros(samples)` → number

- [ ] **Step 1: 自检追加 sampler 段（失败：模块不存在）**

`require` 区加 `var ActSampler = require('../js/sampler.js');`，结尾汇总前加：

```js
console.log('== sampler.js ==');
(function () {
  var xs = ActSampler.sampleInput(ActSampler.makeRng(42), 100000, 1);
  var mv = ActSampler.sampleMeanVar(xs);
  approx('N(0,1) 样本均值≈0', mv.mean, 0, 0.02);
  approx('N(0,1) 样本方差≈1', mv.variance, 1, 0.02);

  var ys = ActSampler.applyActivation(ActFns.byId('relu'), {}, xs);
  approx('ReLU 后 0 的比例≈0.5', ActSampler.countExactZeros(ys) / ys.length, 0.5, 0.01);

  var h = ActSampler.histogram(xs, -4, 4, 80);
  var w = 8 / 80, area = 0;
  for (var i = 0; i < h.density.length; i++) area += h.density[i] * w;
  approx('直方图密度积分+越界≈1', area + (h.under + h.over) / xs.length, 1, 1e-12);

  approx('同种子可复现',
    ActSampler.sampleInput(ActSampler.makeRng(42), 10, 1)[5],
    ActSampler.sampleInput(ActSampler.makeRng(42), 10, 1)[5], 0);
})();
```

- [ ] **Step 2: 跑自检确认失败**

Run: `node web-tools/activation-demo/test/selftest.js`
Expected: `Cannot find module '../js/sampler.js'`

- [ ] **Step 3: 实现 `js/sampler.js`**

docstring：「activation-demo 蒙特卡洛采样层。mulberry32 + Box–Muller 高斯 RNG（可设种子可复现）、输入采样、逐元素激活映射、直方图分箱与样本矩。无 DOM 依赖。」RNG 部分与 noise-ops-demo 的 sampler.js 相同（mulberry32 / makeGaussian 原样照搬），本 demo 特有部分：

```js
  function makeRng(seed) { return makeGaussian(mulberry32(seed)); }

  function sampleInput(gauss, N, sigma) {
    var x = new Float64Array(N);
    for (var i = 0; i < N; i++) x[i] = sigma * gauss();
    return x;
  }
  function applyActivation(act, p, xs) {
    var out = new Float64Array(xs.length);
    for (var i = 0; i < xs.length; i++) out[i] = act.fn(xs[i], p);
    return out;
  }
  // histogram / sampleMeanVar 与 noise-ops-demo 相同（等宽箱、无偏方差）
  function countExactZeros(samples) {
    var c = 0;
    for (var i = 0; i < samples.length; i++) if (samples[i] === 0) c++;
    return c;
  }

  var ActSampler = { makeRng: makeRng, sampleInput: sampleInput,
    applyActivation: applyActivation, histogram: histogram,
    sampleMeanVar: sampleMeanVar, countExactZeros: countExactZeros };
  global.ActSampler = ActSampler;
  if (typeof module !== 'undefined' && module.exports) module.exports = ActSampler;
```

- [ ] **Step 4: 跑自检确认通过**

Run: `node web-tools/activation-demo/test/selftest.js`
Expected: 全部 `ok`，退出码 0

- [ ] **Step 5: Commit**

```bash
git add web-tools/activation-demo/js/sampler.js web-tools/activation-demo/test/selftest.js
git commit -m "feat: activation-demo 采样层"
```

---

### Task 3: 理论层 `js/theory.js`（临界点/求逆/密度/Gauss–Hermite 矩）

**Files:**
- Create: `web-tools/activation-demo/js/theory.js`
- Test: `web-tools/activation-demo/test/selftest.js`（追加）

**Interfaces:**
- Consumes: `ActFns`（`fn`/`dfn`/`helpers`）、`ActSampler`（仅自检用于矩对照）
- Produces: `ActTheory`：
  - `gaussHermite(n)` → `{ x, w }`（对权 e^(−x²) 的求积节点与权重）
  - `findCriticalPoints(act, p, lo, hi)` → `number[]`（dfn=0 临界点，升序）
  - `monotoneSegments(act, p, lo, hi)` → `[{a, b, fa, fb}]`
  - `solvePreimages(act, p, y, segments)` → `number[]`
  - `outputDensity(act, p, y, sigma, segments)` → number（无原像返回 0）
  - `densityGrid(act, p, sigma, lo, hi, nY)` → `{ ys, fs, yLo, yHi }`
  - `outputMoments(act, p, sigma)` → `{ mean, variance, atom }`
  - `suggestRange(act, p, sigma)` → `{ xLo, xHi, yLo, yHi }`（±8σ + 值域 10% 余量）

- [ ] **Step 1: 自检追加 theory 段（失败：模块不存在）**

`require` 区加 `var ActTheory = require('../js/theory.js');`，结尾汇总前加：

```js
console.log('== theory.js ==');
(function () {
  // Gauss–Hermite 自洽：Σw=√π、Σwx²=√π/2
  var gh = ActTheory.gaussHermite(64), s0 = 0, s2 = 0, i;
  for (i = 0; i < 64; i++) { s0 += gh.w[i]; s2 += gh.w[i] * gh.x[i] * gh.x[i]; }
  approx('GH Σw=√π', s0, Math.sqrt(Math.PI), 1e-12);
  approx('GH Σwx²=√π/2', s2, Math.sqrt(Math.PI) / 2, 1e-12);

  // gelu-exact 临界点：x≈−0.7518 唯一极小，f≈−0.16996
  var gelu = ActFns.byId('gelu-exact');
  var segs = ActTheory.monotoneSegments(gelu, {}, -8, 8);
  check('gelu 有 2 段单调段', segs.length === 2);
  approx('gelu 临界点 x≈−0.7518', segs[0].b, -0.7518, 1e-3);
  approx('gelu 极小值≈−0.16996', segs[0].fb, -0.16996, 1e-3);

  // 单调函数求逆：sigmoid y=0.5 的原像≈0
  var sig = ActFns.byId('sigmoid');
  var roots = ActTheory.solvePreimages(sig, {}, 0.5,
    ActTheory.monotoneSegments(sig, {}, -8, 8));
  check('sigmoid y=0.5 单一原像', roots.length === 1);
  approx('原像≈0', roots[0], 0, 1e-8);

  // gelu 在 (f(c), 0) 内有两个原像
  check('gelu y=−0.1 两个原像', ActTheory.solvePreimages(gelu, {}, -0.1, segs).length === 2);

  // 密度归一化：∫f_Y dy + atom ≈ 1（每个函数，σ=1）
  ActFns.list.forEach(function (act) {
    var p = ActFns.defaultParams(act);
    var r = ActTheory.suggestRange(act, p, 1);
    var grid = ActTheory.densityGrid(act, p, 1, r.xLo, r.xHi, 4000);
    var m = ActTheory.outputMoments(act, p, 1);
    var area = 0;
    for (var j = 1; j < grid.ys.length; j++) {
      area += 0.5 * (grid.fs[j] + grid.fs[j - 1]) * (grid.ys[j] - grid.ys[j - 1]);
    }
    approx(act.id + ' 密度积分+atom≈1', area + m.atom, 1, 2e-3);
  });

  // 矩：gelu 均值闭式 1/(2√π)（σ=1）；relu 点质量=0.5
  approx('gelu 均值=1/(2√π)', ActTheory.outputMoments(gelu, {}, 1).mean,
    1 / (2 * Math.sqrt(Math.PI)), 1e-5);
  approx('relu 点质量=0.5', ActTheory.outputMoments(ActFns.byId('relu'), {}, 1).atom, 0.5, 1e-7);

  // 矩 vs 蒙特卡洛（N=2e5，固定种子）
  ActFns.list.forEach(function (act) {
    var p = ActFns.defaultParams(act);
    var xs = ActSampler.sampleInput(ActSampler.makeRng(7), 200000, 1);
    var ys = ActSampler.applyActivation(act, p, xs);
    var mv = ActSampler.sampleMeanVar(ys);
    var mt = ActTheory.outputMoments(act, p, 1);
    approx(act.id + ' 理论均值 vs MC', mt.mean, mv.mean, 0.02);
    approxRel(act.id + ' 理论方差 vs MC', mt.variance, mv.variance, 0.05);
  });
})();
```

- [ ] **Step 2: 跑自检确认失败**

Run: `node web-tools/activation-demo/test/selftest.js`
Expected: `Cannot find module '../js/theory.js'`

- [ ] **Step 3: 实现 `js/theory.js`**

docstring：「activation-demo 理论层：高斯输入过激活函数的输出分布。单调段切分（临界点符号扫描+二分）、输出密度多原像求和、64 点 Gauss–Hermite 矩、relu 族 y=0 点质量 atom=P(x≤0)=0.5。无 DOM 依赖。」完整实现：

```js
(function (global) {
  'use strict';
  var ActFns = global.ActFns
    || (typeof require !== 'undefined' ? require('./functions.js') : null);

  /** Numerical Recipes gauher：对权 e^(−x²) 的 n 点节点与权重 */
  function gaussHermite(n) {
    var xs = new Float64Array(n), ws = new Float64Array(n);
    var m = Math.floor((n + 1) / 2), i, z = 0, pp = 0;
    for (i = 0; i < m; i++) {
      if (i === 0) z = Math.sqrt(2 * n + 1) - 1.85575 * Math.pow(2 * n + 1, -1 / 6);
      else if (i === 1) z = z - 1.14 * Math.pow(n, 0.426) / z;
      else if (i === 2) z = 1.86 * z - 0.86 * xs[0];
      else if (i === 3) z = 1.91 * z - 0.91 * xs[1];
      else z = 2 * z - xs[i - 2];
      for (var iter = 0; iter < 20; iter++) {
        var p1 = Math.pow(Math.PI, -0.25), p2 = 0, p3, j;
        for (j = 0; j < n; j++) {
          p3 = p2; p2 = p1;
          p1 = z * Math.sqrt(2 / (j + 1)) * p2 - Math.sqrt(j / (j + 1)) * p3;
        }
        pp = Math.sqrt(2 * n) * p2;
        var z1 = z;
        z = z1 - p1 / pp;
        if (Math.abs(z - z1) < 1e-14) break;
      }
      xs[i] = -z; xs[n - 1 - i] = z;
      var w = 2 / (pp * pp);
      ws[i] = w; ws[n - 1 - i] = w;
    }
    return { x: xs, w: ws };
  }
  var GH64 = null;
  function gh64() { if (!GH64) GH64 = gaussHermite(64); return GH64; }

  /** dfn=0 临界点：2001 点符号扫描 + 二分 60 次，返回升序数组 */
  function findCriticalPoints(act, p, lo, hi) {
    var M = 2001, roots = [];
    var xPrev = lo, dPrev = act.dfn(lo, p);
    for (var i = 1; i < M; i++) {
      var x = lo + (hi - lo) * i / (M - 1);
      var d = act.dfn(x, p);
      if (dPrev !== 0 && d !== 0 && ((dPrev < 0) !== (d < 0))) {
        var a = xPrev, b = x, da = dPrev;
        for (var it = 0; it < 60; it++) {
          var mid = 0.5 * (a + b), dm = act.dfn(mid, p);
          if (dm === 0) { a = mid; b = mid; break; }
          if ((da < 0) === (dm < 0)) { a = mid; da = dm; } else { b = mid; }
        }
        roots.push(0.5 * (a + b));
      }
      xPrev = x; dPrev = d;
    }
    return roots;
  }

  /** 在临界点切分 [lo,hi]，返回 [{a,b,fa,fb}] */
  function monotoneSegments(act, p, lo, hi) {
    var bounds = [lo].concat(findCriticalPoints(act, p, lo, hi), [hi]);
    var segs = [];
    for (var s = 0; s + 1 < bounds.length; s++) {
      segs.push({ a: bounds[s], b: bounds[s + 1],
        fa: act.fn(bounds[s], p), fb: act.fn(bounds[s + 1], p) });
    }
    return segs;
  }

  /** f(x)=y 的全部原像：逐段查值域、段内二分 80 次，按 1e-10 去重 */
  function solvePreimages(act, p, y, segments) {
    var roots = [];
    segments.forEach(function (seg) {
      var fa = seg.fa - y, fb = seg.fb - y;
      if (fa === 0) { roots.push(seg.a); return; }
      if (fa * fb > 0) return;
      var a = seg.a, b = seg.b;
      for (var it = 0; it < 80; it++) {
        var mid = 0.5 * (a + b), fm = act.fn(mid, p) - y;
        if (fa * fm <= 0) { b = mid; fb = fm; } else { a = mid; fa = fm; }
      }
      roots.push(0.5 * (a + b));
    });
    var uniq = [];
    roots.forEach(function (r) {
      if (!uniq.length || Math.abs(r - uniq[uniq.length - 1]) > 1e-10) uniq.push(r);
    });
    return uniq;
  }

  /** f_Y(y) = Σ f_X(x_i)/|f'(x_i)|；无原像返回 0 */
  function outputDensity(act, p, y, sigma, segments) {
    var roots = solvePreimages(act, p, y, segments);
    var sum = 0;
    roots.forEach(function (x) {
      var d = Math.abs(act.dfn(x, p));
      if (d > 0) sum += ActFns.helpers.normPdf(x, sigma) / d;
    });
    return sum;
  }

  /** 理论曲线网格：值域内取 nY 点算密度（值域留 2% 余量） */
  function densityGrid(act, p, sigma, lo, hi, nY) {
    var segs = monotoneSegments(act, p, lo, hi);
    var yLo = Infinity, yHi = -Infinity;
    segs.forEach(function (s) {
      yLo = Math.min(yLo, s.fa, s.fb);
      yHi = Math.max(yHi, s.fa, s.fb);
    });
    var pad = 0.02 * (yHi - yLo || 1);
    yLo -= pad; yHi += pad;
    var ys = new Float64Array(nY), fs = new Float64Array(nY);
    for (var i = 0; i < nY; i++) {
      ys[i] = yLo + (yHi - yLo) * i / (nY - 1);
      fs[i] = outputDensity(act, p, ys[i], sigma, segs);
    }
    return { ys: ys, fs: fs, yLo: yLo, yHi: yHi };
  }

  /** 输出矩：64 点 GH，E[g(x)] ≈ π^(−1/2) Σ w_i g(σ√2 x_i) */
  function outputMoments(act, p, sigma) {
    var gh = gh64(), t = sigma * Math.SQRT2, m1 = 0, m2 = 0;
    for (var i = 0; i < gh.x.length; i++) {
      var v = act.fn(t * gh.x[i], p);
      m1 += gh.w[i] * v;
      m2 += gh.w[i] * v * v;
    }
    m1 /= Math.sqrt(Math.PI);
    m2 /= Math.sqrt(Math.PI);
    return { mean: m1, variance: Math.max(0, m2 - m1 * m1),
      atom: act.atom ? ActFns.helpers.normCdf(0, sigma) : 0 };
  }

  /** 绘图范围：输入 ±8σ，y 值域留 10% 余量 */
  function suggestRange(act, p, sigma) {
    var xLo = -8 * sigma, xHi = 8 * sigma;
    var segs = monotoneSegments(act, p, xLo, xHi);
    var yLo = Infinity, yHi = -Infinity;
    segs.forEach(function (s) {
      yLo = Math.min(yLo, s.fa, s.fb);
      yHi = Math.max(yHi, s.fa, s.fb);
    });
    var pad = 0.1 * (yHi - yLo || 1);
    return { xLo: xLo, xHi: xHi, yLo: yLo - pad, yHi: yHi + pad };
  }

  var ActTheory = { gaussHermite: gaussHermite, findCriticalPoints: findCriticalPoints,
    monotoneSegments: monotoneSegments, solvePreimages: solvePreimages,
    outputDensity: outputDensity, densityGrid: densityGrid,
    outputMoments: outputMoments, suggestRange: suggestRange };
  global.ActTheory = ActTheory;
  if (typeof module !== 'undefined' && module.exports) module.exports = ActTheory;
})(typeof window !== 'undefined' ? window : globalThis);
```

- [ ] **Step 4: 跑自检确认通过**

Run: `node web-tools/activation-demo/test/selftest.js`
Expected: 全部 `ok`，退出码 0。若 gelu 临界点断言失败，先打印实际值核对（应 x≈−0.75179、f≈−0.16997）再调容差，不改算法。

- [ ] **Step 5: Commit**

```bash
git add web-tools/activation-demo/js/theory.js web-tools/activation-demo/test/selftest.js
git commit -m "feat: activation-demo 理论密度与矩层"
```

---

### Task 4: 页面骨架 `index.html` + `css/style.css`

**Files:**
- Create: `web-tools/activation-demo/index.html`
- Create: `web-tools/activation-demo/css/style.css`

**Interfaces:**
- Produces（app.js 依赖的 DOM id）: `chkSeed`, `inputSeed`, `p1view`（radio name）, `fnChecks`, `paramSliders1`, `sliderX`, `inputX`, `chartFn`, `chartDfn`, `selDist`, `sliderSigma`, `inputSigma`, `inputN2`, `paramSliders2`, `btnSample`, `btnReset`, `chartDist`, `distStats`, `distNote`

- [ ] **Step 1: 写 `index.html`**

要点（完整骨架，样式类名与 noise-ops-demo 一致）：

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>激活函数：曲线、导数与噪音分布</title>
  <link rel="stylesheet" href="css/style.css" />
  <script src="https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js"></script>
</head>
<body>
  <header>
    <div class="header-row"><h1>LLM 激活函数：曲线、导数与噪音分布</h1></div>
    <p class="subtitle">
      现代 LLM 的非线性全在这几个函数里。面板一对比函数与导数形状；面板二把高斯噪音灌进去看输出分布——
      <strong class="legend-note note-blue">ReLU 把一半质量压成 0 处的点质量（精确稀疏）</strong>，
      <strong class="legend-note note-orange">GELU/SiLU 非单调，极小值处密度出现可积奇点</strong>，
      <strong class="legend-note note-red">饱和函数把质量挤向两端</strong>。
    </p>
  </header>

  <section class="controls">
    <div class="control control-mc">
      <label for="chkSeed"><input id="chkSeed" type="checkbox" checked /> 固定种子</label>
      <input id="inputSeed" type="number" value="42" step="1" />
    </div>
    <span class="controls-hint">
      理论曲线随参数即时更新；蒙特卡洛不自动采样——点面板二的「采样」叠加直方图（参数变更后按钮高亮提示重新采样）
    </span>
  </section>

  <main>
    <div class="chart-wrap">
      <div class="panel-head">
        <h2>面板一：函数与导数</h2>
        <div class="op-radios">
          <label><input type="radio" name="p1view" value="fn" checked /> 只看函数</label>
          <label><input type="radio" name="p1view" value="dfn" /> 只看导数</label>
          <label><input type="radio" name="p1view" value="both" /> 双图联动</label>
        </div>
      </div>
      <div class="panel-controls">
        <div id="fnChecks" class="fn-checks"></div>
        <label for="sliderX">x 范围 ±</label>
        <input id="sliderX" type="range" min="1" max="10" step="0.5" value="5" />
        <input id="inputX" type="number" min="1" max="10" step="0.5" value="5" />
      </div>
      <div id="paramSliders1" class="panel-controls param-row"></div>
      <div id="chartFn" class="chart"></div>
      <div id="chartDfn" class="chart" style="display:none"></div>
      <p class="chart-note">
        勾选函数叠加对比；带参数的函数（Leaky ReLU 的 α、ELU 的 α、SiLU 的 β）在参数行调节。
        导数图直观看：sigmoid/tanh 两端梯度→0（饱和）、ReLU 负半轴梯度恒 0（死区）、GELU/SiLU 负半轴有一段平缓的"软死区"。
      </p>
      <table class="model-table">
        <thead><tr><th>模型</th><th>FFN 激活</th><th>备注</th></tr></thead>
        <tbody>
          <tr><td>GPT-2 / GPT-3</td><td>GELU（tanh 近似）</td><td>OpenAI 系惯例</td></tr>
          <tr><td>BERT</td><td>GELU（精确）</td><td></td></tr>
          <tr><td>LLaMA / Qwen / Kimi-K3</td><td>SwiGLU</td><td>门控变体（双分支相乘），后续面板覆盖</td></tr>
          <tr><td>Gemma</td><td>GeGLU</td><td>门控变体，后续面板覆盖</td></tr>
          <tr><td>早期 MLP / CNN</td><td>ReLU / Tanh</td><td></td></tr>
        </tbody>
      </table>
    </div>

    <div class="chart-wrap">
      <div class="panel-head">
        <h2>面板二：噪音过激活的分布</h2>
        <div class="op-radios">
          <label for="selDist">激活函数</label>
          <select id="selDist"></select>
        </div>
      </div>
      <div class="panel-controls">
        <label for="sliderSigma">输入方差 σ²</label>
        <input id="sliderSigma" type="range" min="0.1" max="10" step="0.1" value="1" />
        <input id="inputSigma" type="number" min="0.1" max="10" step="any" value="1" />
        <label for="inputN2">样本量 N</label>
        <input id="inputN2" type="number" min="10000" max="1000000" step="10000" value="100000" />
        <button id="btnSample" type="button">采样</button>
        <button id="btnReset" type="button">重置</button>
        <span class="radio-hint">输入 x ~ N(0, σ²)，看 y = f(x) 的分布</span>
      </div>
      <div id="paramSliders2" class="panel-controls param-row"></div>
      <div id="chartDist" class="chart"></div>
      <p class="chart-note" id="distStats"></p>
      <p class="chart-note" id="distNote"></p>
    </div>
  </main>

  <script src="js/functions.js"></script>
  <script src="js/theory.js"></script>
  <script src="js/sampler.js"></script>
  <script src="js/app.js"></script>
</body>
</html>
```

- [ ] **Step 2: 写 `css/style.css`**

以 noise-ops-demo 的 `css/style.css` 为基础原样复制（body/header/controls/panel/chart 等全部通用类），再追加本 demo 特有样式：

```css
/* 空参数行隐藏 */
.panel-controls:empty { display: none; }

/* 面板一函数勾选 chip：色点 + 名称 */
.fn-checks { display: flex; flex-wrap: wrap; gap: 4px 14px; width: 100%; }
.fn-checks label { display: inline-flex; align-items: center; gap: 5px; white-space: nowrap; cursor: pointer; }
.fn-checks .dot { width: 9px; height: 9px; border-radius: 50%; display: inline-block; }

/* 参数滑块行（两面板共用） */
.param-row .control-group { display: inline-flex; align-items: center; gap: 8px; }
.param-row input[type="range"] { width: 140px; }
.param-row input[type="number"] { width: 64px; }

/* 模型-激活对照表 */
.model-table { margin: 12px 4px 4px; border-collapse: collapse; font-size: 12px; color: #374151; }
.model-table th, .model-table td { border: 1px solid #e2e8f0; padding: 5px 12px; text-align: left; }
.model-table th { background: #f8fafc; font-weight: 600; }
```

- [ ] **Step 3: Commit**

```bash
git add web-tools/activation-demo/index.html web-tools/activation-demo/css/style.css
git commit -m "feat: activation-demo 页面骨架与样式"
```

---

### Task 5: `js/app.js`（一）：状态、面板一、面板二理论曲线

**Files:**
- Create: `web-tools/activation-demo/js/app.js`

**Interfaces:**
- Consumes: `ActFns`、`ActTheory`、Task 4 的全部 DOM id
- Produces（Task 6 依赖）:
  - 全局 `state`（含 `samples`/`sampleKey` 字段，本任务恒为 null）
  - `currentKey()` → `JSON.stringify([state.distId, state.sigma2, state.params[state.distId], state.N])`
  - `renderPanel2()` 渲染理论曲线（Task 6 整体替换为含直方图叠加的版本）
  - `updateSampleButton()`、`boot()`

- [ ] **Step 1: 实现 `js/app.js`**

docstring：「activation-demo UI 层：状态同步与 ECharts 渲染。面板一多函数 fn/dfn 曲线叠加（双图联动共享缩放）；面板二理论输出密度曲线（采样叠加在后续段落）。」完整实现：

```js
(function () {
  'use strict';

  var PALETTE = ['#2563eb', '#dc2626', '#16a34a', '#9333ea', '#ea580c',
    '#0891b2', '#be185d', '#65a30d', '#7c3aed', '#0d9488', '#b45309'];

  var state = {
    view: 'fn',
    selected: { relu: true, 'gelu-exact': true, silu: true, tanh: true },
    params: {},          // id -> {key: value}
    xRange: 5,
    distId: 'gelu-exact',
    sigma2: 1,
    N: 100000,
    samples: null,
    sampleKey: null,
  };
  ActFns.list.forEach(function (act) { state.params[act.id] = ActFns.defaultParams(act); });

  if (typeof echarts === 'undefined') {
    document.getElementById('chartFn').textContent = 'ECharts 加载失败（需联网走 CDN）';
    return;
  }
  var chartFn = echarts.init(document.getElementById('chartFn'));
  var chartDfn = echarts.init(document.getElementById('chartDfn'));
  var chartDist = echarts.init(document.getElementById('chartDist'));
  echarts.connect([chartFn, chartDfn]);

  function colorOf(id) {
    return PALETTE[ActFns.list.findIndex(function (a) { return a.id === id; }) % PALETTE.length];
  }
  function selectedActs() {
    return ActFns.list.filter(function (a) { return state.selected[a.id]; });
  }
  function clampNum(v, lo, hi, fallback) {
    v = parseFloat(v);
    if (!isFinite(v)) return fallback;
    return Math.min(hi, Math.max(lo, v));
  }

  /* ---------- 面板一 ---------- */

  function curveSeries(act, deriv) {
    var p = state.params[act.id];
    var n = 500, data = [];
    for (var i = 0; i < n; i++) {
      var x = -state.xRange + 2 * state.xRange * i / (n - 1);
      data.push([x, deriv ? act.dfn(x, p) : act.fn(x, p)]);
    }
    return { name: act.name, type: 'line', data: data, showSymbol: false,
      lineStyle: { width: 2, color: colorOf(act.id) },
      itemStyle: { color: colorOf(act.id) }, emphasis: { focus: 'series' } };
  }

  function panel1Option(deriv) {
    return {
      animation: false,
      grid: { left: 56, right: 20, top: 36, bottom: 48 },
      legend: { top: 0 },
      tooltip: { trigger: 'axis', axisPointer: { type: 'cross' } },
      xAxis: { type: 'value', name: 'x', min: -state.xRange, max: state.xRange },
      yAxis: { type: 'value', name: deriv ? "f'(x)" : 'f(x)', scale: true },
      dataZoom: [{ type: 'inside' }],
      series: selectedActs().map(function (a) { return curveSeries(a, deriv); }),
    };
  }

  function renderPanel1() {
    document.getElementById('chartDfn').style.display = state.view === 'both' ? '' : 'none';
    if (state.view === 'dfn') {
      chartFn.setOption(panel1Option(true), true);
    } else {
      chartFn.setOption(panel1Option(false), true);
      if (state.view === 'both') {
        chartDfn.resize();
        chartDfn.setOption(panel1Option(true), true);
      }
    }
  }

  function buildPanel1Controls() {
    var box = document.getElementById('fnChecks');
    box.innerHTML = '';
    ActFns.list.forEach(function (act) {
      var label = document.createElement('label');
      var cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.checked = !!state.selected[act.id];
      cb.addEventListener('change', function () {
        state.selected[act.id] = cb.checked;
        buildParamRows();
        renderPanel1();
      });
      var dot = document.createElement('span');
      dot.className = 'dot';
      dot.style.background = colorOf(act.id);
      label.appendChild(cb);
      label.appendChild(dot);
      label.appendChild(document.createTextNode(act.name + '  ' + act.formula));
      box.appendChild(label);
    });
  }

  /** 参数行：面板一列全部选中函数的参数；面板二只列当前函数的参数 */
  function buildParamRows() {
    fillParamRow(document.getElementById('paramSliders1'), selectedActs());
    fillParamRow(document.getElementById('paramSliders2'), [ActFns.byId(state.distId)]);
  }

  function fillParamRow(row, acts) {
    row.innerHTML = '';
    acts.forEach(function (act) {
      act.params.forEach(function (spec) {
        var group = document.createElement('span');
        group.className = 'control-group';
        var lab = document.createElement('label');
        lab.textContent = act.name + ' ' + spec.label;
        var slider = document.createElement('input');
        slider.type = 'range';
        slider.min = spec.min; slider.max = spec.max; slider.step = spec.step;
        slider.value = state.params[act.id][spec.key];
        var num = document.createElement('input');
        num.type = 'number';
        num.min = spec.min; num.max = spec.max; num.step = spec.step;
        num.value = state.params[act.id][spec.key];
        function sync(v) {
          state.params[act.id][spec.key] = clampNum(v, spec.min, spec.max, spec.def);
          slider.value = state.params[act.id][spec.key];
          num.value = state.params[act.id][spec.key];
          renderPanel1();
          renderPanel2();
        }
        slider.addEventListener('input', function () { sync(slider.value); });
        num.addEventListener('change', function () { sync(num.value); });
        group.appendChild(lab); group.appendChild(slider); group.appendChild(num);
        row.appendChild(group);
      });
    });
  }

  /* ---------- 面板二（理论曲线） ---------- */

  function currentKey() {
    return JSON.stringify([state.distId, state.sigma2, state.params[state.distId], state.N]);
  }

  function renderPanel2() {
    var act = ActFns.byId(state.distId);
    var p = state.params[act.id];
    var sigma = Math.sqrt(state.sigma2);
    var range = ActTheory.suggestRange(act, p, sigma);
    var grid = ActTheory.densityGrid(act, p, sigma, range.xLo, range.xHi, 800);
    var line = [];
    for (var i = 0; i < grid.ys.length; i++) line.push([grid.ys[i], grid.fs[i]]);
    var series = [{ name: '理论密度', type: 'line', data: line, showSymbol: false,
      lineStyle: { width: 2, color: '#2563eb' }, itemStyle: { color: '#2563eb' } }];
    var m = ActTheory.outputMoments(act, p, sigma);
    if (act.atom) {
      series[0].markLine = { symbol: 'none', silent: true,
        lineStyle: { type: 'dashed', color: '#dc2626' },
        label: { formatter: '点质量 P(y=0)=' + m.atom.toFixed(3) },
        data: [{ xAxis: 0 }] };
    }
    chartDist.setOption({
      animation: false,
      grid: { left: 56, right: 20, top: 36, bottom: 48 },
      legend: { top: 0 },
      tooltip: { trigger: 'axis' },
      xAxis: { type: 'value', name: 'y', min: grid.yLo, max: grid.yHi },
      yAxis: { type: 'value', name: '密度', scale: true },
      dataZoom: [{ type: 'inside' }],
      series: series,
    }, true);
    document.getElementById('distStats').innerHTML =
      '理论：均值 ' + m.mean.toFixed(4) + '，方差 ' + m.variance.toFixed(4) +
      (act.atom ? '，点质量 ' + m.atom.toFixed(3) : '') +
      ' ｜ <span class="stat-dim">样本：未采样（点「采样」叠加直方图）</span>';
    document.getElementById('distNote').textContent = act.distNote;
    updateSampleButton();
  }

  function updateSampleButton() {
    var btn = document.getElementById('btnSample');
    if (state.samples === null || state.sampleKey !== currentKey()) btn.classList.add('need-sample');
    else btn.classList.remove('need-sample');
  }

  /* ---------- 控件绑定与启动 ---------- */

  function buildDistSelect() {
    var sel = document.getElementById('selDist');
    ActFns.list.forEach(function (act) {
      var opt = document.createElement('option');
      opt.value = act.id;
      opt.textContent = act.name;
      sel.appendChild(opt);
    });
    sel.value = state.distId;
    sel.addEventListener('change', function () {
      state.distId = sel.value;
      buildParamRows();
      renderPanel2();
    });
  }

  function bindGlobal() {
    document.querySelectorAll('input[name="p1view"]').forEach(function (r) {
      r.addEventListener('change', function () { state.view = r.value; renderPanel1(); });
    });
    var sliderX = document.getElementById('sliderX');
    var inputX = document.getElementById('inputX');
    function syncX(v) {
      state.xRange = clampNum(v, 1, 10, 5);
      sliderX.value = state.xRange;
      inputX.value = state.xRange;
      renderPanel1();
    }
    sliderX.addEventListener('input', function () { syncX(sliderX.value); });
    inputX.addEventListener('change', function () { syncX(inputX.value); });

    var sliderS = document.getElementById('sliderSigma');
    var inputS = document.getElementById('inputSigma');
    function syncS(v) {
      state.sigma2 = clampNum(v, 0.1, 10, 1);
      sliderS.value = state.sigma2;
      inputS.value = state.sigma2;
      renderPanel2();
    }
    sliderS.addEventListener('input', function () { syncS(sliderS.value); });
    inputS.addEventListener('change', function () { syncS(inputS.value); });

    document.getElementById('inputN2').addEventListener('change', function () {
      state.N = Math.round(clampNum(this.value, 10000, 1000000, 100000));
      this.value = state.N;
      updateSampleButton();
    });
  }

  window.addEventListener('resize', function () {
    chartFn.resize(); chartDfn.resize(); chartDist.resize();
  });

  function boot() {
    buildPanel1Controls();
    buildDistSelect();
    buildParamRows();
    bindGlobal();
    renderPanel1();
    renderPanel2();
  }
  boot();
})();
```

- [ ] **Step 2: 浏览器人工验证面板一与理论曲线**

启动本地服务：`python -m http.server 8000`，打开
`http://localhost:8000/web-tools/activation-demo/` 检查：
- 面板一默认 4 条曲线（ReLU/GELU/SiLU/Tanh），切换「只看导数」「双图联动」正常；
- 勾选 silu 出现 β 滑块，拖到 5 曲线趋近 ReLU；
- 面板二默认 GELU 理论密度，左侧极小值附近有尖峰；切到 relu 出现 y=0 红色虚线点质量标注；
- 拖 σ² 曲线即时变化。

- [ ] **Step 3: Commit**

```bash
git add web-tools/activation-demo/js/app.js
git commit -m "feat: activation-demo 面板一与面板二理论曲线"
```

---

### Task 6: `js/app.js`（二）：面板二采样叠加与统计行

**Files:**
- Modify: `web-tools/activation-demo/js/app.js`

**Interfaces:**
- Consumes: `ActSampler`（`makeRng`/`sampleInput`/`applyActivation`/`histogram`/`sampleMeanVar`/`countExactZeros`）、Task 5 的 `state`/`currentKey()`/`updateSampleButton()`
- Produces: 完整 `renderPanel2()`（含直方图叠加与样本统计）、`doSample()`

- [ ] **Step 1: 替换 `renderPanel2`，追加采样逻辑并替换 `boot`**

把 Task 5 的 `renderPanel2` 整个替换为（新增直方图轮廓叠加与样本统计）：

```js
  function renderPanel2() {
    var act = ActFns.byId(state.distId);
    var p = state.params[act.id];
    var sigma = Math.sqrt(state.sigma2);
    var range = ActTheory.suggestRange(act, p, sigma);
    var grid = ActTheory.densityGrid(act, p, sigma, range.xLo, range.xHi, 800);
    var line = [];
    for (var i = 0; i < grid.ys.length; i++) line.push([grid.ys[i], grid.fs[i]]);
    var series = [];
    if (state.samples) {
      // 直方图轮廓：逐箱两点的阶梯折线 + 半透明填充
      var hist = ActSampler.histogram(state.samples, grid.yLo, grid.yHi, 120);
      var w = (grid.yHi - grid.yLo) / 120;
      var outline = [[grid.yLo, 0]];
      for (var b = 0; b < hist.centers.length; b++) {
        outline.push([grid.yLo + b * w, hist.density[b]]);
        outline.push([grid.yLo + (b + 1) * w, hist.density[b]]);
      }
      outline.push([grid.yHi, 0]);
      series.push({ name: '蒙特卡洛直方图', type: 'line', data: outline,
        showSymbol: false, silent: true,
        lineStyle: { width: 1, color: '#ea580c' },
        itemStyle: { color: '#ea580c' }, areaStyle: { opacity: 0.25 } });
    }
    series.push({ name: '理论密度', type: 'line', data: line, showSymbol: false,
      lineStyle: { width: 2, color: '#2563eb' }, itemStyle: { color: '#2563eb' } });
    var m = ActTheory.outputMoments(act, p, sigma);
    if (act.atom) {
      series[series.length - 1].markLine = { symbol: 'none', silent: true,
        lineStyle: { type: 'dashed', color: '#dc2626' },
        label: { formatter: '点质量 P(y=0)=' + m.atom.toFixed(3) },
        data: [{ xAxis: 0 }] };
    }
    chartDist.setOption({
      animation: false,
      grid: { left: 56, right: 20, top: 36, bottom: 48 },
      legend: { top: 0 },
      tooltip: { trigger: 'axis' },
      xAxis: { type: 'value', name: 'y', min: grid.yLo, max: grid.yHi },
      yAxis: { type: 'value', name: '密度', scale: true },
      dataZoom: [{ type: 'inside' }],
      series: series,
    }, true);
    var stats = '理论：均值 ' + m.mean.toFixed(4) + '，方差 ' + m.variance.toFixed(4) +
      (act.atom ? '，点质量 ' + m.atom.toFixed(3) : '') + ' ｜ ';
    if (state.samples) {
      var mv = ActSampler.sampleMeanVar(state.samples);
      stats += '样本（N=' + state.samples.length + '）：均值 ' + mv.mean.toFixed(4) +
        '，方差 ' + mv.variance.toFixed(4);
      if (act.atom) {
        stats += '，0 处比例 ' +
          (ActSampler.countExactZeros(state.samples) / state.samples.length).toFixed(3);
      }
    } else {
      stats += '<span class="stat-dim">样本：未采样（点「采样」叠加直方图）</span>';
    }
    document.getElementById('distStats').innerHTML = stats;
    document.getElementById('distNote').textContent = act.distNote;
    updateSampleButton();
  }
```

在 `boot()` 前插入 `doSample`，并把 `boot` 替换为：

```js
  function doSample() {
    var act = ActFns.byId(state.distId);
    var seed = document.getElementById('chkSeed').checked
      ? parseInt(document.getElementById('inputSeed').value, 10) || 0
      : (Math.random() * 2147483647) | 0;
    var gauss = ActSampler.makeRng(seed);
    var xs = ActSampler.sampleInput(gauss, state.N, Math.sqrt(state.sigma2));
    state.samples = ActSampler.applyActivation(act, state.params[act.id], xs);
    state.sampleKey = currentKey();
    renderPanel2();
  }

  function boot() {
    buildPanel1Controls();
    buildDistSelect();
    buildParamRows();
    bindGlobal();
    document.getElementById('btnSample').addEventListener('click', doSample);
    document.getElementById('btnReset').addEventListener('click', function () {
      state.samples = null;
      state.sampleKey = null;
      renderPanel2();
    });
    document.getElementById('chkSeed').addEventListener('change', updateSampleButton);
    document.getElementById('inputSeed').addEventListener('change', updateSampleButton);
    renderPanel1();
    renderPanel2();
  }
```

- [ ] **Step 2: 跑自检确认数值层无回归**

Run: `node web-tools/activation-demo/test/selftest.js`
Expected: 全部 `ok`，退出码 0

- [ ] **Step 3: 浏览器人工验证采样交互**

`http://localhost:8000/web-tools/activation-demo/`：
- 点「采样」叠加橙色直方图，与蓝色理论曲线吻合；统计行出现样本均值/方差；
- 选 ReLU → 采样：统计行"0 处比例 ≈ 0.500"与理论点质量对照；
- 改 σ² 或换函数后「采样」按钮变蓝（失效提示），重采样恢复；
- 「重置」清空直方图；取消"固定种子"两次采样结果不同。

- [ ] **Step 4: Commit**

```bash
git add web-tools/activation-demo/js/app.js
git commit -m "feat: activation-demo 面板二采样叠加与统计"
```

---

### Task 7: README.md + 根目录登记 + 收尾验证

**Files:**
- Create: `web-tools/activation-demo/README.md`
- Modify: `index.html`（根目录，工具列表 `<li>` 卡片末尾追加，noise-ops-demo 卡片之后）

**Interfaces:**
- Consumes: 全部已完成任务

- [ ] **Step 1: 写 `README.md`**

按既有 demo 的 README 体例（标题+简介 → 用法 → 文件结构 → 数学背景）写，包含以下各节与公式（行内公式注意空格规范）：

1. 简介：两个面板各讲什么（面板一 11 函数含 GELU 双版本与带参函数；面板二理论密度 vs 蒙特卡洛，点质量与可积奇点）。
2. 用法：直接开 `index.html`（ECharts CDN 需联网）；采样手动触发、失效高亮、固定种子；自检 `node test/selftest.js`。
3. 文件结构：6 个文件各一行说明。
4. 数学背景 §1 输出密度：变量替换与多原像求和

$$f_Y(y)=\sum_i \frac{f_X(x_i)}{|f'(x_i)|}$$

   说明单调段切分与二分求逆的实现对应关系。
5. §2 可积奇点：极小值 $c$ 附近 $f(x)-f(c)\approx\frac12 f''(c)(x-c)^2$，故
   $f_Y(y)\sim (y-f(c))^{-1/2}$ 可积；GELU $\sigma=1$ 时峰在 $y\approx-0.170$。
6. §3 ReLU 点质量与精确稀疏：$P(y=0)=\Phi(0)=1/2$，$y>0$ 为截断正态右半，
   连续密度只积到 1/2；与直方图"恰好为 0"比例对照。ReLU² 同理且均值 $=\sigma^2/2$。
7. §4 理论矩 Gauss–Hermite：

$$\mathbb E[g(x)]\approx\frac{1}{\sqrt\pi}\sum_{i=1}^{64} w_i\,g\bigl(\sigma\sqrt2\,t_i\bigr)$$

   节点/权重来自 gauher；GELU 均值闭式校验（Stein 引理）：

$$\mathbb E[\mathrm{gelu}(x)]=\frac{\sigma^2}{\sqrt{2\pi(1+\sigma^2)}},
\quad \sigma=1 \Rightarrow \frac{1}{2\sqrt\pi}\approx 0.2821$$
8. §5 各函数来历与 LLM 对应：Sigmoid/Tanh（RNN 时代，饱和梯度消失；sigmoid 以"门"的形式活在 SiLU/SwiGLU 里）；ReLU 族（死区与稀疏）；GELU（BERT/GPT 系，tanh 近似的来历）；SiLU/Swish（LLaMA 系 SwiGLU 的门支）；Mish；ELU/Softplus 的历史地位。

- [ ] **Step 2: 行内公式空格检查**

Run: `python tools/fix_md_math_spacing.py --apply web-tools/activation-demo/README.md`
Expected: 修复全部报告项后退出码 0

- [ ] **Step 3: 根目录 `index.html` 登记卡片**

在 noise-ops-demo 卡片 `</li>` 之后、`</ul>` 之前插入：

```html
        <li class="tool-card">
          <a href="web-tools/activation-demo/">
            <h2>激活函数：曲线、导数与噪音分布<span class="arrow">→</span></h2>
            <p>
              LLM 里的 11 个激活函数（GELU 精确/tanh 近似、SiLU/Swish、Mish、ReLU 族……）：
              函数与导数叠加对比（饱和区、死区、平滑化），高斯噪音过激活的输出分布——
              理论密度（多原像求和、可积奇点）与蒙特卡洛直方图对照，
              ReLU 的 y=0 点质量即"精确稀疏"的几何图像。
            </p>
            <div class="tags">
              <span class="tag">激活函数</span>
              <span class="tag">分布视角</span>
              <span class="tag">理论对照</span>
            </div>
          </a>
        </li>
```

- [ ] **Step 4: 收尾验证**

- `node web-tools/activation-demo/test/selftest.js` 全过；
- 浏览器打开根目录 `http://localhost:8000/` 确认新卡片可点进工具页，两个面板交互正常；
- 停掉 http.server。

- [ ] **Step 5: Commit**

```bash
git add web-tools/activation-demo/README.md index.html
git commit -m "docs: activation-demo README 与首页登记"
```

---

## Self-Review 记录

- Spec 覆盖：spec §2 函数清单 → Task 1；§3 文件结构 → Tasks 1–7；§4 面板一 → Tasks 4/5；
  §5 面板二 → Tasks 3/5/6；§6 数据流 → Task 1/5；§7 错误处理 → clampNum/Task 5 echarts 守卫/README 注明；
  §8 测试 → Tasks 1–3 自检；§9 文档登记 → Task 7。无缺口。
- 类型一致性：`ActFns/ActSampler/ActTheory` 导出与各任务消费签名一致；
  `renderPanel2` 在 Task 6 为整体替换（旧版被完全覆盖，无残留调用）；
  `state.samples/sampleKey` 在 Task 5 已定义，Task 6 填充。
- 已知风险：erf 逼近误差 1.5e-7，GELU 均值闭式断言容差 1e-5 足够；密度归一化容差 2e-3
  对奇点附近梯形积分误差偏大一侧仍安全（4000 网格下实测应远小于容差）。




