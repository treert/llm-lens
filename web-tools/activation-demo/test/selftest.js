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
var ActSampler = require('../js/sampler.js');
var ActTheory = require('../js/theory.js');

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

  // 密度闭式抽查（单调函数，σ=1）：f_Y(y) = φ(g(y))·|g'(y)|
  var H = ActFns.helpers;
  approx('sigmoid fY(0.5)=φ(0)/0.25',
    ActTheory.outputDensity(sig, {}, 0.5, 1, ActTheory.monotoneSegments(sig, {}, -8, 8)),
    H.normPdf(0, 1) / 0.25, 1e-6);
  approx('tanh fY(0)=φ(0)',
    ActTheory.outputDensity(ActFns.byId('tanh'), {}, 0, 1,
      ActTheory.monotoneSegments(ActFns.byId('tanh'), {}, -8, 8)),
    H.normPdf(0, 1), 1e-6);
  approx('relu fY(1)=φ(1)',
    ActTheory.outputDensity(ActFns.byId('relu'), {}, 1, 1,
      ActTheory.monotoneSegments(ActFns.byId('relu'), {}, -8, 8)),
    H.normPdf(1, 1), 1e-6);

  /**
   * 区间 [y1,y2] 的理论质量：逐单调段求交后取原像，
   * 用 normCdf 差精确求质量（绕开可积奇点的数值求积误差）。
   */
  function massInBin(act, p, sigma, segs, y1, y2) {
    var mass = 0;
    segs.forEach(function (seg) {
      var sLo = Math.min(seg.fa, seg.fb), sHi = Math.max(seg.fa, seg.fb);
      var lo = Math.max(y1, sLo), hi = Math.min(y2, sHi);
      if (!(lo < hi)) return;
      var span = hi - lo;
      lo += span * 1e-9; // 避开端点退化（relu 左半整段映射到 0）
      hi -= span * 1e-9;
      var rLo = ActTheory.solvePreimages(act, p, lo, [seg]);
      var rHi = ActTheory.solvePreimages(act, p, hi, [seg]);
      if (!rLo.length || !rHi.length) return;
      mass += Math.abs(H.normCdf(rHi[0], sigma) - H.normCdf(rLo[0], sigma));
    });
    return mass;
  }

  // 每函数：40 个等宽 bin，理论质量 vs MC 比例；总质量（含 atom）=1
  var NMC = 200000;
  ActFns.list.forEach(function (act) {
    var p = ActFns.defaultParams(act);
    var r = ActTheory.suggestRange(act, p, 1);
    var segs = ActTheory.monotoneSegments(act, p, r.xLo, r.xHi);
    var xs = ActSampler.sampleInput(ActSampler.makeRng(7), NMC, 1);
    var ys = ActSampler.applyActivation(act, p, xs);
    var NB = 40, bw = (r.yHi - r.yLo) / NB, total = 0, badBins = 0;
    for (var b = 0; b < NB; b++) {
      var y1 = r.yLo + b * bw, y2 = y1 + bw;
      var mass = massInBin(act, p, 1, segs, y1, y2);
      if (act.atom && y1 <= 0 && 0 < y2) mass += 0.5;
      total += mass;
      var cnt = 0;
      for (var k = 0; k < NMC; k++) if (ys[k] >= y1 && ys[k] < y2) cnt++;
      var tolB = Math.max(0.006, 6 * Math.sqrt(Math.max(mass, 1e-6) * (1 - mass) / NMC));
      if (Math.abs(mass - cnt / NMC) > tolB) badBins++;
    }
    check(act.id + ' 各 bin 理论质量 vs MC', badBins === 0);
    approx(act.id + ' 理论总质量≈1', total, 1, 2e-5);
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

console.log('== gates 注册表 ==');
(function () {
  check('4 个门', ActFns.gates.length === 4);
  ['swiglu', 'glu', 'geglu', 'reglu'].forEach(function (id) {
    var g = ActFns.gateById(id);
    check('gateById(' + id + ')', g && g.id === id);
    check(id + ' 门函数条目存在', !!ActFns.gateAct(g));
  });
  check('reglu 有 atom', ActFns.gateById('reglu').atom === true);
  check('swiglu/glu/geglu 无 atom',
    !ActFns.gateById('swiglu').atom && !ActFns.gateById('glu').atom && !ActFns.gateById('geglu').atom);
  check('swiglu 门是 silu', ActFns.gateAct(ActFns.gateById('swiglu')).id === 'silu');
})();

console.log('== sampler.js GLU ==');
(function () {
  var N = 200000;
  var bv = ActSampler.sampleBivariate(ActSampler.makeRng(11), N, 1, 0.7);
  var mvU = ActSampler.sampleMeanVar(bv.u), mvV = ActSampler.sampleMeanVar(bv.v);
  approx('u 边缘方差≈1', mvU.variance, 1, 0.02);
  approx('v 边缘方差≈1', mvV.variance, 1, 0.02);
  var cov = 0, i;
  for (i = 0; i < N; i++) cov += (bv.u[i] - mvU.mean) * (bv.v[i] - mvV.mean);
  cov /= (N - 1);
  approx('样本协方差≈ρ=0.7', cov, 0.7, 0.01);

  // applyGate：relu 门在 v<0 时输出恒 0
  var relu = ActFns.byId('relu');
  var y = ActSampler.applyGate(relu, {}, bv.u, bv.v);
  var bad = 0;
  for (i = 0; i < N; i++) if (bv.v[i] < 0 && y[i] !== 0) bad++;
  check('relu 门 v<0 输出恒 0', bad === 0);
  approx('relu 门 y=0 比例≈0.5', ActSampler.countExactZeros(y) / N, 0.5, 0.01);
})();

console.log('== theory.js GLU ==');
(function () {
  var NoiseTheory = require('../../noise-ops-demo/js/theory.js');
  var NG = 200000;

  // 矩 vs MC：ρ ∈ {-0.5, 0, 0.7} × 4 门
  [-0.5, 0, 0.7].forEach(function (rho) {
    ActFns.gates.forEach(function (gate) {
      var g = ActFns.gateAct(gate), gp = ActFns.defaultParams(g);
      var bv = ActSampler.sampleBivariate(ActSampler.makeRng(7), NG, 1, rho);
      var ys = ActSampler.applyGate(g, gp, bv.u, bv.v);
      var mv = ActSampler.sampleMeanVar(ys);
      var mt = ActTheory.gluOutputMoments(gate, rho, 1);
      approx(gate.id + ' ρ=' + rho + ' 理论均值 vs MC', mt.mean, mv.mean, 0.02);
      approxRel(gate.id + ' ρ=' + rho + ' 理论方差 vs MC', mt.variance, mv.variance, 0.05);
      if (gate.atom) approx(gate.id + ' atom=0.5', mt.atom, 0.5, 1e-15);
    });
  });

  // 条件分布：swiglu ρ=0.6 v0=1.3，|v−v0|<0.02 的 MC 条件样本
  (function () {
    var gate = ActFns.gateById('swiglu');
    var g = ActFns.gateAct(gate), gp = ActFns.defaultParams(g);
    var bv = ActSampler.sampleBivariate(ActSampler.makeRng(5), 400000, 1, 0.6);
    var ys = ActSampler.applyGate(g, gp, bv.u, bv.v);
    var sel = [];
    for (var i = 0; i < ys.length; i++) if (Math.abs(bv.v[i] - 1.3) < 0.02) sel.push(ys[i]);
    check('条件样本量充足（>800）', sel.length > 800);
    var mv = ActSampler.sampleMeanVar(Float64Array.from(sel));
    var cond = ActTheory.gluConditional(gate, 0.6, 1, 1.3);
    approxRel('条件均值 vs MC', cond.mean, mv.mean, 0.05);
    approxRel('条件 SD vs MC', cond.sd, Math.sqrt(mv.variance), 0.05);
  })();

  // ReGLU ρ=0 闭式：y>0 时 f(y) = K₀(y)/(2π)（σ=1）
  var reglu = ActFns.gateById('reglu');
  [0.5, 1, 2].forEach(function (y) {
    var closed = Math.exp(NoiseTheory.logBesselK(0, y)) / (2 * Math.PI);
    approxRel('ReGLU ρ=0 K₀ 闭式 y=' + y, ActTheory.gluOutputDensity(reglu, 0, 1, y), closed, 1e-4);
  });

  // gluBinMass：30 个等宽 bin（±6）理论质量+atom vs MC；总质量≈1
  ActFns.gates.forEach(function (gate) {
    var g = ActFns.gateAct(gate), gp = ActFns.defaultParams(g);
    var rho = 0.3;
    var bv = ActSampler.sampleBivariate(ActSampler.makeRng(7), NG, 1, rho);
    var ys = ActSampler.applyGate(g, gp, bv.u, bv.v);
    var NB = 30, bw = 12 / NB, total = 0, badBins = 0;
    // 统一用 floor((y+6)/bw) 定位 bin（理论与 MC 同规则，atom 必然同 bin）
    var hist = ActSampler.histogram(ys, -6, 6, NB);
    var atomBin = Math.min(NB - 1, Math.max(0, Math.floor(6 / bw)));
    for (var b = 0; b < NB; b++) {
      var y1 = -6 + b * bw, y2 = y1 + bw;
      var mass = ActTheory.gluBinMass(gate, rho, 1, y1, y2);
      if (gate.atom && b === atomBin) mass += 0.5;
      total += mass;
      var tolB = Math.max(0.006, 6 * Math.sqrt(Math.max(mass, 1e-6) * (1 - mass) / NG));
      if (Math.abs(mass - hist.density[b] * bw) > tolB) badBins++;
    }
    check(gate.id + ' 各 bin 理论质量 vs MC', badBins === 0);
    approx(gate.id + ' 总质量≈1', total, 1, 3e-3);
  });

  // gaussEllipse：逐点满足 Mahalanobis 距离 = k²σ²(1−ρ²)
  (function () {
    var rho = 0.6, sigma = 1.5, k = 2;
    var pts = ActTheory.gaussEllipse(rho, sigma, k, 50);
    var target = k * k * sigma * sigma * (1 - rho * rho), bad = 0;
    pts.forEach(function (pt) {
      var lhs = pt[0] * pt[0] - 2 * rho * pt[0] * pt[1] + pt[1] * pt[1];
      if (Math.abs(lhs - target) / target > 1e-12) bad++;
    });
    check('椭圆点满足等高线方程', bad === 0);
  })();
})();

console.log(failures === 0 ? '\n全部通过' : '\n失败 ' + failures + ' 项');
process.exit(failures === 0 ? 0 : 1);
