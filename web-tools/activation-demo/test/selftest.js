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

console.log(failures === 0 ? '\n全部通过' : '\n失败 ' + failures + ' 项');
process.exit(failures === 0 ? 0 : 1);
