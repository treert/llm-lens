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

console.log(failures === 0 ? '\n全部通过' : '\n失败 ' + failures + ' 项');
process.exit(failures === 0 ? 0 : 1);
