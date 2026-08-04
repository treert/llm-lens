/**
 * montecarlo.js 的 node 自检（对应 monte-carlo-design.md §9）。
 * 用法：node web-tools/orthogonal-demo/test/mc-selftest.js
 * 无测试框架，直接断言；全部通过时退出码 0。
 */
'use strict';

// theory.js 以 (function(global){...})(window) 导出，node 下垫一个 window
globalThis.window = globalThis;
require('../js/theory.js');
require('../js/montecarlo.js');
const T = globalThis.Theory;
const MC = globalThis.MC;

let failures = 0;
function check(name, cond, detail) {
  if (cond) {
    console.log('  PASS ' + name);
  } else {
    failures++;
    console.log('  FAIL ' + name + (detail ? ' -- ' + detail : ''));
  }
}
function drain(gen) {
  let r;
  do {
    r = gen.next();
  } while (!r.done);
  return r.value;
}

// ---------- 1. 预算函数 ----------
console.log('[1] 预算函数');
{
  const MB = 1024 * 1024;
  check('probe 峰值 0 -> 兜底 256MB', MC.budgetFromProbe(0) === 256 * MB);
  check('probe 峰值 1GB -> 0.7GB', MC.budgetFromProbe(1024 * MB) === 0.7 * 1024 * MB);
  check(
    'probe 峰值 10GB -> 封顶 2GB',
    MC.budgetFromProbe(10 * 1024 * MB) === 2 * 1024 * MB
  );
  const k1 = MC.computeKMax(1024, 16, 2 * 1024 * MB, 5e9);
  check('computeKMax(1024,16,2GB,5e9)=781（时间封顶）', k1 === 781, 'got ' + k1);
  const k2 = MC.computeKMax(8192, 64, 256 * MB, 5e15);
  check('computeKMax(8192,64,256MB,大ops)=128（内存封顶）', k2 === 128, 'got ' + k2);
}

// ---------- 2. K=2 单侧中位数 ≈ 0 ----------
console.log('[2] K=2 单侧中位数 ≈ 0');
{
  const s = MC.createSession({ N: 1024, KMax: 2, twoSided: false, seed: 1 });
  drain(s.nextBatch(256));
  const agg = MC.aggregateColumn(s.pool, 2);
  check('n=256', agg.n === 256, 'n=' + agg.n);
  check('|median| < 0.01', Math.abs(agg.median) < 0.01, 'median=' + agg.median);
}

// ---------- 3 & 4. 直方图均值 ≈ 0；跨轨迹中位数 vs F^M ----------
console.log('[3&4] 均值与中位数精度（N=1024, KMax=200, R=64）');
{
  const s = MC.createSession({ N: 1024, KMax: 200, twoSided: false, seed: 42 });
  drain(s.nextBatch(64));
  const mean = s.pool.sumAll / s.pool.cntAll;
  check('|点积均值| < 0.002', Math.abs(mean) < 0.002, 'mean=' + mean);
  for (const k of [10, 50, 200]) {
    const agg = MC.aggregateColumn(s.pool, k);
    const theo = T.maxDotQuantileBeta(0.5, 1024, k, false);
    const rel = Math.abs(agg.median - theo) / theo;
    check('K=' + k + ' 中位数相对偏差 < 10%', rel < 0.1,
      'mc=' + agg.median.toFixed(5) + ' theo=' + theo.toFixed(5) +
      ' rel=' + (rel * 100).toFixed(2) + '%');
  }
}

// ---------- 5. 确定性：同种子两次运行完全一致 ----------
console.log('[5] 固定种子确定性');
{
  const cfg = { N: 64, KMax: 30, twoSided: false, seed: 7 };
  const a = MC.createSession(cfg);
  drain(a.nextBatch(8));
  const b = MC.createSession(cfg);
  drain(b.nextBatch(8));
  let same = a.pool.maxRuns.length === b.pool.maxRuns.length;
  for (let i = 0; same && i < a.pool.maxRuns.length; i++) {
    for (let k = 2; same && k <= 30; k++) {
      if (a.pool.maxRuns[i][k] !== b.pool.maxRuns[i][k]) same = false;
    }
  }
  check('maxRuns 完全一致', same);
  check('cntAll 一致', a.pool.cntAll === b.pool.cntAll,
    a.pool.cntAll + ' vs ' + b.pool.cntAll);
}

// ---------- 6. 分批池化等价：一批16 ≡ 两批各8 ----------
console.log('[6] 分批池化等价性');
{
  const cfg = { N: 64, KMax: 30, twoSided: false, seed: 13 };
  const a = MC.createSession(cfg);
  drain(a.nextBatch(16));
  const b = MC.createSession(cfg);
  drain(b.nextBatch(8));
  drain(b.nextBatch(8));
  let same = a.pool.maxRuns.length === 16 && b.pool.maxRuns.length === 16;
  for (let i = 0; same && i < 16; i++) {
    for (let k = 2; same && k <= 30; k++) {
      if (a.pool.maxRuns[i][k] !== b.pool.maxRuns[i][k]) same = false;
    }
  }
  check('maxRuns 集合一致（轨迹独立子流）', same);
  check('直方图 count 总和一致', a.pool.cntAll === b.pool.cntAll,
    a.pool.cntAll + ' vs ' + b.pool.cntAll);
}

// ---------- 7. 列切片直方图峰值 vs F^M 密度 ----------
console.log('[7] 列切片密度重建（N=1024, K=50, R=256）');
{
  const s = MC.createSession({ N: 1024, KMax: 50, twoSided: false, seed: 99 });
  drain(s.nextBatch(256));
  const k = 50;
  const lo = T.maxDotQuantileBeta(0.0001, 1024, k, false);
  const hi = T.maxDotQuantileBeta(0.9999, 1024, k, false);
  const bins = 40;
  const h = MC.columnHist(s.pool, k, lo, hi, bins);
  // 模拟直方图峰值 bin
  let imax = 0;
  for (let i = 1; i < bins; i++) if (h.ys[i] > h.ys[imax]) imax = i;
  // 理论密度在同网格上的峰值 bin
  let tmax = 0;
  for (let i = 1; i < bins; i++) {
    if (T.maxDotDensityBeta(h.xs[i], 1024, k, false) >
        T.maxDotDensityBeta(h.xs[tmax], 1024, k, false)) tmax = i;
  }
  check('峰值位置偏差 < 1 个 bin', Math.abs(imax - tmax) <= 1,
    'mc bin=' + imax + ' theo bin=' + tmax);
}

// ---------- 8. 双侧：列样本非负且中位数贴近理论 ----------
console.log('[8] 双侧 max|ρ|');
{
  const s = MC.createSession({ N: 256, KMax: 20, twoSided: true, seed: 5 });
  drain(s.nextBatch(64));
  const vals = MC.columnValues(s.pool, 20);
  check('列样本全部 >= 0', vals.every((v) => v >= 0));
  const agg = MC.aggregateColumn(s.pool, 20);
  const theo = T.maxDotQuantileBeta(0.5, 256, 20, true);
  const rel = Math.abs(agg.median - theo) / theo;
  check('双侧中位数相对偏差 < 10%', rel < 0.1,
    'mc=' + agg.median.toFixed(5) + ' theo=' + theo.toFixed(5));
}

console.log('');
if (failures > 0) {
  console.log(failures + ' 项失败');
  process.exit(1);
} else {
  console.log('全部通过');
}
