/**
 * theory.js 的 node 自检：锚点值（与 Python/numpy 独立实现对照）、
 * 归一化、解析矩、支撑端点。
 * 用法：node web-tools/spectrum-demo/test/theory-selftest.js
 * 无测试框架，直接断言；全部通过时退出码 0。
 */
'use strict';

globalThis.window = globalThis;
require('../js/theory.js');
const T = globalThis.SpecTheory;

let failures = 0;
function check(name, cond, detail) {
  if (cond) {
    console.log('  PASS ' + name);
  } else {
    failures++;
    console.log('  FAIL ' + name + (detail ? ' -- ' + detail : ''));
  }
}
function near(name, got, want, tol) {
  check(name, Math.abs(got - want) <= tol, 'got ' + got + ', want ' + want + ' ± ' + tol);
}

/** 对数网格梯形积分 ∫ f(x) x^k dx（网格对数加密以处理 x→0 奇点） */
function logGridMoment(f, k, lo, hi, n) {
  let acc = 0;
  let xPrev = lo;
  let yPrev = Math.pow(lo, k) * f(lo);
  for (let i = 1; i <= n; i++) {
    const x = lo * Math.pow(hi / lo, i / n);
    const y = Math.pow(x, k) * f(x);
    acc += 0.5 * (y + yPrev) * (x - xPrev);
    xPrev = x;
    yPrev = y;
  }
  return acc;
}

// ---------- 1. 乘积谱密度锚点（Python 独立实现对照值） ----------
console.log('[1] 乘积谱 MP_c ⊠ MP_c 密度锚点');
near('p(1.0, c=1)', T.productDensity(1.0, 1.0), 0.1789791275, 1e-9);
near('p(0.5, c=1)', T.productDensity(0.5, 1.0), 0.3183098862, 1e-9);
near('p(2.0, c=1)', T.productDensity(2.0, 1.0), 0.0938604826, 1e-9);
near('p(1.0, c=0.25)', T.productDensity(1.0, 0.25), 0.4271761681, 1e-9);
near('p(0.8, c=1/16)', T.productDensity(0.8, 1 / 16), 1.0436481248, 1e-9);
near('p(1.5, c=1/16)', T.productDensity(1.5, 1 / 16), 0.4778033556, 1e-9);

// ---------- 1b. AB（H×H）曲线 MP_c ⊠ MP_1 锚点 ----------
console.log('[1b] AB 曲线（÷c 归一 = MP_c ⊠ MP_1）密度锚点');
near('p(1.0, c=1/16, b=1)', T.abNormDensity(1.0, 1 / 16), 0.2694010185, 1e-9);
near('p(2.0, c=1/16, b=1)', T.abNormDensity(2.0, 1 / 16), 0.1500775934, 1e-9);
near('p(4.0, c=1/16, b=1)', T.abNormDensity(4.0, 1 / 16), 0.0316464039, 1e-9);
near('p(1.0, c=0.25, b=1)', T.abNormDensity(1.0, 0.25), 0.2476363445, 1e-9);
near('p(3.0, c=0.25, b=1)', T.abNormDensity(3.0, 0.25), 0.0775819485, 1e-9);
check('c=1 时 AB 曲线退化为 FC_2',
  [0.5, 1.0, 2.0].every(function (x) {
    return Math.abs(T.abNormDensity(x, 1) - T.productDensity(x, 1)) < 1e-12;
  }));

// ---------- 2. 支撑端点 ----------
console.log('[2] 支撑端点');
{
  const s1 = T.productSupport(1.0);
  near('乘积 c=1 左端 = 0', s1[0], 0, 1e-12);
  near('乘积 c=1 右端 = 27/4', s1[1], 6.75, 1e-12);
  const s2 = T.productSupport(0.25);
  near('乘积 c=0.25 左端', s2[0], 0.1361674427, 1e-9);
  near('乘积 c=0.25 右端', s2[1], 3.0982075573, 1e-9);
  const s3 = T.productSupport(1 / 16);
  near('乘积 c=1/16 左端', s3[0], 0.4403523666, 1e-9);
  near('乘积 c=1/16 右端', s3[1], 1.8711710709, 1e-9);
  const m1 = T.mpSupport(1.0);
  near('MP c=1 支撑 [0,4]', m1[0] + m1[1], 4, 1e-12);
  const m2 = T.mpSupport(0.25);
  near('MP c=0.25 左端 = 0.25', m2[0], 0.25, 1e-12);
  near('MP c=0.25 右端 = 2.25', m2[1], 2.25, 1e-12);
  const a1 = T.abNormSupport(1.0);
  near('AB(÷c) c=1 右端 = 27/4（与 FC_2 一致）', a1[1], 6.75, 1e-12);
  const a2 = T.abNormSupport(0.25);
  near('AB(÷c) c=0.25 右端', a2[1], 4.8480762114, 1e-9);
  const a3 = T.abNormSupport(1 / 16);
  near('AB(÷c) c=1/16 右端', a3[1], 4.2367346142, 1e-9);
  near('未归一 σ²(AB) 方差 c²(1+c)', T.abVariance(1 / 16), 0.004150390625, 1e-12);
}

// ---------- 3. 支撑外密度为 0、支撑内为正 ----------
console.log('[3] 支撑内外行为');
{
  const se = T.productSupport(0.25);
  check('乘积 c=0.25：右端外为 0', T.productDensity(se[1] * 1.001, 0.25) === 0);
  check('乘积 c=0.25：左端外为 0', T.productDensity(se[0] * 0.99, 0.25) === 0);
  check('乘积 c=0.25：支撑内为正', T.productDensity((se[0] + se[1]) / 2, 0.25) > 0);
  check('乘积 c=1：27/4 之外为 0', T.productDensity(6.751, 1.0) === 0);
  const me = T.mpSupport(0.25);
  check('MP c=0.25：右端外为 0', T.mpDensity(me[1] * 1.001, 0.25) === 0);
  check('MP c=0.25：左端外为 0', T.mpDensity(me[0] * 0.99, 0.25) === 0);
}

// ---------- 4. 归一化与解析矩（对数网格数值积分） ----------
console.log('[4] 归一化与矩');
{
  const N = 200000;
  for (const c of [1.0, 0.25, 1 / 16]) {
    const se = T.productSupport(c);
    const lo = Math.max(1e-8, se[0] * 0.999);
    const f = function (x) { return T.productDensity(x, c); };
    near('乘积 c=' + c + ' 归一化', logGridMoment(f, 0, lo, se[1], N), 1, 3e-3);
    near('乘积 c=' + c + ' 均值 = 1', logGridMoment(f, 1, lo, se[1], N), 1, 1e-3);
    near('乘积 c=' + c + ' 二阶矩 = 1+2c', logGridMoment(f, 2, lo, se[1], N), 1 + 2 * c, 1e-3);
  }
  for (const c of [1.0, 0.25]) {
    const me = T.mpSupport(c);
    const f = function (x) { return T.mpDensity(x, c); };
    near('MP c=' + c + ' 归一化', logGridMoment(f, 0, 1e-10, me[1], N), 1, 1e-4);
    near('MP c=' + c + ' 二阶矩 = 1+c', logGridMoment(f, 2, 1e-10, me[1], N), 1 + c, 1e-4);
  }
  // AB 曲线（MP_c ⊠ MP_1）：均值 1，方差 1+c
  for (const c of [0.25, 1 / 16]) {
    const se = T.abNormSupport(c);
    const f = function (x) { return T.abNormDensity(x, c); };
    near('AB(÷c) c=' + c + ' 归一化', logGridMoment(f, 0, 1e-6, se[1] * 0.99999, N), 1, 3e-3);
    near('AB(÷c) c=' + c + ' 均值 = 1', logGridMoment(f, 1, 1e-6, se[1] * 0.99999, N), 1, 1e-3);
    near('AB(÷c) c=' + c + ' 方差 = 1+c',
      logGridMoment(f, 2, 1e-6, se[1] * 0.99999, N) - 1, 1 + c, 1e-3);
  }
}

// ---------- 5. σ 轴变量替换（c=1 时 MP 的 σ 密度为 quarter-circle） ----------
console.log('[5] σ 轴密度');
{
  // c=1 的 MP：p_σ(s) = √(4-s²)/π（s ∈ [0,2]），s=1 处 = √3/π
  near('quarter-circle p_σ(1)', T.sigmaDensity(1, 1, 'mp'), Math.sqrt(3) / Math.PI, 1e-12);
  check('σ 轴与 σ² 轴概率一致',
    Math.abs(
      T.sigmaDensity(1.2, 0.25, 'product') - 2.4 * T.productDensity(1.44, 0.25)
    ) < 1e-12);
}

if (failures > 0) {
  console.log('\n' + failures + ' 项失败');
  process.exit(1);
}
console.log('\n全部通过');
