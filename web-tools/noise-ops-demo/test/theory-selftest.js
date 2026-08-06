/**
 * theory.js / sampler.js 的 node 自检。
 * 用法：node web-tools/noise-ops-demo/test/theory-selftest.js
 * 无测试框架，直接断言；全部通过时退出码 0。
 */
'use strict';

const T = require('../js/theory.js');
const S = require('../js/sampler.js');

let failures = 0;
function check(name, cond, detail) {
  if (cond) {
    console.log('  PASS ' + name);
  } else {
    failures++;
    console.log('  FAIL ' + name + (detail ? ' -- ' + detail : ''));
  }
}

/** 梯形数值积分 */
function integrate(f, lo, hi, n) {
  const h = (hi - lo) / n;
  let s = 0.5 * (f(lo) + f(hi));
  for (let i = 1; i < n; i++) s += f(lo + i * h);
  return s * h;
}

function relErr(a, b) {
  return Math.abs(a - b) / Math.max(Math.abs(b), 1e-300);
}

// ---------- 1. log 域 Bessel K_ν 精确值 ----------
console.log('[1] logBesselK 精确值与递推');
{
  // 参考值：A&S 表 / 任意精度库
  const K0_1 = 0.4210244382407083;
  const K1_1 = 0.6019072301972346;
  const K2_1 = K0_1 + 2 * K1_1; // 递推关系 K2 = K0 + (2/x)K1，x=1
  check('K0(1)', relErr(Math.exp(T.logBesselK(0, 1)), K0_1) < 1e-6);
  check('K1(1)', relErr(Math.exp(T.logBesselK(1, 1)), K1_1) < 1e-6);
  check('K2(1)（整数阶递推）', relErr(Math.exp(T.logBesselK(2, 1)), K2_1) < 1e-5);
  // 半整数阶初等式：K_{1/2}(x) = √(π/(2x)) e^{-x}
  const x = 2;
  const Khalf = Math.sqrt(Math.PI / (2 * x)) * Math.exp(-x);
  check('K_{1/2}(2)（初等闭式）', relErr(Math.exp(T.logBesselK(0.5, x)), Khalf) < 1e-12);
  check(
    'K_{3/2}(2) = K_{1/2}(2)·(1+1/x)',
    relErr(Math.exp(T.logBesselK(1.5, x)), Khalf * (1 + 1 / x)) < 1e-12
  );
  // 一般阶递推：K_{ν+1} - K_{ν-1} = (2ν/x) K_ν，ν=7/2, x=2.3
  const xx = 2.3;
  const Km = Math.exp(T.logBesselK(2.5, xx));
  const K0v = Math.exp(T.logBesselK(3.5, xx));
  const Kp = Math.exp(T.logBesselK(4.5, xx));
  check(
    '递推 K_{ν+1} − K_{ν−1} = (2ν/x)K_ν（ν=7/2）',
    relErr(Kp - Km, ((2 * 3.5) / xx) * K0v) < 1e-9,
    'lhs=' + (Kp - Km) + ' rhs=' + ((2 * 3.5) / xx) * K0v
  );
  // 大 x 不下溢且符合渐近 K0(x) ~ √(π/(2x))·e^{-x}（e^{-700} 已远低于 float64 下限，log 域仍精确）
  const asymptotic = 0.5 * Math.log(Math.PI / (2 * 700)) - 700;
  check(
    'logK0(700) 有限且符合渐近式',
    Math.abs(T.logBesselK(0, 700) - asymptotic) < 1e-3,
    'got ' + T.logBesselK(0, 700) + ' expect≈' + asymptotic
  );
}

// ---------- 2. 各密度数值归一化 ----------
console.log('[2] 密度函数数值积分 ≈ 1');
{
  const n = 20000;
  check('normalPDF N(0,1)', Math.abs(integrate((z) => T.normalPDF(z, 0, 1), -8, 8, n) - 1) < 1e-6);
  // 乘积正态在 z=0 对数奇异：z=u² 换元后被积函数 ~ u·ln u（连续、可积），梯形法快收敛；
  // 从 u=1e-12 起积避开 z=0 的 Infinity 端点（该处被积函数极限为 0，损失可忽略）
  const intProduct = (sigma, B, m) =>
    2 * integrate((u) => T.productPDF(u * u, sigma) * 2 * u, 1e-12, Math.sqrt(B), m);
  check('productPDF σ=1（z=u² 换元）', Math.abs(intProduct(1, 12, 200000) - 1) < 1e-4);
  check('productPDF σ=0.7（z=u² 换元）', Math.abs(intProduct(0.7, 8, 200000) - 1) < 1e-4);
  // z = t² 换元消掉 z^{-1/2} 奇异；从 t=1e-9 起积：squarePDF(0)=0 是硬截断，
  // 而换元被积函数在 t→0+ 的极限为 2/(σ√2π)，从 0 起积会损失端点区间
  check(
    'squarePDF σ=1（z=t² 换元）',
    Math.abs(integrate((t) => T.squarePDF(t * t, 1) * 2 * t, 1e-9, 6, n) - 1) < 1e-6
  );
  // D=2 时点积密度应精确等于 Laplace(0, σ²)：直接比点值
  const s2 = 0.64; // σ=0.8
  const zTest = [0.05, 0.5, 1.3, 3.0];
  const laplaceOK = zTest.every(
    (z) =>
      relErr(T.dotRandomPDF(z, 2, 0.8), Math.exp(-Math.abs(z) / s2) / (2 * s2)) < 1e-8
  );
  check('dotRandomPDF D=2 恰为 Laplace(0, σ²)', laplaceOK);
  check(
    'dotRandomPDF D=5 σ=0.8',
    Math.abs(integrate((z) => T.dotRandomPDF(z, 5, 0.8), -15, 15, n * 2) - 1) < 5e-3
  );
  check(
    'dotFixedPDF D=16 σ=0.25（即 N(0,1)）',
    Math.abs(integrate((z) => T.dotFixedPDF(z, 16, 0.25), -8, 8, n) - 1) < 1e-6
  );
  check(
    'norm2PDF D=8 σ=0.5',
    Math.abs(integrate((z) => T.norm2PDF(z, 8, 0.5), 1e-9, 14, n) - 1) < 2e-3
  );
}

// ---------- 3. 理论矩 vs 蒙特卡洛 ----------
console.log('[3] 理论均值/方差与采样一致（固定种子）');
{
  const gauss = S.makeRng(20260806);
  const sigma = 0.5;
  const N = 400000;
  const pairs = S.samplePairs(gauss, N, sigma);

  const add = S.sampleMeanVar(S.applyElementOp('add', pairs.x, pairs.y));
  check('add：方差 ≈ 2σ²', relErr(add.variance, 2 * sigma * sigma) < 0.02, 'got ' + add.variance);

  const prod = S.sampleMeanVar(S.applyElementOp('product', pairs.x, pairs.y));
  check('product：均值 ≈ 0', Math.abs(prod.mean) < 0.01, 'got ' + prod.mean);
  check(
    'product：方差 ≈ σ⁴（方差的乘积）',
    relErr(prod.variance, Math.pow(sigma, 4)) < 0.03,
    'got ' + prod.variance
  );

  const sq = S.sampleMeanVar(S.applyElementOp('square', pairs.x, pairs.y));
  check('square：均值 ≈ σ²（漂离 0）', relErr(sq.mean, sigma * sigma) < 0.02, 'got ' + sq.mean);
  check('square：方差 ≈ 2σ⁴', relErr(sq.variance, 2 * Math.pow(sigma, 4)) < 0.05, 'got ' + sq.variance);

  const D = 32;
  const M = 20000;
  const dr = S.sampleMeanVar(S.sampleSum(S.makeRng(1), 'dotRandom', M, D, sigma, null));
  check(
    'dotRandom D=32：方差 ≈ D·σ⁴（=2，σ²=1/D 时将是 1/D）',
    relErr(dr.variance, D * Math.pow(sigma, 4)) < 0.05,
    'got ' + dr.variance
  );

  const fv = S.makeFixedVector(S.makeRng(2), D);
  const df = S.sampleMeanVar(S.sampleSum(S.makeRng(3), 'dotFixed', M, D, sigma, fv));
  check(
    'dotFixed D=32：方差 ≈ D·σ²（=8）',
    relErr(df.variance, D * sigma * sigma) < 0.05,
    'got ' + df.variance
  );

  const n2 = S.sampleMeanVar(S.sampleSum(S.makeRng(4), 'norm2', M, D, sigma, null));
  check('norm2 D=32：均值 ≈ D·σ²（=8）', relErr(n2.mean, D * sigma * sigma) < 0.03, 'got ' + n2.mean);
  check(
    'norm2 D=32：方差 ≈ 2D·σ⁴（=4）',
    relErr(n2.variance, 2 * D * Math.pow(sigma, 4)) < 0.08,
    'got ' + n2.variance
  );

  // 投影点积（attention 分数）：条件于固定 W 的精确方差 = σ⁴·‖W_QᵀW_K‖_F²，
  // 样本方差应贴合该条件值（只含采样误差）；‖M‖_F² 的期望 = H（σw²=1/D），
  // 单个 W 的涨落 ~√2/D（D=32 时 ≈4.4%），后者作弱检查
  const Dp = 32;
  const Hp = 16;
  const gW = S.makeRng(11);
  const wQ = S.makeProjection(gW, Hp, Dp, 1 / Math.sqrt(Dp));
  const wK = S.makeProjection(gW, Hp, Dp, 1 / Math.sqrt(Dp));
  let fro2 = 0;
  for (let a = 0; a < Dp; a++) {
    for (let b = 0; b < Dp; b++) {
      let m = 0;
      for (let i = 0; i < Hp; i++) m += wQ[i * Dp + a] * wK[i * Dp + b];
      fro2 += m * m;
    }
  }
  const pd = S.sampleMeanVar(S.sampleProjDot(S.makeRng(12), 50000, Dp, Hp, sigma, wQ, wK));
  check('projDot：均值 ≈ 0', Math.abs(pd.mean) < 0.05, 'got ' + pd.mean);
  check(
    'projDot：样本方差 = σ⁴‖W_QᵀW_K‖_F²/H（÷√H 后的条件精确式）',
    relErr(pd.variance, (Math.pow(sigma, 4) * fro2) / Hp) < 0.03,
    'got ' + pd.variance + ' expect ' + (Math.pow(sigma, 4) * fro2) / Hp
  );
  check(
    'projDot：‖M‖_F² ≈ H（期望关系，容差覆盖个体涨落）',
    relErr(fro2, Hp) < 0.15,
    'got ' + fro2
  );
}

// ---------- 4. 直方图守恒与边界 ----------
console.log('[4] 直方图与边界行为');
{
  const samples = S.samplePairs(S.makeRng(7), 50000, 1).x;
  const h = S.histogram(samples, -3, 3, 60);
  let total = h.under + h.over;
  for (let i = 0; i < h.density.length; i++) total += h.density[i] * 50000 * (6 / 60);
  check('直方图计数守恒（箱内 + 范围外 = N）', Math.abs(total - 50000) < 1e-6, 'got ' + total);
  check('squarePDF 负半轴为 0', T.squarePDF(-1, 1) === 0);
  check('squarePDF(0) 发散（χ²₁ 的 z^{-1/2} 奇异）', T.squarePDF(0, 1) === Infinity);
  check('norm2PDF z=0：D=1 发散', T.norm2PDF(0, 1, 1) === Infinity);
  check('norm2PDF z=0：D=2 为 1/(2σ²)（指数分布起点）', T.norm2PDF(0, 2, 1) === 0.5);
  check('norm2PDF z=0：D≥3 为 0', T.norm2PDF(0, 3, 1) === 0);
  check('productPDF(0) 发散（对数奇异）', T.productPDF(0, 1) === Infinity);
  check('dotRandomPDF(0)：D=1 发散', T.dotRandomPDF(0, 1, 1) === Infinity);
  check(
    'dotRandomPDF(0)：D=2 为 1/(2a)（Laplace 起点）',
    relErr(T.dotRandomPDF(0, 2, 1), 0.5) < 1e-12,
    'got ' + T.dotRandomPDF(0, 2, 1)
  );
  check(
    'dotRandomPDF(0)：D=3 为 1/π（极限公式 Γ(ν)/(2√π Γ(p) a)）',
    relErr(T.dotRandomPDF(0, 3, 1), 1 / Math.PI) < 1e-12,
    'got ' + T.dotRandomPDF(0, 3, 1)
  );
}

console.log('');
if (failures > 0) {
  console.log('FAILED: ' + failures + ' 项未通过');
  process.exit(1);
} else {
  console.log('全部通过');
}
