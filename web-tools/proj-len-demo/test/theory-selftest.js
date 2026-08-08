/**
 * theory.js 的 node 自检：ψ/ψ′ 锚点、伽马与乘积伽马密度的特例恒等、
 * 归一化与解析矩（对数网格数值积分）、链矩公式、quenched 球面矩。
 * 用法：node web-tools/proj-len-demo/test/theory-selftest.js
 * 无测试框架，直接断言；全部通过时退出码 0。
 */
'use strict';

globalThis.window = globalThis;
require('../js/theory.js');
const T = globalThis.ProjLenTheory;

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
function nearRel(name, got, want, tolRel) {
  const rel = Math.abs(got - want) / Math.max(Math.abs(want), 1e-300);
  check(name, rel <= tolRel, 'got ' + got + ', want ' + want + ' ± ' + (tolRel * 100) + '%');
}

/** 对数网格梯形积分 ∫ s^k f(s) ds（s > 0，网格对数加密处理 s→0 行为） */
function logGridMoment(f, k, lo, hi, n) {
  let acc = 0;
  let sPrev = lo;
  let yPrev = Math.pow(lo, k) * f(lo);
  for (let i = 1; i <= n; i++) {
    const s = lo * Math.pow(hi / lo, i / n);
    const y = Math.pow(s, k) * f(s);
    acc += 0.5 * (y + yPrev) * (s - sPrev);
    sPrev = s;
    yPrev = y;
  }
  return acc;
}

// ---------- 1. digamma / trigamma 锚点 ----------
console.log('[1] digamma / trigamma 锚点');
{
  const GAMMA = 0.5772156649015329;
  near('ψ(1) = −γ', T.digamma(1), -GAMMA, 1e-10);
  near('ψ(2) = 1−γ', T.digamma(2), 1 - GAMMA, 1e-10);
  near('ψ(1/2) = −γ−2ln2', T.digamma(0.5), -GAMMA - 2 * Math.LN2, 1e-10);
  near('ψ(32)（递推一致性）', T.digamma(32),
    -GAMMA + Array.from({ length: 31 }, function (_, i) { return 1 / (i + 1); })
      .reduce(function (a, b) { return a + b; }, 0), 1e-9);
  near("ψ'(1) = π²/6", T.trigamma(1), Math.PI * Math.PI / 6, 1e-8);
  near("ψ'(1/2) = π²/2", T.trigamma(0.5), Math.PI * Math.PI / 2, 1e-7);
  near("ψ'(2) = π²/6 − 1", T.trigamma(2), Math.PI * Math.PI / 6 - 1, 1e-9);
  // 渐近行为：ψ(z) ≈ ln z（大 z）
  near('ψ(1000) ≈ ln1000 − 1/2000', T.digamma(1000),
    Math.log(1000) - 1 / 2000 - 1 / (12e6), 1e-10);
}

// ---------- 2. 伽马密度（L=1）特例 ----------
console.log('[2] 伽马密度（L = 1）');
{
  // H = 2：χ²₂/D = Gamma(1, 2/D) = 指数分布，f(s) = (D/2)e^(−sD/2)
  const theta = 2 / 256;
  for (const s of [0.001, 0.1, 1]) {
    nearRel('H=2 即指数分布, s=' + s, T.gammaDensity(s, 1, theta),
      Math.exp(-s / theta) / theta, 1e-12);
  }
  near('H=2 时 f(0) = D/2', T.gammaDensity(0, 1, theta), 1 / theta, 1e-12);
  check('H=1 时 f(0) 发散', T.gammaDensity(0, 0.5, theta) === Infinity);
  check('H≥3 时 f(0) = 0', T.gammaDensity(0, 2, theta) === 0);
  // 归一化与矩：Gamma(k, θ) 均值 kθ、二阶矩 k(k+1)θ²
  for (const k of [0.5, 1, 2, 8, 32]) {
    const th = 2 / 256;
    const hi = (k + 1) * th * 60; // 覆盖右尾
    // lo 取 1e-16：k=0.5 的原点 s^(−1/2) 奇异段 ∫₀^lo ≈ 2√lo/(Γ(½)√θ)，lo 过高会缺质量
    near('伽马归一化 k=' + k, logGridMoment(function (s) { return T.gammaDensity(s, k, th); }, 0, 1e-16, hi, 100000), 1, 1e-6);
    near('伽马均值 k=' + k, logGridMoment(function (s) { return T.gammaDensity(s, k, th); }, 1, 1e-16, hi, 100000), k * th, 1e-6 * k * th + 1e-12);
    near('伽马二阶矩 k=' + k, logGridMoment(function (s) { return T.gammaDensity(s, k, th); }, 2, 1e-16, hi, 100000), k * (k + 1) * th * th, 1e-5 * k * (k + 1) * th * th + 1e-15);
  }
}

// ---------- 3. 乘积伽马密度（L=2） ----------
console.log('[3] 乘积伽马密度（L = 2）');
{
  const D = 256;
  // H = 2（k = 1）：f(s) = 2K₀(2√(s/(θ1θ2)))/(θ1θ2)，θ1 = 2/D，θ2 = 2/H
  const th1 = 2 / D, th2 = 2 / 2;
  for (const s of [0.01, 0.5, 2]) {
    const z = 2 * Math.sqrt(s / (th1 * th2));
    nearRel('H=2 乘积密度 = K₀ 形式, s=' + s, T.prodGammaDensity(s, 1, th1, th2),
      2 * T.bessk0(z) / (th1 * th2), 1e-12);
  }
  check('H=2 时 f(0) 对数发散', T.prodGammaDensity(0, 1, th1, th2) === Infinity);
  check('H≥4 时 f(0) = 0', T.prodGammaDensity(0, 2, th1, th2) === 0);
  // 归一化与矩：E[P^m] = [Γ(k+m)/Γ(k)]² (θ1θ2)^m
  for (const k of [1, 2, 8, 32]) {
    const t1 = 2 / D, t2 = 2 / (2 * k);
    const f = function (s) { return T.prodGammaDensity(s, k, t1, t2); };
    const hi = Math.pow(k + 1, 2) * t1 * t2 * 80;
    const meanT = k * k * t1 * t2;
    const m2T = k * k * (k + 1) * (k + 1) * t1 * t1 * t2 * t2;
    near('乘积归一化 k=' + k, logGridMoment(f, 0, 1e-8, hi, 200000), 1, 1e-4);
    near('乘积均值 k=' + k, logGridMoment(f, 1, 1e-8, hi, 200000), meanT, 5e-4 * meanT);
    near('乘积二阶矩 k=' + k, logGridMoment(f, 2, 1e-8, hi, 200000), m2T, 5e-4 * m2T);
  }
}

// ---------- 4. 链矩与对数域参数 ----------
console.log('[4] L 层链矩');
{
  const H = 64, D = 256;
  near('均值 = c（任意 L）', T.chainMean(H, D), 0.25, 1e-15);
  near('L=1 方差 = 2H/D²', T.chainVar(H, D, 1), 2 * H / (D * D), 1e-15);
  near('L=2 方差 = 4c²(1+1/H)/…', T.chainVar(H, D, 2),
    0.25 * 0.25 * ((1 + 2 / H) * (1 + 2 / H) - 1), 1e-15);
  // L=1 时 μ_ln = E[ln(χ²_H/D)] = ln2 + ψ(H/2) − lnD
  near('L=1 对数均值', T.chainLogMean(H, D, 1),
    Math.LN2 + T.digamma(H / 2) - Math.log(D), 1e-12);
  near('L=1 对数方差 = ψ′(H/2)', T.chainLogVar(H, 1), T.trigamma(H / 2), 1e-12);
  // 大 H 渐近：E ln(χ²_H/H) ≈ −1/H ⇒ 中位数 ≈ c·e^(−L/H)
  const med = T.chainMedianApprox(H, D, 8);
  near('中位数近似 ≈ c·e^(−L/H)（大 H）', med,
    0.25 * Math.exp(-8 / H), 0.25 * 8 / (H * H));
}

// ---------- 5. quenched 球面二次型矩 ----------
console.log('[5] quenched 实例矩');
{
  near('quenchMean = trW/D', T.quenchMean(72, 256), 72 / 256, 1e-15);
  // 精确公式 Var = 2[trW² − trW²/D]/(D(D+2))；大 D 极限 ≈ 2trW²/D²
  const v = T.quenchVar(72, 320, 256);
  near('quenchVar 精确值', v, 2 * (320 - 72 * 72 / 256) / (256 * 258), 1e-15);
  // 大 D 近似 2trW²/D²：构造 trW² ≫ (trW)²/D 的例子（修正项相对量级 1/D 才可忽略）
  const v2 = T.quenchVar(1, 100, 100000);
  near('quenchVar 大 D 近似', v2, 2 * 100 / 1e10, 1e-3 * 2 * 100 / 1e10);
}

// ---------- 6. 对数正态与高斯 ----------
console.log('[6] 近似密度');
{
  const mu = T.chainLogMean(64, 256, 8);
  const vv = T.chainLogVar(64, 8);
  const f = function (s) { return T.lognormalDensity(s, mu, vv); };
  near('对数正态归一化', logGridMoment(f, 0, 1e-6, Math.exp(mu + 8 * Math.sqrt(vv)), 100000), 1, 1e-6);
  near('对数正态均值 = e^(μ+σ²/2)', logGridMoment(f, 1, 1e-6, Math.exp(mu + 8 * Math.sqrt(vv)), 100000),
    Math.exp(mu + vv / 2), 1e-5);
}

if (failures > 0) {
  console.log('\n' + failures + ' 项失败');
  process.exit(1);
}
console.log('\n全部通过');
