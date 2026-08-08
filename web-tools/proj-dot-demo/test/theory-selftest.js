/**
 * theory.js 的 node 自检：Bessel 锚点值、递推恒等式、VG 密度的特例恒等、
 * 归一化与解析矩（数值积分对照）、大 H 高斯极限。
 * 用法：node web-tools/proj-dot-demo/test/theory-selftest.js
 * 无测试框架，直接断言；全部通过时退出码 0。
 */
'use strict';

globalThis.window = globalThis;
require('../js/theory.js');
const T = globalThis.ProjDotTheory;

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

/** z 域对数网格双侧矩：∫ s^k f(s) ds，s = ±βz，z ∈ [1e-8, 60] */
function vgMoment(H, beta, k, nGrid) {
  const n = nGrid || 200000;
  const lo = 1e-8, hi = 60;
  let acc = 0;
  let zPrev = lo;
  let yPrev = Math.pow(lo, k) * T.vgDensity(beta * lo, H, beta);
  for (let i = 1; i <= n; i++) {
    const z = lo * Math.pow(hi / lo, i / n);
    const y = Math.pow(z, k) * T.vgDensity(beta * z, H, beta);
    acc += 0.5 * (y + yPrev) * (z - zPrev);
    zPrev = z;
    yPrev = y;
  }
  // 偶数阶矩两侧相等；奇数阶严格为 0（密度偶对称），只算一侧乘 2 或 0。
  // 一侧积分 ∫ s^k f(s) ds = β^(k+1) ∫ z^k f(βz) dz（Jacobi 因子 ds = β dz）
  if (k % 2 === 1) return 0;
  return 2 * Math.pow(beta, k + 1) * acc;
}

// ---------- 1. logGamma 与 Bessel 锚点 ----------
console.log('[1] logGamma / Bessel 锚点');
near('logGamma(1) = 0', T.logGamma(1), 0, 1e-12);
near('logGamma(0.5) = ½lnπ', T.logGamma(0.5), 0.5 * Math.log(Math.PI), 1e-12);
near('logGamma(5) = ln24', T.logGamma(5), Math.log(24), 1e-12);
near('logGamma(0.3)', T.logGamma(0.3), 1.0957979948, 1e-9);
near('K_0(1)', T.bessk0(1), 0.4210244382407083, 1e-7);
near('K_1(1)', T.bessk1(1), 0.6019072301972346, 1e-7);
near('K_0(2)', T.bessk0(2), 0.1138938727495334, 1e-7);
near('K_1(2)', T.bessk1(2), 0.1398658818165224, 1e-7);
near('K_0(0.5)', T.bessk0(0.5), 0.9244190712276659, 1e-7);
near('K_1(0.5)', T.bessk1(0.5), 1.6564411200033029, 1e-6);

// ---------- 2. 高阶 Bessel：递推恒等式与半整数闭式 ----------
console.log('[2] 高阶 Bessel K');
{
  // 整数阶递推恒等式 K_{n+1}(z) = K_{n−1}(z) + (2n/z)K_n(z)
  for (const z of [0.3, 1.0, 2.5, 10.0]) {
    for (const n of [1, 2, 5, 9]) {
      const lhs = T.besselK(n + 1, z);
      const rhs = T.besselK(n - 1, z) + ((2 * n) / z) * T.besselK(n, z);
      nearRel('递推 K_' + (n + 1) + '(' + z + ')', lhs, rhs, 1e-9);
    }
  }
  // 半整数闭式 K_{n+1/2}(z) = √(π/(2z)) e^{−z} Σ a_k z^{−k}（a 递推）
  function kHalfClosed(n, z) {
    let sum = 0, a = 1, zk = 1;
    for (let k = 0; k <= n; k++) {
      if (k > 0) {
        a *= ((n + k) * (n - k + 1)) / (2 * k);
        zk *= z;
      }
      sum += a / zk;
    }
    return Math.sqrt(Math.PI / (2 * z)) * Math.exp(-z) * sum;
  }
  for (const z of [0.5, 1.0, 3.0, 12.0]) {
    for (const n of [0, 1, 2, 7]) {
      nearRel('半整数 K_' + n + '.5(' + z + ')', T.besselK(n + 0.5, z), kHalfClosed(n, z), 1e-9);
    }
  }
  // 整数阶绝对锚点（K_2(1) = K_0(1) + 2K_1(1) 的真值）
  near('K_2(1)', T.besselK(2, 1), 1.6248388986351775, 1e-6);
  near('K_2(1.5)', T.besselK(2, 1.5), 0.5836559632586908, 1e-7);
  near('K_3(1.5)', T.besselK(3, 1.5), 1.8338037047620595, 1e-6);
}

// ---------- 3. VG 密度的特例恒等 ----------
console.log('[3] VG 密度特例');
{
  const beta = 1 / 256;
  // H = 2 退化为 Laplace：f(s) = e^{−|s|/β}/(2β)
  for (const s of [0, 0.001, -0.01, 0.05]) {
    nearRel('H=2 即 Laplace, s=' + s, T.vgDensity(s, 2, beta),
      Math.exp(-Math.abs(s) / beta) / (2 * beta), 1e-9);
  }
  // H = 1：f(s) = K_0(|s|/β)/(πβ)
  nearRel('H=1 即 K_0 形式', T.vgDensity(0.01, 1, beta),
    T.bessk0(0.01 / beta) / (Math.PI * beta), 1e-12);
  // 峰值 f(0⁺) = Γ((H−1)/2)/(2β√π Γ(H/2))
  for (const H of [3, 4, 7, 8, 16]) {
    nearRel('f(0) = vgPeak, H=' + H, T.vgDensity(0, H, beta), T.vgPeak(H, beta), 1e-9);
  }
  // 峰值公式独立验算：H=4 时 f(0) = Γ(1.5)/(2β√πΓ(2)) = (√π/2)/(2β√π) = 1/(4β)
  nearRel('H=4 峰值 = 1/(4β)', T.vgPeak(4, beta), 1 / (4 * beta), 1e-12);
}

// ---------- 4. 归一化与解析矩（数值积分） ----------
console.log('[4] VG 归一化与矩');
{
  const beta = 1 / 256;
  for (const H of [1, 2, 3, 5, 8, 16, 64]) {
    near('归一化 H=' + H, vgMoment(H, beta, 0), 1, 1e-4);
    near('二阶矩 = Hβ², H=' + H, vgMoment(H, beta, 2), H * beta * beta, 2e-3 * H * beta * beta);
    near('四阶矩 = 3H(H+2)β⁴, H=' + H, vgMoment(H, beta, 4),
      3 * H * (H + 2) * Math.pow(beta, 4), 5e-3 * 3 * H * (H + 2) * Math.pow(beta, 4));
  }
}

// ---------- 5. 大 H 高斯极限（Edgeworth 一阶偏差 ~ 3/(4H)） ----------
console.log('[5] 大 H 高斯化');
{
  const H = 64, beta = 1 / 256;
  const sd = Math.sqrt(H) * beta;
  for (const xs of [0, 1]) {
    const s = xs * sd;
    nearRel('H=64 vs 高斯, s=' + xs + 'σ', T.vgDensity(s, H, beta),
      T.gaussDensity(s, 0, H * beta * beta), 0.02);
  }
}

// ---------- 6. 方案二矩公式 ----------
console.log('[6] 方案二条件矩');
{
  near('scheme2Mean', T.scheme2Mean(0.8, 5, 256), (0.8 * 5) / 256, 1e-15);
  near('scheme2Var ρ=0', T.scheme2Var(0, 64, 32, 256), 64 / 65536, 1e-15);
  near('scheme2Var ρ=1', T.scheme2Var(1, 64, 32, 256), 64 / 65536, 1e-15);
  near('scheme2Var 一般', T.scheme2Var(0.6, 64, 32, 256),
    (2 * 0.36 * 32 + 0.64 * 64) / 65536, 1e-15);
}

if (failures > 0) {
  console.log('\n' + failures + ' 项失败');
  process.exit(1);
}
console.log('\n全部通过');
