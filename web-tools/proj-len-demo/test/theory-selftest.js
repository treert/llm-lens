/**
 * theory.js 的 node 自检：ψ/ψ′ 锚点、任意阶 Bessel K_ν（对照 K₀/K₁ 系数版与
 * 半整数闭式）、伽马与乘积伽马密度的特例恒等、归一化与解析矩（对数网格数值
 * 积分）、链矩公式、quenched 球面矩与实例中心跳动、MT 伽马采样的蒙特卡洛矩。
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
  near("ψ'(1) = π²/6", T.trigamma(1), Math.PI * Math.PI / 6, 1e-8);
  near("ψ'(1/2) = π²/2", T.trigamma(0.5), Math.PI * Math.PI / 2, 1e-7);
}

// ---------- 2. 任意阶 Bessel K_ν ----------
console.log('[2] logBessKnu：整数阶对照系数版、半整数闭式、递推一致性');
{
  // 整数阶：logBessKnu(·,0/1) 对照 bessk0/bessk1（NR 系数版精度 ~1e-7，容差放宽至此量级）
  for (const x of [0.01, 0.5, 1, 2, 5, 20, 100]) {
    nearRel('logK₀ 对照 x=' + x, T.logBessKnu(x, 0), Math.log(T.bessk0(x)), 1e-6);
    nearRel('logK₁ 对照 x=' + x, T.logBessKnu(x, 1), Math.log(T.bessk1(x)), 1e-6);
  }
  // 半整数闭式：K_{1/2}(x) = √(π/(2x)) e^{−x}
  for (const x of [0.1, 1, 3, 10]) {
    nearRel('K_{1/2} 闭式 x=' + x, T.logBessKnu(x, 0.5),
      0.5 * Math.log(Math.PI / (2 * x)) - x, 1e-8);
  }
  // K_{3/2}(x) = K_{1/2}(x)(1 + 1/x)（递推 + 闭式）
  for (const x of [0.2, 1, 4]) {
    nearRel('K_{3/2} 闭式 x=' + x, T.logBessKnu(x, 1.5),
      0.5 * Math.log(Math.PI / (2 * x)) - x + Math.log(1 + 1 / x), 1e-8);
  }
  // 递推一致性：K_{7/2} 从 μ=1/2 起推，对照 K_{5/2} = K_{1/2}(1+3/x+3/x²) 再递推
  {
    const x = 2.5;
    const k12 = Math.exp(T.logBessKnu(x, 0.5));
    const k32 = k12 * (1 + 1 / x);
    const k52 = k12 * (1 + 3 / x + 3 / (x * x));
    const k72 = k32 + (2 * (5 / 2) / x) * k52;
    nearRel('K_{7/2} 递推一致性', T.logBessKnu(x, 3.5), Math.log(k72), 1e-8);
  }
  // 大阶数不溢出且与渐近一致：K_ν(x) ~ (1/2)Γ(ν)(2/x)^ν（x 小、ν 大）
  {
    const x = 0.05, nu = 40;
    const asym = -Math.LN2 + T.logGamma(nu) + nu * Math.log(2 / x);
    nearRel('K_40(0.05) 渐近', T.logBessKnu(x, nu), asym, 2e-3);
  }
}

// ---------- 3. 伽马密度（中间层 z=Bx）特例 ----------
console.log('[3] 伽马密度（中间层）');
{
  const theta = 2 / 256;
  for (const s of [0.001, 0.1, 1]) {
    nearRel('M=2 即指数分布, s=' + s, T.gammaDensity(s, 1, theta),
      Math.exp(-s / theta) / theta, 1e-12);
  }
  check('M=1 时 f(0) 发散', T.gammaDensity(0, 0.5, theta) === Infinity);
  for (const k of [0.5, 1, 2, 8, 32]) {
    const th = 2 / 256;
    const hi = (k + 1) * th * 60;
    near('伽马归一化 k=' + k, logGridMoment(function (s) { return T.gammaDensity(s, k, th); }, 0, 1e-16, hi, 100000), 1, 1e-6);
    near('伽马均值 k=' + k, logGridMoment(function (s) { return T.gammaDensity(s, k, th); }, 1, 1e-16, hi, 100000), k * th, 1e-6 * k * th + 1e-12);
  }
}

// ---------- 4. 乘积伽马密度（单块 AB，k1 ≠ k2） ----------
console.log('[4] 乘积伽马密度（单块 AB）');
{
  const D = 256, M = 64;
  // k1 = k2 时退化为 K₀ 等形状公式
  {
    const k = 32, t1 = 2 / D, t2 = 2 / (2 * k);
    for (const s of [0.01, 0.5, 2]) {
      const z = 2 * Math.sqrt(s / (t1 * t2));
      nearRel('k1=k2 退化 K₀ 形式, s=' + s, T.prodGammaDensityAB(s, k, t1, k, t2),
        2 * Math.pow(s, k - 1) * T.bessk0(z) /
        (Math.exp(2 * T.logGamma(k)) * Math.pow(t1 * t2, k)), 1e-6);
    }
  }
  // k1 ≠ k2：归一化与解析矩 E[P^m] = [Γ(k1+m)Γ(k2+m)/(Γ(k1)Γ(k2))]·(θ1θ2)^m
  for (const pr of [[M / 2, 2 / D, D / 2, 2 / M], [1 / 2, 2 / D, D / 2, 2], [3, 0.1, 17, 0.02]]) {
    const k1 = pr[0], t1 = pr[1], k2 = pr[2], t2 = pr[3];
    const f = function (s) { return T.prodGammaDensityAB(s, k1, t1, k2, t2); };
    const hi = (k1 + 1) * (k2 + 1) * t1 * t2 * 100;
    const lo = 1e-14 * k1 * k2 * t1 * t2 + 1e-300;
    const momT = function (m) {
      return Math.exp(T.logGamma(k1 + m) + T.logGamma(k2 + m) -
        T.logGamma(k1) - T.logGamma(k2)) * Math.pow(t1 * t2, m);
    };
    near('乘积归一化 k1=' + k1 + ',k2=' + k2, logGridMoment(f, 0, lo, hi, 200000), 1, 1e-4);
    near('乘积均值 k1=' + k1 + ',k2=' + k2, logGridMoment(f, 1, lo, hi, 200000), momT(1), 5e-4 * momT(1));
    near('乘积二阶矩 k1=' + k1 + ',k2=' + k2, logGridMoment(f, 2, lo, hi, 200000), momT(2), 1e-3 * momT(2));
  }
  // 单块 AB 的物理特例：M=64, D=256 时均值恰为 1（配对 fan-in 保长度）
  near('单块 AB 均值 = 1', (M / 2) * (2 / D) * (D / 2) * (2 / M), 1, 1e-15);
}

// ---------- 5. 链矩与对数域参数 ----------
console.log('[5] 链矩（完整链 y=(AB)^L x 与中间层 z=Bx）');
{
  const M = 64, D = 256;
  near('完整链均值恒 1', T.abMean(), 1, 1e-15);
  near('L=1 方差 = (1+2/M)(1+2/D)−1', T.abVar(M, D, 1),
    (1 + 2 / M) * (1 + 2 / D) - 1, 1e-15);
  near('L=8 方差', T.abVar(M, D, 8),
    Math.pow((1 + 2 / M) * (1 + 2 / D), 8) - 1, 1e-15);
  // L=1 时 μ_ln = E[ln χ²_M + ln χ²_D − ln(DM)]
  near('L=1 对数均值', T.abLogMean(M, D, 1),
    Math.LN2 + T.digamma(M / 2) - Math.log(D * M / 2) + T.digamma(D / 2), 1e-12);
  near('L=1 对数方差 = ψ′(M/2)+ψ′(D/2)', T.abLogVar(M, D, 1),
    T.trigamma(M / 2) + T.trigamma(D / 2), 1e-12);
  // 大维度渐近：ψ(z)−ln z ≈ −1/(2z) ⇒ 中位数 ≈ e^{−L(1/M+1/D)}
  const med = T.abMedianApprox(M, D, 8);
  near('中位数近似 ≈ e^{−L(1/M+1/D)}', med,
    Math.exp(-8 * (1 / M + 1 / D)), 8 * (1 / (M * M) + 1 / (D * D)));
  // 中间层（varB = 1/D 配对下投；varB = 1/M 均值归一）
  near('中间层均值 = M/D（varB=1/D）', T.midMean(M, 1 / D), 0.25, 1e-15);
  near('中间层均值 = 1（varB=1/M）', T.midMean(M, 1 / M), 1, 1e-15);
  near('中间层方差 = 2M/D²（varB=1/D）', T.midVar(M, 1 / D), 2 * M / (D * D), 1e-15);
  near('中间层方差 = 2/M（varB=1/M）', T.midVar(M, 1 / M), 2 / M, 1e-15);
  near('中间层对数均值（varB=1/D）', T.midLogMean(M, 1 / D),
    Math.LN2 + T.digamma(M / 2) - Math.log(D), 1e-12);
  near('中间层对数均值（varB=1/M）', T.midLogMean(M, 1 / M),
    Math.LN2 + T.digamma(M / 2) - Math.log(M), 1e-12);
  near('中间层对数方差 = ψ′(M/2)（与 varB 无关）', T.midLogVar(M), T.trigamma(M / 2), 1e-12);
}

// ---------- 6. quenched 球面二次型矩与实例中心跳动 ----------
console.log('[6] quenched 实例矩');
{
  near('quenchMean = trW/D', T.quenchMean(72, 256), 72 / 256, 1e-15);
  const v = T.quenchVar(72, 320, 256);
  near('quenchVar 精确值', v, 2 * (320 - 72 * 72 / 256) / (256 * 258), 1e-15);
  // 中心跳动：中间层 √(2M·varB²/D)；M=D 时单块 AB ≈ √(6)/D（与旧方阵链 L=2 公式一致）
  near('中间层中心跳动（varB=1/D）', T.quenchCenterSdMid(64, 256, 1 / 256),
    Math.sqrt(2 * 64 / Math.pow(256, 3)), 1e-15);
  near('中间层中心跳动（varB=1/M）', T.quenchCenterSdMid(64, 256, 1 / 64),
    Math.sqrt(2 / (64 * 256)), 1e-15);
  nearRel('单块 AB 中心跳动（M=D）≈ √6/D', T.quenchCenterSdAB1(256, 256),
    Math.sqrt(6) / 256, 1e-3);
}

// ---------- 7. 对数正态 ----------
console.log('[7] 近似密度');
{
  const mu = T.abLogMean(64, 256, 8);
  const vv = T.abLogVar(64, 256, 8);
  const f = function (s) { return T.lognormalDensity(s, mu, vv); };
  near('对数正态归一化', logGridMoment(f, 0, 1e-8, Math.exp(mu + 8 * Math.sqrt(vv)), 100000), 1, 1e-6);
  near('对数正态均值 = e^(μ+σ²/2)', logGridMoment(f, 1, 1e-8, Math.exp(mu + 8 * Math.sqrt(vv)), 100000),
    Math.exp(mu + vv / 2), 1e-5);
}

// ---------- 8. Marsaglia–Tsang 伽马/卡方采样（蒙特卡洛矩） ----------
console.log('[8] MT 伽马采样（种子化蒙特卡洛）');
{
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  const rand01 = mulberry32(12345);
  let spare = null;
  const gauss = function () {
    if (spare !== null) { const v = spare; spare = null; return v; }
    let u = 0, v = 0, s = 0;
    do { u = rand01() * 2 - 1; v = rand01() * 2 - 1; s = u * u + v * v; } while (s >= 1 || s === 0);
    const m = Math.sqrt((-2 * Math.log(s)) / s);
    spare = v * m;
    return u * m;
  };
  // Gamma(k,1)：均值 k、方差 k、E ln = ψ(k)；χ²_ν：均值 ν、方差 2ν
  const N = 200000;
  for (const k of [0.3, 1, 2.5, 32]) {
    let m1 = 0, m2 = 0, ml = 0;
    for (let i = 0; i < N; i++) {
      const g = T.gammaSampleMT(k, rand01, gauss);
      m1 += g; m2 += g * g; ml += Math.log(g);
    }
    m1 /= N; m2 /= N; ml /= N;
    const sd = Math.sqrt(m2 - m1 * m1);
    nearRel('Gamma(' + k + ') 均值', m1, k, 4 / Math.sqrt(N));
    nearRel('Gamma(' + k + ') 标准差', sd, Math.sqrt(k), 6 / Math.sqrt(N));
    near('Gamma(' + k + ') E ln = ψ(k)', ml, T.digamma(k), 4 / Math.sqrt(N * Math.max(k, 0.3)));
  }
  {
    let m1 = 0, m2 = 0;
    for (let i = 0; i < N; i++) {
      const c = T.chi2Sample(64, rand01, gauss);
      m1 += c; m2 += c * c;
    }
    m1 /= N; m2 /= N;
    nearRel('χ²_64 均值', m1, 64, 4 / Math.sqrt(N));
    nearRel('χ²_64 方差', m2 - m1 * m1, 128, 6 / Math.sqrt(N));
  }
}

if (failures > 0) {
  console.log('\n' + failures + ' 项失败');
  process.exit(1);
}
console.log('\n全部通过');
