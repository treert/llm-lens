/**
 * 噪音向量基本运算的蒙特卡洛采样层。
 *
 * - mulberry32 + Box–Muller 高斯 RNG（可设种子，结果可复现）；
 * - 按元素运算：采 N 对分量样本 x、y，再按运算映射，五种运算共享同一批样本；
 * - 求和类运算：逐样本采 D 维向量做乘加（不取巧直接采理论分布，保持"模拟"语义）；
 * - 直方图分箱与样本矩估计。
 *
 * 无 DOM 依赖，node 可直接 require 跑 test/theory-selftest.js。
 */
(function (global) {
  'use strict';

  /** mulberry32：32 位可设种子的轻量 PRNG */
  function mulberry32(seed) {
    var a = seed >>> 0;
    return function () {
      a |= 0;
      a = (a + 0x6d2b79f5) | 0;
      var t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  /** Box–Muller 高斯 RNG（标准正态），缓存 sin 支路下次复用 */
  function makeGaussian(rand) {
    var spare = null;
    return function () {
      if (spare !== null) {
        var s = spare;
        spare = null;
        return s;
      }
      var u;
      do {
        u = rand();
      } while (u <= 1e-300);
      var v = 2 * Math.PI * rand();
      var r = Math.sqrt(-2 * Math.log(u));
      spare = r * Math.sin(v);
      return r * Math.cos(v);
    };
  }

  /** 由种子生成标准正态采样函数；再包一层得到 N(0, σ²) 分量 */
  function makeRng(seed) {
    return makeGaussian(mulberry32(seed));
  }

  /**
   * 采 N 对分量样本（按元素运算共用）。
   * 返回 { x, y }：两个 Float64Array(N)，分量 ~ N(0, σ²)。
   */
  function samplePairs(gauss, N, sigma) {
    var x = new Float64Array(N);
    var y = new Float64Array(N);
    for (var i = 0; i < N; i++) {
      x[i] = sigma * gauss();
      y[i] = sigma * gauss();
    }
    return { x: x, y: y };
  }

  /** 把样本对按选中的按元素运算映射成输出样本 */
  function applyElementOp(opId, x, y, c) {
    var N = x.length;
    var out = new Float64Array(N);
    var i;
    if (opId === 'x') {
      return x.slice();
    } else if (opId === 'add') {
      for (i = 0; i < N; i++) out[i] = x[i] + y[i];
    } else if (opId === 'scale') {
      for (i = 0; i < N; i++) out[i] = c * x[i];
    } else if (opId === 'product') {
      for (i = 0; i < N; i++) out[i] = x[i] * y[i];
    } else if (opId === 'square') {
      for (i = 0; i < N; i++) out[i] = x[i] * x[i];
    } else {
      throw new Error('未知按元素运算: ' + opId);
    }
    return out;
  }

  /** 生成固定的 ±1 向量（"一方固定"模式用；‖v‖² = D 恒成立） */
  function makeFixedVector(gauss, D) {
    var v = new Int8Array(D);
    for (var i = 0; i < D; i++) {
      v[i] = gauss() >= 0 ? 1 : -1;
    }
    return v;
  }

  /**
   * 求和类运算采样，返回 M 个输出样本。
   *   dotRandom：z = Σ x_i y_i（双方均重新采样）
   *   dotFixed ：z = Σ v_i x_i（v 为事先生成的固定 ±1 向量）
   *   norm2    ：z = Σ x_i²
   */
  function sampleSum(gauss, modeId, M, D, sigma, fixedVec) {
    var out = new Float64Array(M);
    for (var m = 0; m < M; m++) {
      var acc = 0;
      var i, a;
      if (modeId === 'dotRandom') {
        for (i = 0; i < D; i++) {
          acc += sigma * gauss() * (sigma * gauss());
        }
      } else if (modeId === 'dotFixed') {
        for (i = 0; i < D; i++) {
          acc += fixedVec[i] * (sigma * gauss());
        }
      } else if (modeId === 'norm2') {
        for (i = 0; i < D; i++) {
          a = sigma * gauss();
          acc += a * a;
        }
      } else {
        throw new Error('未知求和模式: ' + modeId);
      }
      out[m] = acc;
    }
    return out;
  }

  /**
   * 直方图分箱：把 samples 投到 [lo, hi] 的 nBins 个等宽箱。
   * 返回 { centers, density, under, over }，density = count/(N·binWidth)，
   * 范围外样本计入 under/over 不进 density。
   */
  function histogram(samples, lo, hi, nBins) {
    var counts = new Float64Array(nBins);
    var w = (hi - lo) / nBins;
    var under = 0;
    var over = 0;
    for (var i = 0; i < samples.length; i++) {
      var v = samples[i];
      if (v < lo) {
        under++;
      } else if (v >= hi) {
        over++;
      } else {
        counts[Math.floor((v - lo) / w)]++;
      }
    }
    var centers = new Float64Array(nBins);
    var density = new Float64Array(nBins);
    for (var b = 0; b < nBins; b++) {
      centers[b] = lo + (b + 0.5) * w;
      density[b] = counts[b] / (samples.length * w);
    }
    return { centers: centers, density: density, under: under, over: over };
  }

  /** 样本均值与（无偏）样本方差 */
  function sampleMeanVar(samples) {
    var n = samples.length;
    var mean = 0;
    var i;
    for (i = 0; i < n; i++) mean += samples[i];
    mean /= n;
    var v = 0;
    for (i = 0; i < n; i++) {
      var d = samples[i] - mean;
      v += d * d;
    }
    v /= n - 1;
    return { mean: mean, variance: v };
  }

  var NoiseSampler = {
    makeRng: makeRng,
    samplePairs: samplePairs,
    applyElementOp: applyElementOp,
    makeFixedVector: makeFixedVector,
    sampleSum: sampleSum,
    histogram: histogram,
    sampleMeanVar: sampleMeanVar,
  };

  global.NoiseSampler = NoiseSampler;
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = NoiseSampler;
  }
})(typeof window !== 'undefined' ? window : globalThis);
