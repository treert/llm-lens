/**
 * activation-demo 蒙特卡洛采样层。
 * mulberry32 + Box–Muller 高斯 RNG（可设种子，可复现）、
 * 输入采样、逐元素激活映射、直方图分箱与样本矩。
 * 无 DOM 依赖，node 可直接 require。
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

  /** 由种子生成标准正态采样函数 */
  function makeRng(seed) {
    return makeGaussian(mulberry32(seed));
  }

  /** 采 N 个输入样本，分量 ~ N(0, σ²) */
  function sampleInput(gauss, N, sigma) {
    var x = new Float64Array(N);
    for (var i = 0; i < N; i++) x[i] = sigma * gauss();
    return x;
  }

  /** 逐元素过激活函数 */
  function applyActivation(act, p, xs) {
    var out = new Float64Array(xs.length);
    for (var i = 0; i < xs.length; i++) out[i] = act.fn(xs[i], p);
    return out;
  }

  /**
   * 直方图分箱：把 samples 投到 [lo, hi] 的 nBins 个等宽箱。
   * density = count/(N·binWidth)，范围外样本计入 under/over。
   */
  function histogram(samples, lo, hi, nBins) {
    var counts = new Float64Array(nBins);
    var w = (hi - lo) / nBins;
    var under = 0, over = 0;
    for (var i = 0; i < samples.length; i++) {
      var v = samples[i];
      if (v < lo) under++;
      else if (v >= hi) over++;
      else counts[Math.floor((v - lo) / w)]++;
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
    var n = samples.length, mean = 0, i;
    for (i = 0; i < n; i++) mean += samples[i];
    mean /= n;
    var v = 0;
    for (i = 0; i < n; i++) {
      var d = samples[i] - mean;
      v += d * d;
    }
    return { mean: mean, variance: v / (n - 1) };
  }

  /** 恰好等于 0 的样本数（ReLU 族 y=0 点质量的实测） */
  function countExactZeros(samples) {
    var c = 0;
    for (var i = 0; i < samples.length; i++) if (samples[i] === 0) c++;
    return c;
  }

  var ActSampler = {
    makeRng: makeRng,
    sampleInput: sampleInput,
    applyActivation: applyActivation,
    histogram: histogram,
    sampleMeanVar: sampleMeanVar,
    countExactZeros: countExactZeros,
  };

  global.ActSampler = ActSampler;
  if (typeof module !== 'undefined' && module.exports) module.exports = ActSampler;
})(typeof window !== 'undefined' ? window : globalThis);
