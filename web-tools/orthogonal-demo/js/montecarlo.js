/**
 * 蒙特卡洛流式模拟：高维球面随机向量的点积与最大点乘（对应 monte-carlo-design.md）。
 *
 * 要点：
 * - 逐向量推进，每对点积只算一次；generator 形式，由调用方按时间片驱动；
 * - 分批累积：每批 R 条新轨迹，批末释放向量内存，maxRuns 曲线入池；
 * - 每条轨迹独立 RNG 子流（seed 与轨迹全局序号派生），结果与分批方式无关；
 * - 无 DOM 依赖，node 可直接 require 跑 test/mc-selftest.js。
 */
(function (global) {
  'use strict';

  var TWO_GB = 2 * 1024 * 1024 * 1024;

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

  /** Box–Muller 高斯 RNG，缓存 sin 支路下次复用 */
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

  /**
   * K 上限：min(内存约束, 运算约束, N²)，至少为 2。
   * 顺序逐条轨迹模型：内存约束为单条轨迹的向量（4·K·N 字节），
   * opsBudget 为单条轨迹的乘加预算（K²·N/2 次，决定单条耗时）。
   */
  function computeKMax(N, memBytes, opsBudget) {
    var byMem = Math.floor(memBytes / (4 * N));
    var byOps = Math.floor(Math.sqrt((2 * opsBudget) / N));
    return Math.max(2, Math.min(byMem, byOps, N * N));
  }

  /** 点积槽位直方图：bins 个等宽内槽 + 首尾两个开口槽；每槽存 sum 与 count */
  function createHist(lo, hi, bins) {
    var width = (hi - lo) / bins;
    var h = {
      lo: lo,
      hi: hi,
      bins: bins,
      width: width,
      sum: new Float64Array(bins + 2),
      count: new Float64Array(bins + 2),
    };
    h.add = function (x) {
      var i = Math.floor((x - lo) / width) + 1;
      if (i < 0) i = 0;
      else if (i > bins + 1) i = bins + 1;
      h.sum[i] += x;
      h.count[i] += 1;
    };
    return h;
  }

  /**
   * 创建模拟会话（顺序逐条轨迹模型）。
   * cfg: { N, KMax, twoSided, seed }（seed 省略则随机）。
   * 返回 { pool, nextTrajectory() }：
   * - pool.maxRuns：已完成轨迹的完整曲线，每条 Float32Array(KMax+1)，
   *   [k] = 该轨迹前 k 个向量的运行最大值（k < 2 不用）；
   * - pool.current：进行中轨迹 { arr, k }（部分曲线也参与聚合）；
   * - pool.runsTotal：已完成轨迹数；hist / sumAll / cntAll 跨轨迹累积；
   * - nextTrajectory() 返回单条轨迹的 generator，每完成一个向量 yield
   *   { k, pairs }，轨迹结束 return { done: true, pairs }，此时曲线入池、
   *   向量内存随 generator 释放。
   * 每条轨迹独立 RNG 子流（seed 与轨迹全局序号派生），结果与推进节奏无关。
   */
  function createSession(cfg) {
    var N = cfg.N;
    var KMax = cfg.KMax;
    var twoSided = !!cfg.twoSided;
    var seed =
      (cfg.seed == null ? Math.floor(Math.random() * 0x100000000) : cfg.seed) >>> 0;
    var sigma = 1 / Math.sqrt(N);
    var pool = {
      N: N,
      KMax: KMax,
      twoSided: twoSided,
      seed: seed,
      maxRuns: [],
      current: null,
      runsTotal: 0,
      hist: twoSided
        ? createHist(0, 8 * sigma, 512)
        : createHist(-8 * sigma, 8 * sigma, 512),
      sumAll: 0,
      cntAll: 0,
    };

    function nextTrajectory() {
      if (pool.current) throw new Error('上一条轨迹未结束');
      var idx = pool.runsTotal; // 轨迹全局序号（独立子流的派生依据）
      var arr = new Float32Array(KMax + 1);
      arr[1] = twoSided ? 0 : -Infinity;
      var cur = { arr: arr, k: 1 };
      pool.current = cur;
      var V = new Float32Array(KMax * N);
      var gauss = makeGaussian(mulberry32((seed + Math.imul(idx + 1, 0x9e3779b9)) >>> 0));
      var pairs = 0;
      /** 采样一个单位球面高斯向量，写入 V 的 [off, off+N) 槽位 */
      function fillVector(off) {
        var g = 0;
        var norm = 0;
        var j = 0;
        for (j = 0; j < N; j++) {
          g = gauss();
          V[off + j] = g;
          norm += g * g;
        }
        norm = Math.sqrt(norm);
        for (j = 0; j < N; j++) V[off + j] /= norm;
      }
      // 第 0 个向量在循环外采样：k 循环只填槽位 1..KMax-1，
      // 槽位 0 若留零，所有 pj=0 的点积恒为 0（直方图 0 处出尖峰）
      fillVector(0);
      return (function* () {
        for (var k = 2; k <= KMax; k++) {
          var off = (k - 1) * N;
          fillVector(off);
          var m = twoSided ? 0 : -Infinity;
          var n4 = N - (N % 4);
          var j = 0;
          for (var pj = 0; pj < k - 1; pj++) {
            var o2 = pj * N;
            // 4 路展开累加（实测吞吐 ~+25%），余数尾部顺序处理
            var d0 = 0;
            var d1 = 0;
            var d2 = 0;
            var d3 = 0;
            for (j = 0; j < n4; j += 4) {
              d0 += V[off + j] * V[o2 + j];
              d1 += V[off + j + 1] * V[o2 + j + 1];
              d2 += V[off + j + 2] * V[o2 + j + 2];
              d3 += V[off + j + 3] * V[o2 + j + 3];
            }
            var d = d0 + d1 + d2 + d3;
            for (; j < N; j++) d += V[off + j] * V[o2 + j];
            var key = twoSided ? Math.abs(d) : d;
            pool.hist.add(key);
            pool.sumAll += key;
            pool.cntAll++;
            if (key > m) m = key;
          }
          pairs += k - 1;
          arr[k] = m > arr[k - 1] ? m : arr[k - 1];
          cur.k = k;
          yield { k: k, pairs: pairs };
        }
        pool.maxRuns.push(arr);
        pool.runsTotal++;
        pool.current = null;
        return { done: true, pairs: pairs };
      })();
    }

    return { pool: pool, nextTrajectory: nextTrajectory };
  }

  /** 已覆盖的最大 K：有完整轨迹即 KMax，否则看进行中轨迹进度 */
  function coveredK(pool) {
    if (pool.runsTotal > 0) return pool.KMax;
    return pool.current ? pool.current.k : 1;
  }

  /**
   * 池化数据结构的实际内存账目（字节）：
   * curves = 曲线池（含进行中轨迹的曲线数组）；hist = 直方图数组。
   * 进行中轨迹的向量缓冲由调用方另计（4 × KMax × N，生成器闭包内）。
   */
  function poolMemoryBytes(pool) {
    var curves =
      (pool.runsTotal + (pool.current ? 1 : 0)) * 4 * (pool.KMax + 1);
    var hist = 2 * (pool.hist.bins + 2) * 8;
    return { curves: curves, hist: hist };
  }

  /** 第 k 列的全部轨迹样本（完整轨迹 + 已覆盖 k 的进行中轨迹） */
  function columnValues(pool, k) {
    var hasCur = pool.current && k <= pool.current.k;
    var n = pool.runsTotal + (hasCur ? 1 : 0);
    var out = new Array(n);
    for (var i = 0; i < pool.runsTotal; i++) out[i] = pool.maxRuns[i][k];
    if (hasCur) out[n - 1] = pool.current.arr[k];
    return out;
  }

  function quantileSorted(sorted, q) {
    if (sorted.length === 0) return NaN;
    var pos = (sorted.length - 1) * q;
    var lo = Math.floor(pos);
    var hi = Math.ceil(pos);
    if (lo === hi) return sorted[lo];
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
  }

  /** 第 k 列跨轨迹聚合：中位数与 IQR */
  function aggregateColumn(pool, k) {
    var vals = columnValues(pool, k);
    vals.sort(function (a, b) {
      return a - b;
    });
    return {
      n: vals.length,
      median: quantileSorted(vals, 0.5),
      q1: quantileSorted(vals, 0.25),
      q3: quantileSorted(vals, 0.75),
    };
  }

  /**
   * 列切片直方图（密度化）：返回 bins 个 bin 的中心与密度
   * { xs, ys, n, width }；调用方可自行展开成阶梯线。
   */
  function columnHist(pool, k, lo, hi, bins) {
    var vals = columnValues(pool, k);
    var width = (hi - lo) / bins;
    var counts = new Float64Array(bins);
    for (var i = 0; i < vals.length; i++) {
      var b = Math.floor((vals[i] - lo) / width);
      if (b >= 0 && b < bins) counts[b]++;
    }
    var n = vals.length || 1;
    var xs = new Array(bins);
    var ys = new Array(bins);
    for (b = 0; b < bins; b++) {
      xs[b] = lo + (b + 0.5) * width;
      ys[b] = counts[b] / (n * width);
    }
    return { xs: xs, ys: ys, n: vals.length, width: width };
  }

  /**
   * 列切片 KDE（高斯核，Silverman 带宽）：在给定网格 xs 上求密度。
   * 样本 < 2 时返回全零。
   */
  function columnKDE(pool, k, xs) {
    var vals = columnValues(pool, k);
    var n = vals.length;
    if (n < 2) {
      return { h: 0, ys: xs.map(function () { return 0; }), n: n };
    }
    var sorted = vals.slice().sort(function (a, b) { return a - b; });
    var mean = 0;
    var i;
    for (i = 0; i < n; i++) mean += sorted[i];
    mean /= n;
    var v = 0;
    for (i = 0; i < n; i++) v += (sorted[i] - mean) * (sorted[i] - mean);
    var std = Math.sqrt(v / (n - 1));
    var iqr = quantileSorted(sorted, 0.75) - quantileSorted(sorted, 0.25);
    var sig = Math.min(std, iqr / 1.34);
    if (!(sig > 0)) sig = std > 0 ? std : 1;
    var h = Math.max(0.9 * sig * Math.pow(n, -0.2), 1e-6);
    var inv = 1 / (h * Math.sqrt(2 * Math.PI) * n);
    var ys = xs.map(function (x) {
      var s = 0;
      for (var j = 0; j < n; j++) {
        var z = (x - vals[j]) / h;
        s += Math.exp(-0.5 * z * z);
      }
      return inv * s;
    });
    return { h: h, ys: ys, n: n };
  }

  /** 全体点积直方图的密度化视图（内槽中心与密度） */
  function pairHistDensity(pool) {
    var hist = pool.hist;
    var n = pool.cntAll || 1;
    var bins = hist.bins;
    var xs = new Array(bins);
    var ys = new Array(bins);
    for (var b = 0; b < bins; b++) {
      xs[b] = hist.lo + (b + 0.5) * hist.width;
      ys[b] = hist.count[b + 1] / (n * hist.width);
    }
    return { xs: xs, ys: ys, n: pool.cntAll, width: hist.width };
  }

  global.MC = {
    mulberry32: mulberry32,
    makeGaussian: makeGaussian,
    computeKMax: computeKMax,
    createHist: createHist,
    createSession: createSession,
    coveredK: coveredK,
    poolMemoryBytes: poolMemoryBytes,
    columnValues: columnValues,
    aggregateColumn: aggregateColumn,
    columnHist: columnHist,
    columnKDE: columnKDE,
    pairHistDensity: pairHistDensity,
    TWO_GB: TWO_GB,
  };
})(typeof window !== 'undefined' ? window : globalThis);
