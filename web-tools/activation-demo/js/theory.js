/**
 * activation-demo 理论层：高斯输入过激活函数的输出分布。
 * - 单调段切分（dfn=0 临界点数值定位：符号扫描 + 二分）；
 * - 输出密度 f_Y(y) = Σ f_X(x_i)/|f'(x_i)|（多原像求和，段内二分求逆）；
 * - 均值/方差走 64 点 Gauss–Hermite 求积（不做密度积分）；
 * - relu 族 y=0 点质量 atom = P(x≤0) = 0.5（输入零均值）。
 * 无 DOM 依赖，node 可直接 require。
 */
(function (global) {
  'use strict';
  var ActFns = global.ActFns
    || (typeof require !== 'undefined' ? require('./functions.js') : null);

  /**
   * Numerical Recipes gauher：对权 e^(−x²) 的 n 点 Gauss–Hermite 节点与权重。
   * 正根降序逐个 Newton 求出（初值递推引用已收敛的正根），再对称装配成升序。
   */
  function gaussHermite(n) {
    var xs = new Float64Array(n), ws = new Float64Array(n);
    var m = Math.floor((n + 1) / 2);
    var pos = []; // 已收敛的 {r, w}，r 降序
    for (var i = 0; i < m; i++) {
      var z;
      if (i === 0) z = Math.sqrt(2 * n + 1) - 1.85575 * Math.pow(2 * n + 1, -1 / 6);
      else if (i === 1) z = pos[0].r - 1.14 * Math.pow(n, 0.426) / pos[0].r;
      else if (i === 2) z = 1.86 * pos[1].r - 0.86 * pos[0].r;
      else if (i === 3) z = 1.91 * pos[2].r - 0.91 * pos[1].r;
      else z = 2 * pos[i - 1].r - pos[i - 2].r;
      var pp = 0;
      for (var iter = 0; iter < 30; iter++) {
        var p1 = Math.pow(Math.PI, -0.25), p2 = 0, p3, j;
        for (j = 0; j < n; j++) {
          p3 = p2;
          p2 = p1;
          p1 = z * Math.sqrt(2 / (j + 1)) * p2 - Math.sqrt(j / (j + 1)) * p3;
        }
        pp = Math.sqrt(2 * n) * p2;
        var z1 = z;
        z = z1 - p1 / pp;
        if (Math.abs(z - z1) < 1e-14) break;
      }
      pos.push({ r: z, w: 2 / (pp * pp) });
    }
    for (i = 0; i < m; i++) {
      xs[i] = -pos[i].r;
      xs[n - 1 - i] = pos[i].r;
      ws[i] = pos[i].w;
      ws[n - 1 - i] = pos[i].w;
    }
    return { x: xs, w: ws };
  }

  var GH64 = null;
  function gh64() {
    if (!GH64) GH64 = gaussHermite(64);
    return GH64;
  }

  /** 在 [lo,hi] 上定位 dfn=0 的临界点（符号扫描 2001 点 + 二分），返回升序数组 */
  function findCriticalPoints(act, p, lo, hi) {
    var M = 2001, roots = [];
    var xPrev = lo, dPrev = act.dfn(lo, p);
    for (var i = 1; i < M; i++) {
      var x = lo + (hi - lo) * i / (M - 1);
      var d = act.dfn(x, p);
      if (dPrev !== 0 && d !== 0 && ((dPrev < 0) !== (d < 0))) {
        var a = xPrev, b = x, da = dPrev;
        for (var it = 0; it < 60; it++) {
          var mid = 0.5 * (a + b), dm = act.dfn(mid, p);
          if (dm === 0) { a = mid; b = mid; break; }
          if ((da < 0) === (dm < 0)) { a = mid; da = dm; } else { b = mid; }
        }
        roots.push(0.5 * (a + b));
      }
      xPrev = x;
      dPrev = d;
    }
    return roots;
  }

  /** 在临界点切分 [lo,hi]，返回单调段 [{a,b,fa,fb}]（fa/fb 为端点函数值） */
  function monotoneSegments(act, p, lo, hi) {
    var bounds = [lo].concat(findCriticalPoints(act, p, lo, hi), [hi]);
    var segs = [];
    for (var s = 0; s + 1 < bounds.length; s++) {
      segs.push({
        a: bounds[s], b: bounds[s + 1],
        fa: act.fn(bounds[s], p), fb: act.fn(bounds[s + 1], p),
      });
    }
    return segs;
  }

  /** f(x)=y 的全部原像：逐段检查值域、段内二分（80 次），按 1e-10 去重 */
  function solvePreimages(act, p, y, segments) {
    var roots = [];
    segments.forEach(function (seg) {
      var fa = seg.fa - y, fb = seg.fb - y;
      if (fa === 0) { roots.push(seg.a); return; }
      if (fa * fb > 0) return;
      var a = seg.a, b = seg.b;
      for (var it = 0; it < 80; it++) {
        var mid = 0.5 * (a + b), fm = act.fn(mid, p) - y;
        if (fa * fm <= 0) { b = mid; fb = fm; } else { a = mid; fa = fm; }
      }
      roots.push(0.5 * (a + b));
    });
    var uniq = [];
    roots.forEach(function (r) {
      if (!uniq.length || Math.abs(r - uniq[uniq.length - 1]) > 1e-10) uniq.push(r);
    });
    return uniq;
  }

  /** 输出密度 f_Y(y) = Σ f_X(x_i)/|f'(x_i)|；无原像返回 0 */
  function outputDensity(act, p, y, sigma, segments) {
    var roots = solvePreimages(act, p, y, segments);
    var sum = 0;
    roots.forEach(function (x) {
      var d = Math.abs(act.dfn(x, p));
      if (d > 0) sum += ActFns.helpers.normPdf(x, sigma) / d;
    });
    return sum;
  }

  /** 理论曲线网格：在函数值域的 y 区间内取 nY 点算密度（留 2% 余量） */
  function densityGrid(act, p, sigma, lo, hi, nY) {
    var segs = monotoneSegments(act, p, lo, hi);
    var yLo = Infinity, yHi = -Infinity;
    segs.forEach(function (s) {
      yLo = Math.min(yLo, s.fa, s.fb);
      yHi = Math.max(yHi, s.fa, s.fb);
    });
    var pad = 0.02 * (yHi - yLo || 1);
    yLo -= pad;
    yHi += pad;
    var ys = new Float64Array(nY), fs = new Float64Array(nY);
    for (var i = 0; i < nY; i++) {
      ys[i] = yLo + (yHi - yLo) * i / (nY - 1);
      fs[i] = outputDensity(act, p, ys[i], sigma, segs);
    }
    return { ys: ys, fs: fs, yLo: yLo, yHi: yHi };
  }

  /** 输出矩：64 点 Gauss–Hermite，E[g(x)] ≈ π^(−1/2) Σ w_i g(σ√2 x_i) */
  function outputMoments(act, p, sigma) {
    var gh = gh64();
    var t = sigma * Math.SQRT2;
    var m1 = 0, m2 = 0;
    for (var i = 0; i < gh.x.length; i++) {
      var v = act.fn(t * gh.x[i], p);
      m1 += gh.w[i] * v;
      m2 += gh.w[i] * v * v;
    }
    m1 /= Math.sqrt(Math.PI);
    m2 /= Math.sqrt(Math.PI);
    return {
      mean: m1,
      variance: Math.max(0, m2 - m1 * m1),
      atom: act.atom ? ActFns.helpers.normCdf(0, sigma) : 0,
    };
  }

  /** 绘图范围建议：输入 ±8σ，y 取函数值域并留 10% 余量 */
  function suggestRange(act, p, sigma) {
    var xLo = -8 * sigma, xHi = 8 * sigma;
    var segs = monotoneSegments(act, p, xLo, xHi);
    var yLo = Infinity, yHi = -Infinity;
    segs.forEach(function (s) {
      yLo = Math.min(yLo, s.fa, s.fb);
      yHi = Math.max(yHi, s.fa, s.fb);
    });
    var pad = 0.1 * (yHi - yLo || 1);
    return { xLo: xLo, xHi: xHi, yLo: yLo - pad, yHi: yHi + pad };
  }

  /* ---------- GLU 门控族：y = u·g(v)，(u,v) 联合高斯相关 ρ ---------- */

  /** GLU 输出矩：E[y]=ρE[v·g(v)]，E[y²]=E[g(v)²(σ²(1−ρ²)+ρ²v²)]，一维 GH */
  function gluOutputMoments(gate, rho, sigma) {
    var g = ActFns.gateAct(gate), gp = ActFns.defaultParams(g);
    var gh = gh64(), t = sigma * Math.SQRT2, m1 = 0, m2 = 0;
    for (var i = 0; i < gh.x.length; i++) {
      var v = t * gh.x[i], gv = g.fn(v, gp);
      m1 += gh.w[i] * v * gv;
      m2 += gh.w[i] * gv * gv * (sigma * sigma * (1 - rho * rho) + rho * rho * v * v);
    }
    m1 = rho * m1 / Math.sqrt(Math.PI);
    m2 /= Math.sqrt(Math.PI);
    return { mean: m1, variance: Math.max(0, m2 - m1 * m1), atom: gate.atom ? 0.5 : 0 };
  }

  /** 条件分布 y|v0 ~ N(ρv0·g(v0), σ²(1−ρ²)g(v0)²)（精确高斯） */
  function gluConditional(gate, rho, sigma, v0) {
    var g = ActFns.gateAct(gate), gp = ActFns.defaultParams(g);
    var gv = g.fn(v0, gp);
    return {
      gv: gv,
      mean: rho * v0 * gv,
      sd: sigma * Math.sqrt(1 - rho * rho) * Math.abs(gv),
    };
  }

  /**
   * 边缘密度：f_Y(y) = ∫ φ(v;σ)·φ(y/g(v) − ρv; s)/|g(v)| dv，s = σ√(1−ρ²)。
   * 用 v 空间梯形积分（2001 箱中点法，±10σ）——被积函数在 relu 门等处
   * 有尖峰，GH 求积收敛差，梯形（光滑快衰）精度 ~1e-8。
   */
  function gluOutputDensity(gate, rho, sigma, y) {
    var g = ActFns.gateAct(gate), gp = ActFns.defaultParams(g);
    var s = sigma * Math.sqrt(1 - rho * rho);
    var nv = 2001, vmax = 10 * sigma, dv = 2 * vmax / nv, sum = 0;
    for (var i = 0; i < nv; i++) {
      var v = -vmax + (i + 0.5) * dv;
      var gv = g.fn(v, gp);
      if (gv === 0) continue; // 门关闭支只对 atom 有贡献
      sum += ActFns.helpers.normPdf(v, sigma)
        * ActFns.helpers.normPdf(y / gv - rho * v, s) / Math.abs(gv);
    }
    return sum * dv;
  }

  /** 区间质量：E_v[ |Φ(y2/g(v)−ρv; s) − Φ(y1/g(v)−ρv; s)| ]，精确一维 GH（不含 atom） */
  function gluBinMass(gate, rho, sigma, y1, y2) {
    var g = ActFns.gateAct(gate), gp = ActFns.defaultParams(g);
    var gh = gh64(), t = sigma * Math.SQRT2;
    var s = sigma * Math.sqrt(1 - rho * rho), sum = 0;
    for (var i = 0; i < gh.x.length; i++) {
      var v = t * gh.x[i], gv = g.fn(v, gp);
      if (gv === 0) continue;
      sum += gh.w[i] * Math.abs(
        ActFns.helpers.normCdf(y2 / gv - rho * v, s) -
        ActFns.helpers.normCdf(y1 / gv - rho * v, s));
    }
    return sum / Math.sqrt(Math.PI);
  }

  /** 联合高斯等高线椭圆：u=kσ·cos t，v=kσ(ρ cos t + √(1−ρ²) sin t) */
  function gaussEllipse(rho, sigma, k, nPts) {
    var pts = [], s = Math.sqrt(1 - rho * rho);
    for (var i = 0; i <= nPts; i++) {
      var t = 2 * Math.PI * i / nPts;
      pts.push([k * sigma * Math.cos(t),
        k * sigma * (rho * Math.cos(t) + s * Math.sin(t))]);
    }
    return pts;
  }

  var ActTheory = {
    gaussHermite: gaussHermite,
    findCriticalPoints: findCriticalPoints,
    monotoneSegments: monotoneSegments,
    solvePreimages: solvePreimages,
    outputDensity: outputDensity,
    densityGrid: densityGrid,
    outputMoments: outputMoments,
    suggestRange: suggestRange,
    gluOutputMoments: gluOutputMoments,
    gluConditional: gluConditional,
    gluOutputDensity: gluOutputDensity,
    gluBinMass: gluBinMass,
    gaussEllipse: gaussEllipse,
  };

  global.ActTheory = ActTheory;
  if (typeof module !== 'undefined' && module.exports) module.exports = ActTheory;
})(typeof window !== 'undefined' ? window : globalThis);
