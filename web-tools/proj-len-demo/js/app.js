/**
 * proj-len-demo 的 UI、蒙特卡洛采样与 ECharts 渲染层。
 * 数学公式全部在 theory.js（window.ProjLenTheory）。
 *
 * 模型：y = (AB)^L x，B 为 M×D（方差 1/D），A 为 D×M（方差 1/M），x 单位向量；
 * 观测目标可选完整链 y 或中间层 z = Bx。
 *
 * 采样设计：
 * - 方案一（随机矩阵，固定 x）：卡方因子化 s = ∏[χ²_M·χ²_D/(DM)] 是精确分布
 *   （各向同性 ⇒ 与 x 方向无关），每层用 Marsaglia–Tsang 抽两次卡方，O(L)/样本
 *   （逐高斯平方和是 O(L(M+D))，M 达 4D 时不可接受）。
 * - 方案二（固定矩阵链，随机 x 球面）：结合律 (AB)^L = A(BA)^{L−1}B 合成 P
 *   （BA 仅 M×M，避免 O(LD³)）；每样本 s = ‖Px‖² 为 O(pD)（p = D 或 M）。
 *   实例矩 trW = ‖P‖²_F、tr(W²) = ‖PPᵀ‖²_F 免构造 D×D 的 W = PᵀP。
 */
(function () {
  'use strict';
  var T = window.ProjLenTheory;

  // ---------- 伪随机数：mulberry32 + Box-Muller（与 proj-dot-demo 同款） ----------
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
  function makeGaussian(rand) {
    let spare = null;
    return function () {
      if (spare !== null) { const v = spare; spare = null; return v; }
      let u = 0, v = 0, s = 0;
      do {
        u = rand() * 2 - 1;
        v = rand() * 2 - 1;
        s = u * u + v * v;
      } while (s >= 1 || s === 0);
      const m = Math.sqrt((-2 * Math.log(s)) / s);
      spare = v * m;
      return u * m;
    };
  }

  // ---------- 矩阵工具（行主序 Float64Array） ----------
  /** C = A·B：A 为 m×n、B 为 n×p */
  function matMul(A, m, n, B, p) {
    const out = new Float64Array(m * p);
    for (let i = 0; i < m; i++) {
      const offI = i * n;
      const offIO = i * p;
      for (let k = 0; k < n; k++) {
        const a = A[offI + k];
        if (a === 0) continue;
        const offK = k * p;
        for (let j = 0; j < p; j++) out[offIO + j] += a * B[offK + j];
      }
    }
    return out;
  }

  /**
   * 合成链 P 与实例矩。
   * obs = 'mid'：P = B（M×D）。
   * obs = 'chain'：P = (AB)^L = A(BA)^{L−1}B（D×D；L=1 时即 AB）。
   * 返回 { P, p, trW, trW2 }：trW = ‖P‖²_F；tr(W²) = ‖PPᵀ‖²_F（PPᵀ 对称 ⇒ tr(G²) = Σ G_kl²）。
   */
  function genChain(M, D, L, obs, varB, seed) {
    const gauss = makeGaussian(mulberry32(seed * 7919 + 17));
    const sigB = Math.sqrt(obs === 'mid' ? varB : 1 / D); // 完整链始终配对 1/D
    const B = new Float64Array(M * D);
    for (let i = 0; i < M * D; i++) B[i] = sigB * gauss();
    let P, p;
    if (obs === 'mid') {
      P = B;
      p = M;
    } else {
      const sigA = 1 / Math.sqrt(M);
      const A = new Float64Array(D * M);
      for (let i = 0; i < D * M; i++) A[i] = sigA * gauss();
      if (L === 1) {
        P = matMul(A, D, M, B, D);
      } else {
        const R = matMul(B, M, D, A, M); // R = BA（M×M）
        let Rp = R; // Rp = R^{L−1}
        for (let l = 2; l < L; l++) Rp = matMul(Rp, M, M, R, M);
        P = matMul(matMul(A, D, M, Rp, M), D, M, B, D);
      }
      p = D;
    }
    let trW = 0;
    for (let i = 0; i < p * D; i++) trW += P[i] * P[i];
    const G = new Float64Array(p * p);
    for (let k = 0; k < p; k++) {
      const offK = k * D;
      for (let l = 0; l < p; l++) {
        const offL = l * D;
        let acc = 0;
        for (let j = 0; j < D; j++) acc += P[offK + j] * P[offL + j];
        G[k * p + l] = acc;
      }
    }
    let trW2 = 0;
    for (let i = 0; i < p * p; i++) trW2 += G[i] * G[i];
    return { P: P, p: p, trW: trW, trW2: trW2 };
  }

  // ---------- 采样 ----------
  /** 方案一：卡方因子化。chain：s = ∏_{l=1..L} χ²_M·χ²_D/(DM)；mid：s = varB·χ²_M */
  function fillScheme1(out, i0, i1, M, D, L, obs, varB, rand01, gauss) {
    for (let i = i0; i < i1; i++) {
      let s = T.chi2Sample(M, rand01, gauss) * (obs === 'mid' ? varB : 1 / D);
      if (obs === 'chain') {
        s *= T.chi2Sample(D, rand01, gauss) / M;
        for (let l = 1; l < L; l++) {
          s *= (T.chi2Sample(M, rand01, gauss) / D) *
            (T.chi2Sample(D, rand01, gauss) / M);
        }
      }
      out[i] = s;
    }
  }

  /** 方案二：固定 P（p×D），x 球面均匀，s = ‖Px‖² */
  function fillScheme2(out, i0, i1, D, P, p, gauss, X) {
    for (let i = i0; i < i1; i++) {
      let nx = 0;
      for (let j = 0; j < D; j++) { X[j] = gauss(); nx += X[j] * X[j]; }
      nx = 1 / Math.sqrt(nx);
      let s = 0;
      for (let k = 0; k < p; k++) {
        const off = k * D;
        let acc = 0;
        for (let j = 0; j < D; j++) acc += P[off + j] * X[j];
        s += acc * acc * nx * nx;
      }
      out[i] = s;
    }
  }

  // ---------- 控件 ----------
  const els = {
    selL: document.getElementById('selL'),
    sliderM: document.getElementById('sliderM'),
    inputM: document.getElementById('inputM'),
    sliderD: document.getElementById('sliderD'),
    inputD: document.getElementById('inputD'),
    selN: document.getElementById('selN'),
    inputSeed: document.getElementById('inputSeed'),
    btnSeed: document.getElementById('btnSeed'),
    btnResample: document.getElementById('btnResample'),
    radioNorm: document.getElementById('radioNorm'),
    radioRaw: document.getElementById('radioRaw'),
    radioLinX: document.getElementById('radioLinX'),
    radioLogX: document.getElementById('radioLogX'),
    radioLinY: document.getElementById('radioLinY'),
    radioLogY: document.getElementById('radioLogY'),
    chkTheory: document.getElementById('chkTheory'),
    chkLN: document.getElementById('chkLN'),
    chkGauss: document.getElementById('chkGauss'),
    chkInst: document.getElementById('chkInst'),
    stats: document.getElementById('stats'),
    statsNote: document.getElementById('statsNote'),
    staleHint: document.getElementById('staleHint'),
  };
  const chart = echarts.init(document.getElementById('chart'));

  const state = {
    scheme: 1, obs: 'chain', L: 1, M: 64, D: 256,
    midVarMode: 'd', // 中间层 B 方差：'d' = 1/D（配对下投），'m' = 1/M（均值归一）
    nSamples: 30000, seed: 1, nonce: 0,
    normAxis: true, logX: false, logY: false,
    showTheory: true, showLN: false, showGauss: false, showInst: true,
  };
  const cache = { samplesKey: '', samples: null, chainKey: '', chain: null };

  // ---------- 滑杆映射 ----------
  // M ∈ [1, 4D] 对数刻度（上限随 D 动）；D ∈ [16, 1024] 对数刻度
  function sliderToM(t) {
    return Math.max(1, Math.min(4 * state.D, Math.round(Math.pow(4 * state.D, t / 1000))));
  }
  function mToSlider(m) {
    return Math.max(0, Math.min(1000, Math.round((Math.log(m) / Math.log(4 * state.D)) * 1000)));
  }
  function sliderToD(t) { return Math.max(16, Math.round(Math.pow(2, 4 + (6 * t) / 1000))); }
  function dToSlider(d) { return Math.max(0, Math.min(1000, Math.round(((Math.log2(d) - 4) / 6) * 1000))); }

  function fmt(x, digits) { return Number(x).toFixed(digits === undefined ? 4 : digits); }
  function fmtAuto(x) {
    if (!isFinite(x)) return String(x);
    if (Math.abs(x) >= 1e4 || (Math.abs(x) < 1e-3 && x !== 0)) return x.toExponential(2);
    return fmt(x, 4);
  }

  /** 中间层 B 的元素方差（仅 obs = 'mid' 时使用；完整链始终配对 1/D） */
  function midVarB() {
    return state.midVarMode === 'm' ? 1 / state.M : 1 / state.D;
  }

  // ---------- 采样主流程（分块异步） ----------
  function samplingKey() {
    return [state.scheme, state.obs, state.L, state.M, state.D,
      state.obs === 'mid' ? state.midVarMode : '',
      state.nSamples, state.seed, state.nonce].join('|');
  }
  function chainKey() {
    return [state.obs, state.M, state.D, state.L,
      state.obs === 'mid' ? state.midVarMode : '', state.seed].join('|');
  }

  function runSampling(onDone) {
    const n = state.nSamples, M = state.M, D = state.D, L = state.L;
    const out = new Float64Array(n);
    const rand01 = mulberry32(state.seed * 131 + 7 + state.nonce * 65537);
    const gauss = makeGaussian(rand01);
    let chain = null;
    if (state.scheme === 2) {
      const key = chainKey();
      if (cache.chainKey === key && cache.chain) {
        chain = cache.chain;
      } else {
        els.statsNote.textContent = '合成矩阵链 P（结合律 A(BA)^{L−1}B，一次；大维度需数秒）…';
        chain = genChain(M, D, L, state.obs, midVarB(), state.seed);
        cache.chainKey = key;
        cache.chain = chain;
      }
    } else {
      cache.chain = null;
      cache.chainKey = '';
    }
    const flops = state.scheme === 1
      ? 200 * (state.obs === 'chain' ? L : 1)   // MT 卡方：每块两次 draw
      : 3 * chain.p * D;
    const chunk = Math.max(200, Math.round(2e7 / flops));
    const X = new Float64Array(D);
    let i = 0;
    els.statsNote.textContent = '采样中 0%…';
    function step() {
      const end = Math.min(n, i + chunk);
      if (state.scheme === 1) fillScheme1(out, i, end, M, D, L, state.obs, midVarB(), rand01, gauss);
      else fillScheme2(out, i, end, D, chain.P, chain.p, gauss, X);
      i = end;
      els.statsNote.textContent = '采样中 ' + Math.round((100 * i) / n) + '%…';
      if (i < n) setTimeout(step, 0);
      else onDone(out, chain);
    }
    step();
  }

  // ---------- 样本统计 ----------
  function sampleStats(samples) {
    const n = samples.length;
    let mean = 0, lmean = 0;
    for (let i = 0; i < n; i++) { mean += samples[i]; lmean += Math.log(samples[i]); }
    mean /= n;
    lmean /= n;
    let m2 = 0, lv = 0;
    for (let i = 0; i < n; i++) {
      const d = samples[i] - mean;
      m2 += d * d;
      const dl = Math.log(samples[i]) - lmean;
      lv += dl * dl;
    }
    const sorted = Float64Array.from(samples).sort();
    return {
      mean: mean, sd: Math.sqrt(m2 / n),
      median: sorted[Math.floor(n / 2)],
      logMean: lmean, logSd: Math.sqrt(lv / n),
    };
  }

  // ---------- 渲染 ----------
  const N_BINS = 121;

  function render(samples, chain) {
    const M = state.M, D = state.D, L = state.L;
    const isMid = state.obs === 'mid';
    const varB = midVarB(); // 仅 isMid 时使用
    const c = isMid ? T.midMean(M, varB) : T.abMean();
    const scale = state.normAxis ? c : 1;
    const muLn = isMid ? T.midLogMean(M, varB) : T.abLogMean(M, D, L);
    const varLn = isMid ? T.midLogVar(M) : T.abLogVar(M, D, L);
    const median = Math.exp(muLn);
    const chainVarV = isMid ? T.midVar(M, varB) : T.abVar(M, D, L);
    const hasExact = isMid || L === 1;
    const hasSamples = !!samples; // 页面刚打开时无样本：只画理论曲线

    // 横轴范围：对数域理论驱动（μ ± 4σ）；有样本时再并入 0.1% / 99.9% 分位
    let lo = Math.exp(muLn - 4 * Math.sqrt(varLn));
    let hi = Math.exp(muLn + 4.5 * Math.sqrt(varLn));
    if (hasSamples) {
      const sorted = Float64Array.from(samples).sort();
      const n = sorted.length;
      lo = Math.min(lo, sorted[Math.floor(n * 0.001)]);
      hi = Math.max(hi, sorted[Math.floor(n * 0.999)]);
    }
    lo /= scale;
    hi /= scale;
    if (state.scheme === 2 && chain) {
      lo = Math.min(lo, T.quenchMean(chain.trW, D) / scale * 0.5);
      hi = Math.max(hi, T.quenchMean(chain.trW, D) / scale * 1.5);
    }
    if (!state.logX) lo = Math.max(lo, 0);
    lo = Math.max(lo, 1e-12);

    // 直方图（等宽或等比 bin，密度高度）；无样本时为空
    const histPts = [];
    if (hasSamples) {
      const edges = new Float64Array(N_BINS + 1);
      for (let b = 0; b <= N_BINS; b++) {
        edges[b] = state.logX
          ? lo * Math.pow(hi / lo, b / N_BINS)
          : lo + ((hi - lo) * b) / N_BINS;
      }
      const counts = new Float64Array(N_BINS);
      for (let i = 0; i < samples.length; i++) {
        const t = samples[i] / scale;
        let b;
        if (state.logX) {
          b = Math.floor((Math.log(t / lo) / Math.log(hi / lo)) * N_BINS);
        } else {
          b = Math.floor(((t - lo) / (hi - lo)) * N_BINS);
        }
        if (b >= 0 && b < N_BINS) counts[b]++;
      }
      for (let b = 0; b < N_BINS; b++) {
        const wBin = edges[b + 1] - edges[b];
        const h = counts[b] / (samples.length * wBin);
        const xm = (edges[b] + edges[b + 1]) / 2;
        histPts.push([xm, h > 0 ? h : null]);
      }
    }

    // 理论曲线（显示域网格；密度变量替换 p_t(t) = scale·f(scale·t)）
    const theoryPts = [], lnPts = [], gaussPts = [], instPts = [];
    const N_PT = 400;
    for (let i = 0; i <= N_PT; i++) {
      const t = state.logX
        ? lo * Math.pow(hi / lo, i / N_PT)
        : lo + ((hi - lo) * i) / N_PT;
      const s = t * scale;
      if (state.showTheory && hasExact) {
        const f = isMid
          ? T.gammaDensity(s, M / 2, 2 * varB)
          : T.prodGammaDensityAB(s, M / 2, 2 / D, D / 2, 2 / M);
        theoryPts.push([t, scale * f]);
      }
      if (state.showLN) lnPts.push([t, scale * T.lognormalDensity(s, muLn, varLn)]);
      if (state.showGauss) gaussPts.push([t, scale * T.gaussDensity(s, c, chainVarV)]);
      if (state.showInst && state.scheme === 2 && chain) {
        const im = T.quenchMean(chain.trW, D);
        const iv = T.quenchVar(chain.trW, chain.trW2, D);
        instPts.push([t, scale * T.gaussDensity(s, im, iv)]);
      }
    }

    // 标线：均值、中位数（对数域近似）、实例中心——全部挂到直方图系列，
    // 不依赖任何理论曲线的显隐状态
    // 归一横轴（÷均值）下均值标线恒在 1，中位数标线的数值才是有效信息
    const medLabel = isMid
      ? (state.normAxis ? '中位数 ≈ e^(−1/M) = ' + fmt(Math.exp(-1 / M), 4)
                        : '中位数 ≈ 均值·e^(−1/M)')
      : (state.normAxis ? '中位数 ≈ e^(−L(1/M+1/D))' : '中位数 ≈ e^(−L(1/M+1/D))');
    const meanLabel = isMid
      ? '均值 ' + (state.midVarMode === 'm' ? '1' : 'M/D')
      : '均值 1';
    const markData = [
      { xAxis: c / scale, lineStyle: { type: 'dashed', color: '#374151' },
        label: { formatter: meanLabel, color: '#374151', position: 'insideEndTop' } },
      { xAxis: median / scale, lineStyle: { type: 'dotted', color: '#b45309' },
        label: { formatter: medLabel, color: '#b45309', position: 'insideStartTop' } },
    ];
    if (state.scheme === 2 && chain) {
      markData.push({ xAxis: T.quenchMean(chain.trW, D) / scale,
        lineStyle: { type: 'dashed', color: '#dc2626' },
        label: { formatter: 'trW/D', color: '#dc2626', position: 'insideEndTop' } });
    }

    const series = [{
      name: '蒙特卡洛直方图',
      type: 'line', step: 'middle', showSymbol: false, connectNulls: false,
      lineStyle: { width: 1.2, color: '#0d9488' },
      itemStyle: { color: '#0d9488' },
      areaStyle: { opacity: 0.3 },
      data: histPts,
      markLine: {
        symbol: 'none', silent: true,
        label: { fontSize: 11 },
        data: markData,
      },
    }];
    if (state.showTheory && hasExact) {
      series.push({
        name: isMid ? '理论：伽马（varB·χ²_M，精确）' : '理论：K_ν 乘积伽马（精确）',
        type: 'line', showSymbol: false,
        lineStyle: { width: 2, color: '#2563eb' },
        itemStyle: { color: '#2563eb' },
        data: theoryPts,
      });
    }
    if (state.showLN) {
      series.push({
        name: '对数正态近似',
        type: 'line', showSymbol: false,
        lineStyle: { width: 1.5, color: '#7c3aed' },
        itemStyle: { color: '#7c3aed' },
        data: lnPts,
      });
    }
    if (state.showGauss) {
      series.push({
        name: isMid ? '高斯近似 N(M·varB, 2M·varB²)' : '高斯近似 N(1, [(1+2/M)(1+2/D)]^L−1)',
        type: 'line', showSymbol: false,
        lineStyle: { width: 1.5, color: '#6b7280', type: 'dashed' },
        itemStyle: { color: '#6b7280' },
        data: gaussPts,
      });
    }
    if (state.showInst && state.scheme === 2 && chain) {
      series.push({
        name: '实例高斯（本批链）',
        type: 'line', showSymbol: false,
        lineStyle: { width: 2, color: '#dc2626' },
        itemStyle: { color: '#dc2626' },
        data: instPts,
      });
    }

    const xName = (state.normAxis ? (isMid ? 's / 均值（归一）' : 's（均值恒 1）')
      : isMid ? 's = ‖Bx‖²' : 's = ‖(AB)^L x‖²') + (state.logX ? '（对数刻度）' : '');
    chart.setOption({
      animation: false,
      grid: { left: 70, right: 30, top: 50, bottom: 50 },
      legend: { top: 8 },
      tooltip: { trigger: 'axis', valueFormatter: function (v) { return fmt(v, 4); } },
      xAxis: {
        type: state.logX ? 'log' : 'value',
        name: xName,
        nameLocation: 'middle', nameGap: 30,
        min: lo, max: hi,
      },
      yAxis: { type: state.logY ? 'log' : 'value', name: '密度' },
      series: series,
    }, true);

    renderStats(hasSamples ? sampleStats(samples) : null, chain, c, chainVarV, muLn, varLn, median);
  }

  function renderStats(ss, chain, c, chainVarV, muLn, varLn, median) {
    const M = state.M, D = state.D, L = state.L;
    const isMid = state.obs === 'mid';
    const hasExact = isMid || L === 1;
    const cols = ['样本（' + (ss ? 'N = ' + state.nSamples : '未采样') + '）',
      hasExact ? '理论（精确）' : '对数正态近似',
      state.scheme === 2 ? '实例（本批链）' : '高斯近似'];
    const gaussSd = Math.sqrt(chainVarV);
    let instMean = null, instSd = null;
    if (state.scheme === 2 && chain) {
      instMean = T.quenchMean(chain.trW, D);
      instSd = Math.sqrt(T.quenchVar(chain.trW, chain.trW2, D));
    }
    const varB = midVarB();
    const medApprox = isMid
      ? '均值·e^(−1/M) = ' + fmtAuto(c * Math.exp(-1 / M))
      : 'e^(−L(1/M+1/D)) = ' + fmtAuto(Math.exp(-L * (1 / M + 1 / D)));
    const varFormula = isMid ? '√(2M·varB²)' : '√([(1+2/M)(1+2/D)]^L−1)';
    const meanNote = isMid
      ? (state.midVarMode === 'm' ? '（= M·(1/M) = 1）' : '（= M/D，配对下投 1/D）')
      : '（恒 1，配对 fan-in）';
    const S = function (f) { return ss ? f(ss) : '—'; }; // 未采样时样本列显示占位
    const rows = [];
    rows.push(['均值', S(function (v) { return fmtAuto(v.mean); }), fmtAuto(c) + meanNote,
      state.scheme === 2 ? fmtAuto(instMean) + '（= trW/D）' : fmtAuto(c)]);
    rows.push(['标准差', S(function (v) { return fmtAuto(v.sd); }), fmtAuto(gaussSd) + '（' + varFormula + '）',
      state.scheme === 2 ? fmtAuto(instSd) + '（≈ annealed，self-average）' : fmtAuto(gaussSd)]);
    rows.push(['中位数', S(function (v) { return fmtAuto(v.median); }), fmtAuto(median) + '（≈ ' + medApprox + '）',
      state.scheme === 2 ? '—' : fmtAuto(c)]);
    rows.push(['均值 / 中位数', S(function (v) { return fmt(v.mean / v.median, 3); }),
      fmt(c / median, 3) + (isMid ? '（≈ e^(1/M)）' : '（≈ e^(L(1/M+1/D))）'),
      state.scheme === 2 ? '≈ 1（已 concentrate）' : '1']);
    rows.push(['E[ln s]', S(function (v) { return fmt(v.logMean, 4); }), fmt(muLn, 4), '—']);
    rows.push(['std(ln s)', S(function (v) { return fmt(v.logSd, 4); }), fmt(Math.sqrt(varLn), 4) +
      (isMid ? '（= √(ψ′(M/2))）' : '（= √(L[ψ′(M/2)+ψ′(D/2)])）'), '—']);
    if (state.scheme === 2 && chain) {
      rows.push(['trW / D（实例中心）', '—', fmtAuto(c) + '（系综均值）', fmtAuto(instMean)]);
      let jumpNote;
      if (isMid) {
        jumpNote = '√(2M·varB²/D) = ' + fmtAuto(T.quenchCenterSdMid(M, D, varB));
      } else if (L === 1) {
        jumpNote = '√((2M+4D+2)/(MD²)) = ' + fmtAuto(T.quenchCenterSdAB1(M, D));
      } else {
        jumpNote = 'L≥2 无简单闭式，量级 O(L/D)，随维度消失';
      }
      rows.push(['中心跳动的理论幅度', '—', jumpNote, '本批偏移 ' + fmtAuto(instMean - c)]);
    }

    let html = '<table class="stats-table"><thead><tr><th>指标</th>';
    for (const col of cols) html += '<th>' + col + '</th>';
    html += '</tr></thead><tbody>';
    for (const r of rows) {
      html += '<tr>';
      for (const cell of r) html += '<td>' + cell + '</td>';
      html += '</tr>';
    }
    html += '</tbody></table>';
    els.stats.innerHTML = html;

    // 附注
    const notes = ['M = ' + M + '，D = ' + D + '，M/D = ' + fmt(M / D, 4) +
      (isMid ? '' : '，L = ' + L + ' 个 AB 块')];
    if (isMid) {
      notes.push('中间层 z = Bx（B 方差 = 1/' + (state.midVarMode === 'm' ? 'M' : 'D') +
        '）：s = varB·χ²_M 精确（伽马分布），均值 M·varB = ' + fmtAuto(c) +
        (state.midVarMode === 'm'
          ? '——均值归一（同完整链的保值刻度），方差 2/M 只随 M 收窄'
          : '——瓶颈压缩把典型长度压到 √(M/D)') +
        '，与 x 方向无关（各向同性遗忘方向）');
    } else if (L === 1) {
      notes.push('单块 y = ABx：s = χ²_M·χ²_D/(DM)，两个不同形状卡方的乘积（K_ν 闭式，ν = (M−D)/2 = ' +
        fmt((M - D) / 2, 1) + '）；均值恰 1——配对 fan-in（1/D 下投、1/M 上投）保长度期望');
    } else {
      notes.push('L = ' + L + ' 块：无初等闭式，对数正态近似 ln s ~ N(μ, σ²)；' +
        '均值恒 1 但中位数 ≈ e^(−L(1/M+1/D)) = ' + fmtAuto(Math.exp(-L * (1 / M + 1 / D))) +
        '——均值被重尾撑着，典型样本指数萎缩');
    }
    if (state.scheme === 2) {
      notes.push('种子 ' + state.seed + '：trW/D = ' + fmtAuto(instMean) +
        '（系综均值 ' + fmtAuto(c) + '）；换种子时整条实例曲线随之平移、形状保持' +
        '——实例内涨落 ≈ annealed 大部分散布，中心跳动随维度消失' +
        '（trW 是平方和的自平均；对照点积的 trM 是符号和，涨落不消失）');
    }
    if (state.logX || state.logY) {
      notes.push('对数刻度下可见尾部层级：中间层为 e^(−s/θ)（直线），单块 AB 为 e^(−2√(s/θ₁θ₂)) 拉伸指数，深链趋近对数正态');
    }
    els.statsNote.textContent = notes.join('；');
  }

  // ---------- 调度 ----------
  // 采样是手动的：改任何重采样控件（方案/观测/方差/L/M/D/样本数/种子）只标脏 +
  // 刷新显示（直方图沿用已缓存样本，理论曲线即时跟随新参数），不重采样；
  // 打开页面也不采样（只画理论曲线）；只有点"运行采样"按钮才真正重采样。
  let pending = false;
  function schedule() {
    render(cache.samples, cache.chain); // samples 为 null 时只画理论曲线
    updateStaleHint();
  }
  function runNow() {
    if (pending) return;
    pending = true;
    runSampling(function (samples, chain) {
      cache.samplesKey = samplingKey();
      cache.samples = samples;
      pending = false;
      if (samplingKey() !== cache.samplesKey) {
        // 采样期间参数又变了：直接标脏，等下一次手动点按钮
        if (cache.samples) render(cache.samples, cache.chain);
        updateStaleHint();
      } else {
        render(cache.samples, cache.chain);
        updateStaleHint();
      }
    });
  }
  /** 无样本提示"尚未采样"；有样本但参数已改时提示"参数已修改" */
  function updateStaleHint() {
    if (!cache.samples) {
      els.staleHint.textContent = '尚未采样，点"运行采样"生成蒙特卡洛直方图';
    } else if (samplingKey() !== cache.samplesKey) {
      els.staleHint.textContent = '参数已修改，点"运行采样"生效（当前直方图仍是旧样本）';
    } else {
      els.staleHint.textContent = '';
    }
  }

  // ---------- 事件 ----------
  function setScheme(sc) {
    state.scheme = sc;
    const fixed = sc === 2;
    els.inputSeed.disabled = !fixed;
    els.btnSeed.disabled = !fixed;
    els.chkInst.disabled = !fixed;
    schedule();
  }
  document.querySelectorAll('input[name="scheme"]').forEach(function (r) {
    r.addEventListener('change', function () { setScheme(Number(r.value)); });
  });

  function setObs(obs) {
    state.obs = obs;
    const isMid = obs === 'mid';
    els.selL.disabled = isMid;
    // 中间层方差选项只在观测中间层时可用（完整链始终配对 1/D）
    document.querySelectorAll('input[name="midvar"]').forEach(function (r) {
      r.disabled = !isMid;
    });
    // 深链无精确闭式：自动开启对数正态近似曲线（用户可再关掉）
    if (!isMid && state.L >= 2 && !state.showLN) {
      state.showLN = true;
      els.chkLN.checked = true;
    }
    schedule();
  }
  document.querySelectorAll('input[name="obs"]').forEach(function (r) {
    r.addEventListener('change', function () { setObs(r.value); });
  });
  document.querySelectorAll('input[name="midvar"]').forEach(function (r) {
    r.addEventListener('change', function () {
      state.midVarMode = r.value;
      schedule();
    });
  });

  function setM(m, fromSlider) {
    state.M = Math.max(1, Math.min(4 * state.D, Math.round(m)));
    if (!fromSlider) els.sliderM.value = mToSlider(state.M);
    if (document.activeElement !== els.inputM) els.inputM.value = state.M;
    schedule();
  }
  function setD(d, fromSlider) {
    state.D = Math.max(16, Math.min(1024, Math.round(d)));
    if (!fromSlider) els.sliderD.value = dToSlider(state.D);
    if (document.activeElement !== els.inputD) els.inputD.value = state.D;
    if (state.M > 4 * state.D) setM(4 * state.D, false); // M 上限 4D，随 D 收拢
    else els.sliderM.value = mToSlider(state.M); // 上限变了，重映射滑杆位置
    els.inputM.max = 4 * state.D;
    schedule();
  }

  els.selL.addEventListener('change', function () {
    state.L = Number(els.selL.value);
    if (state.obs === 'chain' && state.L >= 2 && !state.showLN) {
      state.showLN = true;
      els.chkLN.checked = true;
    }
    schedule();
  });
  els.sliderM.addEventListener('input', function () {
    setM(sliderToM(Number(els.sliderM.value)), true);
    els.inputM.value = state.M;
  });
  els.inputM.addEventListener('change', function () {
    const v = Number(els.inputM.value);
    if (isFinite(v) && v >= 1) setM(v, false);
  });
  els.sliderD.addEventListener('input', function () {
    setD(sliderToD(Number(els.sliderD.value)), true);
    els.inputD.value = state.D;
  });
  els.inputD.addEventListener('change', function () {
    const v = Number(els.inputD.value);
    if (isFinite(v) && v >= 16) setD(v, false);
  });
  els.selN.addEventListener('change', function () {
    state.nSamples = Number(els.selN.value);
    schedule();
  });
  els.inputSeed.addEventListener('change', function () {
    const v = Number(els.inputSeed.value);
    if (isFinite(v)) { state.seed = Math.round(v); schedule(); }
  });
  els.btnSeed.addEventListener('click', function () {
    state.seed += 1;
    els.inputSeed.value = state.seed;
    schedule(); // 只改种子，标脏等手动采样
  });
  els.btnResample.addEventListener('click', function () {
    state.nonce += 1; // 矩阵链不动，只换采样流
    runNow(); // 手动触发采样
  });
  els.radioNorm.addEventListener('change', function () {
    state.normAxis = true; schedule();
  });
  els.radioRaw.addEventListener('change', function () {
    state.normAxis = false; schedule();
  });
  els.radioLinX.addEventListener('change', function () {
    state.logX = false; schedule();
  });
  els.radioLogX.addEventListener('change', function () {
    state.logX = true; schedule();
  });
  els.radioLinY.addEventListener('change', function () {
    state.logY = false; schedule();
  });
  els.radioLogY.addEventListener('change', function () {
    state.logY = true; schedule();
  });
  els.chkTheory.addEventListener('change', function () {
    state.showTheory = els.chkTheory.checked; schedule();
  });
  els.chkLN.addEventListener('change', function () {
    state.showLN = els.chkLN.checked; schedule();
  });
  els.chkGauss.addEventListener('change', function () {
    state.showGauss = els.chkGauss.checked; schedule();
  });
  els.chkInst.addEventListener('change', function () {
    state.showInst = els.chkInst.checked; schedule();
  });
  window.addEventListener('resize', function () { chart.resize(); });

  // ---------- 初始化 ----------
  els.sliderM.value = mToSlider(state.M);
  els.inputM.value = state.M;
  els.inputM.max = 4 * state.D;
  els.sliderD.value = dToSlider(state.D);
  els.inputD.value = state.D;
  els.inputSeed.value = state.seed;
  setObs('chain');
  setScheme(1);
  // 打开页面不采样：只画理论曲线，提示点"运行采样"
  render(null, null);
  updateStaleHint();
})();
