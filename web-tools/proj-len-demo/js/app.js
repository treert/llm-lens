/**
 * proj-len-demo 的 UI、蒙特卡洛采样与 ECharts 渲染层。
 * 数学公式全部在 theory.js（window.ProjLenTheory）。
 *
 * 采样设计：
 * - 方案一（随机矩阵，固定 X）：卡方因子化 s = (1/D)·C₀·∏(C_l/H) 是精确分布
 *   （各向同性 ⇒ 与 X 方向无关），每层直接抽 H 个高斯平方和，O(LH)/样本。
 * - 方案二（固定矩阵链，随机 X 球面）：先把链合成 M = B_{L−1}···B_1 A（O(LH²D) 一次），
 *   每样本 s = ‖MX‖² 只要 O(HD)；实例矩 trW = ‖M‖²_F、tr(W²) = ‖MMᵀ‖²_F 同样免构造
 *   D×D 的 W = MᵀM。
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
  /** C = B·M：B 为 H×H、M 为 H×D */
  function matMulHB(B, M, H, D) {
    const out = new Float64Array(H * D);
    for (let k = 0; k < H; k++) {
      const offK = k * H;
      const offKO = k * D;
      for (let m = 0; m < H; m++) {
        const b = B[offK + m];
        if (b === 0) continue;
        const offM = m * D;
        for (let j = 0; j < D; j++) out[offKO + j] += b * M[offM + j];
      }
    }
    return out;
  }

  function genChain(H, D, L, seed) {
    const gauss = makeGaussian(mulberry32(seed * 7919 + 17));
    const sigA = 1 / Math.sqrt(D);
    const sigB = 1 / Math.sqrt(H);
    let M = new Float64Array(H * D);
    for (let i = 0; i < H * D; i++) M[i] = sigA * gauss();
    for (let l = 1; l < L; l++) {
      const B = new Float64Array(H * H);
      for (let i = 0; i < H * H; i++) B[i] = sigB * gauss();
      M = matMulHB(B, M, H, D);
    }
    // 实例矩：trW = ‖M‖²_F；tr(W²) = ‖MMᵀ‖²_F（MMᵀ 对称 ⇒ tr(G²) = Σ G_kl²）
    let trW = 0;
    for (let i = 0; i < H * D; i++) trW += M[i] * M[i];
    const G = new Float64Array(H * H);
    for (let k = 0; k < H; k++) {
      const offK = k * D;
      for (let l = 0; l < H; l++) {
        const offL = l * D;
        let acc = 0;
        for (let j = 0; j < D; j++) acc += M[offK + j] * M[offL + j];
        G[k * H + l] = acc;
      }
    }
    let trW2 = 0;
    for (let i = 0; i < H * H; i++) trW2 += G[i] * G[i];
    return { M: M, trW: trW, trW2: trW2 };
  }

  // ---------- 采样 ----------
  /** 方案一：s = (1/D)·∏_{l=0..L−1} χ²_H 因子（l ≥ 1 的因子再除 H） */
  function fillScheme1(out, i0, i1, H, D, L, gauss) {
    for (let i = i0; i < i1; i++) {
      let s = 1 / D;
      for (let l = 0; l < L; l++) {
        let c = 0;
        for (let k = 0; k < H; k++) { const g = gauss(); c += g * g; }
        s *= l === 0 ? c : c / H;
      }
      out[i] = s;
    }
  }

  /** 方案二：固定 M（H×D），X 球面均匀，s = ‖MX‖² */
  function fillScheme2(out, i0, i1, H, D, M, gauss, X, Y) {
    for (let i = i0; i < i1; i++) {
      let nx = 0;
      for (let j = 0; j < D; j++) { X[j] = gauss(); nx += X[j] * X[j]; }
      nx = 1 / Math.sqrt(nx);
      let s = 0;
      for (let k = 0; k < H; k++) {
        const off = k * D;
        let acc = 0;
        for (let j = 0; j < D; j++) acc += M[off + j] * X[j];
        s += acc * acc * nx * nx;
      }
      out[i] = s;
    }
  }

  // ---------- 控件 ----------
  const els = {
    selL: document.getElementById('selL'),
    sliderH: document.getElementById('sliderH'),
    inputH: document.getElementById('inputH'),
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
  };
  const chart = echarts.init(document.getElementById('chart'));

  const state = {
    scheme: 1, L: 1, H: 64, D: 256,
    nSamples: 30000, seed: 1, nonce: 0,
    normAxis: true, logX: false, logY: false,
    showTheory: true, showLN: false, showGauss: false, showInst: true,
  };
  const cache = { samplesKey: '', samples: null, chainKey: '', chain: null };

  // ---------- 滑杆映射（与 proj-dot-demo 相同） ----------
  function sliderToH(t) { return Math.max(1, Math.round(Math.pow(2, (8 * t) / 1000))); }
  function hToSlider(h) { return Math.max(0, Math.min(1000, Math.round((Math.log2(h) / 8) * 1000))); }
  function sliderToD(t) { return Math.max(16, Math.round(Math.pow(2, 4 + (6 * t) / 1000))); }
  function dToSlider(d) { return Math.max(0, Math.min(1000, Math.round(((Math.log2(d) - 4) / 6) * 1000))); }

  function fmt(x, digits) { return Number(x).toFixed(digits === undefined ? 4 : digits); }
  function fmtAuto(x) {
    if (!isFinite(x)) return String(x);
    if (Math.abs(x) >= 1e4 || (Math.abs(x) < 1e-3 && x !== 0)) return x.toExponential(2);
    return fmt(x, 4);
  }

  // ---------- 采样主流程（分块异步） ----------
  function samplingKey() {
    return [state.scheme, state.L, state.H, state.D,
      state.nSamples, state.seed, state.nonce].join('|');
  }
  function chainKey() {
    return [state.H, state.D, state.L, state.seed].join('|');
  }

  function runSampling(onDone) {
    const n = state.nSamples, H = state.H, D = state.D, L = state.L;
    const out = new Float64Array(n);
    const gauss = makeGaussian(mulberry32(state.seed * 131 + 7 + state.nonce * 65537));
    let chain = null;
    if (state.scheme === 2) {
      const key = chainKey();
      if (cache.chainKey === key && cache.chain) {
        chain = cache.chain;
      } else {
        els.statsNote.textContent = '合成矩阵链 M = B···BA（O(LH²D)，一次）…';
        chain = genChain(H, D, L, state.seed);
        cache.chainKey = key;
        cache.chain = chain;
      }
    } else {
      cache.chain = null;
      cache.chainKey = '';
    }
    const flops = state.scheme === 1 ? 3 * H * L : 3 * H * D;
    const chunk = Math.max(200, Math.round(2e7 / flops));
    const X = new Float64Array(D), Y = new Float64Array(H);
    let i = 0;
    els.statsNote.textContent = '采样中 0%…';
    function step() {
      const end = Math.min(n, i + chunk);
      if (state.scheme === 1) fillScheme1(out, i, end, H, D, L, gauss);
      else fillScheme2(out, i, end, H, D, chain.M, gauss, X, Y);
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
    const H = state.H, D = state.D, L = state.L;
    const c = T.chainMean(H, D);
    const scale = state.normAxis ? c : 1;
    const muLn = T.chainLogMean(H, D, L);
    const varLn = T.chainLogVar(H, L);
    const median = Math.exp(muLn);
    const chainVarV = T.chainVar(H, D, L);
    const k = H / 2;
    const hasExact = L <= 2;

    // 横轴范围：对数域理论驱动（μ ± 4σ）并入样本 0.1% / 99.9% 分位
    const sorted = Float64Array.from(samples).sort();
    const n = sorted.length;
    let lo = Math.min(Math.exp(muLn - 4 * Math.sqrt(varLn)), sorted[Math.floor(n * 0.001)]);
    let hi = Math.max(Math.exp(muLn + 4.5 * Math.sqrt(varLn)), sorted[Math.floor(n * 0.999)]);
    lo /= scale;
    hi /= scale;
    if (state.scheme === 2 && chain) {
      lo = Math.min(lo, T.quenchMean(chain.trW, D) / scale * 0.5);
      hi = Math.max(hi, T.quenchMean(chain.trW, D) / scale * 1.5);
    }
    if (!state.logX) lo = Math.max(lo, 0);
    lo = Math.max(lo, 1e-12);

    // 直方图（等宽或等比 bin，密度高度）
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
    const histPts = [];
    for (let b = 0; b < N_BINS; b++) {
      const wBin = edges[b + 1] - edges[b];
      const h = counts[b] / (samples.length * wBin);
      const xm = (edges[b] + edges[b + 1]) / 2;
      histPts.push([xm, h > 0 ? h : null]);
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
        const f = L === 1
          ? T.gammaDensity(s, k, 2 / D)
          : T.prodGammaDensity(s, k, 2 / D, 2 / H);
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

    // 标线：均值、中位数（LN 近似）、实例中心——全部挂到直方图系列，
    // 不依赖任何理论曲线的显隐状态（L ≥ 3 无精确曲线时中位数标线依然在）
    const markData = [
      { xAxis: c / scale, lineStyle: { type: 'dashed', color: '#374151' },
        label: { formatter: '均值 c', color: '#374151', position: 'insideEndTop' } },
      { xAxis: median / scale, lineStyle: { type: 'dotted', color: '#b45309' },
        label: { formatter: '中位数 ≈ c·e^(−L/H)', color: '#b45309', position: 'insideStartTop' } },
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
        name: L === 1 ? '理论：伽马（χ²_H/D，精确）' : '理论：K₀ 乘积伽马（精确）',
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
        name: '高斯近似 N(c, c²((1+2/H)^L−1))',
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

    chart.setOption({
      animation: false,
      grid: { left: 70, right: 30, top: 50, bottom: 50 },
      legend: { top: 8 },
      tooltip: { trigger: 'axis', valueFormatter: function (v) { return fmt(v, 4); } },
      xAxis: {
        type: state.logX ? 'log' : 'value',
        name: (state.normAxis ? 's / c（归一，c = H/D）' : 's = ‖B···BA X‖²') +
          (state.logX ? '（对数刻度）' : ''),
        nameLocation: 'middle', nameGap: 30,
        min: lo, max: hi,
      },
      yAxis: { type: state.logY ? 'log' : 'value', name: '密度' },
      series: series,
    }, true);

    renderStats(sampleStats(samples), chain, c, chainVarV, muLn, varLn, median, scale);
  }

  function renderStats(ss, chain, c, chainVarV, muLn, varLn, median, scale) {
    const H = state.H, D = state.D, L = state.L;
    const hasExact = L <= 2;
    const cols = ['样本（N = ' + state.nSamples + '）',
      hasExact ? '理论（精确）' : '对数正态近似',
      state.scheme === 2 ? '实例（本批链）' : '高斯近似'];
    const gaussSd = Math.sqrt(chainVarV);
    let instMean = null, instSd = null;
    if (state.scheme === 2 && chain) {
      instMean = T.quenchMean(chain.trW, D);
      instSd = Math.sqrt(T.quenchVar(chain.trW, chain.trW2, D));
    }
    const rows = [];
    rows.push(['均值', fmtAuto(ss.mean), fmtAuto(c),
      state.scheme === 2 ? fmtAuto(instMean) + '（= trW/D）' : fmtAuto(c)]);
    rows.push(['标准差', fmtAuto(ss.sd), fmtAuto(gaussSd) + '（c²((1+2/H)^L−1) 开方）',
      state.scheme === 2 ? fmtAuto(instSd) + '（≈ annealed，self-average）' : fmtAuto(gaussSd)]);
    rows.push(['中位数', fmtAuto(ss.median), fmtAuto(median) + '（≈ c·e^(−L/H) = ' +
      fmtAuto(c * Math.exp(-L / H)) + '）', state.scheme === 2 ? '—' : fmtAuto(gaussSd > 0 ? c : c)]);
    rows.push(['均值 / 中位数', fmt(ss.mean / ss.median, 3),
      fmt(c / median, 3) + '（≈ e^(L/H)）', state.scheme === 2 ? '≈ 1（已 concentrate）' : '1']);
    rows.push(['E[ln s]', fmt(ss.logMean, 4), fmt(muLn, 4), '—']);
    rows.push(['std(ln s)', fmt(ss.logSd, 4), fmt(Math.sqrt(varLn), 4) + '（= √(L·ψ′(H/2))）', '—']);
    if (state.scheme === 2 && chain) {
      rows.push(['trW / D（实例中心）', '—', fmtAuto(c) + '（系综均值）', fmtAuto(instMean)]);
      rows.push(['中心跳动的理论幅度', '—',
        '√(L(L−1+2c))/D = ' + fmtAuto(Math.sqrt(L * (L - 1 + 2 * H / D)) / D),
        '本批偏移 ' + fmtAuto(instMean - c)]);
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
    const notes = ['H = ' + H + '，D = ' + D + '，c = H/D = ' + fmt(H / D, 4) +
      '，L = ' + L + ' 层'];
    if (state.scheme === 1) {
      if (L === 1) {
        notes.push('单层：s = χ²_H/D 精确（伽马分布），与 X 方向无关——各向同性再次遗忘方向');
      } else if (L === 2) {
        notes.push('双层：s = χ²_H·χ²_H/(DH)，两个独立卡方乘积（K₀ 闭式）；均值仍 c，相对方差从 2/H 涨到 ≈ 4/H');
      } else {
        notes.push('L = ' + L + ' 层：无初等闭式（Meijer G），对数正态近似 ln s ~ N(μ, σ²)；' +
          '均值 c 不变但中位数 ≈ c·e^(−L/H) = ' + fmtAuto(c * Math.exp(-L / H)) +
          '——均值被重尾撑着，典型样本指数萎缩');
      }
    } else {
      notes.push('种子 ' + state.seed + '：trW/D = ' + fmtAuto(instMean) +
        '（系综均值 c = ' + fmtAuto(c) + '）；换种子时整条实例曲线随之平移、形状保持' +
        '——实例内涨落 ≈ annealed 大部分散布，中心跳动 √(L(L−1+2c))/D 随维度消失' +
        '（trW 是平方和；对照点积的 trM 是符号和，涨落 √(H/D) 不消失）');
    }
    if (state.logX || state.logY) {
      notes.push('对数刻度下可见尾部层级：L=1 为 e^(−s/θ)（直线），L=2 为 e^(−2√(s/θ₁θ₂)) 拉伸指数，L 层趋近对数正态');
    }
    els.statsNote.textContent = notes.join('；');
  }

  // ---------- 调度 ----------
  let pending = false;
  function schedule() {
    const key = samplingKey();
    if (key === cache.samplesKey && cache.samples) {
      render(cache.samples, cache.chain);
      return;
    }
    if (pending) return;
    pending = true;
    runSampling(function (samples, chain) {
      cache.samplesKey = samplingKey();
      cache.samples = samples;
      pending = false;
      if (samplingKey() !== cache.samplesKey) schedule();
      else render(cache.samples, cache.chain);
    });
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

  function setH(h, fromSlider) {
    state.H = Math.max(1, Math.min(256, Math.round(h)));
    if (!fromSlider) els.sliderH.value = hToSlider(state.H);
    if (document.activeElement !== els.inputH) els.inputH.value = state.H;
    schedule();
  }
  function setD(d, fromSlider) {
    state.D = Math.max(16, Math.min(1024, Math.round(d)));
    if (!fromSlider) els.sliderD.value = dToSlider(state.D);
    if (document.activeElement !== els.inputD) els.inputD.value = state.D;
    schedule();
  }

  els.selL.addEventListener('change', function () {
    state.L = Number(els.selL.value);
    // L ≥ 3 无精确闭式：自动开启对数正态近似曲线（用户可再关掉）
    if (state.L >= 3 && !state.showLN) {
      state.showLN = true;
      els.chkLN.checked = true;
    }
    schedule();
  });
  els.sliderH.addEventListener('input', function () {
    setH(sliderToH(Number(els.sliderH.value)), true);
    els.inputH.value = state.H;
  });
  els.inputH.addEventListener('change', function () {
    const v = Number(els.inputH.value);
    if (isFinite(v) && v >= 1) setH(v, false);
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
    schedule();
  });
  els.btnResample.addEventListener('click', function () {
    state.nonce += 1; // 矩阵链不动，只换采样流
    schedule();
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
  els.sliderH.value = hToSlider(state.H);
  els.inputH.value = state.H;
  els.sliderD.value = dToSlider(state.D);
  els.inputD.value = state.D;
  els.inputSeed.value = state.seed;
  setScheme(1);
})();
