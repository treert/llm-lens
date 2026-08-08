/**
 * proj-dot-demo 的 UI、蒙特卡洛采样与 ECharts 渲染层。
 * 数学公式全部在 theory.js（window.ProjDotTheory）。
 *
 * 采样设计：
 * - 方案一（随机矩阵）：u = Ma A ~ N(0, βI_H)、v = Mb B ~ N(0, βI_H) 且独立（β = 1/D，
 *   单位向量口径），与 A、B 的具体取值和夹角 ρ 无关——直接抽 2H 个高斯即可，
 *   与显式生成矩阵严格同分布。「显式矩阵模式」取 A = e₁、B = ρe₁ + √(1−ρ²)e₂
 *   （不失一般性：双各向同性下分布只依赖范数与夹角），代码里 ρ 出现但统计上消失。
 * - 方案二/三（固定矩阵）：种子化生成 Ma、Mb 并精确计算不变量
 *   trM、‖M‖²_F = ⟨MaMaᵀ, MbMbᵀ⟩_F、tr(M²) = tr((MaMbᵀ)²)（O(H²D)，免构造 D×D 的 M），
 *   球面构造 A、B（精确 cos = ρ），每样本两次矩阵向量 O(HD)。
 * - 退火平均：24 个种子的矩阵各抽一段样本合并，展示 ρ = 0 时与方案一恒等。
 */
(function () {
  'use strict';
  var T = window.ProjDotTheory;

  // ---------- 伪随机数：mulberry32 + Box-Muller ----------
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

  // ---------- 矩阵工具（行主序 Float64Array，H×D） ----------
  function matVec(M, H, D, x, out) {
    for (let k = 0; k < H; k++) {
      const off = k * D;
      let acc = 0;
      for (let j = 0; j < D; j++) acc += M[off + j] * x[j];
      out[k] = acc;
    }
  }

  /** 生成 Ma、Mb（元素 N(0, 1/D)）；方案三 Mb = αMa + √(1−α²)G */
  function genMatrices(H, D, alpha, scheme, seed) {
    const gauss = makeGaussian(mulberry32(seed * 7919 + 17));
    const sig = 1 / Math.sqrt(D);
    const n = H * D;
    const Ma = new Float64Array(n);
    const Mb = new Float64Array(n);
    for (let i = 0; i < n; i++) Ma[i] = sig * gauss();
    if (scheme === 3) {
      const sa = Math.sqrt(1 - alpha * alpha);
      for (let i = 0; i < n; i++) Mb[i] = alpha * Ma[i] + sa * sig * gauss();
    } else {
      for (let i = 0; i < n; i++) Mb[i] = sig * gauss();
    }
    return { Ma: Ma, Mb: Mb };
  }

  /**
   * M = MaᵀMb（D×D）的不变量，免显式构造 M：
   *   trM      = Σ Ma·Mb（O(HD)）
   *   ‖M‖²_F   = tr(MaMaᵀ·MbMbᵀ) = ⟨Ga, Gb⟩_F（行 Gram，O(H²D)）
   *   tr(M²)   = tr((MaMbᵀ)²) = Σ Gx_kl·Gx_lk（互 Gram，O(H²D)）
   *   ‖M_s‖²_F = (‖M‖²_F + tr M²)/2
   */
  function matrixStats(Ma, Mb, H, D) {
    let trM = 0;
    for (let i = 0; i < H * D; i++) trM += Ma[i] * Mb[i];
    // 行 Gram Ga = MaMaᵀ、Gb = MbMbᵀ（对称，只算上三角）、互 Gram Gx = MaMbᵀ
    const Ga = new Float64Array(H * H);
    const Gb = new Float64Array(H * H);
    const Gx = new Float64Array(H * H);
    for (let k = 0; k < H; k++) {
      const offK = k * D;
      for (let l = 0; l < H; l++) {
        const offL = l * D;
        let a = 0, b = 0, x = 0;
        for (let j = 0; j < D; j++) {
          const ma = Ma[offK + j];
          const mb = Mb[offK + j];
          a += ma * Ma[offL + j];
          b += mb * Mb[offL + j];
          x += ma * Mb[offL + j];
        }
        Ga[k * H + l] = a;
        Gb[k * H + l] = b;
        Gx[k * H + l] = x;
      }
    }
    let normF2 = 0, trM2 = 0;
    for (let k = 0; k < H; k++) {
      for (let l = 0; l < H; l++) {
        normF2 += Ga[k * H + l] * Gb[k * H + l];
        trM2 += Gx[k * H + l] * Gx[l * H + k];
      }
    }
    return { trM: trM, normF2: normF2, trM2: trM2, normMs2: (normF2 + trM2) / 2 };
  }

  // ---------- 采样 ----------
  /** 方案一：直接抽 u_k、v_k ~ N(0, β)（β = 1/D）。explicit 模式显式走两列矩阵元 */
  function fillScheme1(out, i0, i1, H, D, rho, explicit, gauss) {
    const beta = 1 / D;
    if (!explicit) {
      for (let i = i0; i < i1; i++) {
        let acc = 0;
        for (let k = 0; k < H; k++) acc += gauss() * gauss();
        out[i] = beta * acc;
      }
      return;
    }
    // 显式：A = e₁，B = ρe₁ + √(1−ρ²)e₂。u = Ma A = (Ma)₁ 列 ×σ；
    // v = Mb B = σ(ρ·g₁ + √(1−ρ²)·g₂)。注意 ρg₁ + √(1−ρ²)g₂ 与 g₁ 同分布且与 u 独立：
    // ρ 在单次实现中存在，在分布中消失。
    const sig = 1 / Math.sqrt(D);
    const sr = Math.sqrt(1 - rho * rho);
    for (let i = i0; i < i1; i++) {
      let acc = 0;
      for (let k = 0; k < H; k++) {
        const u = sig * gauss();
        const v = sig * (rho * gauss() + sr * gauss());
        acc += u * v;
      }
      out[i] = acc;
    }
  }

  /** 方案二/三：固定 Ma、Mb，球面构造精确 cos = ρ 的单位向量对 */
  function fillFixedM(out, i0, i1, H, D, rho, Ma, Mb, gauss, buf) {
    const sr = Math.sqrt(1 - rho * rho);
    const A = buf.A, B = buf.B, Z = buf.Z, u = buf.u, v = buf.v;
    for (let i = i0; i < i1; i++) {
      let na = 0;
      for (let j = 0; j < D; j++) { A[j] = gauss(); na += A[j] * A[j]; }
      na = 1 / Math.sqrt(na);
      for (let j = 0; j < D; j++) A[j] *= na;
      let za = 0;
      for (let j = 0; j < D; j++) { Z[j] = gauss(); za += Z[j] * A[j]; }
      let nz = 0;
      for (let j = 0; j < D; j++) { Z[j] -= za * A[j]; nz += Z[j] * Z[j]; }
      nz = 1 / Math.sqrt(nz);
      for (let j = 0; j < D; j++) B[j] = rho * A[j] + sr * nz * Z[j];
      matVec(Ma, H, D, A, u);
      matVec(Mb, H, D, B, v);
      let s = 0;
      for (let k = 0; k < H; k++) s += u[k] * v[k];
      out[i] = s;
    }
  }

  // ---------- 控件 ----------
  const els = {
    sliderH: document.getElementById('sliderH'),
    inputH: document.getElementById('inputH'),
    sliderD: document.getElementById('sliderD'),
    inputD: document.getElementById('inputD'),
    sliderRho: document.getElementById('sliderRho'),
    inputRho: document.getElementById('inputRho'),
    sliderAlpha: document.getElementById('sliderAlpha'),
    inputAlpha: document.getElementById('inputAlpha'),
    selN: document.getElementById('selN'),
    inputSeed: document.getElementById('inputSeed'),
    btnSeed: document.getElementById('btnSeed'),
    btnResample: document.getElementById('btnResample'),
    chkAnneal: document.getElementById('chkAnneal'),
    chkExplicit: document.getElementById('chkExplicit'),
    chkVg: document.getElementById('chkVg'),
    chkGauss: document.getElementById('chkGauss'),
    chkInst: document.getElementById('chkInst'),
    radioNorm: document.getElementById('radioNorm'),
    radioRaw: document.getElementById('radioRaw'),
    stats: document.getElementById('stats'),
    statsNote: document.getElementById('statsNote'),
  };
  const chart = echarts.init(document.getElementById('chart'));

  const state = {
    scheme: 1, H: 64, D: 256, rho: 0.8, alpha: 0.2,
    nSamples: 30000, seed: 1, anneal: false, explicit: false, nonce: 0,
    normAxis: true, showVg: true, showGauss: false, showInst: true,
  };
  // 缓存：samples 与 matrices 按 key 复用，避免滑杆联动时无谓重算
  const cache = { samplesKey: '', samples: null, matsKey: '', mats: null };
  const ANNEAL_K = 24;

  // ---------- 滑杆映射 ----------
  // H：2^0..2^8 = 1..256；D：2^4..2^10 = 16..1024（对数）
  function sliderToH(t) { return Math.max(1, Math.round(Math.pow(2, (8 * t) / 1000))); }
  function hToSlider(h) { return Math.max(0, Math.min(1000, Math.round((Math.log2(h) / 8) * 1000))); }
  function sliderToD(t) { return Math.max(16, Math.round(Math.pow(2, 4 + (6 * t) / 1000))); }
  function dToSlider(d) { return Math.max(0, Math.min(1000, Math.round(((Math.log2(d) - 4) / 6) * 1000))); }
  function sliderToRho(t) { return -1 + (2 * t) / 1000; }
  function rhoToSlider(r) { return Math.max(0, Math.min(1000, Math.round(((r + 1) / 2) * 1000))); }

  function fmt(x, digits) { return Number(x).toFixed(digits === undefined ? 4 : digits); }
  function fmtAuto(x) {
    if (!isFinite(x)) return String(x);
    if (Math.abs(x) >= 1e4 || (Math.abs(x) < 1e-3 && x !== 0)) return x.toExponential(2);
    return fmt(x, 4);
  }

  // ---------- 采样主流程（分块异步，防 UI 卡死） ----------
  function samplingKey() {
    // 方案一中 ρ 不进分布：故意不放入 key——调 ρ 连样本都不换，正是演示点
    const rhoPart = state.scheme === 1 ? '' : '|r' + state.rho;
    return [state.scheme, state.H, state.D, rhoPart, state.alpha,
      state.nSamples, state.seed, state.anneal, state.explicit, state.nonce].join('|');
  }
  function matsKey() {
    return [state.scheme, state.H, state.D, state.alpha, state.seed].join('|');
  }

  function runSampling(onDone) {
    const n = state.nSamples, H = state.H, D = state.D;
    const out = new Float64Array(n);
    const sampleGauss = makeGaussian(mulberry32(state.seed * 131 + 7 + state.nonce * 65537));
    const anneal = state.scheme !== 1 && state.anneal;
    const segLen = Math.ceil(n / ANNEAL_K);
    const buf = {
      A: new Float64Array(D), B: new Float64Array(D), Z: new Float64Array(D),
      u: new Float64Array(H), v: new Float64Array(H),
    };
    let mats = null;
    let seg = -1;
    if (state.scheme !== 1 && !anneal) {
      const key = matsKey();
      if (cache.matsKey === key && cache.mats) {
        mats = cache.mats;
      } else {
        const gen = genMatrices(H, D, state.alpha, state.scheme, state.seed);
        gen.stats = matrixStats(gen.Ma, gen.Mb, H, D);
        cache.matsKey = key;
        cache.mats = gen;
        mats = gen;
      }
    } else {
      cache.mats = null;
      cache.matsKey = '';
    }
    const flopsPerSample = state.scheme === 1 ? 3 * H : 2 * H * D;
    const chunk = Math.max(200, Math.round(2e7 / flopsPerSample));
    let i = 0;
    els.statsNote.textContent = '采样中 0%…';
    function step() {
      let end = Math.min(n, i + chunk);
      if (anneal) {
        const needSeg = Math.floor(i / segLen);
        if (needSeg !== seg) {
          seg = needSeg;
          mats = genMatrices(H, D, state.alpha, state.scheme, state.seed + seg);
        }
        end = Math.min(end, (seg + 1) * segLen);
      }
      if (state.scheme === 1) {
        fillScheme1(out, i, end, H, D, state.rho, state.explicit, sampleGauss);
      } else {
        fillFixedM(out, i, end, H, D, state.rho, mats.Ma, mats.Mb, sampleGauss, buf);
      }
      i = end;
      els.statsNote.textContent = '采样中 ' + Math.round((100 * i) / n) + '%…';
      if (i < n) setTimeout(step, 0);
      else onDone(out, mats);
    }
    step();
  }

  // ---------- 样本统计 ----------
  function sampleStats(samples) {
    const n = samples.length;
    let mean = 0;
    for (let i = 0; i < n; i++) mean += samples[i];
    mean /= n;
    let m2 = 0, m4 = 0;
    for (let i = 0; i < n; i++) {
      const d = samples[i] - mean;
      const d2 = d * d;
      m2 += d2;
      m4 += d2 * d2;
    }
    m2 /= n;
    m4 /= n;
    return { mean: mean, sd: Math.sqrt(m2), exkurt: m4 / (m2 * m2) - 3 };
  }

  // ---------- 渲染 ----------
  const N_BINS = 121;

  function render(samples, mats) {
    const H = state.H, D = state.D;
    const beta = 1 / D;
    const sig0 = Math.sqrt(H) / D; // VG 标准差 √Hβ
    const scale = state.normAxis ? sig0 : 1;
    const isFixed = state.scheme !== 1;
    const inst = isFixed && mats && mats.stats ? mats.stats : null;

    // 理论矩（原始 s 域）
    const vgVar = T.vgVariance(H, beta);
    let instMean = 0, instVar = vgVar;
    if (inst) {
      instMean = T.scheme2Mean(state.rho, inst.trM, D);
      instVar = T.scheme2Var(state.rho, inst.normF2, inst.normMs2, D);
    }
    const ss = sampleStats(samples);

    // 显示域范围：±xMax；σ 取 VG 与实例的较大者；样本极值截到 9σ 防重尾拉爆
    const sdShow = Math.max(Math.sqrt(vgVar), Math.sqrt(instVar)) / scale;
    let xMax = 5 * sdShow;
    for (let i = 0; i < samples.length; i++) {
      const a = Math.abs(samples[i]) / scale;
      if (a > xMax) xMax = Math.min(a, 9 * sdShow);
    }
    const w = (2 * xMax) / N_BINS;

    // 直方图（显示域等宽 bin，密度高度；step line 填充）
    const counts = new Float64Array(N_BINS);
    for (let i = 0; i < samples.length; i++) {
      const t = samples[i] / scale;
      const b = Math.floor((t + xMax) / w);
      if (b >= 0 && b < N_BINS) counts[b]++;
    }
    const histPts = [[-xMax, 0]];
    for (let b = 0; b < N_BINS; b++) {
      histPts.push([-xMax + (b + 0.5) * w, counts[b] / (samples.length * w)]);
    }
    histPts.push([xMax, 0]);

    // 理论曲线（显示域 400 点；密度变量替换 p_t(t) = scale·f(scale·t)）
    const vgPts = [], gaussPts = [], instPts = [];
    const N_PT = 400;
    for (let i = 0; i <= N_PT; i++) {
      const t = -xMax + (2 * xMax * i) / N_PT;
      const s = t * scale;
      if (state.showVg) vgPts.push([t, scale * T.vgDensity(s, H, beta)]);
      if (state.showGauss) gaussPts.push([t, scale * T.gaussDensity(s, 0, vgVar)]);
      if (state.showInst && inst) {
        instPts.push([t, scale * T.gaussDensity(s, instMean, instVar)]);
      }
    }

    const series = [{
      name: '蒙特卡洛直方图',
      type: 'line', step: 'middle', showSymbol: false,
      lineStyle: { width: 1.2, color: '#0d9488' },
      itemStyle: { color: '#0d9488' },
      areaStyle: { opacity: 0.3 },
      data: histPts,
    }];
    if (state.showVg) {
      series.push({
        name: state.scheme === 1 ? '理论：Variance-Gamma（精确）' : 'VG 参考（方案一曲线）',
        type: 'line', showSymbol: false, smooth: false,
        lineStyle: { width: 2, color: '#2563eb' },
        itemStyle: { color: '#2563eb' },
        data: vgPts,
      });
    }
    if (state.showGauss) {
      series.push({
        name: '高斯近似 N(0, H/D²)',
        type: 'line', showSymbol: false,
        lineStyle: { width: 1.5, color: '#6b7280', type: 'dashed' },
        itemStyle: { color: '#6b7280' },
        data: gaussPts,
      });
    }
    if (state.showInst && inst) {
      series.push({
        name: '实例高斯（本批矩阵）',
        type: 'line', showSymbol: false,
        lineStyle: { width: 2, color: '#dc2626' },
        itemStyle: { color: '#dc2626' },
        data: instPts,
        markLine: {
          symbol: 'none', silent: true,
          lineStyle: { type: 'dotted', color: '#dc2626', width: 1.5 },
          label: { formatter: 'ρ·trM/D = ' + fmtAuto(instMean / scale), fontSize: 11, color: '#dc2626' },
          data: [{ xAxis: instMean / scale }],
        },
      });
    }

    chart.setOption({
      animation: false,
      grid: { left: 60, right: 30, top: 50, bottom: 50 },
      legend: { top: 8 },
      tooltip: { trigger: 'axis', valueFormatter: function (v) { return fmt(v, 4); } },
      xAxis: {
        type: 'value',
        name: state.normAxis ? 't = s / (√H/D)（归一）' : 's = (Ma A)·(Mb B)',
        nameLocation: 'middle', nameGap: 30,
        min: -xMax, max: xMax,
      },
      yAxis: { type: 'value', name: '密度' },
      series: series,
    }, true);

    renderStats(ss, inst, instMean, instVar, vgVar, scale, sdShow);
  }

  function renderStats(ss, inst, instMean, instVar, vgVar, scale, sdShow) {
    const H = state.H, D = state.D;
    const isFixed = state.scheme !== 1;
    const cols = ['样本（N = ' + state.nSamples + '）',
      'VG 理论（方案一精确）', isFixed ? '实例高斯（本批矩阵）' : '高斯近似'];
    const rows = [];
    rows.push(['均值', fmtAuto(ss.mean), '0', isFixed ? fmtAuto(instMean) : '0']);
    rows.push(['标准差', fmtAuto(ss.sd), fmtAuto(Math.sqrt(vgVar)),
      isFixed ? fmtAuto(Math.sqrt(instVar)) : fmtAuto(Math.sqrt(vgVar))]);
    rows.push(['超额峰度', fmt(ss.exkurt, 3), fmt(T.vgExcessKurtosis(H), 3) + '（= 6/H）',
      isFixed ? '≈ 0（大维高斯化）' : '0']);
    if (inst) {
      rows.push(['tr M', '—', '—', fmtAuto(inst.trM) + '（std ~ √(H/D) = ' +
        fmt(Math.sqrt(H / D), 3) + '）']);
      rows.push(['‖M‖²_F', '—', '—', fmtAuto(inst.normF2) + '（≈ H = ' + H + '）']);
      rows.push(['均值信噪比 |E[s]|/σ', '—', '0',
        fmt(Math.abs(instMean) / Math.sqrt(instVar), 3) + '（~ ρ/√D 量级）']);
    }

    let html = '<table class="stats-table"><thead><tr><th>指标</th>';
    for (const c of cols) html += '<th>' + c + '</th>';
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
      '，β = 1/D = ' + fmtAuto(1 / D)];
    if (state.scheme === 1) {
      notes.push('当前 ρ = ' + fmt(state.rho, 2) +
        '：分布与直方图不随 ρ 变化——M = MaᵀMb 双各向同性，A、B 可被各自独立旋转，' +
        '相对夹角不是分布的不变量（调 ρ 时样本甚至没有重抽）');
      if (state.explicit) {
        notes.push('显式模式：v = σ(ρ·g₁ + √(1−ρ²)·g₂) 中 ρ 出现，但该组合与 g₁ 同分布且与 u 独立——ρ 在单次实现中存在，在分布中消失');
      }
    } else if (state.anneal) {
      notes.push('退火平均：' + ANNEAL_K + ' 批随机矩阵各抽一段合并；ρ = 0 时与方案一严格同一分布，' +
        'ρ ≠ 0 时各批均值偏移 ρ·trM/D 随机变号、平均后抵消，直方图仍回落到 VG 曲线');
    } else {
      notes.push('种子 ' + state.seed + '：trM = ' + fmtAuto(inst.trM) +
        '，均值偏移 ρ·trM/D = ' + fmtAuto(instMean) + '（≈ ' +
        fmt(Math.abs(instMean) / Math.sqrt(instVar), 2) + 'σ）；换一批矩阵可看到偏移方向随机翻转');
      if (state.scheme === 3) {
        notes.push('相关矩阵 Mb = αMa + √(1−α²)G（α = ' + fmt(state.alpha, 2) +
          '）：trM ≈ αH = ' + fmt(state.alpha * H, 1) +
          ' 是确定性信号，ρ 效应随 α 稳定显现——对应 QK 学出结构之后');
      }
    }
    els.statsNote.textContent = notes.join('；');
  }

  // ---------- 调度 ----------
  let pending = false;
  function schedule() {
    const key = samplingKey();
    if (key === cache.samplesKey && cache.samples) {
      render(cache.samples, cache.mats);
      return;
    }
    if (pending) return; // 上一次采样进行中；结束后由 onDone 再触发
    pending = true;
    runSampling(function (samples, mats) {
      cache.samplesKey = samplingKey();
      cache.samples = samples;
      cache.mats = state.scheme === 1 || state.anneal ? cache.mats : mats;
      pending = false;
      if (samplingKey() !== cache.samplesKey) {
        schedule(); // 采样期间参数又变了，重跑
      } else {
        render(cache.samples, cache.mats);
      }
    });
  }

  // ---------- 事件 ----------
  function setScheme(sc) {
    state.scheme = sc;
    const fixed = sc !== 1;
    els.inputSeed.disabled = !fixed;
    els.btnSeed.disabled = !fixed;
    els.chkAnneal.disabled = !fixed;
    els.chkExplicit.disabled = fixed;
    els.sliderAlpha.disabled = sc !== 3;
    els.inputAlpha.disabled = sc !== 3;
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
  function setRho(r, fromSlider) {
    state.rho = Math.max(-1, Math.min(1, r));
    if (!fromSlider) els.sliderRho.value = rhoToSlider(state.rho);
    if (document.activeElement !== els.inputRho) els.inputRho.value = fmt(state.rho, 2);
    schedule();
  }
  function setAlpha(a, fromSlider) {
    state.alpha = Math.max(0, Math.min(1, a));
    if (!fromSlider) els.sliderAlpha.value = Math.round(state.alpha * 1000);
    if (document.activeElement !== els.inputAlpha) els.inputAlpha.value = fmt(state.alpha, 2);
    schedule();
  }

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
  els.sliderRho.addEventListener('input', function () {
    setRho(sliderToRho(Number(els.sliderRho.value)), true);
    els.inputRho.value = fmt(state.rho, 2);
  });
  els.inputRho.addEventListener('change', function () {
    const v = Number(els.inputRho.value);
    if (isFinite(v)) setRho(v, false);
  });
  els.sliderAlpha.addEventListener('input', function () {
    setAlpha(Number(els.sliderAlpha.value) / 1000, true);
    els.inputAlpha.value = fmt(state.alpha, 2);
  });
  els.inputAlpha.addEventListener('change', function () {
    const v = Number(els.inputAlpha.value);
    if (isFinite(v)) setAlpha(v, false);
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
    // 矩阵种子不动（同一批矩阵），只换采样流：nonce 进入 samplingKey 与采样种子
    state.nonce += 1;
    schedule();
  });
  els.chkAnneal.addEventListener('change', function () {
    state.anneal = els.chkAnneal.checked;
    schedule();
  });
  els.chkExplicit.addEventListener('change', function () {
    state.explicit = els.chkExplicit.checked;
    schedule();
  });
  els.chkVg.addEventListener('change', function () {
    state.showVg = els.chkVg.checked;
    schedule();
  });
  els.chkGauss.addEventListener('change', function () {
    state.showGauss = els.chkGauss.checked;
    schedule();
  });
  els.chkInst.addEventListener('change', function () {
    state.showInst = els.chkInst.checked;
    schedule();
  });
  els.radioNorm.addEventListener('change', function () {
    state.normAxis = true;
    schedule();
  });
  els.radioRaw.addEventListener('change', function () {
    state.normAxis = false;
    schedule();
  });
  window.addEventListener('resize', function () { chart.resize(); });

  // ---------- 初始化 ----------
  els.sliderH.value = hToSlider(state.H);
  els.inputH.value = state.H;
  els.sliderD.value = dToSlider(state.D);
  els.inputD.value = state.D;
  els.sliderRho.value = rhoToSlider(state.rho);
  els.inputRho.value = fmt(state.rho, 2);
  els.sliderAlpha.value = Math.round(state.alpha * 1000);
  els.inputAlpha.value = fmt(state.alpha, 2);
  els.inputSeed.value = state.seed;
  setScheme(1);
})();
