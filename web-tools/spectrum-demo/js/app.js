/**
 * spectrum-demo 的 UI 与渲染层：c = H/D 与 λ = σ_w²D 滑杆、σ²/σ 轴切换、ECharts 曲线。
 * 数学公式全部在 theory.js（window.SpecTheory）。
 */
(function () {
  'use strict';
  const T = window.SpecTheory;

  const els = {
    slider: document.getElementById('sliderC'),
    input: document.getElementById('inputC'),
    axisSq: document.getElementById('radioSq'),
    axisSigma: document.getElementById('radioSigma'),
    chkMp: document.getElementById('chkMp'),
    chkProd: document.getElementById('chkProd'),
    chkAB: document.getElementById('chkAB'),
    sliderLam: document.getElementById('sliderLam'),
    inputLam: document.getElementById('inputLam'),
    stats: document.getElementById('stats'),
    statsNote: document.getElementById('statsNote'),
  };
  const chart = echarts.init(document.getElementById('chart'));

  // λ = σ_w²D：单矩阵谱的均值（乘积谱均值 λ²），滑杆直接选择它，不再显式出现 D
  const state = { c: 0.07143, lam: 1, sigmaAxis: false, showMp: true, showProd: true, showAB: true };

  // c 滑杆对数映射：t ∈ [0,1000] -> c = 10^(-3 + 3t/1000) ∈ [1e-3, 1]
  function sliderToC(t) { return Math.pow(10, -3 + (3 * t) / 1000); }
  function cToSlider(c) {
    return Math.max(0, Math.min(1000, Math.round(((Math.log10(c) + 3) / 3) * 1000)));
  }

  // λ 滑杆对数映射：t ∈ [0,1000] -> λ = 10^(-4 + 8t/1000) ∈ [1e-4, 1e4]
  // （λ 跨 8 个数量级，线性滑杆在小 λ 端无分辨率，故仍用对数；默认 λ=1 在正中 t=500）
  function sliderToLam(t) { return Math.pow(10, -4 + (8 * t) / 1000); }
  function lamToSlider(l) {
    return Math.max(0, Math.min(1000, Math.round(((Math.log10(l) + 4) / 8) * 1000)));
  }

  function logspace(a, b, n) {
    const out = new Array(n);
    const r = Math.log(b / a);
    for (let i = 0; i < n; i++) out[i] = a * Math.exp((r * i) / (n - 1));
    return out;
  }

  function fmt(x, digits) {
    return Number(x).toFixed(digits === undefined ? 4 : digits);
  }

  // c 的显示用 4 位有效数字：c 跨 1e-3 ~ 1，固定小数位会在小 c 处丢精度
  // （如 512/7168 = 0.07143 用 toFixed(4) 会截成 0.0714）
  function fmtC(c) {
    return Number(c).toPrecision(4);
  }

  // 大数紧凑显示（未归一模式下 λ² 可达 1e6 量级）
  function fmtAuto(x) {
    if (x >= 1e5) return x.toExponential(2);
    return fmt(x, 4);
  }

  // λ 的显示同 c：4 位有效数字（跨 1e-4 ~ 1e4）
  function fmtLam(l) {
    return Number(l).toPrecision(4);
  }

  // σ 轴（奇异值本身）均值/方差的数值积分，仅供乘积/ABᵀ 谱使用
  // （单矩阵列直接用 T.mpSigmaMean 的 ₂F₁ 精确闭式，不走这里）：
  // p_σ(s) = 2s·p_σ²(s²)，对数网格 4000 段梯形
  function sigmaMoments(kind, c) {
    const sup = kind === 'mp' ? T.mpSupport(c)
      : kind === 'product' ? T.productSupport(c) : T.abNormSupport(c);
    const dens = kind === 'mp' ? T.mpDensity
      : kind === 'product' ? T.productDensity : T.abNormDensity;
    const lo = sup[0] > 0 ? sup[0] * 0.99 : 1e-7;
    const hi = sup[1] * 0.999999;
    const n = 4000;
    let m0 = 0, m1 = 0, m2 = 0;
    let sPrev = lo;
    let pPrev = 2 * lo * dens(lo * lo, c);
    for (let i = 1; i <= n; i++) {
      const s = lo * Math.pow(hi / lo, i / n);
      const p = 2 * s * dens(s * s, c);
      const ds = s - sPrev;
      m0 += 0.5 * (p + pPrev) * ds;
      m1 += 0.5 * (s * p + sPrev * pPrev) * ds;
      m2 += 0.5 * (s * s * p + sPrev * sPrev * pPrev) * ds;
      sPrev = s;
      pPrev = p;
    }
    const mean = m1 / m0;
    return [mean, m2 / m0 - mean * mean];
  }

  function render() {
    const c = state.c;
    const pSup = T.productSupport(c);
    const mSup = T.mpSupport(c);
    const abSup = T.abNormSupport(c); // ABᵀ（H×H 摆法）÷c 归一的支撑

    // 元素方差尺度：奇异值随矩阵倍数线性缩放（σ(αM) = α·σ(M)），
    // σ² 谱横轴 = 归一谱 × λ；单矩阵 λ1 = σ_w²D（即其均值），乘积 λ2 = λ1²（收缩维数不同）。
    // 未归一时 ABᵀ 也按原始形态（均值 c·λ2）显示
    const lam1 = state.lam;
    const lam2 = lam1 * lam1;
    // λ = 1（即 σ_w² = 1/D）为归一模式：ABᵀ 显示 ÷c 归一版
    const rawAB = Math.abs(lam1 - 1) > 1e-9;
    const scM = lam1;
    const scP = lam2;
    const scA = rawAB ? c * lam2 : 1;

    const showM = state.showMp, showP = state.showProd, showA = state.showAB;
    const anyCurve = showM || showP || showA;

    // σ² 轴对数加密网格（网格画在归一坐标上，各曲线按自身 λ 伸缩后显示）。
    // c=1 时分布在 x→0 发散、ABᵀ 谱左端恒贴 0（按 x^(-1/2) 发散），
    // 这两种情况网格起点取 1e-3 以展示近 0 行为
    const xEnd = (anyCurve
      ? Math.max(showP ? pSup[1] : 0, showM ? mSup[1] : 0, showA ? abSup[1] : 0)
      : 1) * 1.04;
    const xStart = (c >= 0.9999 && anyCurve) || showA
      ? 1e-3
      : Math.min(showP ? pSup[0] : Infinity, showM ? mSup[0] : Infinity) * 0.9;
    const xs = logspace(xStart, xEnd, 900);

    const mpPts = [];
    const prodPts = [];
    const abPts = [];
    // 纵轴截断只看体部（按实际显示值），避免奇点顶爆。
    // σ 轴的发散更温和（s^(-1/3)），阈值收紧到 0.005 以展示近 0 段；
    // σ² 轴维持 0.05（x^(-2/3) 发散太强，再低会压扁体部）
    const capFrom = state.sigmaAxis ? 0.005 : 0.05;
    let cap = 0;
    for (let i = 0; i < xs.length; i++) {
      const x = xs[i];
      let ym = showM ? T.mpDensity(x, c) / scM : 0;
      let yp = showP ? T.productDensity(x, c) / scP : 0;
      let ya = showA ? T.abNormDensity(x, c) / scA : 0;
      let xm = x * scM, xp = x * scP, xa = x * scA;
      if (state.sigmaAxis) {
        const sm = Math.sqrt(xm), sp = Math.sqrt(xp), sa = Math.sqrt(xa);
        ym = 2 * sm * ym;
        yp = 2 * sp * yp;
        ya = 2 * sa * ya;
        if (showM) mpPts.push([sm, ym]);
        if (showP) prodPts.push([sp, yp]);
        if (showA) abPts.push([sa, ya]);
      } else {
        if (showM) mpPts.push([xm, ym]);
        if (showP) prodPts.push([xp, yp]);
        if (showA) abPts.push([xa, ya]);
      }
      if (x >= capFrom) cap = Math.max(cap, ym, yp, ya);
    }
    // 纵轴截断：只看归一坐标 x ≥ 0.05 的体部；向上取整到 1 位有效数字
    // （未归一模式下密度可至 1e-4 量级，固定 0.1 粒度会把曲线压扁）
    let yMax = 1;
    if (cap > 0) {
      const raw = cap * 1.25;
      const unit = Math.pow(10, Math.floor(Math.log10(raw)));
      yMax = Math.ceil(raw / unit) * unit;
    }

    const edgeFmt = function (prm) { return fmt(prm.value, 3); };
    const toAxis = function (v) { return state.sigmaAxis ? Math.sqrt(v) : v; };
    function edgeLines(sup, color) {
      const data = [];
      // 左端为 0（c=1，或 ABᵀ 任意 c）时不画，贴在纵轴上没有信息量
      if (sup[0] > 1e-12) data.push({ xAxis: toAxis(sup[0]) });
      data.push({ xAxis: toAxis(sup[1]) });
      return {
        symbol: 'none', silent: true,
        lineStyle: { type: 'dotted', color: color, width: 1 },
        label: { formatter: edgeFmt, fontSize: 11, color: color },
        data: data,
      };
    }

    // notMerge：系列数量随 AB 开关变化，合并模式会残留旧系列
    chart.setOption({
      animation: false,
      grid: { left: 60, right: 30, top: 50, bottom: 50 },
      legend: { top: 8 },
      tooltip: {
        trigger: 'axis',
        valueFormatter: function (v) { return fmt(v, 4); },
      },
      xAxis: {
        type: 'value',
        name: state.sigmaAxis ? 's = σ（奇异值）' : 'x = σ²（平方奇异值）',
        nameLocation: 'middle',
        nameGap: 30,
      },
      yAxis: { type: 'value', name: '密度', max: yMax },
      series: (function () {
        const s = [];
        if (showM) {
          s.push({
            name: 'A（单矩阵）：MP_c',
            type: 'line', showSymbol: false, smooth: true,
            lineStyle: { width: 2, color: '#2563eb' },
            itemStyle: { color: '#2563eb' },
            data: mpPts,
            markLine: edgeLines([mSup[0] * scM, mSup[1] * scM], '#2563eb'),
          });
        }
        if (showP) {
          s.push({
            name: 'AᵀB（乘积）：MP_c ⊠ MP_c',
            type: 'line', showSymbol: false, smooth: true,
            lineStyle: { width: 2, color: '#dc2626' },
            itemStyle: { color: '#dc2626' },
            data: prodPts,
            markLine: edgeLines([pSup[0] * scP, pSup[1] * scP], '#dc2626'),
          });
        }
        if (showA) {
          s.push({
            name: rawAB ? 'ABᵀ（H×H，原始）：c·(MP_c ⊠ MP_1)'
              : 'ABᵀ（H×H，÷c 归一）：MP_c ⊠ MP_1',
            type: 'line', showSymbol: false, smooth: true,
            lineStyle: { width: 2, color: '#7c3aed' },
            itemStyle: { color: '#7c3aed' },
            data: abPts,
            markLine: edgeLines([abSup[0] * scA, abSup[1] * scA], '#7c3aed'),
          });
        }
        return s;
      })(),
    }, true);

    // 统计表：列随曲线显隐开关同步；行随 σ²/σ 轴同步变换
    const isC1 = c >= 0.9999;
    const onSq = !state.sigmaAxis;
    const tv = function (v) { return onSq ? v : Math.sqrt(v); };
    // 均值/方差：归一模式为精确公式；未归一模式按 λ / λ² 伸缩；σ 轴数值积分
    let meanDisp, varDisp, meanLabel = '均值', varLabel = '方差';
    if (onSq) {
      meanDisp = rawAB
        ? ['λ = ' + fmtAuto(lam1), 'λ² = ' + fmtAuto(lam2), 'cλ² = ' + fmtAuto(c * lam2)]
        : ['1', '1', '1'];
      varDisp = rawAB
        ? ['cλ² = ' + fmtAuto(c * lam1 * lam1), '2cλ⁴ = ' + fmtAuto(2 * c * lam2 * lam2),
          'c²(1+c)λ⁴ = ' + fmtAuto(c * c * (1 + c) * lam2 * lam2)]
        : ['c = ' + fmt(T.mpVariance(c), 4),
          '2c = ' + fmt(T.productVariance(c), 4),
          '1+c = ' + fmt(1 + c, 4)];
    } else {
      // σ 轴：单矩阵列用精确闭式 E[σ] = √λ·₂F₁(−1/2, 1/2; 2; c)、Var = λ − E[σ]²
      // （E[σ²] = E[x] = λ 平凡）；乘积/ABᵀ 列无干净形式，数值积分
      const f = T.mpSigmaMean(c);
      const mm = [f, 1 - f * f];
      const pm = sigmaMoments('product', c);
      const am = sigmaMoments('ab', c);
      meanDisp = [fmtAuto(Math.sqrt(scM) * mm[0]), fmtAuto(Math.sqrt(scP) * pm[0]),
        fmtAuto(Math.sqrt(scA) * am[0])];
      varDisp = [fmtAuto(scM * mm[1]), fmtAuto(scP * pm[1]), fmtAuto(scA * am[1])];
      meanLabel = '均值（σ 轴）';
      varLabel = '方差（σ 轴）';
    }
    // 左端点与近 0 行为。σ² 轴：c<1 时 MP/乘积有空隙、ABᵀ 按 x^(-1/2) 发散，
    // c=1 时 MP 按 x^(-1/2)、乘积/ABᵀ 按 x^(-2/3) 发散；
    // σ 轴（p_σ(s) = 2s·p_σ²(s²)）：MP 与 ABᵀ 在 0 处变为有限值，乘积按 s^(-1/3) 发散
    const leftDisp = onSq ? [
      isC1 ? '0（x^(-1/2) 发散）' : fmtAuto(mSup[0] * scM) + '（有空隙）',
      isC1 ? '0（x^(-2/3) 发散）' : fmtAuto(pSup[0] * scP) + '（有空隙）',
      isC1 ? '0（x^(-2/3) 发散）' : '0（~1/(π√((1−c)x)) 发散）',
    ] : [
      isC1 ? '0（p(0⁺) = 2/π ≈ 0.637）' : fmtAuto(Math.sqrt(mSup[0] * scM)) + '（有空隙）',
      isC1 ? '0（s^(-1/3) 发散）' : fmtAuto(Math.sqrt(pSup[0] * scP)) + '（有空隙）',
      isC1 ? '0（s^(-1/3) 发散）'
        : '0（p(0⁺) = 2/(π√(1−c)) ≈ ' + fmt(2 / (Math.PI * Math.sqrt((1 - c) * scA)), 3) + '）',
    ];
    // 右端点：两轴下密度都按 √(端点−自变量) 平方根式归零
    const rightDisp = [fmtAuto(tv(mSup[1] * scM)), fmtAuto(tv(pSup[1] * scP)),
      fmtAuto(tv(abSup[1] * scA))];
    const smaxDisp = [fmtAuto(Math.sqrt(mSup[1] * scM)), fmtAuto(Math.sqrt(pSup[1] * scP)),
      fmtAuto(Math.sqrt(abSup[1] * scA))];

    const vis = [showM, showP, showA];
    const names = ['A（单矩阵）', 'AᵀB（D×D 乘积）',
      rawAB ? 'ABᵀ（H×H，原始）' : 'ABᵀ（H×H，÷c 归一）'];
    const idx = [0, 1, 2].filter(function (i) { return vis[i]; });
    const pick = function (arr) { return idx.map(function (i) { return arr[i]; }); };

    const rows = [
      ['极限律'].concat(pick(['MP_c', 'MP_c ⊠ MP_c', rawAB ? 'c·(MP_c ⊠ MP_1)' : 'MP_c ⊠ MP_1'])),
      [meanLabel].concat(pick(meanDisp)),
      [varLabel].concat(pick(varDisp)),
      [onSq ? '左端点 x₋（x→0⁺）' : '左端点 s₋（s→0⁺）'].concat(pick(leftDisp)),
      [onSq ? '右端点 x₊（√ 式归零）' : '右端点 s₊（√ 式归零）'].concat(pick(rightDisp)),
    ];
    // σ_max 在 σ² 轴是派生量；σ 轴上与右端点重复，不显示
    if (onSq) rows.push(['σ_max = √x₊'].concat(pick(smaxDisp)));

    let html = '<table class="stats-table"><thead><tr><th>指标</th>';
    for (const i of idx) html += '<th>' + names[i] + '</th>';
    html += '</tr></thead><tbody>';
    for (const r of rows) {
      html += '<tr>';
      for (const cell of r) html += '<td>' + cell + '</td>';
      html += '</tr>';
    }
    html += '</tbody></table>';
    els.stats.innerHTML = html;

    // 表下附注
    const notes = ['c = H/D = ' + fmtC(c)];
    if (rawAB) {
      notes.push('λ = σ_w²D = ' + fmtAuto(lam1) +
        '：横轴整体伸缩，单矩阵 ×λ、乘积 ×λ² = ' + fmtAuto(lam2) +
        '——形状不变，但三条曲线伸缩倍数不同（λ vs λ²），未归一时量级悬殊');
    } else if (state.showAB) {
      notes.push('ABᵀ 列为 ÷c 归一后的曲线：原始 σ²(ABᵀ) 均值 c = ' + fmtC(c) +
        '、方差 c²(1+c) = ' + fmt(T.abVariance(c), 5) +
        '、支撑 [0, ' + fmt(c * abSup[1], 3) + ']');
    }
    if (state.sigmaAxis) {
      notes.push('σ 轴均值/方差：A 列为精确值 √λ·₂F₁(−1/2, 1/2; 2; c)（Var = λ − E[σ]²），乘积/ABᵀ 列为数值积分');
    }
    if (c >= 0.9999) {
      notes.push('c = 1 时乘积谱即 Fuss-Catalan FC₂：支撑 [0, 27/4]，x→0 按 x^(-2/3) 发散（纵轴已截断），此时与 ABᵀ 曲线重合');
    }
    els.statsNote.textContent = notes.join('；');
  }

  function setC(c, fromSlider) {
    state.c = Math.max(1e-3, Math.min(1, c));
    if (!fromSlider) els.slider.value = cToSlider(state.c);
    if (document.activeElement !== els.input) els.input.value = fmtC(state.c);
    render();
  }

  els.slider.addEventListener('input', function () {
    setC(sliderToC(Number(els.slider.value)), true);
    els.input.value = fmtC(state.c);
  });
  els.input.addEventListener('change', function () {
    const v = Number(els.input.value);
    if (isFinite(v) && v > 0) setC(v, false);
  });

  function setLam(l, fromSlider) {
    state.lam = Math.max(1e-4, Math.min(1e4, l));
    if (!fromSlider) els.sliderLam.value = lamToSlider(state.lam);
    if (document.activeElement !== els.inputLam) els.inputLam.value = fmtLam(state.lam);
    render();
  }

  els.sliderLam.addEventListener('input', function () {
    setLam(sliderToLam(Number(els.sliderLam.value)), true);
    els.inputLam.value = fmtLam(state.lam);
  });
  els.inputLam.addEventListener('change', function () {
    const v = Number(els.inputLam.value);
    if (isFinite(v) && v > 0) setLam(v, false);
  });
  document.querySelectorAll('[data-lam]').forEach(function (btn) {
    btn.addEventListener('click', function () { setLam(Number(btn.dataset.lam), false); });
  });
  els.axisSq.addEventListener('change', function () {
    state.sigmaAxis = false; render();
  });
  els.axisSigma.addEventListener('change', function () {
    state.sigmaAxis = true; render();
  });
  els.chkMp.addEventListener('change', function () {
    state.showMp = els.chkMp.checked; render();
  });
  els.chkProd.addEventListener('change', function () {
    state.showProd = els.chkProd.checked; render();
  });
  els.chkAB.addEventListener('change', function () {
    state.showAB = els.chkAB.checked; render();
  });
  document.querySelectorAll('[data-preset]').forEach(function (btn) {
    btn.addEventListener('click', function () { setC(Number(btn.dataset.preset), false); });
  });
  window.addEventListener('resize', function () { chart.resize(); });

  els.slider.value = cToSlider(state.c);
  els.input.value = fmtC(state.c);
  els.sliderLam.value = lamToSlider(state.lam);
  els.inputLam.value = fmtLam(state.lam);
  render();
})();
