/**
 * 交互与渲染：滑杆/输入框状态同步，调用 Theory 生成曲线数据，ECharts 绘图。
 * 依赖：echarts（CDN）、Theory（js/theory.js）。
 */
(function () {
  'use strict';

  const T = window.Theory;

  const N_MIN = 64;
  const N_MAX = 8192;

  const els = {
    sliderN: document.getElementById('sliderN'),
    inputN: document.getElementById('inputN'),
    sliderK: document.getElementById('sliderK'),
    inputK: document.getElementById('inputK'),
    chkTwoSided: document.getElementById('chkTwoSided'),
    stats: document.getElementById('stats'),
  };

  const chart1 = echarts.init(document.getElementById('chart1'));
  const chart2 = echarts.init(document.getElementById('chart2'));

  const state = { N: 1024, K: 128, twoSided: false };

  // ---- 滑杆 <-> 数值：指数映射 ----
  function sliderToValue(s, lo, hi) {
    return Math.round(
      Math.exp(Math.log(lo) + (s / 1000) * (Math.log(hi) - Math.log(lo)))
    );
  }
  function valueToSlider(v, lo, hi) {
    return Math.round(
      (1000 * (Math.log(v) - Math.log(lo))) / (Math.log(hi) - Math.log(lo))
    );
  }

  function syncControls() {
    els.inputN.value = state.N;
    els.sliderN.value = valueToSlider(state.N, N_MIN, N_MAX);
    els.inputK.max = state.N;
    els.inputK.value = state.K;
    els.sliderK.value = valueToSlider(state.K, 2, state.N);
    els.chkTwoSided.checked = state.twoSided;
  }

  // ---- 曲线 1：max ρ ~ K ----
  function renderChart1() {
    const { N, K, twoSided } = state;
    const samples = 160;
    const firstOrder = [];
    const gumbelMedian = [];
    const betaMedian = [];
    for (let i = 0; i <= samples; i++) {
      const k = Math.exp(
        Math.log(2) + (i / samples) * (Math.log(N) - Math.log(2))
      );
      firstOrder.push([k, T.firstOrderMean(N, k)]);
      gumbelMedian.push([k, T.maxDotQuantile(0.5, N, k, twoSided)]);
      betaMedian.push([k, T.maxDotQuantileBeta(0.5, N, k, twoSided)]);
    }

    chart1.setOption(
      {
        title: { text: '最大点乘的中位数水平（N = ' + N + '）', left: 'center', textStyle: { fontSize: 14 } },
        // 系列顺序：一阶近似(橙)、F^M(蓝)、Gumbel(黄)
        color: ['#ff7f0e', '#2563eb', '#eab308'],
        tooltip: {
          trigger: 'axis',
          valueFormatter: (v) => (typeof v === 'number' ? v.toFixed(4) : v),
        },
        legend: { bottom: 0 },
        grid: { left: 60, right: 30, top: 50, bottom: 60 },
        xAxis: { type: 'log', name: 'K（向量个数）', min: 2, max: N },
        yAxis: { type: 'value', name: 'max ρ', min: 0 },
        series: [
          {
            name: '一阶近似 2√(lnK/N)',
            type: 'line',
            showSymbol: false,
            smooth: true,
            data: firstOrder,
          },
          {
            name: 'F^M 中位数（beta 幂次，全 K 适用）',
            type: 'line',
            showSymbol: false,
            smooth: true,
            data: betaMedian,
            markLine: {
              silent: true,
              symbol: 'none',
              lineStyle: { type: 'dashed', color: '#999' },
              label: { formatter: 'K = ' + K, position: 'insideEndTop' },
              data: [{ xAxis: K }],
            },
          },
          {
            name: 'Gumbel 中位数（渐近）',
            type: 'line',
            showSymbol: false,
            smooth: true,
            lineStyle: { type: 'dashed' },
            data: gumbelMedian,
          },
        ],
      },
      { notMerge: true }
    );
  }

  // ---- 曲线 2：给定 (N,K) 的密度 ----
  function renderChart2() {
    const { N, K, twoSided } = state;

    // 横轴范围：覆盖 F^M 分布的 [0.1%, 99.9%] 分位区间与单对密度可见范围（≈6σ）
    const q001 = T.maxDotQuantileBeta(0.001, N, K, twoSided);
    const q999 = Math.max(
      T.maxDotQuantileBeta(0.999, N, K, twoSided),
      T.maxDotQuantile(0.999, N, K, twoSided)
    );
    // 双侧 max|ρ| 无负支撑；单侧小 K 时负半轴有可观质量，自动扩展
    const xLo = twoSided ? 0 : Math.min(0, q001 * 1.1);
    const xHi = Math.min(1, Math.max(q999, 6 / Math.sqrt(N)) * 1.05);

    const samples = 400;
    const maxDotGumbel = [];
    const maxDotBeta = [];
    const singlePair = [];
    for (let i = 0; i <= samples; i++) {
      const r = xLo + (i / samples) * (xHi - xLo);
      maxDotGumbel.push([r, T.maxDotDensity(r, N, K, twoSided)]);
      maxDotBeta.push([r, T.maxDotDensityBeta(r, N, K, twoSided)]);
      // 双侧模式下对比基线应为单个 |ρ| 的密度：负半轴折叠到正半轴，高度翻倍
      const g = T.pairDotDensity(r, N);
      singlePair.push([r, twoSided ? 2 * g : g]);
    }

    chart2.setOption(
      {
        title: {
          text: '密度曲线（N = ' + N + ', K = ' + K + '）',
          left: 'center',
          textStyle: { fontSize: 14 },
        },
        // 系列顺序：F^M(蓝)、Gumbel(黄)、单对(绿)
        color: ['#2563eb', '#eab308', '#2ca02c'],
        tooltip: {
          trigger: 'axis',
          valueFormatter: (v) => (typeof v === 'number' ? v.toFixed(3) : v),
        },
        legend: { bottom: 0 },
        grid: { left: 60, right: 30, top: 50, bottom: 60 },
        xAxis: { type: 'value', name: 'ρ（点乘）', min: xLo, max: xHi },
        yAxis: { type: 'value', name: '密度', min: 0 },
        series: [
          {
            name: 'max ρ 密度（F^M，beta 幂次）',
            type: 'line',
            showSymbol: false,
            smooth: true,
            areaStyle: { opacity: 0.08 },
            data: maxDotBeta,
          },
          {
            name: 'max ρ 密度（Gumbel 渐近）',
            type: 'line',
            showSymbol: false,
            smooth: true,
            lineStyle: { type: 'dashed' },
            data: maxDotGumbel,
          },
          {
            name: twoSided
              ? '单对 |ρ| 密度（精确 beta ×2 折叠）'
              : '单对 ρ 密度（精确 beta）',
            type: 'line',
            showSymbol: false,
            smooth: true,
            data: singlePair,
          },
        ],
      },
      { notMerge: true }
    );

    const med = T.maxDotQuantileBeta(0.5, N, K, twoSided);
    const medGumbel = T.maxDotQuantile(0.5, N, K, twoSided);
    const first = T.firstOrderMean(N, K);
    const angleDeg = ((Math.acos(Math.max(-1, Math.min(1, med))) * 180) / Math.PI).toFixed(2);
    els.stats.innerHTML =
      'max ρ 中位数：<strong class="legend-note note-blue">F^M ≈ ' + med.toFixed(4) + '</strong>' +
      '，<strong class="legend-note note-yellow">Gumbel ≈ ' + medGumbel.toFixed(4) + '</strong>' +
      '；<strong class="legend-note note-orange">一阶近似（众数口径）≈ ' + first.toFixed(4) + '</strong>' +
      '；对应最小夹角 ≈ <strong class="legend-note note-gray">' + angleDeg + '°</strong>。' +
      '单对 ρ 的散布 σ ≈ 1/√N ≈ ' + (1 / Math.sqrt(N)).toFixed(4) + '。';
  }

  function render() {
    syncControls();
    renderChart1();
    renderChart2();
  }

  // ---- 事件 ----
  els.sliderN.addEventListener('input', () => {
    state.N = sliderToValue(Number(els.sliderN.value), N_MIN, N_MAX);
    state.K = Math.min(state.K, state.N);
    render();
  });
  els.inputN.addEventListener('change', () => {
    let v = Math.round(Number(els.inputN.value));
    if (!isFinite(v)) v = state.N;
    state.N = Math.max(N_MIN, Math.min(N_MAX, v));
    state.K = Math.min(state.K, state.N);
    render();
  });
  els.sliderK.addEventListener('input', () => {
    state.K = sliderToValue(Number(els.sliderK.value), 2, state.N);
    render();
  });
  els.inputK.addEventListener('change', () => {
    let v = Math.round(Number(els.inputK.value));
    if (!isFinite(v)) v = state.K;
    state.K = Math.max(2, Math.min(state.N, v));
    render();
  });
  els.chkTwoSided.addEventListener('change', () => {
    state.twoSided = els.chkTwoSided.checked;
    render();
  });
  window.addEventListener('resize', () => {
    chart1.resize();
    chart2.resize();
  });

  render();
})();
