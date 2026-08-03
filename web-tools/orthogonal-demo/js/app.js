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
    for (let i = 0; i <= samples; i++) {
      const k = Math.exp(
        Math.log(2) + (i / samples) * (Math.log(N) - Math.log(2))
      );
      firstOrder.push([k, T.firstOrderMean(N, k)]);
      gumbelMedian.push([k, T.maxDotQuantile(0.5, N, k, twoSided)]);
    }

    chart1.setOption(
      {
        title: { text: '最大点乘的期望水平（N = ' + N + '）', left: 'center', textStyle: { fontSize: 14 } },
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
            name: 'Gumbel 中位数',
            type: 'line',
            showSymbol: false,
            smooth: true,
            data: gumbelMedian,
            markLine: {
              silent: true,
              symbol: 'none',
              lineStyle: { type: 'dashed', color: '#999' },
              label: { formatter: 'K = ' + K, position: 'insideEndTop' },
              data: [{ xAxis: K }],
            },
          },
        ],
      },
      { notMerge: true }
    );
  }

  // ---- 曲线 2：给定 (N,K) 的密度 ----
  function renderChart2() {
    const { N, K, twoSided } = state;

    // 横轴上限：Gumbel 0.999 分位数与单对密度可见范围（≈6σ, σ≈1/√N）取大者
    const q999 = T.maxDotQuantile(0.999, N, K, twoSided);
    const xHi = Math.min(1, Math.max(q999, 6 / Math.sqrt(N)) * 1.05);

    const samples = 320;
    const maxDot = [];
    const singlePair = [];
    for (let i = 0; i <= samples; i++) {
      const r = (i / samples) * xHi;
      maxDot.push([r, T.maxDotDensity(r, N, K, twoSided)]);
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
        tooltip: {
          trigger: 'axis',
          valueFormatter: (v) => (typeof v === 'number' ? v.toFixed(3) : v),
        },
        legend: { bottom: 0 },
        grid: { left: 60, right: 30, top: 50, bottom: 60 },
        xAxis: { type: 'value', name: 'ρ（点乘）', min: 0, max: xHi },
        yAxis: { type: 'value', name: '密度', min: 0 },
        series: [
          {
            name: 'max ρ 密度（Gumbel 渐近）',
            type: 'line',
            showSymbol: false,
            smooth: true,
            areaStyle: { opacity: 0.08 },
            data: maxDot,
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

    const med = T.maxDotQuantile(0.5, N, K, twoSided);
    const meanApprox = T.gumbelMeanApprox(N, K, twoSided);
    const first = T.firstOrderMean(N, K);
    const angleDeg = ((Math.acos(Math.min(1, med)) * 180) / Math.PI).toFixed(2);
    els.stats.textContent =
      'max ρ：中位数 ≈ ' + med.toFixed(4) +
      '，近似均值 ≈ ' + meanApprox.toFixed(4) +
      '，一阶近似 ≈ ' + first.toFixed(4) +
      '；对应最小夹角 ≈ ' + angleDeg + '°。' +
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
