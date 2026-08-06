/**
 * UI 层：控件状态、采样驱动、ECharts 渲染。
 *
 * 两个面板：
 *   1. 按元素运算：5 条理论密度曲线同图对比 + 选中运算的蒙特卡洛直方图；
 *   2. 求和类：点积（双方随机 / 一方固定）与长度平方的直方图 + 精确理论曲线；
 *      两种点积模式的方差对比即回答"σ²=1/D 时点积方差是否为 1"。
 */
(function () {
  'use strict';

  var T = window.NoiseTheory;
  var S = window.NoiseSampler;

  // ---------- 小工具 ----------
  function $(id) {
    return document.getElementById(id);
  }

  /** 紧凑数字格式：常规范围定点，极端范围科学计数 */
  function fmt(x) {
    if (!isFinite(x)) return String(x);
    var a = Math.abs(x);
    if (a !== 0 && (a < 1e-4 || a >= 1e5)) return x.toExponential(2);
    return Number(x.toPrecision(4)).toString();
  }

  // ---------- 控件 ----------
  var el = {
    sliderD: $('sliderD'),
    inputD: $('inputD'),
    inputSigma2: $('inputSigma2'),
    inputC: $('inputC'),
    inputN: $('inputN'),
    chkSeed: $('chkSeed'),
    inputSeed: $('inputSeed'),
    btnResample: $('btnResample'),
    elementStats: $('elementStats'),
    sumStats: $('sumStats'),
  };
  var presetButtons = Array.prototype.slice.call(
    document.querySelectorAll('.preset-group button')
  );

  // ---------- 状态 ----------
  var state = {
    D: 16,
    sigma2preset: '1/D', // '1' | '1/D' | '1/sqrtD' | null（自定义）
    sigma2: 1 / 16,
    c: 2,
    N: 200000,
    useSeed: true,
    seed: 42,
    elementOp: 'product',
    sumMode: 'dotRandom',
  };

  var pairs = null; // 按元素运算共享的样本对 {x, y}
  var sumCache = {}; // modeId -> { samples, M }
  var charts = {};

  // ---------- σ² 预设 ----------
  function sigma2FromPreset(preset, D) {
    if (preset === '1') return 1;
    if (preset === '1/D') return 1 / D;
    if (preset === '1/sqrtD') return 1 / Math.sqrt(D);
    return null;
  }

  function refreshSigma2() {
    if (state.sigma2preset) {
      state.sigma2 = sigma2FromPreset(state.sigma2preset, state.D);
    }
    el.inputSigma2.value = Number(state.sigma2.toPrecision(6));
    presetButtons.forEach(function (b) {
      b.classList.toggle('active', b.dataset.preset === state.sigma2preset);
    });
  }

  // ---------- 采样 ----------
  function currentSeed(salt) {
    var base = state.useSeed ? state.seed : (Math.random() * 1e9) | 0;
    return (base + salt) >>> 0;
  }

  function resample() {
    var sigma = Math.sqrt(state.sigma2);
    pairs = S.samplePairs(S.makeRng(currentSeed(0)), state.N, sigma);
    sumCache = {};
    renderElement();
    renderSum();
  }

  /** 求和类样本：M 随 D 自适应（总采样量封顶 ~1.6e7 个分量），按模式缓存 */
  function getSumSamples(modeId) {
    if (sumCache[modeId]) return sumCache[modeId];
    var D = state.D;
    var sigma = Math.sqrt(state.sigma2);
    var M = Math.max(2000, Math.min(60000, Math.floor(1.6e7 / (2 * D))));
    var salt = modeId === 'dotRandom' ? 11 : modeId === 'dotFixed' ? 23 : 37;
    var seed = currentSeed(salt);
    var fixedVec = null;
    if (modeId === 'dotFixed') {
      // 固定向量只生成一次：±1 分量（‖v‖² = D 恒成立）
      fixedVec = S.makeFixedVector(S.makeRng(seed ^ 0x9e3779b9), D);
    }
    var samples = S.sampleSum(S.makeRng(seed), modeId, M, D, sigma, fixedVec);
    sumCache[modeId] = { samples: samples, M: M };
    return sumCache[modeId];
  }

  // ---------- 绘图数据 ----------
  function histData(samples, lo, hi, nBins) {
    var h = S.histogram(samples, lo, hi, nBins);
    var data = [];
    var peak = 0;
    for (var i = 0; i < nBins; i++) {
      data.push([h.centers[i], h.density[i]]);
      if (h.density[i] > peak) peak = h.density[i];
    }
    return { data: data, under: h.under, over: h.over, peak: peak };
  }

  function theoryLine(pdf, lo, hi, n) {
    var data = [];
    for (var i = 0; i <= n; i++) {
      var z = lo + ((hi - lo) * i) / n;
      data.push([z, pdf(z)]);
    }
    return data;
  }

  // ---------- 面板 1：按元素运算 ----------
  function renderElement() {
    var sigma = Math.sqrt(state.sigma2);
    var c = state.c;
    // 横轴取五种运算建议范围的并集：同图对比时能看到"平方被压到正半轴 0 附近"
    var lo = Infinity;
    var hi = -Infinity;
    T.ELEMENT_OPS.forEach(function (op) {
      var r = op.range(sigma, c);
      lo = Math.min(lo, r[0]);
      hi = Math.max(hi, r[1]);
    });

    var samples = S.applyElementOp(state.elementOp, pairs.x, pairs.y, c);
    var hist = histData(samples, lo, hi, 140);

    var series = T.ELEMENT_OPS.map(function (op) {
      var active = op.id === state.elementOp;
      return {
        name: op.label,
        type: 'line',
        showSymbol: false,
        animation: false,
        data: theoryLine(
          function (z) {
            return op.pdf(z, sigma, c);
          },
          lo,
          hi,
          260
        ),
        lineStyle: { color: op.color, width: active ? 3 : 1.2, opacity: active ? 1 : 0.55 },
        emphasis: { disabled: true },
        z: active ? 4 : 3,
      };
    });
    series.push({
      name: '蒙特卡洛（选中运算）',
      type: 'line',
      step: 'middle',
      showSymbol: false,
      animation: false,
      data: hist.data,
      lineStyle: { color: '#1d4ed8', width: 1 },
      areaStyle: { color: 'rgba(59,130,246,0.22)' },
      emphasis: { disabled: true },
      z: 2,
    });

    charts.element.setOption(
      {
        grid: { left: 80, right: 24, top: 40, bottom: 44 },
        legend: { top: 4, type: 'scroll' },
        tooltip: { trigger: 'axis', valueFormatter: fmt },
        xAxis: {
          type: 'value',
          name: 'z',
          min: lo,
          max: hi,
          axisLabel: { formatter: fmt },
        },
        yAxis: {
          type: 'value',
          name: '密度',
          nameLocation: 'middle',
          nameRotate: 90,
          nameGap: 50,
          max: hist.peak > 0 ? hist.peak * 1.3 : null,
          axisLabel: { formatter: fmt },
        },
        series: series,
      },
      true
    );

    var op = T.ELEMENT_OPS.filter(function (o) {
      return o.id === state.elementOp;
    })[0];
    var mv = S.sampleMeanVar(samples);
    var outCount = hist.under + hist.over;
    el.elementStats.innerHTML =
      '选中 <b>' +
      op.label +
      '</b>：理论 均值 ' +
      fmt(op.mean(sigma, c)) +
      '、方差 ' +
      fmt(op.variance(sigma, c)) +
      ' ｜ 样本 均值 ' +
      fmt(mv.mean) +
      '、方差 ' +
      fmt(mv.variance) +
      '（N=' +
      state.N.toLocaleString() +
      (outCount > 0 ? '，绘图范围外 ' + outCount + ' 个' : '') +
      '）';
  }

  // ---------- 面板 2：求和类 ----------
  function renderSum() {
    var sigma = Math.sqrt(state.sigma2);
    var D = state.D;
    var mode = T.SUM_MODES.filter(function (m) {
      return m.id === state.sumMode;
    })[0];
    var r = mode.range(D, sigma);
    var lo = r[0];
    var hi = r[1];

    var pack = getSumSamples(mode.id);
    var hist = histData(pack.samples, lo, hi, 140);

    var series = [
      {
        name: mode.label + '（理论）',
        type: 'line',
        showSymbol: false,
        animation: false,
        data: theoryLine(
          function (z) {
            return mode.pdf(z, D, sigma);
          },
          lo,
          hi,
          300
        ),
        lineStyle: { color: mode.color, width: 2.5 },
        emphasis: { disabled: true },
        z: 3,
      },
      {
        name: '蒙特卡洛',
        type: 'line',
        step: 'middle',
        showSymbol: false,
        animation: false,
        data: hist.data,
        lineStyle: { color: '#1d4ed8', width: 1 },
        areaStyle: { color: 'rgba(59,130,246,0.22)' },
        emphasis: { disabled: true },
        z: 2,
      },
    ];

    charts.sum.setOption(
      {
        grid: { left: 80, right: 24, top: 40, bottom: 44 },
        legend: { top: 4 },
        tooltip: { trigger: 'axis', valueFormatter: fmt },
        xAxis: {
          type: 'value',
          name: 'z',
          min: lo,
          max: hi,
          axisLabel: { formatter: fmt },
        },
        yAxis: {
          type: 'value',
          name: '密度',
          nameLocation: 'middle',
          nameRotate: 90,
          nameGap: 50,
          max: hist.peak > 0 ? hist.peak * 1.3 : null,
          axisLabel: { formatter: fmt },
        },
        series: series,
      },
      true
    );

    var mv = S.sampleMeanVar(pack.samples);
    el.sumStats.innerHTML =
      'D=' +
      D +
      '、σ²=' +
      fmt(state.sigma2) +
      '：<b>' +
      mode.label +
      '</b> 理论 均值 ' +
      fmt(mode.mean(D, sigma)) +
      '、方差 ' +
      fmt(mode.variance(D, sigma)) +
      ' ｜ 样本 均值 ' +
      fmt(mv.mean) +
      '、方差 ' +
      fmt(mv.variance) +
      '（M=' +
      pack.M.toLocaleString() +
      ' 个独立向量）';
  }

  // ---------- 控件事件 ----------
  function onDChange(D) {
    state.D = D;
    refreshSigma2(); // 依赖 D 的预设（1/D、1/√D）随动
    resample();
  }

  function bindControls() {
    // D：滑块走 2 的幂，输入框允许任意 1~8192
    el.sliderD.addEventListener('input', function () {
      var D = Math.pow(2, +el.sliderD.value);
      el.inputD.value = D;
      onDChange(D);
    });
    el.inputD.addEventListener('change', function () {
      var D = Math.round(+el.inputD.value);
      if (!isFinite(D)) {
        el.inputD.value = state.D;
        return;
      }
      D = Math.max(1, Math.min(8192, D));
      el.inputD.value = D;
      el.sliderD.value = Math.max(0, Math.min(12, Math.round(Math.log2(D))));
      onDChange(D);
    });

    presetButtons.forEach(function (b) {
      b.addEventListener('click', function () {
        state.sigma2preset = b.dataset.preset;
        refreshSigma2();
        resample();
      });
    });
    el.inputSigma2.addEventListener('change', function () {
      var v = +el.inputSigma2.value;
      if (!(v > 0)) {
        refreshSigma2();
        return;
      }
      state.sigma2preset = null; // 自定义后取消预设高亮
      state.sigma2 = v;
      refreshSigma2();
      resample();
    });

    // c 只影响映射，无需重新采样
    el.inputC.addEventListener('change', function () {
      var v = +el.inputC.value;
      if (isFinite(v) && v !== 0) {
        state.c = v;
        renderElement();
      } else {
        el.inputC.value = state.c;
      }
    });

    el.inputN.addEventListener('change', function () {
      var v = Math.round(+el.inputN.value);
      if (isFinite(v) && v >= 1000) {
        state.N = Math.min(v, 2000000);
        el.inputN.value = state.N;
        resample();
      } else {
        el.inputN.value = state.N;
      }
    });

    el.chkSeed.addEventListener('change', function () {
      state.useSeed = el.chkSeed.checked;
      el.inputSeed.disabled = !state.useSeed;
      resample();
    });
    el.inputSeed.addEventListener('change', function () {
      var v = Math.round(+el.inputSeed.value);
      if (isFinite(v)) {
        state.seed = v;
        resample();
      }
    });
    el.btnResample.addEventListener('click', resample);

    document.querySelectorAll('input[name="elementOp"]').forEach(function (r) {
      r.addEventListener('change', function () {
        state.elementOp = r.value;
        renderElement();
      });
    });
    document.querySelectorAll('input[name="sumMode"]').forEach(function (r) {
      r.addEventListener('change', function () {
        state.sumMode = r.value;
        renderSum();
      });
    });
  }

  // ---------- 初始化 ----------
  function init() {
    charts.element = echarts.init($('chartElement'));
    charts.sum = echarts.init($('chartSum'));
    bindControls();
    refreshSigma2();
    resample();
    window.addEventListener('resize', function () {
      charts.element.resize();
      charts.sum.resize();
    });
  }

  init();
})();
