/**
 * UI 层：控件状态、采样驱动、ECharts 渲染。
 *
 * 两个相互独立的面板，参数各管各的（σ²、D、样本量均不共享）：
 *   1. 按元素运算：4 条理论密度曲线同图对比 + 选中运算的蒙特卡洛直方图。
 *      逐分量标量分布，与维数 D 无关，σ²、N 独立设置；
 *   2. 求和类：点积（双方随机 / 一方固定）与长度平方的直方图 + 精确理论曲线。
 *      维数 D 与 σ²（预设 1 / 1/D / 1/√D 随 D 联动）独立设置，
 *      两种点积模式的方差对比即回答"σ²=1/D 时点积方差是否为 1"。
 *   全局仅共享随机种子（影响复现性，不影响分布结论）。
 *
 * 交互约定：理论曲线随参数即时重画；蒙特卡洛不自动采样——
 * 点各面板参数行的「采样」按钮才生成样本并叠加直方图；
 * 参数（σ²、N、D、种子）变更后旧样本失效：直方图移除、采样按钮高亮提示。
 * 仅切换运算/模式（radio）不失效：面板一用同一批样本重新映射，
 * 面板二切到有缓存的模式直接显示。
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
    // 全局
    chkSeed: $('chkSeed'),
    inputSeed: $('inputSeed'),
    // 面板一
    inputSigmaElem: $('inputSigmaElem'),
    inputN: $('inputN'),
    btnSampleElem: $('btnSampleElem'),
    elementStats: $('elementStats'),
    // 面板二
    sliderD: $('sliderD'),
    inputD: $('inputD'),
    hControl: $('hControl'),
    sliderH: $('sliderH'),
    inputH: $('inputH'),
    inputSigmaSum: $('inputSigmaSum'),
    btnSampleSum: $('btnSampleSum'),
    sumStats: $('sumStats'),
  };
  var presetButtons = Array.prototype.slice.call(
    document.querySelectorAll('.preset-group button')
  );

  // ---------- 状态：两个面板各自独立 ----------
  var state = {
    useSeed: true,
    seed: 42,
    elem: { sigma2: 1, N: 200000, op: 'product' },
    sum: { D: 8192, H: 128, sigma2preset: '1/D', sigma2: 1 / 8192, mode: 'dotRandom' },
  };

  var pairs = null; // 面板一样本对 {x, y}；null = 未采样/已失效
  var sumCache = {}; // 面板二缓存：modeId -> { samples, M }；参数变更即清空
  var projW = null; // projDot 的固定投影矩阵 {D, H, key, wQ, wK}；随参数/种子失效
  var charts = {};

  /** 随机种子（全局共享）；salt 区分各条随机流，保证两面板样本无交集 */
  function currentSeed(salt) {
    var base = state.useSeed ? state.seed : (Math.random() * 1e9) | 0;
    return (base + salt) >>> 0;
  }

  /** 采样按钮高亮切换：样本失效时提示需要重新采样 */
  function markNeedSample(btn, need) {
    btn.classList.toggle('need-sample', need);
  }

  // ---------- 绘图数据小函数（两面板共用） ----------
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
      var y = pdf(z);
      // 密度在奇点可发散（如乘积正态 z=0 对数奇异），断点处理避免污染坐标轴范围
      data.push([z, isFinite(y) ? y : null]);
    }
    return data;
  }

  // ---------- 面板一：按元素运算 ----------
  function renderElement() {
    var sigma = Math.sqrt(state.elem.sigma2);
    // 横轴取四种运算建议范围的并集：同图对比时能看到"平方被压到正半轴 0 附近"
    var lo = Infinity;
    var hi = -Infinity;
    T.ELEMENT_OPS.forEach(function (op) {
      var r = op.range(sigma);
      lo = Math.min(lo, r[0]);
      hi = Math.max(hi, r[1]);
    });

    var hasSample = pairs !== null;
    var samples = hasSample ? S.applyElementOp(state.elem.op, pairs.x, pairs.y) : null;
    var hist = hasSample ? histData(samples, lo, hi, 140) : null;

    var series = T.ELEMENT_OPS.map(function (op) {
      var active = op.id === state.elem.op;
      // 理论线裁剪到该运算的支撑集：平方只画 z≥0（卡方无负半轴），避免贴地 0 值长线
      var or = op.range(sigma);
      return {
        name: op.label,
        type: 'line',
        showSymbol: false,
        animation: false,
        data: theoryLine(
          function (z) {
            return op.pdf(z, sigma);
          },
          Math.max(lo, or[0]),
          Math.min(hi, or[1]),
          260
        ),
        lineStyle: { color: op.color, width: active ? 3 : 1.2, opacity: active ? 1 : 0.55 },
        emphasis: { disabled: true },
        z: active ? 4 : 3,
      };
    });
    if (hasSample) {
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
    }

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
          max: hist && hist.peak > 0 ? hist.peak * 1.3 : null,
          axisLabel: { formatter: fmt },
        },
        series: series,
      },
      true
    );

    var op = T.ELEMENT_OPS.filter(function (o) {
      return o.id === state.elem.op;
    })[0];
    var text =
      'σ²=' +
      fmt(state.elem.sigma2) +
      '：<b>' +
      op.label +
      '</b> 理论 均值 ' +
      fmt(op.mean(sigma)) +
      '、方差 ' +
      fmt(op.variance(sigma));
    if (hasSample) {
      var mv = S.sampleMeanVar(samples);
      var outCount = hist.under + hist.over;
      text +=
        ' ｜ 样本 均值 ' +
        fmt(mv.mean) +
        '、方差 ' +
        fmt(mv.variance) +
        '（N=' +
        state.elem.N.toLocaleString() +
        (outCount > 0 ? '，绘图范围外 ' + outCount + ' 个' : '') +
        '）';
    } else {
      text += ' ｜ <span class="stat-dim">未采样——点参数行「采样」叠加蒙特卡洛直方图</span>';
    }
    el.elementStats.innerHTML = text;
  }

  // ---------- 面板二：求和类 ----------
  function sigma2FromPreset(preset, D) {
    if (preset === '1') return 1;
    if (preset === '1/D') return 1 / D;
    if (preset === '1/sqrtD') return 1 / Math.sqrt(D);
    return null;
  }

  function refreshSigma2() {
    if (state.sum.sigma2preset) {
      state.sum.sigma2 = sigma2FromPreset(state.sum.sigma2preset, state.sum.D);
    }
    el.inputSigmaSum.value = Number(state.sum.sigma2.toPrecision(6));
    presetButtons.forEach(function (b) {
      b.classList.toggle('active', b.dataset.preset === state.sum.sigma2preset);
    });
  }

  /**
   * projDot 的投影矩阵：σ_w² = 1/D（标准初始化），由种子生成后固定；
   * D、H 或种子变化时重建。
   */
  function getProjW() {
    var D = state.sum.D;
    var H = state.sum.H;
    var key = D + '|' + H + '|' + currentSeed(0);
    if (projW && projW.key === key) return projW;
    var g = S.makeRng(currentSeed(41));
    var sigmaW = 1 / Math.sqrt(D);
    projW = {
      key: key,
      wQ: S.makeProjection(g, H, D, sigmaW),
      wK: S.makeProjection(g, H, D, sigmaW),
    };
    return projW;
  }

  /** 采指定模式并存入缓存；M 随 D 自适应（总采样量封顶 ~1.6e7 个分量） */
  function sampleSumMode(modeId) {
    if (modeId === 'projDot') {
      var D = state.sum.D;
      var H = state.sum.H;
      var sigma = Math.sqrt(state.sum.sigma2);
      var w = getProjW();
      // 每样本 2HD 次乘加：预算 ~3e8，M 随之自适应
      var M = Math.max(500, Math.min(20000, Math.floor(3e8 / (2 * H * D))));
      var samples = S.sampleProjDot(S.makeRng(currentSeed(43)), M, D, H, sigma, w.wQ, w.wK);
      sumCache.projDot = { samples: samples, M: M };
      return;
    }
    var D = state.sum.D;
    var sigma = Math.sqrt(state.sum.sigma2);
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
  }

  function renderSum() {
    var sigma = Math.sqrt(state.sum.sigma2);
    var D = state.sum.D;
    var H = state.sum.H; // 仅 projDot 模式使用
    var mode = T.SUM_MODES.filter(function (m) {
      return m.id === state.sum.mode;
    })[0];
    var r = mode.range(D, sigma, H);
    var lo = r[0];
    var hi = r[1];

    var pack = sumCache[mode.id] || null; // 不自动采样：无缓存则只画理论线
    var hist = pack ? histData(pack.samples, lo, hi, 140) : null;

    var series = [
      {
        name: mode.label + '（理论）',
        type: 'line',
        showSymbol: false,
        animation: false,
        data: theoryLine(
          function (z) {
            return mode.pdf(z, D, sigma, H);
          },
          lo,
          hi,
          300
        ),
        lineStyle: { color: mode.color, width: 2.5 },
        emphasis: { disabled: true },
        z: 3,
      },
    ];
    if (pack) {
      series.push({
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
      });
    }

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
          max: hist && hist.peak > 0 ? hist.peak * 1.3 : null,
          axisLabel: { formatter: fmt },
        },
        series: series,
      },
      true
    );

    var text =
      'D=' +
      D +
      (mode.id === 'projDot' ? '、H=' + H : '') +
      '、σ²=' +
      fmt(state.sum.sigma2) +
      '：<b>' +
      mode.label +
      '</b> 理论 均值 ' +
      fmt(mode.mean(D, sigma, H)) +
      '、方差 ' +
      fmt(mode.variance(D, sigma, H));
    if (pack) {
      var mv = S.sampleMeanVar(pack.samples);
      text +=
        ' ｜ 样本 均值 ' +
        fmt(mv.mean) +
        '、方差 ' +
        fmt(mv.variance) +
        '（M=' +
        pack.M.toLocaleString() +
        ' 个独立向量）';
    } else {
      text += ' ｜ <span class="stat-dim">未采样——点参数行「采样」叠加蒙特卡洛直方图</span>';
    }
    el.sumStats.innerHTML = text;
  }

  /** H 控件只在投影点积模式下显示 */
  function refreshHVisibility() {
    el.hControl.style.display = state.sum.mode === 'projDot' ? '' : 'none';
  }

  // ---------- 样本失效 ----------
  function invalidateElement() {
    pairs = null;
    markNeedSample(el.btnSampleElem, true);
    renderElement();
  }

  function invalidateSum() {
    sumCache = {};
    projW = null;
    markNeedSample(el.btnSampleSum, true);
    renderSum();
  }

  // ---------- 控件事件 ----------
  function bindControls() {
    // —— 面板一 ——
    el.inputSigmaElem.addEventListener('change', function () {
      var v = +el.inputSigmaElem.value;
      if (!(v > 0)) {
        el.inputSigmaElem.value = state.elem.sigma2;
        return;
      }
      state.elem.sigma2 = v;
      invalidateElement();
    });
    el.inputN.addEventListener('change', function () {
      var v = Math.round(+el.inputN.value);
      if (isFinite(v) && v >= 1000) {
        state.elem.N = Math.min(v, 2000000);
        el.inputN.value = state.elem.N;
        invalidateElement();
      } else {
        el.inputN.value = state.elem.N;
      }
    });
    document.querySelectorAll('input[name="elementOp"]').forEach(function (r) {
      r.addEventListener('change', function () {
        state.elem.op = r.value;
        renderElement(); // 运算切换不失效样本：同一批 pairs 重新映射
      });
    });
    el.btnSampleElem.addEventListener('click', function () {
      pairs = S.samplePairs(
        S.makeRng(currentSeed(0)),
        state.elem.N,
        Math.sqrt(state.elem.sigma2)
      );
      markNeedSample(el.btnSampleElem, false);
      renderElement();
    });

    // —— 面板二 ——
    function onDChange(D) {
      state.sum.D = D;
      refreshSigma2(); // 依赖 D 的预设（1/D、1/√D）随动
      invalidateSum();
    }
    el.sliderD.addEventListener('input', function () {
      var D = Math.pow(2, +el.sliderD.value);
      el.inputD.value = D;
      onDChange(D);
    });
    el.inputD.addEventListener('change', function () {
      var D = Math.round(+el.inputD.value);
      if (!isFinite(D)) {
        el.inputD.value = state.sum.D;
        return;
      }
      D = Math.max(1, Math.min(8192, D));
      el.inputD.value = D;
      el.sliderD.value = Math.max(0, Math.min(13, Math.round(Math.log2(D))));
      onDChange(D);
    });
    presetButtons.forEach(function (b) {
      b.addEventListener('click', function () {
        state.sum.sigma2preset = b.dataset.preset;
        refreshSigma2();
        invalidateSum();
      });
    });
    el.inputSigmaSum.addEventListener('change', function () {
      var v = +el.inputSigmaSum.value;
      if (!(v > 0)) {
        refreshSigma2();
        return;
      }
      state.sum.sigma2preset = null; // 自定义后取消预设高亮
      state.sum.sigma2 = v;
      refreshSigma2();
      invalidateSum();
    });
    // H：滑块走 2 的幂，输入框允许任意 1~512
    function onHChange(H) {
      state.sum.H = H;
      invalidateSum();
    }
    el.sliderH.addEventListener('input', function () {
      var H = Math.pow(2, +el.sliderH.value);
      el.inputH.value = H;
      onHChange(H);
    });
    el.inputH.addEventListener('change', function () {
      var H = Math.round(+el.inputH.value);
      if (!isFinite(H)) {
        el.inputH.value = state.sum.H;
        return;
      }
      H = Math.max(1, Math.min(512, H));
      el.inputH.value = H;
      el.sliderH.value = Math.max(0, Math.min(9, Math.round(Math.log2(H))));
      onHChange(H);
    });
    document.querySelectorAll('input[name="sumMode"]').forEach(function (r) {
      r.addEventListener('change', function () {
        state.sum.mode = r.value;
        refreshHVisibility();
        renderSum(); // 切换模式不失效：有缓存则显示直方图，无则纯理论线
      });
    });
    el.btnSampleSum.addEventListener('click', function () {
      sampleSumMode(state.sum.mode); // 只采当前模式，其余模式切换后再按需采
      markNeedSample(el.btnSampleSum, false);
      renderSum();
    });

    // —— 全局 ——
    el.chkSeed.addEventListener('change', function () {
      state.useSeed = el.chkSeed.checked;
      el.inputSeed.disabled = !state.useSeed;
      invalidateElement();
      invalidateSum();
    });
    el.inputSeed.addEventListener('change', function () {
      var v = Math.round(+el.inputSeed.value);
      if (isFinite(v)) {
        state.seed = v;
        invalidateElement();
        invalidateSum();
      }
    });
  }

  // ---------- 初始化：只画理论线，采样由用户手动触发 ----------
  function init() {
    charts.element = echarts.init($('chartElement'));
    charts.sum = echarts.init($('chartSum'));
    bindControls();
    refreshSigma2();
    refreshHVisibility();
    markNeedSample(el.btnSampleElem, true);
    markNeedSample(el.btnSampleSum, true);
    renderElement();
    renderSum();
    window.addEventListener('resize', function () {
      charts.element.resize();
      charts.sum.resize();
    });
  }

  init();
})();
