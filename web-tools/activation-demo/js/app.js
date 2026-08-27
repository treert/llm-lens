/**
 * activation-demo UI 层：状态同步与 ECharts 渲染。
 * 面板一：多函数 fn/dfn 曲线叠加（双图联动模式共享缩放）；
 * 面板二：理论输出密度曲线 + 蒙特卡洛直方图叠加（采样逻辑见文件后段）。
 */
(function () {
  'use strict';

  var PALETTE = ['#2563eb', '#dc2626', '#16a34a', '#9333ea', '#ea580c',
    '#0891b2', '#be185d', '#65a30d', '#7c3aed', '#0d9488', '#b45309'];

  var state = {
    view: 'fn',
    selected: { relu: true, 'gelu-exact': true, silu: true, tanh: true },
    params: {},          // id -> {key: value}
    xRange: 5,
    distId: 'gelu-exact',
    sigma2: 1,
    N: 100000,
    samples: null,
    sampleKey: null,
    // 面板三：GLU 门控族
    gate: 'swiglu',
    rho: 0,
    sigma2G: 1,
    NG: 100000,
    v0: 1,
    sliceMode: 'dist',
    slices: [],
    samplesG: null,
    sampleKeyG: null,
  };
  ActFns.list.forEach(function (act) { state.params[act.id] = ActFns.defaultParams(act); });

  if (typeof echarts === 'undefined') {
    document.getElementById('chartFn').textContent = 'ECharts 加载失败（需联网走 CDN）';
    return;
  }
  var chartFn = echarts.init(document.getElementById('chartFn'));
  var chartDfn = echarts.init(document.getElementById('chartDfn'));
  var chartDist = echarts.init(document.getElementById('chartDist'));
  echarts.connect([chartFn, chartDfn]);

  function colorOf(id) {
    return PALETTE[ActFns.list.findIndex(function (a) { return a.id === id; }) % PALETTE.length];
  }
  function selectedActs() {
    return ActFns.list.filter(function (a) { return state.selected[a.id]; });
  }
  function clampNum(v, lo, hi, fallback) {
    v = parseFloat(v);
    if (!isFinite(v)) return fallback;
    return Math.min(hi, Math.max(lo, v));
  }

  /** 统一 tooltip：轴指示标签与数值截断小数位，避免全精度浮点刷屏（ECharts 要求返回字符串） */
  function tipOpt(pointerType) {
    return {
      trigger: 'axis',
      valueFormatter: function (v) { return String(parseFloat(v.toFixed(4))); },
      axisPointer: {
        type: pointerType || 'line',
        label: { formatter: function (p) { return String(parseFloat(p.value.toFixed(3))); } },
      },
    };
  }

  /** 坐标轴标签统一截断 2 位小数（min/max 设为浮点余量时端点标签会显示全精度） */
  function axisLbl() {
    return { formatter: function (v) { return String(parseFloat(v.toFixed(2))); } };
  }

  /* ---------- 面板一 ---------- */

  function curveSeries(act, deriv) {
    var p = state.params[act.id];
    var n = 500, data = [];
    for (var i = 0; i < n; i++) {
      var x = -state.xRange + 2 * state.xRange * i / (n - 1);
      data.push([x, deriv ? act.dfn(x, p) : act.fn(x, p)]);
    }
    return {
      name: act.name, type: 'line', data: data, showSymbol: false,
      lineStyle: { width: 2, color: colorOf(act.id) },
      itemStyle: { color: colorOf(act.id) },
      emphasis: { focus: 'series' },
    };
  }

  function panel1Option(deriv) {
    return {
      animation: false,
      grid: { left: 56, right: 20, top: 36, bottom: 48 },
      legend: { top: 0 },
      tooltip: tipOpt('cross'),
      xAxis: { type: 'value', name: 'x', min: -state.xRange, max: state.xRange },
      yAxis: { type: 'value', name: deriv ? "f'(x)" : 'f(x)', scale: true, axisLine: { onZero: false } },
      dataZoom: [{ type: 'inside' }],
      series: selectedActs().map(function (a) { return curveSeries(a, deriv); }),
    };
  }

  function renderPanel1() {
    document.getElementById('chartDfn').style.display = state.view === 'both' ? '' : 'none';
    if (state.view === 'dfn') {
      chartFn.setOption(panel1Option(true), true);
    } else {
      chartFn.setOption(panel1Option(false), true);
      if (state.view === 'both') {
        chartDfn.resize();
        chartDfn.setOption(panel1Option(true), true);
      }
    }
  }

  function buildPanel1Controls() {
    var box = document.getElementById('fnChecks');
    box.innerHTML = '';
    ActFns.list.forEach(function (act) {
      var label = document.createElement('label');
      var cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.checked = !!state.selected[act.id];
      cb.addEventListener('change', function () {
        state.selected[act.id] = cb.checked;
        buildParamRows();
        renderPanel1();
      });
      var dot = document.createElement('span');
      dot.className = 'dot';
      dot.style.background = colorOf(act.id);
      label.appendChild(cb);
      label.appendChild(dot);
      label.appendChild(document.createTextNode(act.name + '  ' + act.formula));
      box.appendChild(label);
    });
  }

  /** 参数滑块行：面板一显示所有选中函数的参数；面板二只显示当前函数的参数 */
  function buildParamRows() {
    fillParamRow(document.getElementById('paramSliders1'), selectedActs());
    fillParamRow(document.getElementById('paramSliders2'), [ActFns.byId(state.distId)]);
  }

  function fillParamRow(row, acts) {
    row.innerHTML = '';
    acts.forEach(function (act) {
      act.params.forEach(function (spec) {
        var group = document.createElement('span');
        group.className = 'control-group';
        var lab = document.createElement('label');
        lab.textContent = act.name + ' ' + spec.label;
        var slider = document.createElement('input');
        slider.type = 'range';
        slider.min = spec.min; slider.max = spec.max; slider.step = spec.step;
        slider.value = state.params[act.id][spec.key];
        var num = document.createElement('input');
        num.type = 'number';
        num.min = spec.min; num.max = spec.max; num.step = spec.step;
        num.value = state.params[act.id][spec.key];
        function sync(v) {
          state.params[act.id][spec.key] = clampNum(v, spec.min, spec.max, spec.def);
          slider.value = state.params[act.id][spec.key];
          num.value = state.params[act.id][spec.key];
          renderPanel1();
          renderPanel2();
        }
        slider.addEventListener('input', function () { sync(slider.value); });
        num.addEventListener('change', function () { sync(num.value); });
        group.appendChild(lab);
        group.appendChild(slider);
        group.appendChild(num);
        row.appendChild(group);
      });
    });
  }

  /* ---------- 面板二（理论曲线） ---------- */

  function currentKey() {
    return JSON.stringify([state.distId, state.sigma2, state.params[state.distId], state.N]);
  }

  function renderPanel2() {
    var act = ActFns.byId(state.distId);
    var p = state.params[act.id];
    var sigma = Math.sqrt(state.sigma2);
    var range = ActTheory.suggestRange(act, p, sigma);
    var grid = ActTheory.densityGrid(act, p, sigma, range.xLo, range.xHi, 800);
    var line = [];
    for (var i = 0; i < grid.ys.length; i++) line.push([grid.ys[i], grid.fs[i]]);
    var series = [];
    if (state.samples) {
      // 直方图轮廓：逐箱两点的阶梯折线 + 半透明填充
      var hist = ActSampler.histogram(state.samples, grid.yLo, grid.yHi, 120);
      var w = (grid.yHi - grid.yLo) / 120;
      var outline = [[grid.yLo, 0]];
      for (var b = 0; b < hist.centers.length; b++) {
        outline.push([grid.yLo + b * w, hist.density[b]]);
        outline.push([grid.yLo + (b + 1) * w, hist.density[b]]);
      }
      outline.push([grid.yHi, 0]);
      series.push({
        name: '蒙特卡洛直方图', type: 'line', data: outline,
        showSymbol: false, silent: true,
        lineStyle: { width: 1, color: '#ea580c' },
        itemStyle: { color: '#ea580c' }, areaStyle: { opacity: 0.25 },
      });
    }
    series.push({
      name: '理论密度', type: 'line', data: line, showSymbol: false,
      lineStyle: { width: 2, color: '#2563eb' }, itemStyle: { color: '#2563eb' },
    });
    var m = ActTheory.outputMoments(act, p, sigma);
    if (act.atom) {
      series[series.length - 1].markLine = {
        symbol: 'none', silent: true,
        lineStyle: { type: 'dashed', color: '#dc2626' },
        label: { formatter: '点质量 P(y=0)=' + m.atom.toFixed(3) },
        data: [{ xAxis: 0 }],
      };
    }
    chartDist.setOption({
      animation: false,
      grid: { left: 56, right: 20, top: 36, bottom: 48 },
      legend: { top: 0 },
      tooltip: tipOpt(),
      xAxis: { type: 'value', name: 'y', min: grid.yLo, max: grid.yHi, axisLabel: axisLbl() },
      yAxis: { type: 'value', name: '密度', scale: true, axisLine: { onZero: false } },
      dataZoom: [{ type: 'inside' }],
      series: series,
    }, true);
    var stats = '理论：均值 ' + m.mean.toFixed(4) + '，方差 ' + m.variance.toFixed(4) +
      (act.atom ? '，点质量 ' + m.atom.toFixed(3) : '') + ' ｜ ';
    if (state.samples) {
      var mv = ActSampler.sampleMeanVar(state.samples);
      stats += '样本（N=' + state.samples.length + '）：均值 ' + mv.mean.toFixed(4) +
        '，方差 ' + mv.variance.toFixed(4);
      if (act.atom) {
        stats += '，0 处比例 ' +
          (ActSampler.countExactZeros(state.samples) / state.samples.length).toFixed(3);
      }
    } else {
      stats += '<span class="stat-dim">样本：未采样（点「采样」叠加直方图）</span>';
    }
    document.getElementById('distStats').innerHTML = stats;
    document.getElementById('distNote').textContent = act.distNote;
    updateSampleButton();
  }

  function updateSampleButton() {
    var btn = document.getElementById('btnSample');
    if (state.samples === null || state.sampleKey !== currentKey()) btn.classList.add('need-sample');
    else btn.classList.remove('need-sample');
  }

  /* ---------- 控件绑定与启动 ---------- */

  function buildDistSelect() {
    var sel = document.getElementById('selDist');
    ActFns.list.forEach(function (act) {
      var opt = document.createElement('option');
      opt.value = act.id;
      opt.textContent = act.name;
      sel.appendChild(opt);
    });
    sel.value = state.distId;
    sel.addEventListener('change', function () {
      state.distId = sel.value;
      buildParamRows();
      renderPanel2();
    });
  }

  function bindGlobal() {
    document.querySelectorAll('input[name="p1view"]').forEach(function (r) {
      r.addEventListener('change', function () {
        state.view = r.value;
        renderPanel1();
      });
    });
    var sliderX = document.getElementById('sliderX');
    var inputX = document.getElementById('inputX');
    function syncX(v) {
      state.xRange = clampNum(v, 1, 10, 5);
      sliderX.value = state.xRange;
      inputX.value = state.xRange;
      renderPanel1();
    }
    sliderX.addEventListener('input', function () { syncX(sliderX.value); });
    inputX.addEventListener('change', function () { syncX(inputX.value); });

    var sliderS = document.getElementById('sliderSigma');
    var inputS = document.getElementById('inputSigma');
    function syncS(v) {
      state.sigma2 = clampNum(v, 0.1, 10, 1);
      sliderS.value = state.sigma2;
      inputS.value = state.sigma2;
      renderPanel2();
    }
    sliderS.addEventListener('input', function () { syncS(sliderS.value); });
    inputS.addEventListener('change', function () { syncS(inputS.value); });

    document.getElementById('inputN2').addEventListener('change', function () {
      state.N = Math.round(clampNum(this.value, 10000, 1000000, 100000));
      this.value = state.N;
      updateSampleButton();
    });
  }

  /* ---------- 面板三：GLU 门控族 ---------- */

  var chartHeat = echarts.init(document.getElementById('chartHeat'));
  var chartSlice = echarts.init(document.getElementById('chartSlice'));
  var chartGlu = echarts.init(document.getElementById('chartGlu'));

  function gate() { return ActFns.gateById(state.gate); }
  function gateActP() {
    var g = ActFns.gateAct(gate());
    return { g: g, gp: ActFns.defaultParams(g) };
  }

  /** 面板三 y 轴范围：矩半宽法（向上取整到 0.1，坐标轴标签整洁） */
  function gluYRange() {
    var m = ActTheory.gluOutputMoments(gate(), state.rho, Math.sqrt(state.sigma2G));
    var sd = Math.sqrt(m.variance);
    var hw = Math.max(4 * sd, Math.abs(m.mean) + 2.5 * sd, 1);
    hw = Math.ceil(hw * 10) / 10;
    return { yLo: -hw, yHi: hw };
  }

  /** 发散色插值：t∈[-1,1]，负蓝零白正红 */
  function divergeColor(t) {
    var neg = [33, 102, 172], mid = [247, 247, 247], pos = [178, 24, 43];
    var b = t < 0 ? neg : pos, s = Math.min(1, Math.abs(t));
    return 'rgb(' + Math.round(mid[0] + (b[0] - mid[0]) * s) + ','
      + Math.round(mid[1] + (b[1] - mid[1]) * s) + ','
      + Math.round(mid[2] + (b[2] - mid[2]) * s) + ')';
  }

  /** 边缘密度网格（按参数缓存，切片视图与分布图共用） */
  var gluMargCache = { key: null, pts: null, yLo: 0, yHi: 0 };
  function gluMarginal() {
    var key = JSON.stringify([state.gate, state.rho, state.sigma2G]);
    if (gluMargCache.key === key) return gluMargCache;
    var sigma = Math.sqrt(state.sigma2G);
    var yr = gluYRange(), n = 400, pts = [];
    for (var i = 0; i < n; i++) {
      var y = yr.yLo + (yr.yHi - yr.yLo) * i / (n - 1);
      pts.push([y, ActTheory.gluOutputDensity(gate(), state.rho, sigma, y)]);
    }
    gluMargCache = { key: key, pts: pts, yLo: yr.yLo, yHi: yr.yHi };
    return gluMargCache;
  }

  /* ---- 热力图 ---- */

  function renderHeat() {
    var gap = gateActP(), sigma = Math.sqrt(state.sigma2G);
    var R = 4 * sigma, M = 121, i, j;
    var data = [], zMax = 0;
    for (i = 0; i < M; i++) {
      for (j = 0; j < M; j++) {
        var u = -R + 2 * R * i / (M - 1);
        var v = -R + 2 * R * j / (M - 1);
        var z = u * gap.g.fn(v, gap.gp);
        data.push([u, v, z]);
        if (Math.abs(z) > zMax) zMax = Math.abs(z);
      }
    }
    // ECharts heatmap 在 value 轴上单元格尺寸推断不可靠，改用 custom 逐格画 rect
    var du = 2 * R / (M - 1), dv = 2 * R / (M - 1);
    var series = [{
      name: 'z=u·g(v)', type: 'custom', data: data, z: 1,
      renderItem: function (params, api) {
        var u = api.value(0), v = api.value(1), z = api.value(2);
        var p0 = api.coord([u - du / 2, v - dv / 2]);
        var p1 = api.coord([u + du / 2, v + dv / 2]);
        return {
          type: 'rect',
          shape: { x: p0[0], y: p1[1], width: p1[0] - p0[0], height: p0[1] - p1[1] },
          style: { fill: divergeColor(z / zMax) },
        };
      },
    }];
    [1, 2, 3].forEach(function (k) {
      series.push({
        name: k + 'σ', type: 'line',
        data: ActTheory.gaussEllipse(state.rho, sigma, k, 121),
        showSymbol: false, silent: true, z: 3,
        lineStyle: { width: 1, color: 'rgba(75,85,99,0.85)', type: k === 1 ? 'solid' : 'dashed' },
      });
    });
    series[series.length - 1].markLine = {
      symbol: 'none', silent: true, z: 4,
      lineStyle: { type: 'dashed', color: '#7c3aed', width: 2 },
      label: { formatter: 'v₀=' + state.v0.toFixed(2), position: 'insideEndTop' },
      data: [{ yAxis: state.v0 }],
    };
    chartHeat.setOption({
      animation: false,
      grid: { left: 56, right: 20, top: 30, bottom: 48 },
      xAxis: { type: 'value', name: 'u', min: -R, max: R, axisLabel: axisLbl() },
      yAxis: { type: 'value', name: 'v', min: -R, max: R, axisLabel: axisLbl() },
      series: series,
    }, true);
  }

  /* ---- 切片视图（分布切片 / 曲面切片） ---- */

  function renderSlice() {
    var sigma = Math.sqrt(state.sigma2G);
    var statsEl = document.getElementById('sliceStats');
    if (state.sliceMode === 'surf') {
      var gap = gateActP();
      var R = 4 * sigma;
      var series = state.slices.map(function (v0, idx) {
        var gv = gap.g.fn(v0, gap.gp);
        var pts = [];
        for (var i = 0; i <= 100; i++) {
          var u = -R + 2 * R * i / 100;
          pts.push([u, u * gv]);
        }
        return {
          name: 'v₀=' + v0.toFixed(2), type: 'line', data: pts, showSymbol: false,
          lineStyle: { width: 2, color: PALETTE[idx % PALETTE.length] },
          itemStyle: { color: PALETTE[idx % PALETTE.length] },
        };
      });
      chartSlice.setOption({
        animation: false,
        grid: { left: 56, right: 20, top: 36, bottom: 48 },
        legend: { top: 0, type: 'scroll' },
        tooltip: tipOpt(),
        xAxis: { type: 'value', name: 'u', min: -R, max: R, axisLabel: axisLbl() },
        yAxis: { type: 'value', name: 'y = u·g(v₀)', scale: true, axisLine: { onZero: false }, axisLabel: axisLbl() },
        series: series,
      }, true);
      statsEl.textContent = state.slices.length
        ? '每条直线斜率 = g(v₀)——门控就是用 v 调制 u 通道的斜率（点击热力图累加，上限 8 条）'
        : '点击左侧热力图累加切片（v₀<0 与 v₀>0 对比最直观）';
      return;
    }
    // 分布切片模式
    var marg = gluMarginal();
    var c = ActTheory.gluConditional(gate(), state.rho, sigma, state.v0);
    var series = [{
      name: '边缘密度 f_Y', type: 'line', data: marg.pts, showSymbol: false,
      lineStyle: { width: 1.5, color: '#9ca3af', type: 'dashed' },
      itemStyle: { color: '#9ca3af' },
    }];
    if (c.sd > 1e-9) {
      var cond = [];
      for (var i = 0; i < marg.pts.length; i++) {
        var y = marg.pts[i][0];
        cond.push([y, ActFns.helpers.normPdf(y - c.mean, c.sd)]);
      }
      series.push({
        name: '条件分布 y|v₀', type: 'line', data: cond, showSymbol: false,
        lineStyle: { width: 2, color: '#7c3aed' }, itemStyle: { color: '#7c3aed' },
        areaStyle: { opacity: 0.15 },
      });
      statsEl.textContent = 'v₀=' + state.v0.toFixed(2) + '，g(v₀)=' + c.gv.toFixed(3) +
        '，y|v₀ ~ N(' + c.mean.toFixed(3) + ', ' + c.sd.toFixed(3) + '²)——' +
        '边缘分布是所有 v 切片高斯的加权混合';
    } else {
      statsEl.textContent = 'v₀=' + state.v0.toFixed(2) + '，g(v₀)≈0：该切片门关闭，y≈0' +
        (gate().atom ? '（质量进入 y=0 点质量）' : '');
    }
    chartSlice.setOption({
      animation: false,
      grid: { left: 56, right: 20, top: 36, bottom: 48 },
      legend: { top: 0 },
      tooltip: tipOpt(),
      xAxis: { type: 'value', name: 'y', min: marg.yLo, max: marg.yHi },
      yAxis: { type: 'value', name: '密度', scale: true, axisLine: { onZero: false } },
      series: series,
    }, true);
  }

  /* ---- 边缘分布 + 采样 ---- */

  function currentKeyG() {
    return JSON.stringify([state.gate, state.rho, state.sigma2G, state.NG]);
  }

  function updateSampleButtonG() {
    var btn = document.getElementById('btnSampleG');
    if (state.samplesG === null || state.sampleKeyG !== currentKeyG()) btn.classList.add('need-sample');
    else btn.classList.remove('need-sample');
  }

  function renderPanel3() {
    var g0 = gate(), sigma = Math.sqrt(state.sigma2G);
    var marg = gluMarginal();
    var series = [];
    if (state.samplesG) {
      var hist = ActSampler.histogram(state.samplesG, marg.yLo, marg.yHi, 120);
      var w = (marg.yHi - marg.yLo) / 120;
      var outline = [[marg.yLo, 0]];
      for (var b = 0; b < hist.centers.length; b++) {
        outline.push([marg.yLo + b * w, hist.density[b]]);
        outline.push([marg.yLo + (b + 1) * w, hist.density[b]]);
      }
      outline.push([marg.yHi, 0]);
      series.push({
        name: '蒙特卡洛直方图', type: 'line', data: outline,
        showSymbol: false, silent: true,
        lineStyle: { width: 1, color: '#ea580c' },
        itemStyle: { color: '#ea580c' }, areaStyle: { opacity: 0.25 },
      });
    }
    series.push({
      name: '理论密度', type: 'line', data: marg.pts, showSymbol: false,
      lineStyle: { width: 2, color: '#2563eb' }, itemStyle: { color: '#2563eb' },
    });
    var m = ActTheory.gluOutputMoments(g0, state.rho, sigma);
    if (g0.atom) {
      series[series.length - 1].markLine = {
        symbol: 'none', silent: true,
        lineStyle: { type: 'dashed', color: '#dc2626' },
        label: { formatter: '点质量 P(y=0)=' + m.atom.toFixed(3) },
        data: [{ xAxis: 0 }],
      };
    }
    chartGlu.setOption({
      animation: false,
      grid: { left: 56, right: 20, top: 36, bottom: 48 },
      legend: { top: 0 },
      tooltip: tipOpt(),
      xAxis: { type: 'value', name: 'y', min: marg.yLo, max: marg.yHi },
      yAxis: { type: 'value', name: '密度', scale: true, axisLine: { onZero: false } },
      dataZoom: [{ type: 'inside' }],
      series: series,
    }, true);
    var stats = '理论：均值 ' + m.mean.toFixed(4) + '，方差 ' + m.variance.toFixed(4) +
      (g0.atom ? '，点质量 ' + m.atom.toFixed(3) : '') + ' ｜ ';
    if (state.samplesG) {
      var mv = ActSampler.sampleMeanVar(state.samplesG);
      stats += '样本（N=' + state.samplesG.length + '）：均值 ' + mv.mean.toFixed(4) +
        '，方差 ' + mv.variance.toFixed(4);
      if (g0.atom) {
        stats += '，0 处比例 ' +
          (ActSampler.countExactZeros(state.samplesG) / state.samplesG.length).toFixed(3);
      }
    } else {
      stats += '<span class="stat-dim">样本：未采样（点「采样」叠加直方图）</span>';
    }
    document.getElementById('gluStats').innerHTML = stats;
    document.getElementById('gluNote').textContent = g0.note;
    updateSampleButtonG();
  }

  function doSampleG() {
    var gap = gateActP();
    var seed = document.getElementById('chkSeed').checked
      ? parseInt(document.getElementById('inputSeed').value, 10) || 0
      : (Math.random() * 2147483647) | 0;
    var bv = ActSampler.sampleBivariate(ActSampler.makeRng(seed), state.NG,
      Math.sqrt(state.sigma2G), state.rho);
    state.samplesG = ActSampler.applyGate(gap.g, gap.gp, bv.u, bv.v);
    state.sampleKeyG = currentKeyG();
    renderPanel3();
  }

  function renderGluAll() {
    renderHeat();
    renderSlice();
    renderPanel3();
  }

  function bindGlu() {
    document.querySelectorAll('input[name="gate"]').forEach(function (r) {
      r.addEventListener('change', function () {
        state.gate = r.value;
        state.slices = [];
        renderGluAll();
      });
    });
    var sliderR = document.getElementById('sliderRho');
    var inputR = document.getElementById('inputRho');
    function syncR(v) {
      state.rho = clampNum(v, -0.95, 0.95, 0);
      sliderR.value = state.rho;
      inputR.value = state.rho;
      renderGluAll();
    }
    sliderR.addEventListener('input', function () { syncR(sliderR.value); });
    inputR.addEventListener('change', function () { syncR(inputR.value); });

    var sliderSG = document.getElementById('sliderSigmaG');
    var inputSG = document.getElementById('inputSigmaG');
    function syncSG(v) {
      state.sigma2G = clampNum(v, 0.1, 4, 1);
      sliderSG.value = state.sigma2G;
      inputSG.value = state.sigma2G;
      renderGluAll();
    }
    sliderSG.addEventListener('input', function () { syncSG(sliderSG.value); });
    inputSG.addEventListener('change', function () { syncSG(inputSG.value); });

    document.getElementById('inputNG').addEventListener('change', function () {
      state.NG = Math.round(clampNum(this.value, 10000, 1000000, 100000));
      this.value = state.NG;
      updateSampleButtonG();
    });
    document.getElementById('btnSampleG').addEventListener('click', doSampleG);
    document.getElementById('btnResetG').addEventListener('click', function () {
      state.samplesG = null;
      state.sampleKeyG = null;
      renderPanel3();
    });

    document.querySelectorAll('input[name="sliceMode"]').forEach(function (r) {
      r.addEventListener('change', function () {
        state.sliceMode = r.value;
        renderSlice();
      });
    });
    document.getElementById('btnClearSlices').addEventListener('click', function () {
      state.slices = [];
      renderSlice();
    });

    chartHeat.on('click', function (p) {
      if (!p.value) return;
      var R = 4 * Math.sqrt(state.sigma2G);
      var v = clampNum(p.value[1], -R, R, state.v0);
      state.v0 = Math.round(v * 100) / 100;
      if (state.sliceMode === 'surf') {
        state.slices.push(state.v0);
        if (state.slices.length > 8) state.slices.shift();
      }
      renderHeat();
      renderSlice();
    });
  }

  window.addEventListener('resize', function () {
    chartFn.resize();
    chartDfn.resize();
    chartDist.resize();
    chartHeat.resize();
    chartSlice.resize();
    chartGlu.resize();
  });

  function doSample() {
    var act = ActFns.byId(state.distId);
    var seed = document.getElementById('chkSeed').checked
      ? parseInt(document.getElementById('inputSeed').value, 10) || 0
      : (Math.random() * 2147483647) | 0;
    var gauss = ActSampler.makeRng(seed);
    var xs = ActSampler.sampleInput(gauss, state.N, Math.sqrt(state.sigma2));
    state.samples = ActSampler.applyActivation(act, state.params[act.id], xs);
    state.sampleKey = currentKey();
    renderPanel2();
  }

  function boot() {
    buildPanel1Controls();
    buildDistSelect();
    buildParamRows();
    bindGlobal();
    document.getElementById('btnSample').addEventListener('click', doSample);
    document.getElementById('btnReset').addEventListener('click', function () {
      state.samples = null;
      state.sampleKey = null;
      renderPanel2();
    });
    document.getElementById('chkSeed').addEventListener('change', updateSampleButton);
    document.getElementById('inputSeed').addEventListener('change', updateSampleButton);
    bindGlu();
    renderPanel1();
    renderPanel2();
    renderGluAll();
  }
  boot();
})();
