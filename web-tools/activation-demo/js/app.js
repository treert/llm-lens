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
      tooltip: { trigger: 'axis', axisPointer: { type: 'cross' } },
      xAxis: { type: 'value', name: 'x', min: -state.xRange, max: state.xRange },
      yAxis: { type: 'value', name: deriv ? "f'(x)" : 'f(x)', scale: true },
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
    var series = [{
      name: '理论密度', type: 'line', data: line, showSymbol: false,
      lineStyle: { width: 2, color: '#2563eb' }, itemStyle: { color: '#2563eb' },
    }];
    var m = ActTheory.outputMoments(act, p, sigma);
    if (act.atom) {
      series[0].markLine = {
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
      tooltip: { trigger: 'axis' },
      xAxis: { type: 'value', name: 'y', min: grid.yLo, max: grid.yHi },
      yAxis: { type: 'value', name: '密度', scale: true },
      dataZoom: [{ type: 'inside' }],
      series: series,
    }, true);
    document.getElementById('distStats').innerHTML =
      '理论：均值 ' + m.mean.toFixed(4) + '，方差 ' + m.variance.toFixed(4) +
      (act.atom ? '，点质量 ' + m.atom.toFixed(3) : '') +
      ' ｜ <span class="stat-dim">样本：未采样（点「采样」叠加直方图）</span>';
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

  window.addEventListener('resize', function () {
    chartFn.resize();
    chartDfn.resize();
    chartDist.resize();
  });

  function boot() {
    buildPanel1Controls();
    buildDistSelect();
    buildParamRows();
    bindGlobal();
    renderPanel1();
    renderPanel2();
  }
  boot();
})();
