/**
 * 交互与渲染：滑杆/输入框状态同步，调用 Theory 生成理论曲线、MC 驱动
 * 蒙特卡洛流式模拟（generator + 时间片），ECharts 绘图。
 * 依赖：echarts（CDN）、Theory（js/theory.js）、MC（js/montecarlo.js）。
 */
(function () {
  'use strict';

  const T = window.Theory;
  const MC = window.MC;

  const N_MIN = 64;
  const N_MAX = 8192;

  // K 上限：N³（多项式增长，lnK/N = 3lnN/N 全程保持在次指数区域）
  function kMax(N) {
    return N * N * N;
  }
  // 渐近区域阈值：lnK/N 小于 SUB_EXP 视为次指数区域（理论线可信），
  // 大于 TRANSITION 视为超出定理覆盖范围
  const SUB_EXP = 0.05;
  const TRANSITION = 0.25;

  // 单条轨迹的默认运算预算（乘加次数；由"单条时长"档位覆盖）
  const OPS_BUDGET_DEFAULT = 2.5e9;
  // MC 叠加层配色
  const MC_COLOR = '#7c3aed';
  const MC_BAND = 'rgba(124,58,237,0.15)';
  const MC_HIST = '#6b7280';

  function regimeOf(N, K) {
    const ratio = Math.log(K) / N;
    if (ratio < SUB_EXP) return { cls: 'note-green', text: '次指数区域' };
    if (ratio < TRANSITION) return { cls: 'note-yellow', text: '过渡区' };
    return { cls: 'note-red', text: '超出定理覆盖范围' };
  }

  // ρ → 夹角度数；clamp 防御 acos 定义域（浮点尾数可能微超 ±1）
  function rhoToDeg(r) {
    return (Math.acos(Math.max(-1, Math.min(1, r))) * 180) / Math.PI;
  }

  const els = {
    sliderN: document.getElementById('sliderN'),
    inputN: document.getElementById('inputN'),
    sliderK: document.getElementById('sliderK'),
    inputK: document.getElementById('inputK'),
    chkTwoSided: document.getElementById('chkTwoSided'),
    stats: document.getElementById('stats'),
    btnMcStart: document.getElementById('btnMcStart'),
    btnMcPause: document.getElementById('btnMcPause'),
    btnMcReset: document.getElementById('btnMcReset'),
    selOps: document.getElementById('selOps'),
    selRuns: document.getElementById('selRuns'),
    chkAuto: document.getElementById('chkAuto'),
    chkSeed: document.getElementById('chkSeed'),
    inputSeed: document.getElementById('inputSeed'),
    mcStatus: document.getElementById('mcStatus'),
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
    els.inputK.max = kMax(state.N);
    els.inputK.value = state.K;
    els.sliderK.value = valueToSlider(state.K, 2, kMax(state.N));
    els.chkTwoSided.checked = state.twoSided;
  }

  // ---- 理论数据缓存：MC 运行中 4fps 重绘时避免重算理论曲线 ----
  const cache1 = { key: '', firstOrder: null, gumbelMedian: null, betaMedian: null };
  function theoryData1() {
    const { N, twoSided } = state;
    const key = N + '|' + twoSided;
    if (cache1.key === key) return cache1;
    const kHi = kMax(N);
    const samples = 160;
    // 整数 K 网格（去重）：模拟叠加层复用同一网格，
    // 保证 axis 触发的 tooltip 中理论与模拟严格同 K 全部显示
    const ks = [];
    for (let i = 0; i <= samples; i++) {
      const k = Math.round(
        Math.exp(Math.log(2) + (i / samples) * (Math.log(kHi) - Math.log(2)))
      );
      if (ks.length === 0 || k > ks[ks.length - 1]) ks.push(k);
    }
    const firstOrder = [];
    const gumbelMedian = [];
    const betaMedian = [];
    for (const k of ks) {
      firstOrder.push([k, T.firstOrderMean(N, k, twoSided)]);
      gumbelMedian.push([k, T.maxDotQuantile(0.5, N, k, twoSided)]);
      betaMedian.push([k, T.maxDotQuantileBeta(0.5, N, k, twoSided)]);
    }
    cache1.key = key;
    cache1.ks = ks;
    cache1.firstOrder = firstOrder;
    cache1.gumbelMedian = gumbelMedian;
    cache1.betaMedian = betaMedian;
    return cache1;
  }

  const cache2 = { key: '', xLo: 0, xHi: 0, maxDotGumbel: null, maxDotBeta: null, singlePair: null, probPair: null, probMax: null };
  function theoryData2() {
    const { N, K, twoSided } = state;
    const key = N + '|' + K + '|' + twoSided;
    if (cache2.key === key) return cache2;
    // 横轴范围：覆盖 F^M 分布的 [0.1%, 99.9%] 分位区间与单对密度可见范围（≈6σ）
    const q001 = T.maxDotQuantileBeta(0.001, N, K, twoSided);
    const q999 = Math.max(
      T.maxDotQuantileBeta(0.999, N, K, twoSided),
      T.maxDotQuantile(0.999, N, K, twoSided)
    );
    // 双侧 max|ρ| 无负支撑；单侧小 K 时负半轴有可观质量，自动扩展
    const xLo = twoSided ? 0 : Math.min(0, q001 * 1.1);
    const xHi = Math.min(1, Math.max(q999, 6 / Math.sqrt(N)) * 1.05);
    // 大 K 时密度峰极窄（宽度 ~1/(2N·maxρ)），采样点要足够密才能画光滑
    const samples = 1200;
    const maxDotGumbel = [];
    const maxDotBeta = [];
    const singlePair = [];
    const probPair = [];
    const probMax = [];
    for (let i = 0; i <= samples; i++) {
      const r = xLo + (i / samples) * (xHi - xLo);
      maxDotGumbel.push([r, T.maxDotDensity(r, N, K, twoSided)]);
      maxDotBeta.push([r, T.maxDotDensityBeta(r, N, K, twoSided)]);
      // 双侧模式下对比基线应为单个 |ρ| 的密度：负半轴折叠到正半轴，高度翻倍
      const g = T.pairDotDensity(r, N);
      singlePair.push([r, twoSided ? 2 * g : g]);
      // 右轴近似正交概率：始终按 |ρ|（双侧）口径，与单双侧开关无关；
      // r ≤ 0 时概率为 0（保持网格一致，axis tip 负半段也能显示该系列）
      probPair.push([r, r > 0 ? T.pairAbsDotCDF(r, N) : 0]);
      probMax.push([r, r > 0 ? T.maxDotCDFBeta(r, N, K, true) : 0]);
    }
    cache2.key = key;
    cache2.xLo = xLo;
    cache2.xHi = xHi;
    // 理论系列的 x 网格：模拟叠加层插值到同一网格，保证 tip 全程同 ρ 显示
    cache2.xs = maxDotBeta.map((p) => p[0]);
    cache2.maxDotGumbel = maxDotGumbel;
    cache2.maxDotBeta = maxDotBeta;
    cache2.singlePair = singlePair;
    cache2.probPair = probPair;
    cache2.probMax = probMax;
    return cache2;
  }

  // ================= 蒙特卡洛 =================

  const mc = {
    session: null, // MC.createSession 返回 { pool, nextTrajectory }
    gen: null,
    timer: 0,
    running: false,
    paused: false,
    memBytes: MC.TWO_GB, // 固定 2 GB 名义上限（早期版本曾首跑实测，已移除）
    batchStart: 0,
    lastPairs: 0,
    totalPairs: 0,
    lastFrame: 0,
  };

  function fmtBytes(b) {
    if (b >= 1024 * 1024 * 1024) return (b / (1024 * 1024 * 1024)).toFixed(1) + ' GB';
    if (b >= 1024 * 1024) return (b / (1024 * 1024)).toFixed(1) + ' MB';
    return (b / 1024).toFixed(1) + ' KB';
  }

  function seedSelected() {
    if (!els.chkSeed.checked) return null;
    const v = Math.floor(Number(els.inputSeed.value));
    return isFinite(v) ? v >>> 0 : 42;
  }

  /** 实际内存账目：向量缓冲 + 曲线池 + 直方图；附浏览器页面堆（若可用） */
  function mcMemoryText() {
    let s = '';
    if (mc.session) {
      const p = mc.session.pool;
      const mem = MC.poolMemoryBytes(p);
      const vec = mc.gen ? 4 * p.KMax * p.N : 0;
      s +=
        ' · 内存 ' + fmtBytes(vec + mem.curves + mem.hist) +
        '（向量 ' + fmtBytes(vec) + ' + 曲线池 ' + fmtBytes(mem.curves) + '）';
    }
    if (window.performance && performance.memory && performance.memory.usedJSHeapSize) {
      s += ' · 页面堆 ' + fmtBytes(performance.memory.usedJSHeapSize);
    }
    return s;
  }

  function opsSelected() {
    const v = Number(els.selOps.value);
    return isFinite(v) && v > 0 ? v : OPS_BUDGET_DEFAULT;
  }

  /** 轨迹数上限：0 表示不限。上限只约束追加，不清空已有数据 */
  function runsLimit() {
    const v = Number(els.selRuns.value);
    return v > 0 ? v : Infinity;
  }

  /** 当前 (N, 预算, 单条时长档位) 下的 K 上限与单条轨迹内存 */
  function mcPlan() {
    const kM = MC.computeKMax(state.N, mc.memBytes, opsSelected());
    const bytes = 4 * kM * state.N;
    return { kM: kM, bytes: bytes };
  }

  function updateMcButtons() {
    const hasSession = !!mc.session;
    const completed = hasSession && mc.session.pool.runsTotal > 0;
    const limReached =
      hasSession && mc.session.pool.runsTotal >= runsLimit();
    els.btnMcStart.textContent = completed ? '追加一条' : '开始模拟';
    els.btnMcStart.disabled = mc.running || mc.paused || limReached;
    els.btnMcPause.disabled = !(mc.running || mc.paused);
    els.btnMcPause.textContent = mc.paused ? '继续' : '暂停';
    els.btnMcReset.disabled = !hasSession;
    els.chkSeed.disabled = mc.running || mc.paused;
    els.inputSeed.disabled = mc.running || mc.paused || !els.chkSeed.checked;
  }

  function setMcStatus(html) {
    els.mcStatus.innerHTML = html;
  }

  function mcInfoText() {
    const plan = mcPlan();
    let s =
      '内存预算 <strong>' + fmtBytes(mc.memBytes) + '</strong>（固定上限）' +
      '；K 上限 ≈ <strong>' + plan.kM.toLocaleString('en-US') + '</strong>' +
      '（每条轨迹内存 ≈ ' + fmtBytes(plan.bytes) + '）';
    if (mc.session && mc.session.pool.runsTotal > 0) {
      const p = mc.session.pool;
      const lim = runsLimit();
      const mean = p.cntAll > 0 ? p.sumAll / p.cntAll : 0;
      const rel = 1.25 * 0.055 / Math.sqrt(p.runsTotal);
      s +=
        '；累计轨迹 <strong class="legend-note note-blue">R=' +
        p.runsTotal + (isFinite(lim) ? '/' + lim : '') + '</strong>' +
        '（中位数涨落 ≈ ±' + (rel * 100).toFixed(1) + '%）' +
        '；点积均值 ' + mean.toFixed(5) + '（应 ≈ 0）';
    }
    return s + mcMemoryText();
  }

  function resetMc() {
    if (mc.timer) clearTimeout(mc.timer);
    mc.session = null;
    mc.gen = null;
    mc.timer = 0;
    mc.running = false;
    mc.paused = false;
    updateMcButtons();
    setMcStatus(mcInfoText());
  }

  function startMc() {
    if (!mc.session) {
      const plan = mcPlan();
      mc.session = MC.createSession({
        N: state.N,
        KMax: plan.kM,
        twoSided: state.twoSided,
        seed: seedSelected(),
      });
    }
    mc.gen = mc.session.nextTrajectory();
    mc.totalPairs = (mc.session.pool.KMax * (mc.session.pool.KMax - 1)) / 2;
    mc.lastPairs = 0;
    mc.batchStart = performance.now();
    mc.running = true;
    mc.paused = false;
    updateMcButtons();
    drive();
  }

  function pauseMc() {
    if (mc.paused) {
      mc.paused = false;
      updateMcButtons();
      drive();
    } else {
      mc.paused = true;
      if (mc.timer) clearTimeout(mc.timer);
      setMcStatus('已暂停 · ' + mcProgressText());
      updateMcButtons();
    }
  }

  function mcProgressText() {
    const p = mc.session.pool;
    const elapsed = (performance.now() - mc.batchStart) / 1000;
    const rate = elapsed > 0.1 ? mc.lastPairs / elapsed : 0;
    const remain = rate > 0 ? (mc.totalPairs - mc.lastPairs) / rate : NaN;
    return (
      '第 ' + (p.runsTotal + 1) + ' 条轨迹 · K=' + (p.current ? p.current.k : 1) + '/' + p.KMax +
      ' · 已完成 R=' + p.runsTotal +
      mcMemoryText() +
      ' · 本条已用 ' + elapsed.toFixed(1) + ' s' +
      (isFinite(remain) ? ' · 剩余 ≈ ' + remain.toFixed(0) + ' s' : '')
    );
  }

  function drive() {
    if (!mc.gen) return;
    const t0 = performance.now();
    let res = null;
    do {
      res = mc.gen.next();
      if (!res.done) mc.lastPairs = res.value.pairs;
    } while (!res.done && performance.now() - t0 < 30);

    if (res.done) {
      onTrajectoryDone();
      return;
    }
    setMcStatus('运行中 · ' + mcProgressText());
    // 帧率节流：最多 ~4fps 重绘
    const now = performance.now();
    if (now - mc.lastFrame > 250) {
      mc.lastFrame = now;
      render();
    }
    mc.timer = setTimeout(drive, 0);
  }

  function onTrajectoryDone() {
    render();
    const lim = runsLimit();
    if (els.chkAuto.checked && mc.session.pool.runsTotal < lim) {
      // 自动追加：曲线入池后立即开始下一条轨迹（受轨迹数上限约束）
      mc.gen = mc.session.nextTrajectory();
      mc.totalPairs =
        (mc.session.pool.KMax * (mc.session.pool.KMax - 1)) / 2;
      mc.lastPairs = 0;
      mc.batchStart = performance.now();
      mc.timer = setTimeout(drive, 0);
      return;
    }
    mc.gen = null;
    mc.running = false;
    mc.paused = false;
    setMcStatus(
      (mc.session.pool.runsTotal >= lim ? '已达轨迹上限 · ' : '本条完成 · ') +
        mcInfoText()
    );
    updateMcButtons();
  }

  // ---- MC 叠加层：中位数曲线（ksAll 复用理论曲线的整数 K 网格） ----
  function mcOverlay1(ksAll) {
    if (!mc.session) return { series: [], legendData: [] };
    const p = mc.session.pool;
    const covered = MC.coveredK(p);
    if (covered < 2) return { series: [], legendData: [] };
    const n = MC.columnValues(p, covered).length;
    if (n === 0) return { series: [], legendData: [] };

    // 仅一条轨迹时：画它的原始"破纪录"路径
    if (n === 1) {
      const arr = p.runsTotal > 0 ? p.maxRuns[p.runsTotal - 1] : p.current.arr;
      const path = [];
      for (let k = 2; k <= covered; k++) path.push([k, arr[k]]);
      return {
        series: [
          {
            name: '模拟轨迹（单条原始路径）',
            type: 'line',
            showSymbol: false,
            color: MC_COLOR,
            lineStyle: { width: 1.5 },
            data: path,
          },
        ],
        legendData: ['模拟轨迹（单条原始路径）'],
      };
    }

    // 多条轨迹：与理论曲线同一整数 K 网格（≤ covered 部分），
    // 跨轨迹中位数 + IQR 带（n ≥ 8 时）；网格一致 tip 才能同 K 全显
    const ks = [];
    for (const k of ksAll) if (k >= 2 && k <= covered) ks.push(k);
    const med = [];
    const q1 = [];
    const q3 = [];
    for (const k of ks) {
      const agg = MC.aggregateColumn(p, k);
      med.push([k, agg.median]);
      q1.push([k, agg.q1]);
      q3.push([k, agg.q3]);
    }
    const series = [
      {
        name: '模拟中位数（R=' + n + '）',
        type: 'line',
        showSymbol: false,
        color: MC_COLOR,
        lineStyle: { width: 2 },
        data: med,
      },
    ];
    const legendData = ['模拟中位数（R=' + n + '）'];
    if (n >= 8) {
      // IQR 带：闭合多边形（正向 q1 + 反向 q3）。log 轴下 stack 面积
      // 会填充到 y=0 而非下边界，故不用堆叠方案
      const polygon = q1.concat(q3.reverse());
      series.push({
        name: '模拟 IQR 带（25~75%）',
        type: 'line',
        showSymbol: false,
        silent: true,
        lineStyle: { opacity: 0 },
        areaStyle: { color: MC_BAND },
        data: polygon,
      });
      legendData.push('模拟 IQR 带（25~75%）');
    }
    return { series: series, legendData: legendData };
  }

  // ---- MC 叠加层：密度曲线（xsTheory 复用理论曲线的 x 网格） ----
  function mcOverlay2(xLo, xHi, xsTheory) {
    const out = { series: [], legendData: [], note: '' };
    if (!mc.session) return out;
    const p = mc.session.pool;

    // ① 点积直方图（K 无关，跨批累积）
    if (p.cntAll > 0) {
      const h = MC.pairHistDensity(p);
      // 阶梯直方图按 bin 取值映射到理论网格（x 与其他系列严格一致，
      // 否则 axis tip 以最近数据点为锚，网格不同的系列整行缺失）
      const lo = h.xs[0] - h.width / 2;
      const nb = h.ys.length;
      const data = xsTheory.map((x) => {
        let b = Math.floor((x - lo) / h.width);
        if (b < 0) b = 0;
        else if (b >= nb) b = nb - 1;
        return [x, h.ys[b]];
      });
      out.series.push({
        name: '模拟点积直方图（M=' + p.cntAll.toExponential(1) + ' 对）',
        type: 'line',
        showSymbol: false,
        color: MC_HIST,
        lineStyle: { width: 1.5 },
        areaStyle: { opacity: 0.08 },
        data: data,
      });
      out.legendData.push(out.series[out.series.length - 1].name);
    }

    // ② 当前 K_select 处的 max ρ 模拟密度
    const covered = MC.coveredK(p);
    if (state.K > covered) {
      out.note =
        '当前 K=' + state.K + ' 超出模拟进度（K=' + covered +
        (p.runsTotal > 0 ? '，上限 ' + p.KMax : '，第 1 条进行中') +
        '），仅显示理论曲线。';
      return out;
    }
    const n = MC.columnValues(p, state.K).length;
    if (n < 8) {
      out.note = 'K=' + state.K + ' 处模拟样本 n=' + n + ' < 8，密度暂不显示。';
      return out;
    }
    if (n >= 32) {
      // KDE 在 121 点粗网格求值（成本 O(n×121)），再线性插值到理论曲线
      // 的网格：高斯核光滑，插值无损观感；x 严格一致，tip 全程显示模拟行
      const grid = [];
      for (let i = 0; i <= 120; i++) grid.push(xLo + (i / 120) * (xHi - xLo));
      const kde = MC.columnKDE(p, state.K, grid);
      const step = (xHi - xLo) / 120;
      const data = xsTheory.map((x) => {
        let t = (x - xLo) / step;
        if (t < 0) t = 0;
        else if (t > 120) t = 120;
        const i0 = Math.floor(t);
        const i1 = i0 >= 120 ? 120 : i0 + 1;
        const f = t - i0;
        return [x, kde.ys[i0] * (1 - f) + kde.ys[i1] * f];
      });
      out.series.push({
        name: '模拟 max ρ 密度（K=' + state.K + ', n=' + n + ', KDE）',
        type: 'line',
        showSymbol: false,
        color: MC_COLOR,
        lineStyle: { width: 2 },
        data: data,
      });
    } else {
      const h = MC.columnHist(p, state.K, xLo, xHi, 24);
      // 阶梯直方图按 bin 取值映射到理论网格（x 一致，tip 全程显示）
      const data = xsTheory.map((x) => {
        let b = Math.floor((x - xLo) / h.width);
        if (b < 0) b = 0;
        else if (b > 23) b = 23;
        return [x, h.ys[b]];
      });
      out.series.push({
        name: '模拟 max ρ 直方图（K=' + state.K + ', n=' + n + '）',
        type: 'line',
        showSymbol: false,
        color: MC_COLOR,
        lineStyle: { width: 1.5 },
        data: data,
      });
    }
    out.legendData.push(out.series[out.series.length - 1].name);
    return out;
  }

  // ---- 中位数曲线：max ρ ~ K ----
  function renderChart1() {
    const { N, K } = state;
    const kHi = kMax(N);
    const td = theoryData1();
    // y 轴显式上限：一阶近似全程为最高线（README §5.3），×1.05 吸收模拟涨落；
    // 右侧角度副轴的范围据此换算
    const yMax = 1.05 * T.firstOrderMean(N, kHi, state.twoSided);

    // ECharts log 轴在跨度 ~12 个数量级时只生成个位数刻度，且
    // customValues / interval / splitNumber 对 log 轴均无效（5.5.0 实测）；
    // 改用 value 轴 + 数据 x 手动取 log10(K)，刻度间隔 interval:1（每个数量级一个）
    const lg = Math.log10;
    const logData = (data) => data.map((p) => [lg(p[0]), p[1]]);

    // lnK/N > SUB_EXP 的区间铺浅灰背景：此区域理论线开始偏离
    const kThreshold = Math.exp(SUB_EXP * N);
    const markAreaData =
      kThreshold < kHi
        ? [[{ xAxis: lg(kThreshold) }, { xAxis: lg(kHi) }]]
        : [];

    const mcOv = mcOverlay1(td.ks);
    // MC 叠加层数据同样是 [K, ρ]，统一做 log10 变换
    const mcSeries = mcOv.series.map((s) =>
      Object.assign({}, s, { data: logData(s.data) })
    );

    chart1.setOption(
      {
        title: { text: '最大点乘的中位数水平（N = ' + N + '）', left: 'center', textStyle: { fontSize: 14 } },
        // 系列顺序：一阶近似(橙)、F^M(蓝)、Gumbel(黄)
        color: ['#ff7f0e', '#2563eb', '#eab308'],
        tooltip: {
          trigger: 'axis',
          // 理论与模拟系列共享同一整数 K 网格（theoryData1.ks），
          // axis 触发下同 K 全部显示；IQR 带是闭合多边形（每个 K 两个点），不进 tip。
          // 横轴是 log10(K)，显示时还原为 K（round 吸收浮点尾数）
          formatter: (params) => {
            let html =
              'K = ' + Math.round(Math.pow(10, Number(params[0].axisValue))).toLocaleString('en-US');
            for (const p of params) {
              if (p.seriesName.indexOf('IQR') >= 0) continue;
              const v = p.value[1];
              if (typeof v !== 'number') continue;
              html +=
                '<br/>' + p.marker + p.seriesName + '：' +
                v.toFixed(4) + '（θ≈' + rhoToDeg(v).toFixed(2) + '°）';
            }
            return html;
          },
        },
        legend: { bottom: 0 },
        grid: { left: 60, right: 40, top: 30, bottom: 45 },
        xAxis: {
          // 名义上是对数刻度：坐标值 = log10(K)（见上面 lg/logData 变换）
          type: 'value',
          name: 'K（向量个数）',
          // 轴名在刻度数值行之下、右对齐收于轴端内侧：
          // nameGap 对 end 放置不生效，用 verticalAlign top + padding 下移；
          // padding 右侧留白与右轴 90° 标签隔开
          nameTextStyle: { align: 'right', verticalAlign: 'top', padding: [24, 0, 0, 0] },
          // min 取 log 空间整数（K=1）：value 轴 interval 网格从 min 起对齐，
          // 这样 interval:1 的刻度恰好落在每个数量级上（数据从 K=2 起，
          // 左端空出 0.3 个数量级可忽略）
          min: 0,
          max: lg(kHi),
          // 每个数量级一个刻度；刻度值为整数 log10(K)，标签还原为 K 的整数码
          interval: 1,
          axisLabel: {
            formatter: (v) => Math.round(Math.pow(10, v)).toLocaleString('en-US'),
            // 右端点（K 上限）标签贴轴缘会被截断，且位数随 N 变化，直接隐藏；
            // 上限可在控件输入框看到
            showMaxLabel: false,
          },
        },
        yAxis: [
          // 显式 max 是浮点数，ECharts 会在轴顶端画出原始 max 标签，隐藏之
          { type: 'value', name: 'max ρ', min: 0, max: yMax, axisLabel: { showMaxLabel: false } },
          // 右侧角度副轴：θ=arccos(ρ) 非线性，但本图 |ρ| 范围内近线性，
          // 端点严格对齐（ρ=0 ↔ 90°），中间刻度像素偏差 <0.5%；
          // inverse 下 nameLocation 'start' 使轴名落在顶端（避免与 x 轴名相撞）
          {
            type: 'value',
            name: '夹角 θ(°)',
            nameLocation: 'start',
            position: 'right',
            min: rhoToDeg(yMax),
            max: 90,
            inverse: true,
            splitLine: { show: false },
            axisLabel: { formatter: (v) => String(Number(v.toFixed(1))) },
          },
        ],
        series: [
          {
            name: state.twoSided ? '一阶近似 √(2·ln(2M)/N)' : '一阶近似 √(2·lnM/N)',
            type: 'line',
            showSymbol: false,
            smooth: true,
            data: logData(td.firstOrder),
            markArea: {
              silent: true,
              itemStyle: { color: 'rgba(0,0,0,0.045)' },
              label: {
                show: markAreaData.length > 0,
                position: 'insideTop',
                color: '#999',
                fontSize: 11,
                formatter: 'lnK/N > ' + SUB_EXP + '：理论线开始偏离',
              },
              data: markAreaData,
            },
          },
          {
            name: 'F^M 中位数（beta 幂次，全 K 适用）',
            type: 'line',
            showSymbol: false,
            smooth: true,
            data: logData(td.betaMedian),
            markLine: {
              silent: true,
              // markLine 不参与新旧数据的差值过渡，每次 setOption 都重建；
              // 不关动画的话 MC 流式重绘时竖线会反复播入场动画
              animation: false,
              symbol: 'none',
              lineStyle: { type: 'dashed', color: '#999' },
              label: { formatter: 'K = ' + K, position: 'insideEndTop' },
              data: [{ xAxis: lg(K) }],
            },
          },
          {
            name: 'Gumbel 中位数（渐近）',
            type: 'line',
            showSymbol: false,
            smooth: true,
            lineStyle: { type: 'dashed' },
            data: logData(td.gumbelMedian),
          },
        ].concat(mcSeries),
      },
      { notMerge: true }
    );
  }

  // ---- 密度曲线：给定 (N,K) 的密度 ----
  function renderChart2() {
    const { N, K, twoSided } = state;
    const td = theoryData2();
    const mcOv = mcOverlay2(td.xLo, td.xHi, td.xs);

    chart2.setOption(
      {
        title: {
          text: '密度曲线（N = ' + N + ', K = ' + K + '）',
          left: 'center',
          textStyle: { fontSize: 14 },
        },
        // 系列顺序：F^M(蓝)、Gumbel(黄)、单对(绿)；两条正交概率曲线自带显式配色
        color: ['#2563eb', '#eab308', '#2ca02c'],
        tooltip: {
          trigger: 'axis',
          // 角度对应横轴 ρ，valueFormatter 管不到，自定义整行
          formatter: (params) => {
            const rho = Number(params[0].axisValue);
            let html =
              'ρ = ' + rho.toFixed(4) + '（θ≈' + rhoToDeg(rho).toFixed(2) + '°）';
            for (const p of params) {
              const v = Number(p.value[1]);
              if (!isFinite(v)) continue;
              // 正交概率曲线在右轴（0~1），按百分比显示
              const isProb = p.seriesName.indexOf('正交概率') >= 0;
              html +=
                '<br/>' + p.marker + p.seriesName + '：' +
                (isProb ? (v * 100).toFixed(2) + '%' : v.toFixed(3));
            }
            return html;
          },
        },
        legend: { bottom: 0, type: 'scroll' },
        // 右侧概率轴占位：right 由 30 加宽到 56
        grid: { left: 60, right: 56, top: 45, bottom: 45 },
        xAxis: [
          {
            type: 'value',
            name: 'ρ（点乘）',
            // 轴名在刻度数值行之下、右对齐收于轴端内侧（同中位数曲线的处理）
            nameTextStyle: { align: 'right', verticalAlign: 'top', padding: [24, 0, 0, 0] },
            min: td.xLo,
            max: td.xHi,
            // 轴端点默认显示 xHi 的完整浮点精度；限制为 3 位有效数字并去尾零
            axisLabel: {
              formatter: (v) => String(Number(v.toPrecision(3))),
            },
          },
          // 顶部角度副轴：ρ 越大 θ 越小，inverse 使角度左大右小；
          // 单侧小 K 时左端 >90°（负 ρ），符合几何事实
          {
            type: 'value',
            name: '夹角 θ(°)',
            // 轴名在右端、刻度行上方（与中位数曲线角度轴名同侧）：inverse 下
            // 'start' 即视觉右端；start/end 放置时 nameGap 不生效、轴名贴轴线，
            // 用 verticalAlign bottom + padding 抬到刻度行上方
            nameLocation: 'start',
            nameTextStyle: { align: 'right', verticalAlign: 'bottom', padding: [0, 0, 26, 0] },
            position: 'top',
            min: rhoToDeg(td.xHi),
            max: rhoToDeg(td.xLo),
            inverse: true,
            splitLine: { show: false },
            axisLabel: { formatter: (v) => String(Number(v.toFixed(1))) },
          },
        ],
        // 轴名放中部：顶部要留给角度副轴的 90° 端点标签
        yAxis: [
          { type: 'value', name: '密度', nameLocation: 'middle', nameGap: 40, min: 0 },
          // 右侧概率轴（0~1）：近似正交概率曲线专用
          {
            type: 'value',
            name: '概率',
            nameLocation: 'middle',
            nameGap: 32,
            position: 'right',
            min: 0,
            max: 1,
            splitLine: { show: false },
            axisLabel: { formatter: (v) => String(Number(v.toFixed(1))) },
          },
        ],
        series: [
          {
            name: 'max ρ 密度（F^M，beta 幂次）',
            type: 'line',
            showSymbol: false,
            smooth: true,
            areaStyle: { opacity: 0.08 },
            data: td.maxDotBeta,
          },
          {
            name: 'max ρ 密度（Gumbel 渐近）',
            type: 'line',
            showSymbol: false,
            smooth: true,
            lineStyle: { type: 'dashed' },
            data: td.maxDotGumbel,
          },
          {
            name: twoSided
              ? '单对 |ρ| 密度（精确 beta ×2 折叠）'
              : '单对 ρ 密度（精确 beta）',
            type: 'line',
            showSymbol: false,
            smooth: true,
            data: td.singlePair,
            // σ 参考线：单对点乘标准差 1/√N 的 1/2/3 倍处画竖线
            markLine: {
              silent: true,
              // 同 chart1：markLine 每次 setOption 都重建，关闭入场动画防闪烁
              animation: false,
              symbol: 'none',
              lineStyle: { type: 'dashed', color: '#999' },
              label: { position: 'insideEndTop', color: '#999', fontSize: 11 },
              data: [
                { xAxis: 1 / Math.sqrt(N), label: { formatter: 'σ' } },
                { xAxis: 2 / Math.sqrt(N), label: { formatter: '2σ' } },
                { xAxis: 3 / Math.sqrt(N), label: { formatter: '3σ' } },
              ],
            },
          },
          // 近似正交概率曲线（右轴）：把 |ρ| < t 当作正交；始终双侧口径，
          // 与 UI 单双侧开关无关。数据网格与密度系列一致（负半轴为 0）
          {
            name: 'P(|ρ|≤t) 单对正交概率',
            type: 'line',
            showSymbol: false,
            yAxisIndex: 1,
            color: '#dc2626',
            lineStyle: { width: 2 },
            data: td.probPair,
          },
          {
            name: 'P(max|ρ|≤t) 全部正交概率',
            type: 'line',
            showSymbol: false,
            yAxisIndex: 1,
            color: '#0d9488',
            lineStyle: { width: 2 },
            data: td.probMax,
          },
        ].concat(mcOv.series),
      },
      { notMerge: true }
    );

    const med = T.maxDotQuantileBeta(0.5, N, K, twoSided);
    const medGumbel = T.maxDotQuantile(0.5, N, K, twoSided);
    const first = T.firstOrderMean(N, K, twoSided);
    const angleDeg = rhoToDeg(med).toFixed(2);
    const regime = regimeOf(N, K);
    els.stats.innerHTML =
      'max ρ 中位数：<strong class="legend-note note-blue">F^M ≈ ' + med.toFixed(4) + '</strong>' +
      '，<strong class="legend-note note-yellow">Gumbel ≈ ' + medGumbel.toFixed(4) + '</strong>' +
      '；<strong class="legend-note note-orange">一阶近似（众数口径）≈ ' + first.toFixed(4) + '</strong>' +
      '；对应最小夹角 ≈ <strong class="legend-note note-gray">' + angleDeg + '°</strong>。' +
      '单对 ρ 的标准差 σ ≈ 1/√N ≈ ' + (1 / Math.sqrt(N)).toFixed(4) + '。' +
      ' lnK/N = ' + (Math.log(K) / N).toFixed(4) +
      ' <strong class="legend-note ' + regime.cls + '">' + regime.text + '</strong>' +
      (mcOv.note ? ' <strong class="legend-note note-gray">' + mcOv.note + '</strong>' : '');
  }

  function render() {
    syncControls();
    renderChart1();
    renderChart2();
  }

  // ---- 事件 ----
  els.sliderN.addEventListener('input', () => {
    state.N = sliderToValue(Number(els.sliderN.value), N_MIN, N_MAX);
    state.K = Math.min(state.K, kMax(state.N));
    resetMc(); // N 变化：模拟全部作废（数据严格以 N 为参数）
    render();
  });
  els.inputN.addEventListener('change', () => {
    let v = Math.round(Number(els.inputN.value));
    if (!isFinite(v)) v = state.N;
    state.N = Math.max(N_MIN, Math.min(N_MAX, v));
    state.K = Math.min(state.K, kMax(state.N));
    resetMc();
    render();
  });
  els.sliderK.addEventListener('input', () => {
    state.K = sliderToValue(Number(els.sliderK.value), 2, kMax(state.N));
    render(); // K 是视图游标，不影响模拟
  });
  els.inputK.addEventListener('change', () => {
    let v = Math.round(Number(els.inputK.value));
    if (!isFinite(v)) v = state.K;
    state.K = Math.max(2, Math.min(kMax(state.N), v));
    render();
  });
  els.chkTwoSided.addEventListener('change', () => {
    state.twoSided = els.chkTwoSided.checked;
    resetMc(); // 单双侧数据不可换算，清空
    render();
  });
  els.chkSeed.addEventListener('change', () => {
    els.inputSeed.disabled = !els.chkSeed.checked;
    resetMc(); // 种子口径变化等同重置
    render();
  });
  els.selOps.addEventListener('change', () => {
    resetMc(); // 单条时长变化会改变 K_max，已有曲线长度不一致，等同重置
    render();
  });
  els.selRuns.addEventListener('change', () => {
    // 轨迹上限是控制参数：只影响后续追加，不清空已有数据
    updateMcButtons();
    setMcStatus(mcInfoText());
  });
  els.inputSeed.addEventListener('change', () => {
    resetMc();
    render();
  });
  els.btnMcStart.addEventListener('click', startMc);
  els.btnMcPause.addEventListener('click', pauseMc);
  els.btnMcReset.addEventListener('click', () => {
    resetMc();
    render();
  });
  window.addEventListener('resize', () => {
    chart1.resize();
    chart2.resize();
  });

  updateMcButtons();
  setMcStatus(mcInfoText());
  render();
})();
