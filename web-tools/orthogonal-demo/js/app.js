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

  // K 上限：N²（多项式增长，lnK/N = 2lnN/N 全程保持在次指数区域）
  function kMax(N) {
    return N * N;
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
    const firstOrder = [];
    const gumbelMedian = [];
    const betaMedian = [];
    for (let i = 0; i <= samples; i++) {
      const k = Math.exp(
        Math.log(2) + (i / samples) * (Math.log(kHi) - Math.log(2))
      );
      firstOrder.push([k, T.firstOrderMean(N, k, twoSided)]);
      gumbelMedian.push([k, T.maxDotQuantile(0.5, N, k, twoSided)]);
      betaMedian.push([k, T.maxDotQuantileBeta(0.5, N, k, twoSided)]);
    }
    cache1.key = key;
    cache1.firstOrder = firstOrder;
    cache1.gumbelMedian = gumbelMedian;
    cache1.betaMedian = betaMedian;
    return cache1;
  }

  const cache2 = { key: '', xLo: 0, xHi: 0, maxDotGumbel: null, maxDotBeta: null, singlePair: null };
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
    for (let i = 0; i <= samples; i++) {
      const r = xLo + (i / samples) * (xHi - xLo);
      maxDotGumbel.push([r, T.maxDotDensity(r, N, K, twoSided)]);
      maxDotBeta.push([r, T.maxDotDensityBeta(r, N, K, twoSided)]);
      // 双侧模式下对比基线应为单个 |ρ| 的密度：负半轴折叠到正半轴，高度翻倍
      const g = T.pairDotDensity(r, N);
      singlePair.push([r, twoSided ? 2 * g : g]);
    }
    cache2.key = key;
    cache2.xLo = xLo;
    cache2.xHi = xHi;
    cache2.maxDotGumbel = maxDotGumbel;
    cache2.maxDotBeta = maxDotBeta;
    cache2.singlePair = singlePair;
    return cache2;
  }

  // ================= 蒙特卡洛 =================

  const mc = {
    session: null, // MC.createSession 返回 { pool, nextTrajectory }
    gen: null,
    timer: 0,
    running: false,
    paused: false,
    probed: false,
    probedPeak: -1,
    memBytes: MC.TWO_GB,
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
      '内存预算 <strong>' + fmtBytes(mc.memBytes) + '</strong>' +
      (mc.probed
        ? '（实测峰值 ' + fmtBytes(mc.probedPeak) + ' × 70%）'
        : '（名义值，首次运行时实测）') +
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
    if (!mc.probed) {
      // 首次运行：先实测可分配内存（同步 ~0.5-2 s），再开批
      setMcStatus('正在探测可用内存…');
      setTimeout(() => {
        mc.probedPeak = MC.probeMemory();
        mc.memBytes = MC.budgetFromProbe(mc.probedPeak);
        mc.probed = true;
        startMc();
      }, 50);
      return;
    }
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

  // ---- MC 叠加层：中位数曲线 ----
  function mcOverlay1() {
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

    // 多条轨迹：对数抽稀 K 点，跨轨迹中位数 + IQR 带（n ≥ 8 时）
    const ks = [];
    const S = 160;
    let last = 1;
    for (let i = 0; i <= S; i++) {
      const k = Math.round(
        Math.exp(Math.log(2) + (i / S) * (Math.log(covered) - Math.log(2)))
      );
      if (k > last && k <= covered) {
        ks.push(k);
        last = k;
      }
    }
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

  // 阶梯线展开：bin 中心序列 → 平顶阶梯点列
  function toStepPairs(xs, ys, width) {
    const pts = [];
    for (let i = 0; i < xs.length; i++) {
      pts.push([xs[i] - width / 2, ys[i]], [xs[i] + width / 2, ys[i]]);
    }
    return pts;
  }

  // ---- MC 叠加层：密度曲线 ----
  function mcOverlay2(xLo, xHi) {
    const out = { series: [], legendData: [], note: '' };
    if (!mc.session) return out;
    const p = mc.session.pool;

    // ① 点积直方图（K 无关，跨批累积）
    if (p.cntAll > 0) {
      const h = MC.pairHistDensity(p);
      out.series.push({
        name: '模拟点积直方图（M=' + p.cntAll.toExponential(1) + ' 对）',
        type: 'line',
        showSymbol: false,
        color: MC_HIST,
        lineStyle: { width: 1.5 },
        areaStyle: { opacity: 0.08 },
        data: toStepPairs(h.xs, h.ys, h.width),
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
      const grid = [];
      for (let i = 0; i <= 120; i++) grid.push(xLo + (i / 120) * (xHi - xLo));
      const kde = MC.columnKDE(p, state.K, grid);
      const data = grid.map((x, i) => [x, kde.ys[i]]);
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
      out.series.push({
        name: '模拟 max ρ 直方图（K=' + state.K + ', n=' + n + '）',
        type: 'line',
        showSymbol: false,
        color: MC_COLOR,
        lineStyle: { width: 1.5 },
        data: toStepPairs(h.xs, h.ys, h.width),
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

    // lnK/N > SUB_EXP 的区间铺浅灰背景：此区域理论线开始偏离
    const kThreshold = Math.exp(SUB_EXP * N);
    const markAreaData =
      kThreshold < kHi
        ? [[{ xAxis: kThreshold }, { xAxis: kHi }]]
        : [];

    const mcOv = mcOverlay1();

    chart1.setOption(
      {
        title: { text: '最大点乘的中位数水平（N = ' + N + '）', left: 'center', textStyle: { fontSize: 14 } },
        // 系列顺序：一阶近似(橙)、F^M(蓝)、Gumbel(黄)
        color: ['#ff7f0e', '#2563eb', '#eab308'],
        tooltip: {
          trigger: 'axis',
          // 本图系列值均为 ρ，顺带显示对应夹角
          valueFormatter: (v) =>
            typeof v === 'number'
              ? v.toFixed(4) + '（θ≈' + rhoToDeg(v).toFixed(2) + '°）'
              : v,
        },
        legend: { bottom: 0 },
        grid: { left: 60, right: 40, top: 30, bottom: 45 },
        xAxis: {
          type: 'log',
          name: 'K（向量个数）',
          // 轴名在刻度数值行之下、右对齐收于轴端内侧：
          // nameGap 对 end 放置不生效，用 verticalAlign top + padding 下移；
          // padding 右侧留白与右轴 90° 标签隔开
          nameTextStyle: { align: 'right', verticalAlign: 'top', padding: [24, 0, 0, 0] },
          min: 2,
          max: kHi,
          // K 是整数计数；log 轴末端刻度由 10^log10(N²) 反算，带浮点尾数，需取整
          axisLabel: {
            formatter: (v) => Math.round(v).toLocaleString('en-US'),
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
            data: td.firstOrder,
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
            data: td.betaMedian,
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
            data: td.gumbelMedian,
          },
        ].concat(mcOv.series),
      },
      { notMerge: true }
    );
  }

  // ---- 密度曲线：给定 (N,K) 的密度 ----
  function renderChart2() {
    const { N, K, twoSided } = state;
    const td = theoryData2();
    const mcOv = mcOverlay2(td.xLo, td.xHi);

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
          // 角度对应横轴 ρ，valueFormatter 管不到，自定义整行
          formatter: (params) => {
            const rho = Number(params[0].axisValue);
            let html =
              'ρ = ' + rho.toFixed(4) + '（θ≈' + rhoToDeg(rho).toFixed(2) + '°）';
            for (const p of params) {
              html +=
                '<br/>' + p.marker + p.seriesName + '：' +
                Number(p.value[1]).toFixed(3);
            }
            return html;
          },
        },
        legend: { bottom: 0 },
        grid: { left: 60, right: 30, top: 45, bottom: 45 },
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
        yAxis: { type: 'value', name: '密度', nameLocation: 'middle', nameGap: 40, min: 0 },
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
