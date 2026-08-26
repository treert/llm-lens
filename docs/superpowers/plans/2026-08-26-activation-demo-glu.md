# activation-demo 面板三（GLU 门控族）实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 activation-demo 新增面板三：GLU 门控族二维可视化（热力图 + 切片联动 + 输出分布）。

**Architecture:** 复用既有分层：`functions.js` 加门控注册表，`theory.js` 加 GLU 理论段（一维 GH 积分），`sampler.js` 加相关双变量采样，`app.js` 加三个联动图表。数学框架见 spec §10。

**设计文档:** `docs/superpowers/specs/2026-08-26-activation-demo-design.md` §10

## Global Constraints

- 同前一份计划：中文提交信息 `<type>: <摘要>`；注释/文档中文；数值层无 DOM；
  markdown 公式空格规范；自检 `node web-tools/activation-demo/test/selftest.js` 非零退出。
- 沿用 master + 逐步提交（用户已确认的工作流）。

**Interfaces 总览（各任务产出）:**
- `ActFns.gates`：`[{id, name, gateId, atom, note}]`，gateId 引用已有条目 id
- `ActFns.gateById(id)`；`ActFns.gateAct(gate)` → 门函数条目本身
- `ActSampler.sampleBivariate(gauss, N, sigma, rho)` → `{u, v}`（Float64Array）
- `ActSampler.applyGate(gact, gp, u, v)` → `Float64Array`，`y[i]=u[i]*gact.fn(v[i],gp)`
- `ActTheory.gluOutputMoments(gate, rho, sigma)` → `{mean, variance, atom}`
- `ActTheory.gluOutputDensity(gate, rho, sigma, y)` → number
- `ActTheory.gluConditional(gate, rho, sigma, v0)` → `{gv, mean, sd}`
- `ActTheory.gluBinMass(gate, rho, sigma, y1, y2)` → number（不含 atom，调用方补）
- `ActTheory.gaussEllipse(rho, sigma, k, nPts)` → `[[u,v],...]` 闭合点列
- app.js 状态扩展：`state.gate/rho/sigma2G/NG/v0/sliceMode/slices/samplesG/sampleKeyG`

---

### Task 1: `functions.js` 门控注册表 + 自检

- [ ] **Step 1: 自检追加（失败）**：`check('4 个门', ActFns.gates.length===4)`；逐门 `gateById`/`gateAct` 正确；`gateById('reglu').atom===true`，其余 false。
- [ ] **Step 2: 实现**：在 `list` 之后追加并挂到导出：

```js
  var gates = [
    { id: 'swiglu', name: 'SwiGLU', gateId: 'silu', atom: false,
      note: 'y = u·silu(v)。ρ=0 时 y≈u·v/2，y=0 处有乘积正态型对数尖峰（可积）；E[y]=ρ·E[v·silu(v)]，相关性的符号决定输出均值方向。' },
    { id: 'glu', name: 'GLU(σ)', gateId: 'sigmoid', atom: false,
      note: 'y = u·σ(v)，原始 GLU（Dauphin 2016）。门恒正，输出符号跟随 u。' },
    { id: 'geglu', name: 'GEGLU', gateId: 'gelu-exact', atom: false,
      note: 'y = u·gelu(v)，Gemma 在用。门比 sigmoid 更"软"，v 负时门近似关但不严格为 0。' },
    { id: 'reglu', name: 'ReGLU', gateId: 'relu', atom: true,
      note: 'y = u·relu(v)。v≤0 门关闭 → y=0 有 0.5 点质量（与面板二 ReLU 呼应）；ρ=0 时 y>0 支有闭式 K₀ 密度。' },
  ];
  function gateById(id) {
    for (var i = 0; i < gates.length; i++) if (gates[i].id === id) return gates[i];
    return undefined;
  }
  function gateAct(gate) { return byId(gate.gateId); }
```

- [ ] **Step 3: 跑自检通过 → Commit** `feat: activation-demo 门控注册表`

---

### Task 2: `sampler.js` 相关双变量采样 + 自检

- [ ] **Step 1: 自检追加（失败）**：

```js
  var bv = ActSampler.sampleBivariate(ActSampler.makeRng(11), 200000, 1, 0.7);
  // 样本相关 ≈ 0.7（±0.01）；u、v 边缘方差 ≈ 1（±0.02）
  // applyGate 逐元素正确：relu 门在 v<0 时输出恒 0
```

- [ ] **Step 2: 实现**：

```js
  /** 相关双变量高斯：v=σz1，u=σ(ρz1+√(1−ρ²)z2) */
  function sampleBivariate(gauss, N, sigma, rho) {
    var u = new Float64Array(N), v = new Float64Array(N);
    var s = Math.sqrt(1 - rho * rho);
    for (var i = 0; i < N; i++) {
      var z1 = gauss(), z2 = gauss();
      v[i] = sigma * z1;
      u[i] = sigma * (rho * z1 + s * z2);
    }
    return { u: u, v: v };
  }

  /** 门控输出：y[i] = u[i] · g(v[i]) */
  function applyGate(gact, gp, u, v) {
    var y = new Float64Array(u.length);
    for (var i = 0; i < u.length; i++) y[i] = u[i] * gact.fn(v[i], gp);
    return y;
  }
```

挂到 `ActSampler` 导出。
- [ ] **Step 3: 跑自检通过 → Commit** `feat: activation-demo 相关双变量采样`

---

### Task 3: `theory.js` GLU 理论段 + 自检

- [ ] **Step 1: 自检追加（失败）**：

```js
// 矩 vs MC：ρ ∈ {-0.5, 0, 0.7} × 4 门，N=2e5，均值 abs 0.02、方差 rel 5%
// reglu atom=0.5
// 条件分布：swiglu ρ=0.6 v0=1.3，|v−v0|<0.02 的 MC 条件样本均值/sd vs gluConditional（rel 5%）
// ReGLU ρ=0 闭式：gluOutputDensity(reglu,0,1,y) === exp(logBesselK(0,y))/(2π)，y∈{0.5,1,2}（rel 1e-4）
//   复用 require('../../noise-ops-demo/js/theory.js').logBesselK
// gluBinMass：30 个等宽 bin（±6σ 范围）理论质量+atom vs MC 比例（同面板二容差规则），总质量≈1（2e-3）
// gaussEllipse：逐点 (u²−2ρuv+v²)/(k²σ²(1−ρ²)) ≈ 1
```

- [ ] **Step 2: 实现**（接在 `suggestRange` 之后，内部用 `gh64()`）：

```js
  /** GLU 输出矩：E[y]=ρE[v·g(v)]，E[y²]=E[g(v)²(σ²(1−ρ²)+ρ²v²)]，一维 GH */
  function gluOutputMoments(gate, rho, sigma) {
    var g = ActFns.gateAct(gate), gp = ActFns.defaultParams(g);
    var gh = gh64(), t = sigma * Math.SQRT2, m1 = 0, m2 = 0;
    for (var i = 0; i < gh.x.length; i++) {
      var v = t * gh.x[i], gv = g.fn(v, gp);
      m1 += gh.w[i] * v * gv;
      m2 += gh.w[i] * gv * gv * (sigma * sigma * (1 - rho * rho) + rho * rho * v * v);
    }
    m1 = rho * m1 / Math.sqrt(Math.PI);
    m2 /= Math.sqrt(Math.PI);
    return { mean: m1, variance: Math.max(0, m2 - m1 * m1), atom: gate.atom ? 0.5 : 0 };
  }

  /** 条件分布 y|v0 ~ N(ρv0·g(v0), σ²(1−ρ²)g(v0)²) */
  function gluConditional(gate, rho, sigma, v0) {
    var g = ActFns.gateAct(gate), gp = ActFns.defaultParams(g);
    var gv = g.fn(v0, gp);
    return { gv: gv, mean: rho * v0 * gv,
      sd: sigma * Math.sqrt(1 - rho * rho) * Math.abs(gv) };
  }

  /** 边缘密度：f_Y(y) = E_v[ φ(y/g(v) − ρv; s)/|g(v)| ]，s = σ√(1−ρ²)，一维 GH */
  function gluOutputDensity(gate, rho, sigma, y) {
    var g = ActFns.gateAct(gate), gp = ActFns.defaultParams(g);
    var gh = gh64(), t = sigma * Math.SQRT2;
    var s = sigma * Math.sqrt(1 - rho * rho), sum = 0;
    for (var i = 0; i < gh.x.length; i++) {
      var v = t * gh.x[i], gv = g.fn(v, gp);
      if (gv === 0) continue; // 门关闭支只对 atom 有贡献
      sum += gh.w[i] * ActFns.helpers.normPdf(y / gv - rho * v, s) / Math.abs(gv);
    }
    return sum / Math.sqrt(Math.PI);
  }

  /** 区间质量：E_v[ |Φ(y2/g(v)−ρv; s) − Φ(y1/g(v)−ρv; s)| ]，精确一维 GH */
  function gluBinMass(gate, rho, sigma, y1, y2) {
    var g = ActFns.gateAct(gate), gp = ActFns.defaultParams(g);
    var gh = gh64(), t = sigma * Math.SQRT2;
    var s = sigma * Math.sqrt(1 - rho * rho), sum = 0;
    for (var i = 0; i < gh.x.length; i++) {
      var v = t * gh.x[i], gv = g.fn(v, gp);
      if (gv === 0) continue;
      sum += gh.w[i] * Math.abs(
        ActFns.helpers.normCdf(y2 / gv - rho * v, s) -
        ActFns.helpers.normCdf(y1 / gv - rho * v, s));
    }
    return sum / Math.sqrt(Math.PI);
  }

  /** 联合高斯等高线椭圆：u=kσ·cos t，v=kσ(ρ cos t + √(1−ρ²) sin t) */
  function gaussEllipse(rho, sigma, k, nPts) {
    var pts = [], s = Math.sqrt(1 - rho * rho);
    for (var i = 0; i <= nPts; i++) {
      var t = 2 * Math.PI * i / nPts;
      pts.push([k * sigma * Math.cos(t),
        k * sigma * (rho * Math.cos(t) + s * Math.sin(t))]);
    }
    return pts;
  }
```

挂到 `ActTheory` 导出。
- [ ] **Step 3: 跑自检通过 → Commit** `feat: activation-demo GLU 理论层`

---

### Task 4: 面板三 HTML/CSS

- [ ] **Step 1: `index.html`** 在面板二 `</div>` 后追加面板三（结构：panel-head 含门选择 radios `name="gate"` value swiglu/glu/geglu/reglu；panel-controls 含 `sliderRho/inputRho`（−0.95~0.95 step 0.05 默认 0）、`sliderSigmaG/inputSigmaG`（0.1~4 默认 1）、`inputNG`、`btnSampleG/btnResetG`；`.glu-row` 双列：左 `#chartHeat`+说明，右 slice radios（`name="sliceMode"` dist/surf）+`btnClearSlices`+`#chartSlice`+`#sliceStats`；通栏 `#chartGlu`+`#gluStats`+`#gluNote`）。
- [ ] **Step 2: `css/style.css`** 追加：

```css
/* 面板三：热力图与切片并排 */
.glu-row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.chart-heat { height: 400px; }
.slice-radios { margin: 4px 0; }
@media (max-width: 1100px) { .glu-row { grid-template-columns: 1fr; } }
```

- [ ] **Step 3: Commit** `feat: activation-demo 面板三骨架`

---

### Task 5: app.js 热力图 + 等高线 + 点击

- [ ] **Step 1: 实现**：
  - state 扩展（见 Interfaces 总览，`v0: 1`，`sliceMode: 'dist'`，`slices: []`，`samplesG: null`）；
  - `renderHeat()`：121×121 网格 $[-4\sigma,4\sigma]^2$ 算 `z=u*g(v)`，ECharts heatmap + visualMap（min=−zMax, max=+zMax，色带 `'#2166ac','#f7f7f7','#b2182b'`，`show:false`）；
    叠加 1/2/3σ 椭圆线系（`gaussEllipse(rho, sigma, k, 121)`，灰 `#6b7280` 细线）；
    当前 `v0` 用 `markLine` 水平虚线（`yAxis: state.v0`，色 `#7c3aed`）；
  - `chartHeat.on('click')`：取 `p.value[1]` 为 v0；`sliceMode==='surf'` 时 push 进 `slices`（上限 8，超出 shift）；然后 `renderHeat()` + `renderSlice()`；
  - gate radios / ρ / σ² 滑块绑定：变更 → `renderHeat()` + `renderSlice()` + `renderPanel3()` + `updateSampleButtonG()`。
- [ ] **Step 2: 浏览器验证**：热力图渲染、椭圆随 ρ 旋转、点击出虚线 → Commit `feat: activation-demo 门控热力图`

---

### Task 6: app.js 切片双模式

- [ ] **Step 1: 实现 `renderSlice()`**：
  - 公共：`gluYRange()` → `{yLo, yHi}`，半宽 = `max(4*sd, |mean|+2.5*sd, 1)`（moments 来自 `gluOutputMoments`）；
  - 模式 dist：500 点网格算边缘密度（灰 `#9ca3af` 线），条件高斯曲线（`gluConditional` 的 mean/sd，normPdf，紫 `#7c3aed`）；`sd < 1e-9` 时不画曲线并在 `#sliceStats` 注"该切片门关闭，y≈0"；sliceStats 显示 `v₀, g(v₀), 条件均值/SD`；
  - 模式 surf：`state.slices` 每条直线 `[u, u*g(v)]`（u 扫 ±4σ，色按 PALETTE 循环），图例标 v 值；`btnClearSlices` 清空后重绘；
  - sliceMode radios 切换重绘。
- [ ] **Step 2: 浏览器验证**：v0=1 默认有曲线；切 surf 多次点击成斜率族 → Commit `feat: activation-demo 切片视图`

---

### Task 7: app.js 边缘分布 + 采样统计

- [ ] **Step 1: 实现 `renderPanel3()` + `doSampleG()` + `updateSampleButtonG()`**：
  - 理论密度线（蓝）+ atom 门（reglu）markLine 点质量标注；
  - 采样：`sampleBivariate` + `applyGate`，直方图轮廓叠加（同面板二的 120 箱阶梯折线，橙）；
  - `currentKeyG()` = JSON `[gate, rho, sigma2G, NG]`；失效高亮逻辑同面板二；
  - `#gluStats` 理论/样本均值方差（atom 门加 0 处比例）；`#gluNote` 显示 `gate.note`。
- [ ] **Step 2: 自检无回归 + 浏览器验证**（SwiGLU ρ=0 的 0 处尖峰、ReGLU 点质量与 0 处比例、ρ=0.7 均值漂正）→ Commit `feat: activation-demo 面板三边缘分布与采样`

---

### Task 8: README + 根卡片更新

- [ ] **Step 1: README 更新**：简介改三个面板；文件结构不变；数学背景加 §6「GLU 门控族」
  （条件高斯混合推导、矩公式、ReGLU 点质量与 K₀ 闭式、SwiGLU 对数尖峰、E[y]=ρE[vg(v)] 的"相关性即注意力"）。
- [ ] **Step 2: 根 `index.html` 卡片**：描述改为含面板三（"……另附 GLU 门控族二维面板：门控地形热力图 + 切片联动 + 输出分布随 ρ 变化"）。
- [ ] **Step 3:** `python tools/fix_md_math_spacing.py --apply web-tools/activation-demo/README.md`；
  自检全过；浏览器三面板回归 → Commit `docs: activation-demo 面板三文档与登记`

## Self-Review 记录

- Spec §10 全覆盖：数学框架→Task 3；注册表→Task 1；采样→Task 2；布局→Task 4；热力图/切片/分布→Tasks 5–7；文档→Task 8。
- gluOutputDensity 在 swiglu ρ=0、y≈0 处为对数尖峰：GH 节点不命中 v=0（64 为偶数），值有限，显示正常。
- gluBinMass 不含 atom：测试与 app 中由 `gate.atom` 显式补 0.5（bin 覆盖 y=0 时）。
