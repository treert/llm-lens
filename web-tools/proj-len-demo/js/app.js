/**
 * proj-len-demo 的 UI、蒙特卡洛采样与 ECharts 渲染层。
 * 数学公式全部在 theory.js（window.ProjLenTheory）。
 *
 * 模型：y = ABx（单块瓶颈），B 为 M×D（方差 1/D 下投），A 为 D×M（方差 1/M 上投），
 * x 单位向量；观测目标可选完整块 y 或中间层 z = Bx。
 *
 * 采样设计：
 * - 方案一（随机矩阵，固定 x）：卡方因子化 s = χ²_M·χ²_D/(DM) 是精确分布
 *   （各向同性 ⇒ 与 x 方向无关），用 Marsaglia–Tsang 抽两次卡方，O(1)/样本。
 * - 方案二（固定矩阵，随机 x 球面）：**不构造 D×D 的 P = AB**（O(D²M) 内存/计算）。
 *   采样每样本分步 z = Bx（O(MD)）、s = ‖Az‖²（O(DM)），共 O(MD)/样本；
 *   实例矩用 M×M 小矩阵 G = (AᵀA)(BBᵀ)：trW = ‖AB‖²_F = tr G、
 *   tr(W²) = tr(G²)（迹轮换），O(M²D)，免 O(D³) 的 D×D Gram。
 *   中间层 P = B（M×D）直接用，trW = ‖B‖²_F、tr(W²) = ‖BBᵀ‖²_F（M×M Gram）。
 */
(function () {
  'use strict';
  var T = window.ProjLenTheory;

  // ---------- 伪随机数：mulberry32 + Box-Muller（与 proj-dot-demo 同款） ----------
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function makeGaussian(rand) {
    let spare = null;
    return function () {
      if (spare !== null) { const v = spare; spare = null; return v; }
      let u = 0, v = 0, s = 0;
      do {
        u = rand() * 2 - 1;
        v = rand() * 2 - 1;
        s = u * u + v * v;
      } while (s >= 1 || s === 0);
      const m = Math.sqrt((-2 * Math.log(s)) / s);
      spare = v * m;
      return u * m;
    };
  }

  // ---------- 矩阵工具（行主序 Float64Array） ----------
  /**
   * 对称 Gram G = X·Xᵀ（X 为 m×n，G 为 m×m）行分片异步；或带权 G = (XᵀX) 见下。
   * 这里只需要 X·Xᵀ：每行 k 算 G[k][l] = Σ_j X[k][j]·X[l][j]（l = 0..m-1）。
   * 利用对称性只算上三角再镜像，省一半。
   */
  function gramAsync(X, m, n, ctrl, label, onProgress, onDone) {
    const G = new Float64Array(m * m);
    let k = 0;
    (function rowStep() {
      if (ctrl.aborted) return;
      if (ctrl.paused) { setTimeout(rowStep, 100); return; }
      const tStart = performance.now();
      while (k < m && performance.now() - tStart < 24) {
        const offK = k * n;
        for (let l = 0; l <= k; l++) {
          const offL = l * n;
          let acc = 0;
          for (let j = 0; j < n; j++) acc += X[offK + j] * X[offL + j];
          G[k * m + l] = acc;
          G[l * m + k] = acc;
        }
        k++;
      }
      if (k < m) {
        onProgress(label, k / m);
        setTimeout(rowStep, 0);
      } else {
        onDone(G);
      }
    })();
  }

  /** G = (AᵀA)(BBᵀ)：两个 m×m 对称矩阵的乘积，行分片异步（非对称，全算） */
  function matMulAsync(A, m, B, ctrl, label, onProgress, onDone) {
    const out = new Float64Array(m * m);
    let i = 0;
    (function rowStep() {
      if (ctrl.aborted) return;
      if (ctrl.paused) { setTimeout(rowStep, 100); return; }
      const tStart = performance.now();
      while (i < m && performance.now() - tStart < 24) {
        const offI = i * m;
        for (let j = 0; j < m; j++) {
          let acc = 0;
          for (let k = 0; k < m; k++) acc += A[offI + k] * B[k * m + j];
          out[offI + j] = acc;
        }
        i++;
      }
      if (i < m) {
        onProgress(label, i / m);
        setTimeout(rowStep, 0);
      } else {
        onDone(out);
      }
    })();
  }

  /** 通用矩形 C = A·B（A 为 m×n、B 为 n×p），行分片异步（M>D 路径构造 P = AB 用） */
  function matMulRectAsync(A, m, n, B, p, ctrl, label, onProgress, onDone) {
    const out = new Float64Array(m * p);
    let i = 0;
    (function rowStep() {
      if (ctrl.aborted) return;
      if (ctrl.paused) { setTimeout(rowStep, 100); return; }
      const tStart = performance.now();
      while (i < m && performance.now() - tStart < 24) {
        const offI = i * n;
        const offIO = i * p;
        for (let k = 0; k < n; k++) {
          const a = A[offI + k];
          if (a === 0) continue;
          const offK = k * p;
          for (let j = 0; j < p; j++) out[offIO + j] += a * B[offK + j];
        }
        i++;
      }
      if (i < m) {
        onProgress(label, i / m);
        setTimeout(rowStep, 0);
      } else {
        onDone(out);
      }
    })();
  }

  /**
   * Hutchinson 随机迹估计实例矩（大矩阵时替代精确 Gram/小矩阵法，避免 O(min(M,D)³)）。
   * 只作用矩阵-向量乘法，O(k·MD)，k 为探针数（几十个），几秒内出近似值。
   * W = PᵀP，P = AB：
   *   trW = E_v[‖A(Bv)‖²]（v ∈ R^D 标准正态，E[vvᵀ]=I ⇒ E[vᵀWv]=trW）
   *   trW2 = E_v[‖Wv‖²]（W 对称 ⇒ E[vᵀW²v]=tr(W²)）；Wv = Pᵀ(Pv)，
   *     Pv = A(Bv)（D 维），Pᵀw = Bᵀ(Aᵀw)（w=Pv，Aᵀ 为 M×D、Bᵀ 为 D×M）
   * 中间层（P=B）：trW = ‖B‖²_F 已精确（O(MD) 顺手算）；trW2 = E_v[‖BBᵀv‖²]（v ∈ R^M）。
   * 分片异步；onDone(trW, trW2)。
   */
  function hutchinsonAsync(M, D, A, B, isMid, gauss, kProbes, ctrl, onProgress, onDone) {
    let sumTrW = 0, sumTrW2 = 0;
    let done = 0;
    const v = new Float64Array(isMid ? M : D);
    const z = new Float64Array(Math.max(M, D));
    const w = new Float64Array(Math.max(M, D));
    const w2 = new Float64Array(Math.max(M, D));

    // y = X·x（X 行主序 r×c）
    function mv(X, r, c, x, y) {
      for (let i = 0; i < r; i++) {
        const off = i * c;
        let acc = 0;
        for (let j = 0; j < c; j++) acc += X[off + j] * x[j];
        y[i] = acc;
      }
    }
    // y = Xᵀ·x（X 行主序 r×c，结果 c 维）
    function mtv(X, r, c, x, y) {
      for (let j = 0; j < c; j++) y[j] = 0;
      for (let i = 0; i < r; i++) {
        const off = i * c;
        const xi = x[i];
        for (let j = 0; j < c; j++) y[j] += X[off + j] * xi;
      }
    }

    (function probeStep() {
      if (ctrl.aborted) return;
      if (ctrl.paused) { setTimeout(probeStep, 100); return; }
      const tStart = performance.now();
      while (done < kProbes && performance.now() - tStart < 24) {
        let nrm = 0, i;
        if (isMid) {
          // v ∈ R^M；trW2 探针：u = BBᵀv（先 Bᵀv 得 D 维，再 B·() 得 M 维），‖u‖²
          for (i = 0; i < M; i++) v[i] = gauss();
          mtv(B, M, D, v, z);          // z = Bᵀv（D 维）
          mv(B, M, D, z, w);           // w = B z = BBᵀv（M 维）
          for (i = 0; i < M; i++) nrm += w[i] * w[i];
          sumTrW2 += nrm;
          // trW 顺手用 ‖Bv‖² 不行（v 在 R^M）；中间层 trW 用精确 ‖B‖²_F（外面已算），这里跳过
        } else {
          // v ∈ R^D
          for (i = 0; i < D; i++) v[i] = gauss();
          mv(B, M, D, v, z);           // z = Bv（M 维）
          mv(A, D, M, z, w);           // w = Az = ABv = Pv（D 维）
          for (i = 0; i < D; i++) nrm += w[i] * w[i];
          sumTrW += nrm;               // trW 探针：‖Pv‖²
          // trW2 探针：‖Wv‖²，Wv = Pᵀ(Pv) = Bᵀ(Aᵀw)
          mtv(A, D, M, w, z);          // z = Aᵀw（M 维）
          mtv(B, M, D, z, w2);         // w2 = Bᵀz = PᵀPv = Wv（D 维）
          let nrm2 = 0;
          for (i = 0; i < D; i++) nrm2 += w2[i] * w2[i];
          sumTrW2 += nrm2;
        }
        done++;
      }
      if (done < kProbes) {
        onProgress('Hutchinson 估计实例矩', done / kProbes);
        setTimeout(probeStep, 0);
      } else {
        // 中间层 trW 由调用方精确提供；这里返回 NaN 占位
        onDone(isMid ? NaN : sumTrW / kProbes, sumTrW2 / kProbes);
      }
    })();
  }

  /**
   * 分片异步生成本批矩阵并算实例矩，**按 min(M,D) 选小的一侧**，绝不构造大的一侧。
   * obs = 'mid'：P = B（M×D）直接用；trW = ‖B‖²_F，tr(W²) = ‖BBᵀ‖²_F（BBᵀ 为 M×M Gram）。
   * obs = 'chain'：y = ABx，P = AB（D×D）。
   *   - M ≤ D（瓶颈）：不构造 P。实例矩用 M×M 小矩阵 G = (AᵀA)(BBᵀ)：
   *     trW = ‖AB‖²_F = tr G，tr(W²) = tr(G²)（迹轮换）。采样分步 z=Bx、‖Az‖²（O(MD)/样本）。
   *     返回 {A, B, P:null, ...}。
   *   - M > D（宽瓶颈）：构造 P = AB（D×D，比 M×M 小）。trW = ‖P‖²_F，
   *     tr(W²) = ‖PPᵀ‖²_F（PPᵀ 为 D×D Gram）。采样直接用 P（O(D²)/样本，比 O(MD) 省）。
   *     返回 {A:null, B:null, P, ...}。
   * onProgress(label, frac)；onDone({A, B, P, p, trW, trW2})。
   */
  /**
   * 精确实例矩的粗估成本（flops，用于决定是否改用 Hutchinson）。
   * 瓶颈侧：Gram BBᵀ+AᵀA 各 ~M²D（对称减半，常数忽略）+ G 乘 M³；
   * 宽瓶颈侧：P=AB 为 D²M + Gram PPᵀ 为 D³；中间层：Gram BBᵀ ~M²D。
   */
  function exactMomentFlops(M, D, obs) {
    if (obs === 'mid') return M * M * D;
    if (M > D) return D * D * M + D * D * D;
    return 2 * M * M * D + M * M * M;
  }
  // 精确法预估超此秒数则改用 Hutchinson 随机估计（~1e8 flops/s 粗估）
  var MOMENT_EXACT_MAX_SEC = 5;

  function genChainAsync(M, D, obs, varB, alpha, seed, ctrl, onProgress, onDone) {
    const gauss = makeGaussian(mulberry32(seed * 7919 + 17));
    const sigB = Math.sqrt(obs === 'mid' ? varB : 1 / D); // 完整链始终配对 1/D
    const B = new Float64Array(M * D);
    for (let i = 0; i < M * D; i++) B[i] = sigB * gauss();
    const useHutch = exactMomentFlops(M, D, obs) / 1e8 > MOMENT_EXACT_MAX_SEC;

    if (obs === 'mid') {
      let trW = 0;
      for (let i = 0; i < M * D; i++) trW += B[i] * B[i];
      if (useHutch) {
        // 中间层：trW 精确（上式 O(MD)），trW2 = ‖BBᵀ‖²_F 用 Hutchinson
        const k = hutchProbes(M, D);
        hutchinsonAsync(M, D, null, B, true, gauss, k, ctrl, onProgress, function (_, trW2) {
          onDone({ A: null, B: B, P: B, p: M, trW: trW, trW2: trW2, approx: true });
        });
        return;
      }
      gramAsync(B, M, D, ctrl, 'Gram BBᵀ', onProgress, function (BBt) {
        let trW2 = 0;
        for (let i = 0; i < M * M; i++) trW2 += BBt[i] * BBt[i];
        onDone({ A: null, B: B, P: B, p: M, trW: trW, trW2: trW2, approx: false });
      });
      return;
    }

    // 完整链 y = ABx：生成 A（D×M）。
    // 矩阵相关：A = α√(D/M)·Bᵀ + √(1−α²)·G，G 为 D×M 高斯（方差 1/M）。
    // B 行主序 M×D，Bᵀ[d][k] = B[k][d]（A[d][k] 对应 Bᵀ 的第 d 行第 k 列）。
    // 方差校验：α²(D/M)(1/D) + (1−α²)/M = 1/M，A 边缘仍是方差 1/M 各向同性高斯。
    const sigA = 1 / Math.sqrt(M);
    const a1 = alpha * Math.sqrt(D / M);
    const a2 = Math.sqrt(1 - alpha * alpha) * sigA;
    const A = new Float64Array(D * M);
    for (let d = 0; d < D; d++) {
      const offA = d * M;
      for (let k = 0; k < M; k++) {
        A[offA + k] = a1 * B[k * D + d] + a2 * gauss();
      }
    }

    if (useHutch) {
      // Hutchinson：不构造任何方阵，trW、trW2 都随机估计；采样仍分步用 A、B
      const k = hutchProbes(M, D);
      hutchinsonAsync(M, D, A, B, false, gauss, k, ctrl, onProgress, function (trW, trW2) {
        onDone({ A: A, B: B, P: null, p: D, trW: trW, trW2: trW2, approx: true });
      });
      return;
    }

    if (M > D) {
      // 宽瓶颈：构造 P = AB（D×D），Gram PPᵀ（D×D）
      matMulRectAsync(A, D, M, B, D, ctrl, '合成 P = AB', onProgress, function (P) {
        let trW = 0;
        for (let i = 0; i < D * D; i++) trW += P[i] * P[i];
        gramAsync(P, D, D, ctrl, 'Gram PPᵀ', onProgress, function (G) {
          let trW2 = 0;
          for (let i = 0; i < D * D; i++) trW2 += G[i] * G[i];
          onDone({ A: null, B: null, P: P, p: D, trW: trW, trW2: trW2, approx: false });
        });
      });
      return;
    }

    // 瓶颈 M ≤ D：不构造 P，用 M×M 小矩阵
    gramAsync(B, M, D, ctrl, 'Gram BBᵀ', onProgress, function (BBt) {
      // AᵀA：M×M（A 列内积，A 行主序 D×M ⇒ AᵀA[i][j] = Σ_d A[d*M+i]·A[d*M+j]）
      const AtA = new Float64Array(M * M);
      let i = 0;
      (function colStep() {
        if (ctrl.aborted) return;
        if (ctrl.paused) { setTimeout(colStep, 100); return; }
        const tStart = performance.now();
        while (i < M && performance.now() - tStart < 24) {
          for (let j = 0; j <= i; j++) {
            let acc = 0;
            for (let d = 0; d < D; d++) acc += A[d * M + i] * A[d * M + j];
            AtA[i * M + j] = acc;
            AtA[j * M + i] = acc;
          }
          i++;
        }
        if (i < M) {
          onProgress('Gram AᵀA', i / M);
          setTimeout(colStep, 0);
          return;
        }
        matMulAsync(AtA, M, BBt, ctrl, 'G = (AᵀA)(BBᵀ)', onProgress, function (G) {
          let trW = 0;
          for (let k = 0; k < M; k++) trW += G[k * M + k];
          let trW2 = 0;
          for (let a = 0; a < M; a++) {
            for (let b = 0; b < M; b++) trW2 += G[a * M + b] * G[b * M + a];
          }
          onDone({ A: A, B: B, P: null, p: D, trW: trW, trW2: trW2, approx: false });
        });
      })();
    });
  }

  /** Hutchinson 探针数：让估计耗时 ~3s（3e8 flops），每探针 ~4MD flops，下限 16 */
  function hutchProbes(M, D) {
    return Math.max(16, Math.min(200, Math.round(3e8 / (4 * M * D))));
  }

  // ---------- 采样 ----------
  /** 方案一：卡方因子化。chain：s = χ²_M·χ²_D/(DM)；mid：s = varB·χ²_M */
  function fillScheme1(out, i0, i1, M, D, obs, varB, rand01, gauss) {
    for (let i = i0; i < i1; i++) {
      let s = T.chi2Sample(M, rand01, gauss) * (obs === 'mid' ? varB : 1 / D);
      if (obs === 'chain') {
        s *= T.chi2Sample(D, rand01, gauss) / M;
      }
      out[i] = s;
    }
  }

  /**
   * 方案二：x 球面均匀，s = ‖Px‖²。P 由 chain 提供，两种存储：
   * - chain.P 非空（中间层 P=B，或宽瓶颈完整链 P=AB 的 D×D）：直接 s = ‖P·x‖²，O(pD)/样本。
   * - chain.P 为空（瓶颈完整链，存 A、B）：分步 z = Bx（O(MD)）、s = ‖Az‖²（O(DM)），共 O(MD)/样本。
   * X 为 D 缓冲、Z 为 M 缓冲。
   */
  function fillScheme2(out, i0, i1, M, D, chain, gauss, X, Z) {
    const P = chain.P, A = chain.A, B = chain.B;
    const p = chain.p;
    for (let i = i0; i < i1; i++) {
      let nx = 0;
      for (let j = 0; j < D; j++) { X[j] = gauss(); nx += X[j] * X[j]; }
      nx = 1 / Math.sqrt(nx);
      let s = 0;
      if (P) {
        for (let k = 0; k < p; k++) {
          const off = k * D;
          let acc = 0;
          for (let j = 0; j < D; j++) acc += P[off + j] * X[j];
          s += acc * acc * nx * nx;
        }
      } else {
        // 分步：z = Bx（M 维），s = ‖Az‖²
        for (let k = 0; k < M; k++) {
          const off = k * D;
          let acc = 0;
          for (let j = 0; j < D; j++) acc += B[off + j] * X[j];
          Z[k] = acc * nx;
        }
        for (let d = 0; d < D; d++) {
          const off = d * M;
          let acc = 0;
          for (let k = 0; k < M; k++) acc += A[off + k] * Z[k];
          s += acc * acc;
        }
      }
      out[i] = s;
    }
  }

  // ---------- 控件 ----------
  const els = {
    sliderM: document.getElementById('sliderM'),
    inputM: document.getElementById('inputM'),
    sliderD: document.getElementById('sliderD'),
    inputD: document.getElementById('inputD'),
    sliderAlpha: document.getElementById('sliderAlpha'),
    inputAlpha: document.getElementById('inputAlpha'),
    selN: document.getElementById('selN'),
    inputSeed: document.getElementById('inputSeed'),
    btnSeed: document.getElementById('btnSeed'),
    btnResample: document.getElementById('btnResample'),
    btnPause: document.getElementById('btnPause'),
    btnReset: document.getElementById('btnReset'),
    radioNorm: document.getElementById('radioNorm'),
    radioRaw: document.getElementById('radioRaw'),
    radioLinX: document.getElementById('radioLinX'),
    radioLogX: document.getElementById('radioLogX'),
    radioLinY: document.getElementById('radioLinY'),
    radioLogY: document.getElementById('radioLogY'),
    chkTheory: document.getElementById('chkTheory'),
    chkGauss: document.getElementById('chkGauss'),
    chkInst: document.getElementById('chkInst'),
    stats: document.getElementById('stats'),
    statsNote: document.getElementById('statsNote'),
    staleHint: document.getElementById('staleHint'),
  };
  const chart = echarts.init(document.getElementById('chart'));

  const state = {
    scheme: 1, obs: 'chain', M: 64, D: 256,
    midVarMode: 'd', // 中间层 B 方差：'d' = 1/D（配对下投），'m' = 1/M（均值归一）
    alpha: 0, // 矩阵相关：A = α√(D/M)·Bᵀ + √(1−α²)·G（仅方案二完整块；α=0 独立）
    nSamples: 30000, seed: 1, nonce: 0,
    normAxis: true, logX: false, logY: false,
    showTheory: true, showGauss: false, showInst: true,
  };
  const cache = { samplesKey: '', samples: null, chainKey: '', chain: null };

  // ---------- 滑杆映射 ----------
  // M ∈ [1, 4D] 对数刻度（上限随 D 动）；D ∈ [16, 1024] 对数刻度
  function sliderToM(t) {
    return Math.max(1, Math.min(4 * state.D, Math.round(Math.pow(4 * state.D, t / 1000))));
  }
  function mToSlider(m) {
    return Math.max(0, Math.min(1000, Math.round((Math.log(m) / Math.log(4 * state.D)) * 1000)));
  }
  function sliderToD(t) { return Math.max(16, Math.round(Math.pow(2, 4 + (9 * t) / 1000))); }
  function dToSlider(d) { return Math.max(0, Math.min(1000, Math.round(((Math.log2(d) - 4) / 9) * 1000))); }
  // α ∈ [0, 0.99] 线性
  function sliderToAlpha(t) { return (0.99 * t) / 1000; }
  function alphaToSlider(a) { return Math.max(0, Math.min(1000, Math.round((a / 0.99) * 1000))); }

  function fmt(x, digits) { return Number(x).toFixed(digits === undefined ? 4 : digits); }
  function fmtAuto(x) {
    if (!isFinite(x)) return String(x);
    if (Math.abs(x) >= 1e4 || (Math.abs(x) < 1e-3 && x !== 0)) return x.toExponential(2);
    return fmt(x, 4);
  }

  /** 中间层 B 的元素方差（仅 obs = 'mid' 时使用；完整链始终配对 1/D） */
  function midVarB() {
    return state.midVarMode === 'm' ? 1 / state.M : 1 / state.D;
  }

  /**
   * 方案二实际生成矩阵的峰值内存估算（字节，Float64 = 8B），**按 min(M,D) 选小侧**。
   * 中间层：B（M×D）+ Gram BBᵀ（M×M）。
   * 完整链 M ≤ D（瓶颈）：A（D×M）+ B（M×D）+ AᵀA、BBᵀ、G（各 M×M），峰值在算 G 时。
   * 完整链 M > D（宽瓶颈）：P = AB（D×D）+ Gram PPᵀ（D×D）（A、B 用完即弃，峰值含 A、B）。
   * 两种都与 min(M,D) 的平方/立方相关 ⇒ D=8192 配小 M 也可行。
   */
  var MEM_LIMIT = 2e9; // 2 GB
  function scheme2Memory() {
    const M = state.M, D = state.D;
    let cells;
    if (state.obs === 'mid') {
      cells = M * D + M * M;
    } else if (M <= D) {
      cells = 2 * M * D + 3 * M * M; // A、B + AᵀA、BBᵀ、G
    } else {
      cells = 2 * M * D + 2 * D * D; // A、B（生成期）+ P + Gram
    }
    return 8 * cells;
  }
  function scheme2OverMem() {
    return state.scheme === 2 && scheme2Memory() > MEM_LIMIT;
  }

  // ---------- 采样主流程（分块异步） ----------
  /** 矩阵相关 α 只在方案二完整块生效（中间层没有 A；方案一与矩阵无关） */
  function alphaEff() {
    return (state.scheme === 2 && state.obs === 'chain') ? state.alpha : 0;
  }
  function samplingKey() {
    return [state.scheme, state.obs, state.M, state.D,
      state.obs === 'mid' ? state.midVarMode : '',
      alphaEff(), state.nSamples, state.seed, state.nonce].join('|');
  }
  function chainKey() {
    return [state.obs, state.M, state.D,
      state.obs === 'mid' ? state.midVarMode : '', alphaEff(), state.seed].join('|');
  }

  /**
   * 分片流式采样：固定时间片（~24ms/块，按实测自适应块大小），每块后 setTimeout(0)
   * 让出主线程，并把已采样本流式回调 onProgress 供增量重绘。方案二不构造 D×D 的 P：
   * 中间层 Gram BBᵀ、完整链小矩阵 AᵀA/BBᵀ/G 都按 M×M 分片——O(M²D)，与 D 的高次无关。
   * ctrl = {aborted, paused} 控制中止/暂停。
   */
  function runSampling(ctrl, onProgress, onDone) {
    const n = state.nSamples, M = state.M, D = state.D;
    const out = new Float64Array(n);
    const rand01 = mulberry32(state.seed * 131 + 7 + state.nonce * 65537);
    const gauss = makeGaussian(rand01);
    let chain = null;
    let count = 0; // 已采样本数（流式）
    let chunk = 64; // 初始块大小，按实测耗时自适应
    const X = new Float64Array(D);
    const Z = new Float64Array(M);

    function sampleLoop() {
      (function step() {
        if (ctrl.aborted) return;
        if (ctrl.paused) { setTimeout(step, 100); return; }
        const tStart = performance.now();
        const end = Math.min(n, count + chunk);
        if (state.scheme === 1) {
          fillScheme1(out, count, end, M, D, state.obs, midVarB(), rand01, gauss);
        } else {
          fillScheme2(out, count, end, M, D, chain, gauss, X, Z);
        }
        count = end;
        const elapsed = performance.now() - tStart;
        if (elapsed > 0) {
          const target = Math.max(1, Math.round((chunk * 24) / elapsed));
          chunk = Math.min(n, Math.max(1, target));
        }
        onProgress(out.subarray(0, count), count / n, chain);
        if (count < n) setTimeout(step, 0);
        else onDone(out, chain);
      })();
    }

    if (state.scheme === 2) {
      const key = chainKey();
      if (cache.chainKey === key && cache.chain) {
        chain = cache.chain;
        sampleLoop();
      } else {
        genChainAsync(M, D, state.obs, midVarB(), alphaEff(), state.seed, ctrl,
          function (label, frac) {
            onProgress(null, -1, null, label + ' ' + Math.round(100 * frac) + '%…');
          },
          function (c) {
            if (ctrl.aborted) return;
            cache.chainKey = key;
            cache.chain = c;
            chain = c;
            sampleLoop();
          });
      }
    } else {
      cache.chain = null;
      cache.chainKey = '';
      sampleLoop();
    }
  }

  // ---------- 样本统计 ----------
  function sampleStats(samples) {
    const n = samples.length;
    let mean = 0, lmean = 0;
    for (let i = 0; i < n; i++) { mean += samples[i]; lmean += Math.log(samples[i]); }
    mean /= n;
    lmean /= n;
    let m2 = 0, lv = 0;
    for (let i = 0; i < n; i++) {
      const d = samples[i] - mean;
      m2 += d * d;
      const dl = Math.log(samples[i]) - lmean;
      lv += dl * dl;
    }
    const sorted = Float64Array.from(samples).sort();
    return {
      mean: mean, sd: Math.sqrt(m2 / n),
      median: sorted[Math.floor(n / 2)],
      logMean: lmean, logSd: Math.sqrt(lv / n),
    };
  }

  // ---------- 渲染 ----------
  const N_BINS = 121;

  function render(samples, chain) {
    const M = state.M, D = state.D;
    const isMid = state.obs === 'mid';
    const varB = midVarB(); // 仅 isMid 时使用
    const c = isMid ? T.midMean(M, varB) : T.abMean();
    const scale = state.normAxis ? c : 1;
    const muLn = isMid ? T.midLogMean(M, varB) : T.abLogMean(M, D, 1);
    const varLn = isMid ? T.midLogVar(M) : T.abLogVar(M, D, 1);
    const median = Math.exp(muLn);
    const chainVarV = isMid ? T.midVar(M, varB) : T.abVar(M, D, 1);
    const hasExact = true; // L=1：中间层伽马、完整链 K_ν 乘积都是精确闭式
    const hasSamples = !!samples; // 页面刚打开时无样本：只画理论曲线

    // 横轴范围：对数域理论驱动（μ ± 4σ）；有样本时再并入 0.1% / 99.9% 分位
    let lo = Math.exp(muLn - 4 * Math.sqrt(varLn));
    let hi = Math.exp(muLn + 4.5 * Math.sqrt(varLn));
    if (hasSamples) {
      const sorted = Float64Array.from(samples).sort();
      const n = sorted.length;
      lo = Math.min(lo, sorted[Math.floor(n * 0.001)]);
      hi = Math.max(hi, sorted[Math.floor(n * 0.999)]);
    }
    lo /= scale;
    hi /= scale;
    if (state.scheme === 2 && chain) {
      lo = Math.min(lo, T.quenchMean(chain.trW, D) / scale * 0.5);
      hi = Math.max(hi, T.quenchMean(chain.trW, D) / scale * 1.5);
    }
    if (!state.logX) lo = Math.max(lo, 0);
    lo = Math.max(lo, 1e-12);

    // 直方图（等宽或等比 bin，密度高度）；无样本时为空
    const histPts = [];
    if (hasSamples) {
      const edges = new Float64Array(N_BINS + 1);
      for (let b = 0; b <= N_BINS; b++) {
        edges[b] = state.logX
          ? lo * Math.pow(hi / lo, b / N_BINS)
          : lo + ((hi - lo) * b) / N_BINS;
      }
      const counts = new Float64Array(N_BINS);
      for (let i = 0; i < samples.length; i++) {
        const t = samples[i] / scale;
        let b;
        if (state.logX) {
          b = Math.floor((Math.log(t / lo) / Math.log(hi / lo)) * N_BINS);
        } else {
          b = Math.floor(((t - lo) / (hi - lo)) * N_BINS);
        }
        if (b >= 0 && b < N_BINS) counts[b]++;
      }
      for (let b = 0; b < N_BINS; b++) {
        const wBin = edges[b + 1] - edges[b];
        const h = counts[b] / (samples.length * wBin);
        const xm = (edges[b] + edges[b + 1]) / 2;
        histPts.push([xm, h > 0 ? h : null]);
      }
    }

    // 理论曲线（显示域网格；密度变量替换 p_t(t) = scale·f(scale·t)）
    const theoryPts = [], gaussPts = [], instPts = [];
    const N_PT = 400;
    for (let i = 0; i <= N_PT; i++) {
      const t = state.logX
        ? lo * Math.pow(hi / lo, i / N_PT)
        : lo + ((hi - lo) * i) / N_PT;
      const s = t * scale;
      if (state.showTheory && hasExact) {
        const f = isMid
          ? T.gammaDensity(s, M / 2, 2 * varB)
          : T.prodGammaDensityAB(s, M / 2, 2 / D, D / 2, 2 / M);
        theoryPts.push([t, scale * f]);
      }
      if (state.showGauss) gaussPts.push([t, scale * T.gaussDensity(s, c, chainVarV)]);
      if (state.showInst && state.scheme === 2 && chain) {
        const im = T.quenchMean(chain.trW, D);
        const iv = T.quenchVar(chain.trW, chain.trW2, D);
        instPts.push([t, scale * T.gaussDensity(s, im, iv)]);
      }
    }

    // 标线：均值、中位数（对数域近似）、实例中心——全部挂到直方图系列，
    // 不依赖任何理论曲线的显隐状态
    // 归一横轴（÷均值）下均值标线恒在 1，中位数标线的数值才是有效信息
    const medLabel = isMid
      ? (state.normAxis ? '中位数 ≈ e^(−1/M) = ' + fmt(Math.exp(-1 / M), 4)
                        : '中位数 ≈ 均值·e^(−1/M)')
      : '中位数 ≈ e^(−(1/M+1/D))';
    const meanLabel = isMid
      ? '均值 ' + (state.midVarMode === 'm' ? '1' : 'M/D')
      : '均值 1';
    const markData = [
      { xAxis: c / scale, lineStyle: { type: 'dashed', color: '#374151' },
        label: { formatter: meanLabel, color: '#374151', position: 'insideEndTop' } },
      { xAxis: median / scale, lineStyle: { type: 'dotted', color: '#b45309' },
        label: { formatter: medLabel, color: '#b45309', position: 'insideStartTop' } },
    ];
    if (state.scheme === 2 && chain) {
      markData.push({ xAxis: T.quenchMean(chain.trW, D) / scale,
        lineStyle: { type: 'dashed', color: '#dc2626' },
        label: { formatter: 'trW/D', color: '#dc2626', position: 'insideEndTop' } });
    }

    const series = [{
      name: '蒙特卡洛直方图',
      type: 'line', step: 'middle', showSymbol: false, connectNulls: false,
      lineStyle: { width: 1.2, color: '#0d9488' },
      itemStyle: { color: '#0d9488' },
      areaStyle: { opacity: 0.3 },
      data: histPts,
      markLine: {
        symbol: 'none', silent: true,
        label: { fontSize: 11 },
        data: markData,
      },
    }];
    if (state.showTheory && hasExact) {
      // 矩阵相关（α>0）时方案一理论假设 A、B 独立，不再精确——标注参考
      const corrNote = alphaEff() > 0 && !isMid ? '（α>0 矩阵相关，仅参考）' : '（精确）';
      series.push({
        name: isMid ? '理论：伽马（varB·χ²_M，精确）' : '理论：K_ν 乘积伽马' + corrNote,
        type: 'line', showSymbol: false,
        lineStyle: { width: 2, color: '#2563eb' },
        itemStyle: { color: '#2563eb' },
        data: theoryPts,
      });
    }
    if (state.showGauss) {
      series.push({
        name: isMid ? '高斯近似 N(M·varB, 2M·varB²)' : '高斯近似 N(1, 2/M+2/D+4/(MD))',
        type: 'line', showSymbol: false,
        lineStyle: { width: 1.5, color: '#6b7280', type: 'dashed' },
        itemStyle: { color: '#6b7280' },
        data: gaussPts,
      });
    }
    if (state.showInst && state.scheme === 2 && chain) {
      series.push({
        name: '实例高斯（本批链' + (chain.approx ? '，Hutchinson 估计' : '') + '）',
        type: 'line', showSymbol: false,
        lineStyle: { width: 2, color: '#dc2626' },
        itemStyle: { color: '#dc2626' },
        data: instPts,
      });
    }

    const xName = (state.normAxis ? (isMid ? 's / 均值（归一）' : 's（均值恒 1）')
      : isMid ? 's = ‖Bx‖²' : 's = ‖ABx‖²') + (state.logX ? '（对数刻度）' : '');
    chart.setOption({
      animation: false,
      grid: { left: 70, right: 30, top: 50, bottom: 50 },
      legend: { top: 8 },
      tooltip: { trigger: 'axis', valueFormatter: function (v) { return fmt(v, 4); } },
      xAxis: {
        type: state.logX ? 'log' : 'value',
        name: xName,
        nameLocation: 'middle', nameGap: 30,
        min: lo, max: hi,
      },
      yAxis: { type: state.logY ? 'log' : 'value', name: '密度' },
      series: series,
    }, true);

    renderStats(hasSamples ? sampleStats(samples) : null, chain, c, chainVarV, muLn, varLn, median);
  }

  function renderStats(ss, chain, c, chainVarV, muLn, varLn, median) {
    const M = state.M, D = state.D;
    const isMid = state.obs === 'mid';
    const cols = ['样本（' + (ss ? 'N = ' + state.nSamples : '未采样') + '）',
      '理论（精确）',
      state.scheme === 2 ? '实例（本批链）' : '高斯近似'];
    const gaussSd = Math.sqrt(chainVarV);
    let instMean = null, instSd = null;
    const hasChain = state.scheme === 2 && chain;
    if (hasChain) {
      instMean = T.quenchMean(chain.trW, D);
      instSd = Math.sqrt(T.quenchVar(chain.trW, chain.trW2, D));
    }
    // 方案二但链未生成（未采样/已重置/参数已改）时，实例列显示占位
    const instReady = state.scheme === 2 && hasChain;
    const approx = instReady && chain.approx; // Hutchinson 估计：标注近似
    const I = function (f) { return instReady ? f() : '待重采样'; };
    const IA = function (f) { // 实例值 + 近似标注
      return instReady ? f() + (approx ? '（Hutchinson 估计）' : '') : '待重采样';
    };
    const varB = midVarB();
    const medApprox = isMid
      ? '均值·e^(−1/M) = ' + fmtAuto(c * Math.exp(-1 / M))
      : 'e^(−(1/M+1/D)) = ' + fmtAuto(Math.exp(-(1 / M + 1 / D)));
    const varFormula = isMid ? '√(2M·varB²)' : '√(2/M+2/D+4/(MD))';
    const meanNote = isMid
      ? (state.midVarMode === 'm' ? '（= M·(1/M) = 1）' : '（= M/D，配对下投 1/D）')
      : '（恒 1，配对 fan-in）';
    const S = function (f) { return ss ? f(ss) : '—'; }; // 未采样时样本列显示占位
    const rows = [];
    rows.push(['均值', S(function (v) { return fmtAuto(v.mean); }), fmtAuto(c) + meanNote,
      state.scheme === 2 ? IA(function () { return fmtAuto(instMean) + '（= trW/D）'; }) : fmtAuto(c)]);
    rows.push(['标准差', S(function (v) { return fmtAuto(v.sd); }), fmtAuto(gaussSd) + '（' + varFormula + '）',
      state.scheme === 2 ? IA(function () { return fmtAuto(instSd) + '（≈ annealed，self-average）'; }) : fmtAuto(gaussSd)]);
    rows.push(['中位数', S(function (v) { return fmtAuto(v.median); }), fmtAuto(median) + '（≈ ' + medApprox + '）',
      state.scheme === 2 ? '—' : fmtAuto(c)]);
    rows.push(['均值 / 中位数', S(function (v) { return fmt(v.mean / v.median, 3); }),
      fmt(c / median, 3) + (isMid ? '（≈ e^(1/M)）' : '（≈ e^(1/M+1/D)）'),
      state.scheme === 2 ? '≈ 1（已 concentrate）' : '1']);
    rows.push(['E[ln s]', S(function (v) { return fmt(v.logMean, 4); }), fmt(muLn, 4), '—']);
    rows.push(['std(ln s)', S(function (v) { return fmt(v.logSd, 4); }), fmt(Math.sqrt(varLn), 4) +
      (isMid ? '（= √(ψ′(M/2))）' : '（= √(ψ′(M/2)+ψ′(D/2))）'), '—']);
    if (state.scheme === 2) {
      rows.push(['trW / D（实例中心）', '—', fmtAuto(c) + '（系综均值）',
        IA(function () { return fmtAuto(instMean); })]);
      const jumpNote = isMid
        ? '√(2M·varB²/D) = ' + fmtAuto(T.quenchCenterSdMid(M, D, varB))
        : '√((2M+4D+2)/(MD²)) = ' + fmtAuto(T.quenchCenterSdAB1(M, D));
      rows.push(['中心跳动的理论幅度', '—', jumpNote,
        IA(function () { return '本批偏移 ' + fmtAuto(instMean - c); })]);
    }

    let html = '<table class="stats-table"><thead><tr><th>指标</th>';
    for (const col of cols) html += '<th>' + col + '</th>';
    html += '</tr></thead><tbody>';
    for (const r of rows) {
      html += '<tr>';
      for (const cell of r) html += '<td>' + cell + '</td>';
      html += '</tr>';
    }
    html += '</tbody></table>';
    els.stats.innerHTML = html;

    // 附注
    const notes = ['M = ' + M + '，D = ' + D + '，M/D = ' + fmt(M / D, 4)];
    if (isMid) {
      notes.push('中间层 z = Bx（B 方差 = 1/' + (state.midVarMode === 'm' ? 'M' : 'D') +
        '）：s = varB·χ²_M 精确（伽马分布），均值 M·varB = ' + fmtAuto(c) +
        (state.midVarMode === 'm'
          ? '——均值归一（同完整块的保值刻度），方差 2/M 只随 M 收窄'
          : '——瓶颈压缩把典型长度压到 √(M/D)') +
        '，与 x 方向无关（各向同性遗忘方向）');
    } else {
      notes.push('单块 y = ABx：s = χ²_M·χ²_D/(DM)，两个不同形状卡方的乘积（K_ν 闭式，ν = (M−D)/2 = ' +
        fmt((M - D) / 2, 1) + '）；均值恰 1——配对 fan-in（1/D 下投、1/M 上投）保长度期望');
    }
    const aE = alphaEff();
    if (state.scheme === 2 && instReady) {
      if (aE > 0) {
        notes.push('矩阵相关 α = ' + fmt(aE, 2) + '（A = α√(D/M)·Bᵀ + √(1−α²)G）：trW/D = ' +
          fmtAuto(instMean) + '（系综均值 1）——A 学出了 B 的结构，trW 偏离 1、实例中心显著偏移' +
          '（对照 α=0 独立时的自平均 trW/D≈1）；种子 ' + state.seed);
      } else {
        notes.push('种子 ' + state.seed + '：trW/D = ' + fmtAuto(instMean) +
          '（系综均值 ' + fmtAuto(c) + '）；换种子时整条实例曲线随之平移、形状保持' +
          '——实例内涨落 ≈ annealed 大部分散布，中心跳动随维度消失' +
          '（trW 是平方和的自平均；对照点积的 trM 是符号和，涨落不消失）');
      }
    } else if (state.scheme === 2 && !instReady) {
      notes.push('尚未生成本批矩阵（点"运行采样"后显示实例统计）' +
        (aE > 0 ? '；当前 α = ' + fmt(aE, 2) + '（矩阵相关）' : ''));
    }
    if (state.logX || state.logY) {
      notes.push('对数刻度下可见尾部层级：中间层为 e^(−s/θ)（直线），单块 AB 为 e^(−2√(s/θ₁θ₂)) 拉伸指数');
    }
    els.statsNote.textContent = notes.join('；');
  }

  // ---------- 调度 ----------
  // 采样是手动的：改任何重采样控件（方案/观测/方差/M/D/样本数/种子）只标脏 +
  // 刷新显示（直方图沿用已缓存样本，理论曲线即时跟随新参数），不重采样；
  // 打开页面也不采样（只画理论曲线）；只有点"运行采样"才真正重采样。
  // 采样中改参数 → 直接重置（停止并清空，回"尚未采样"，等手动点运行）。
  let pending = false;
  let ctrl = null; // 当前采样的控制对象 {aborted, paused}
  let progressCount = 0; // 采样中已采样本数（供提示文本显示）

  function schedule() {
    if (pending) {
      resetSample();
      return;
    }
    render(cache.samples, cache.chain); // samples 为 null 时只画理论曲线
    setButtons(); // 刷新运行按钮禁用态（方案二内存超限检测）与提示
  }

  function setButtons() {
    els.btnResample.disabled = pending || scheme2OverMem();
    els.btnPause.disabled = !pending;
    els.btnPause.textContent = (pending && ctrl && ctrl.paused) ? '继续' : '暂停';
    els.btnReset.disabled = !pending && !cache.samples;
    updateStaleHint();
  }

  function fmtMem(bytes) {
    return bytes >= 1e9 ? (bytes / 1e9).toFixed(1) + ' GB' : Math.round(bytes / 1e6) + ' MB';
  }

  function runNow() {
    if (pending) return;
    if (scheme2OverMem()) {
      // 方案二实际矩阵内存超限：不采样，只提示
      els.statsNote.textContent = '方案二需实际生成矩阵，当前参数约需 ' + fmtMem(scheme2Memory()) +
        '（超 ' + fmtMem(MEM_LIMIT) + ' 上限）——请减小 M/D 或改用方案一（方案一与矩阵无关，任意维度都可）';
      updateStaleHint();
      return;
    }
    pending = true;
    ctrl = { aborted: false, paused: false };
    progressCount = 0;
    setButtons();
    let lastDraw = 0;
    runSampling(ctrl,
      // onProgress：流式重绘（限频 ~15fps）；frac < 0 表示链合成阶段（stage 为文案）
      function (partial, frac, chain, stage) {
        const now = performance.now();
        if (now - lastDraw < 66 && frac < 1) return;
        lastDraw = now;
        if (frac < 0) {
          els.statsNote.textContent = stage;
          els.staleHint.textContent = stage;
          return;
        }
        progressCount = partial.length;
        els.statsNote.textContent = '采样中 ' + Math.round(100 * frac) + '%（' +
          partial.length + ' / ' + state.nSamples + ' 样本）…';
        els.staleHint.textContent = '采样中 ' + partial.length + ' / ' + state.nSamples + ' 样本…';
        render(partial, chain);
      },
      function (samples, chain) {
        cache.samplesKey = samplingKey();
        cache.samples = samples;
        pending = false;
        ctrl = null;
        setButtons();
        render(cache.samples, cache.chain);
        updateStaleHint();
      });
  }

  /** 暂停/继续切换 */
  function togglePause() {
    if (!pending || !ctrl) return;
    ctrl.paused = !ctrl.paused;
    setButtons();
    if (ctrl.paused) els.statsNote.textContent = '已暂停（点"继续"恢复采样）';
  }

  /** 重置：中止当前采样（若有）并清空已采样本与链缓存，回"尚未采样" */
  function resetSample() {
    if (ctrl) ctrl.aborted = true; // 让分片循环自然退出
    pending = false;
    ctrl = null;
    progressCount = 0;
    cache.samples = null;
    cache.samplesKey = '';
    cache.chain = null;
    cache.chainKey = '';
    setButtons();
    render(null, null);
    updateStaleHint();
  }

  /** 无样本提示"尚未采样"；有样本但参数已改时提示"参数已修改"；采样中显示进度 */
  function updateStaleHint() {
    if (pending) {
      const cnt = progressCount + ' / ' + state.nSamples + ' 样本';
      els.staleHint.textContent = (ctrl && ctrl.paused)
        ? '已暂停（' + cnt + '）'
        : '采样中 ' + cnt + '…';
      return;
    }
    if (scheme2OverMem()) {
      els.staleHint.textContent = '方案二矩阵内存约 ' + fmtMem(scheme2Memory()) + '（超 ' + fmtMem(MEM_LIMIT) + '），请减小 M/D 或用方案一';
      return;
    }
    if (!cache.samples) {
      els.staleHint.textContent = '尚未采样，点"运行采样"生成蒙特卡洛直方图';
    } else if (samplingKey() !== cache.samplesKey) {
      els.staleHint.textContent = '参数已修改，点"运行采样"生效（当前直方图仍是旧样本）';
    } else {
      els.staleHint.textContent = '';
    }
  }

  // ---------- 事件 ----------
  /** α 只在方案二完整块有意义（中间层没有 A、方案一与矩阵无关） */
  function updateAlphaEnabled() {
    const en = state.scheme === 2 && state.obs === 'chain';
    els.sliderAlpha.disabled = !en;
    els.inputAlpha.disabled = !en;
  }
  function setScheme(sc) {
    state.scheme = sc;
    const fixed = sc === 2;
    els.inputSeed.disabled = !fixed;
    els.btnSeed.disabled = !fixed;
    els.chkInst.disabled = !fixed;
    updateAlphaEnabled();
    schedule();
  }
  document.querySelectorAll('input[name="scheme"]').forEach(function (r) {
    r.addEventListener('change', function () { setScheme(Number(r.value)); });
  });

  function setObs(obs) {
    state.obs = obs;
    const isMid = obs === 'mid';
    // 中间层方差选项只在观测中间层时可用（完整块始终配对 1/D）
    document.querySelectorAll('input[name="midvar"]').forEach(function (r) {
      r.disabled = !isMid;
    });
    updateAlphaEnabled();
    schedule();
  }
  document.querySelectorAll('input[name="obs"]').forEach(function (r) {
    r.addEventListener('change', function () { setObs(r.value); });
  });
  document.querySelectorAll('input[name="midvar"]').forEach(function (r) {
    r.addEventListener('change', function () {
      state.midVarMode = r.value;
      schedule();
    });
  });

  function setM(m, fromSlider) {
    state.M = Math.max(1, Math.min(4 * state.D, Math.round(m)));
    if (!fromSlider) els.sliderM.value = mToSlider(state.M);
    if (document.activeElement !== els.inputM) els.inputM.value = state.M;
    schedule();
  }
  function setD(d, fromSlider) {
    state.D = Math.max(16, Math.min(8192, Math.round(d)));
    if (!fromSlider) els.sliderD.value = dToSlider(state.D);
    if (document.activeElement !== els.inputD) els.inputD.value = state.D;
    if (state.M > 4 * state.D) setM(4 * state.D, false); // M 上限 4D，随 D 收拢
    else els.sliderM.value = mToSlider(state.M); // 上限变了，重映射滑杆位置
    els.inputM.max = 4 * state.D;
    schedule();
  }

  els.sliderM.addEventListener('input', function () {
    setM(sliderToM(Number(els.sliderM.value)), true);
    els.inputM.value = state.M;
  });
  els.inputM.addEventListener('change', function () {
    const v = Number(els.inputM.value);
    if (isFinite(v) && v >= 1) setM(v, false);
  });
  els.sliderD.addEventListener('input', function () {
    setD(sliderToD(Number(els.sliderD.value)), true);
    els.inputD.value = state.D;
  });
  els.inputD.addEventListener('change', function () {
    const v = Number(els.inputD.value);
    if (isFinite(v) && v >= 16) setD(v, false);
  });
  function setAlpha(a, fromSlider) {
    state.alpha = Math.max(0, Math.min(0.99, a));
    if (!fromSlider) els.sliderAlpha.value = alphaToSlider(state.alpha);
    if (document.activeElement !== els.inputAlpha) els.inputAlpha.value = fmt(state.alpha, 2);
    schedule();
  }
  els.sliderAlpha.addEventListener('input', function () {
    setAlpha(sliderToAlpha(Number(els.sliderAlpha.value)), true);
    els.inputAlpha.value = fmt(state.alpha, 2);
  });
  els.inputAlpha.addEventListener('change', function () {
    const v = Number(els.inputAlpha.value);
    if (isFinite(v) && v >= 0) setAlpha(v, false);
  });
  els.selN.addEventListener('change', function () {
    state.nSamples = Number(els.selN.value);
    schedule();
  });
  els.inputSeed.addEventListener('change', function () {
    const v = Number(els.inputSeed.value);
    if (isFinite(v)) { state.seed = Math.round(v); schedule(); }
  });
  els.btnSeed.addEventListener('click', function () {
    state.seed += 1;
    els.inputSeed.value = state.seed;
    schedule(); // 只改种子，标脏等手动采样
  });
  els.btnResample.addEventListener('click', function () {
    state.nonce += 1; // 矩阵链不动，只换采样流
    runNow(); // 手动触发采样
  });
  els.btnPause.addEventListener('click', togglePause);
  els.btnReset.addEventListener('click', resetSample);

  // 显示控件（横轴/纵轴刻度、归一、各理论曲线勾选）：不重采样、不重置，只重绘当前缓存
  function refreshView() {
    if (pending) return; // 采样中由流式回调负责重绘，此处不动
    render(cache.samples, cache.chain);
  }
  els.radioNorm.addEventListener('change', function () {
    state.normAxis = true; refreshView();
  });
  els.radioRaw.addEventListener('change', function () {
    state.normAxis = false; refreshView();
  });
  els.radioLinX.addEventListener('change', function () {
    state.logX = false; refreshView();
  });
  els.radioLogX.addEventListener('change', function () {
    state.logX = true; refreshView();
  });
  els.radioLinY.addEventListener('change', function () {
    state.logY = false; refreshView();
  });
  els.radioLogY.addEventListener('change', function () {
    state.logY = true; refreshView();
  });
  els.chkTheory.addEventListener('change', function () {
    state.showTheory = els.chkTheory.checked; refreshView();
  });
  els.chkGauss.addEventListener('change', function () {
    state.showGauss = els.chkGauss.checked; refreshView();
  });
  els.chkInst.addEventListener('change', function () {
    state.showInst = els.chkInst.checked; refreshView();
  });
  window.addEventListener('resize', function () { chart.resize(); });

  // ---------- 初始化 ----------
  els.sliderM.value = mToSlider(state.M);
  els.inputM.value = state.M;
  els.inputM.max = 4 * state.D;
  els.sliderD.value = dToSlider(state.D);
  els.inputD.value = state.D;
  els.sliderAlpha.value = alphaToSlider(state.alpha);
  els.inputAlpha.value = fmt(state.alpha, 2);
  els.inputSeed.value = state.seed;
  setObs('chain');
  setScheme(1);
  updateAlphaEnabled();
  setButtons();
  // 打开页面不采样：只画理论曲线，提示点"运行采样"
  render(null, null);
  updateStaleHint();
})();
