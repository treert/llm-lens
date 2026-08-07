/**
 * 随机矩阵奇异值谱的理论密度（纯数学层，无 DOM 依赖）。
 *
 * 设定：A、B 为 H×D 独立高斯矩阵，条目方差 1/D（对应初始化缩放），
 * c = H/D ∈ (0, 1]。大维极限下谱密度只依赖比值 c：
 *
 * 1. 单矩阵 A：平方奇异值服从 Marchenko-Pastur 律 MP_c（均值归一为 1），
 *    支撑 [(1-√c)², (1+√c)²]，方差 c。
 * 2. 乘积 M = AᵀB：非零平方奇异值 = 两个 Wishart 之积的特征值，
 *    服从自由乘性卷积 MP_c ⊠ MP_c（均值 1，方差 2c）。
 *    c = 1 时即 Fuss-Catalan 分布 FC_2，支撑 [0, 27/4]。
 *    其矩生成函数 M(w) 满足三次方程（S-变换推出）：
 *      w c² M³ + 2w(1-c)c M² + (w(1-c)² - 1) M + 1 = 0,  w = 1/x
 *    密度 p(x) = |Im M(1/x)| / (π x)（取复根，支撑内存在一对共轭复根）。
 *
 * 参考：docs/qk-spectrum.md §3；Fuss-Catalan 数为矩的分布。
 */
(function (global) {
  'use strict';

  // ---------- Marchenko-Pastur（单矩阵） ----------

  /** MP_c 支撑端点 [(1-√c)², (1+√c)²] */
  function mpSupport(c) {
    const s = Math.sqrt(c);
    return [(1 - s) * (1 - s), (1 + s) * (1 + s)];
  }

  /**
   * MP_c 密度（均值归一为 1 的参数化）：
   * p(x) = √((x₊-x)(x-x₋)) / (2π c x)，x ∈ [x₋, x₊]
   */
  function mpDensity(x, c) {
    const se = mpSupport(c);
    if (x <= se[0] || x >= se[1]) return 0;
    return Math.sqrt((se[1] - x) * (x - se[0])) / (2 * Math.PI * c * x);
  }

  // ---------- 乘积谱 MP_c ⊠ MP_c ----------

  /**
   * 乘积谱的支撑端点。
   * 分支点：M_± = (3c ± √(c²+8c)) / (4c)，x = 1/g(M)，
   * g(M) = (M-1) / [M(1-c+cM)²]。
   * c = 1 时 M₋ = 0，x₋ = 0（左端贴 0，密度按 x^(-2/3) 发散）。
   */
  function productSupport(c) {
    const s = Math.sqrt(c * c + 8 * c);
    const g = function (M) {
      const t = 1 - c + c * M;
      return (M - 1) / (M * t * t);
    };
    const xPlus = 1 / g((3 * c + s) / (4 * c));
    if (c >= 1) return [0, xPlus];
    return [1 / g((3 * c - s) / (4 * c)), xPlus];
  }

  /**
   * 一般自由乘积 MP_a ⊠ MP_b 的密度 p(x)：解三次方程取复根虚部
   * （Cardano，实系数避免复数运算）。
   * S-变换 S(z) = 1/[(1+az)(1+bz)] 给出矩生成函数方程
   *   w ab M³ + w(a+b-2ab) M² + (w(1-a)(1-b) - 1) M + 1 = 0,  w = 1/x
   * 首一化系数（除以 w·ab）：
   *   A2 = 1/a + 1/b - 2,  A1 = ((1-a)(1-b) - x)/(ab),  A0 = x/(ab)
   * 支撑内恰有一对共轭复根；支撑外三根皆实，密度为 0。
   */
  function freePoissonProductDensity(x, a, b) {
    if (x <= 0) return 0;
    const A2 = 1 / a + 1 / b - 2;
    const A1 = ((1 - a) * (1 - b) - x) / (a * b);
    const A0 = x / (a * b);
    //  depressed cubic t³ + pt + q = 0（M = t - A2/3，平移为实数不影响虚部）
    const p = A1 - (A2 * A2) / 3;
    const q = (2 * A2 * A2 * A2) / 27 - (A2 * A1) / 3 + A0;
    const disc = (q * q) / 4 + (p * p * p) / 27;
    if (disc <= 1e-18) return 0;
    const sq = Math.sqrt(disc);
    const u = Math.cbrt(-q / 2 + sq);
    const v = Math.cbrt(-q / 2 - sq);
    const imAbs = (Math.sqrt(3) / 2) * Math.abs(u - v);
    return imAbs / (Math.PI * x);
  }

  /** AᵀB（D×D 摆法）的非零平方奇异值谱：MP_c ⊠ MP_c */
  function productDensity(x, c) {
    return freePoissonProductDensity(x, c, c);
  }

  /**
   * AB（H×H 摆法，A: H×D, B: D×H）平方奇异值谱，÷c 归一后的密度：
   * 即 MP_c ⊠ MP_1（均值 1，方差 1+c）。未归一的 σ²(AB) 均值 c、方差 c²(1+c)。
   */
  function abNormDensity(x, c) {
    return freePoissonProductDensity(x, c, 1);
  }

  /**
   * MP_c ⊠ MP_1 的支撑 [0, x₊]（b=1 时左端贴 0）。
   * 分支点方程退化为二次 2cM² + (1-4c)M - 2(1-c) = 0，
   * M₊ = (4c - 1 + √(1+8c)) / (4c)，x₊ = 1/w(M₊)，w(M) = (M-1)/[M²(1-c+cM)]。
   * c=1 时与 FC_2 一致：M₊ = 3/2，x₊ = 27/4。
   */
  function abNormSupport(c) {
    const mPlus = (4 * c - 1 + Math.sqrt(1 + 8 * c)) / (4 * c);
    const w = (mPlus - 1) / (mPlus * mPlus * (1 - c + c * mPlus));
    return [0, 1 / w];
  }

  /** 未归一 σ²(AB) 的方差 c²(1+c)（均值 c） */
  function abVariance(c) {
    return c * c * (1 + c);
  }

  /**
   * σ 轴（奇异值本身，非平方）密度的变量替换：p_σ(s) = 2s·p_σ²(s²)。
   * kind: 'mp' | 'product'
   */
  function sigmaDensity(s, c, kind) {
    if (s <= 0) return 0;
    const f = kind === 'mp' ? mpDensity : productDensity;
    return 2 * s * f(s * s, c);
  }

  global.SpecTheory = {
    mpSupport: mpSupport,
    mpDensity: mpDensity,
    productSupport: productSupport,
    productDensity: productDensity,
    freePoissonProductDensity: freePoissonProductDensity,
    abNormDensity: abNormDensity,
    abNormSupport: abNormSupport,
    abVariance: abVariance,
    sigmaDensity: sigmaDensity,
    // 解析矩：均值均归一为 1；方差 MP_c 为 c，乘积为 2c，AB(÷c) 为 1+c
    mpVariance: function (c) { return c; },
    productVariance: function (c) { return 2 * c; },
  };
})(typeof window !== 'undefined' ? window : globalThis);
