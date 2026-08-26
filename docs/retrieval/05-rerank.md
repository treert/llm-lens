# 精排:Rerank 模型的训练与蒸馏

[01-rag.md](01-rag.md) §6 讲了"召回求快、精排求准"的两阶段结构,
本篇深挖精排这一侧:rerank 模型怎么训、三种损失目标的差别、
蒸馏路线,以及 LLM 直接做 rerank 的成本账。

## 1. 为什么召回之后还要精排

bi-encoder 的结构上限:整篇文档**预先**压成一个向量,
query 到来时只能做一次向量点积——所有的相关性判断,
被压缩成"两个点在近不近"。细粒度的词项对齐、逻辑关系
("A 导致 B" vs "B 导致 A" 向量几乎一样)在这个表示里已经丢了。

cross-encoder 把 query 与文档**拼接**输入同一个模型:

```
[CLS] query [SEP] document [SEP] → 相关性分数(单个 logit)
```

query 的每个 token 都能注意力到文档的每个 token,交互是逐词级的——
准,但每对 (query, doc) 都要一次完整前向,$O(k)$ 次推理、无法预计算。
所以它只能放在漏斗末端,处理召回给的 top-50~100。

## 2. 三种训练目标

| 目标 | 形式 | 优点 | 缺点 |
|---|---|---|---|
| pointwise | 每对独立打分(分类/回归) | 简单、数据要求低 | 优化目标 ≠ 排序目标:分数的绝对值校准了,相对顺序却未必对 |
| pairwise | $\max(0,\; m - s^{+} + s^{-})$ 或 $\log \sigma(s^{+} - s^{-})$ | 直接学相对偏好,标注只需"哪个更相关" | 忽略列表整体分布 |
| listwise | 整个候选列表上 softmax / LambdaRank 梯度 | 与 nDCG 等排序指标直接对齐 | 训练复杂、需要完整列表标注 |

实践主流是 **pairwise**:标注成本与效果的平衡点。
listwise 在有完整列表标注时(如蒸馏场景,老师能给全列表打分)收益明显。

## 3. 训练数据:负例必须是"难"的

rerank 的训练数据来自**召回结果 + 相关性标注**:

- 正例:标注的相关文档;
- 负例:**召回阶段返回的 top-$k$ 中不相关的文档**(难负例),
  而不是随机文档——rerank 的战场就是"这些看起来都相关",
  用随机负例训出来的模型学不会这个战场。

这与 [04-embedding.md](04-embedding.md) §3.3 的难负例挖掘是同一思想,
只是阶段不同。MS MARCO passage ranking 是标准训练集。

## 4. 蒸馏:排序能力的接力

rerank 侧有两条方向相反的蒸馏:

- **LLM → cross-encoder**:用 LLM 生成排序标注
  (RankZephyr 思路:让 LLM 对列表排序,蒸馏给小模型),
  免去人工标注,得到可在生产环境跑的小 reranker;
- **cross-encoder → bi-encoder**:MarginMSE
  (见 [04-embedding.md](04-embedding.md) §5),把精排能力反哺回召回模型。

共同风险:**老师的偏差被学生继承甚至放大**——
LLM 老师的位置偏置、长度偏置(见 [07-rag-evaluation.md](07-rag-evaluation.md) §4.2)
会写进学生的权重里。

## 5. 第三条路线:ColBERT 的 late interaction

双塔与 cross-encoder 之间还有折中:ColBERT 把文档的**每个 token**
都编码并预计算存储,查询时计算 token 级相似度的 MaxSim:

$$s(q, d) = \sum_{t \in q} \max_{t' \in d} \; E_t \cdot E_{t'}$$

保留了细粒度交互(比双塔准),又能离线预计算文档侧(比 cross-encoder 快);
代价是索引体积大一个量级(每篇文档存 $L$ 个向量)。

## 6. LLM 直接做 reranker

不训模型,直接让 LLM 排序,三种姿势:

| 姿势 | 调用次数 | 说明 |
|---|---|---|
| pointwise | $O(k)$ | 逐段问"相关吗,0~10 分" |
| pairwise | $O(k \log k) \sim O(k^2)$ | 两两比较后归并成排序 |
| listwise(RankGPT) | 1 | 一次给全部候选,直接输出名次 |

listwise 最省,但有两个坑:上下文窗口限制(一次一般 ≤ 20 个候选)、
以及**位置偏置**(倾向把排在前面的候选排前面,需打乱顺序测两次取平均)。

**成本账**:LLM rerank 适合"低 QPS、高精度、不想维护模型"的场景;
高 QPS 生产环境,蒸馏出的小 cross-encoder 在成本上碾压。

## 7. 工程账本

- **rerank 深度**:对 top-50 还是 top-100 精排?收益曲线通常在
  50~100 之后急剧变平,而延迟随深度线性涨——按"召回-延迟曲线"
  (同 [02-vector-index.md](02-vector-index.md) §6 的方法)选工作点;
- **接口冗余**:给 LLM 的 top-$k$ 必须小于 rerank 深度,
  中间留出 rerank 淘汰的冗余量;
- **长文档截断**:cross-encoder 输入有长度上限,长 chunk 被截断会丢信息——
  对策是滑窗打分取 max/mean 聚合;
- **OOD 退化**:cross-encoder 跨领域退化比双塔更明显
  (交互特征对训练域过拟合),换领域时它是第一个要重训的组件。

## 参考

- Nogueira & Cho, 2019. *Passage Re-ranking with BERT*(cross-encoder 范式)
- Burges, 2010. *From RankNet to LambdaRank to LambdaMART*(listwise 主线)
- Khattab & Zaharia, 2020. *ColBERT*(late interaction)
- Sun et al., 2023. *Is ChatGPT Good at Search? RankGPT 研究*
- Pradeep et al., 2023. *RankZephyr: Effective and Robust Zero-Shot
  Listwise Reranking*(LLM 蒸馏 reranker)
