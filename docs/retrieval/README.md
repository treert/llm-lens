# 检索与 RAG(Retrieval)笔记

目标:从信息检索(IR)视角理解"给 LLM 外接知识"这件事——
检索侧的数学与系统(embedding、向量索引、BM25、排序),
以及应用侧的 RAG 架构、评估与失败模式。

与仓库其他文档的分工:

- `docs/inference/` 关注模型**本身**怎么跑得快(kernel、serving、量化);
- 本目录关注模型**之外**的知识怎么进 prompt——模型权重全程不动,
  变的是输入。

## 笔记索引

**总览**:

1. [01-rag.md](01-rag.md):RAG 全景——三条知识注入路线的取舍、
   embedding 与对比学习、ANN 索引(IVF/HNSW/PQ)、BM25 与混合检索、
   rerank 两阶段结构、chunking 策略、检索/生成分开评估、
   失败模式与进阶变体(HyDE、Agentic RAG、GraphRAG)、RAG vs 长上下文。

**检索理论线**(模型怎么算"相似",建议按序):

2. [02-vector-index.md](02-vector-index.md):向量索引原理深挖——
   高维距离集中与维度灾难、IVF 的复杂度账(手算)、PQ 乘积量化与 ADC、
   HNSW 小世界图导航 + 分层结构(手算例子)、参数权衡与选型速查。
3. [03-bm25.md](03-bm25.md):稀疏检索的概率模型——BIM 二值独立模型
   推出 IDF、2-Poisson 推出词频饱和 $k_1$、长度归一 $b$;
   倒排索引工程(差分压缩、跳表、WAND 精确剪枝)、BM25 为何至今打不死。
4. [10-grep.md](10-grep.md):grep 与文本扫描——正则的线性时间保证、
   Boyer-Moore 亚线性字面量搜索、扫描的带宽账、编码 agent
   为何用"零索引 + 强策略"而非向量检索、"无索引→倒排→向量→长上下文"频谱。
   (编号 10 是后补的,主题上紧承 03,建议读完 03 就读它。)
5. [04-embedding.md](04-embedding.md):稠密检索模型训练——双塔与 pooling、
   InfoNCE 深挖($\tau$ 的梯度聚焦效应、in-batch negatives、难负例挖掘
   与假阴性)、cross-encoder 蒸馏(MarginMSE)、指令式与 MRL 可变维度、
   LLM 改造 embedding、BEIR/MTEB 榜单读法。
6. [05-rerank.md](05-rerank.md):精排模型——pointwise/pairwise/listwise
   损失对比、难负例训练数据、双向蒸馏路线(LLM↔cross-encoder↔bi-encoder)、
   ColBERT late interaction、LLM 做 reranker 的成本账、工程参数与失效模式。

**应用与系统线**(RAG 怎么把检索用好):

7. [06-query-and-retrieval-strategy.md](06-query-and-retrieval-strategy.md):
   查询侧策略——多轮指代消解改写、HyDE 原理(答案找答案的分布对齐)、
   子问题分解与多跳、混合检索融合(RRF 为什么稳)、迭代检索。
8. [07-rag-evaluation.md](07-rag-evaluation.md):评估体系——
   Recall/MRR/nDCG 的定义与选用、pooling 标注法、
   LLM-as-judge 的四类偏差与对策、RAGAS 拆解、归因消融与线上指标。
9. [08-advanced-rag.md](08-advanced-rag.md):进阶形态——Agentic RAG
   (检索作为 agent 工具的循环结构与成本账)、GraphRAG(图谱+社区摘要,
   回答全局性问题)、长期记忆与 RAG 的边界、形态对比与演进路径。

**动手**:

10. [09-hands-on.md](09-hands-on.md):本机最小实操——纯 numpy TF-IDF
    检索 `docs/` 语料、自写 IVF 实测 recall-nprobe 曲线;
    配套脚本在 `demos/retrieval/`。

原则:先讲清"检索为什么有效"(向量空间、概率模型),
再落到"工程上怎么调"(索引参数、chunk 粒度、评估指标),
能在本机验证的尽量写脚本实测。
