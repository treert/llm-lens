# 实操:本机最小检索 demo

把 [01-rag.md](01-rag.md) ~ [03-bm25.md](03-bm25.md) 的概念变成可观察的数字。
遵守仓库原则:**不跑大模型、不下载任何权重**,只用 numpy(matplotlib 画图),
语料直接用本仓库 `docs/` 的 markdown。

配套脚本在 `demos/retrieval/`(模型无关的独立工具,
按仓库规则放根级目录而非 `analysis/`)。

## 脚本 1:`tfidf_retrieval.py` —— 稀疏检索全流程

```
python demos/retrieval/tfidf_retrieval.py
python demos/retrieval/tfidf_retrieval.py --query "FlashAttention 为什么快" --top 5
```

实现链路(每步都对应前面文档的概念):

1. **chunk**:按空行切段,过滤过短(< 30 字)/过长(> 300 字)段
   ——对应 [01-rag.md](01-rag.md) §7 的结构切分;
2. **tokenize**:英文数字按词、中文按相邻 bigram 切(无分词器的折中);
3. **TF-IDF 矩阵**:tf 取 $1 + \log f$,idf 取 $\log\frac{N+1}{n+0.5}$,
   行向量 L2 归一后,检索 = 一次矩阵乘;
4. 打印 top-$k$ 的分数、来源文件与片段。

与 [03-bm25.md](03-bm25.md) 的关系:这个打分是
"无饱和( $k_1 \to \infty$ )、无长度归一( $b = 0$ )"的 BM25 近亲。

**观察点**:

- 换几个 query(中文、英文、术语缩写),看命中的 chunk 是否符合直觉;
- 找一个**应该命中却没命中**的 case:多半是 query 用词与文档错位
  ——这正是稠密检索要解决的"语义 gap"([01-rag.md](01-rag.md) §3.3)。

## 脚本 2:`ivf_recall.py` —— ANN 召回率实测

```
python demos/retrieval/ivf_recall.py
python demos/retrieval/ivf_recall.py --clusters 16 --nprobes 1 2 4 8 16
```

实现链路:

1. 复用脚本 1 的 TF-IDF 向量;
2. 自写 k-means(余弦版,k-means++ 初始化)把向量分簇——这就是
   [02-vector-index.md](02-vector-index.md) §2 的 IVF;
3. 抽样 64 个 chunk 当查询,以**暴力结果为 ground truth**,
   测不同 `nprobe` 下的 recall@10 与扫描比例;
4. 输出表格 + 画曲线到 `output/ivf_recall.png`。

这就是 [02-vector-index.md](02-vector-index.md) §6"召回率怎么测"的最小实现。

**注意**:查询用的是语料 chunk 自身(自检索),最近邻必含它自己,
所以 recall@10 的区分度体现在**其余 9 个近邻**上;
指标本身读的是"nprobe 从 1 加到 16 时召回的爬升形状"。

**预期现象**(具体数值以运行输出为准):

- `nprobe=1` 时 recall 明显 < 1(边界上的近邻被切到别的簇,02 §2.2 的硬分配损失);
- recall 随 `nprobe` 上升趋近 1,扫描比例同步线性上升;
- 曲线就是"召回 vs 代价"的权衡——ANN 调参的本质。

## 延伸练习(按难度排序)

1. **补全 BM25**:给脚本 1 加长度归一与 $k_1$ 饱和
   (公式见 [03-bm25.md](03-bm25.md) §3~§5),对比 top-$k$ 变化;
2. **换 chunk 策略**:段落切分改成 200 字滑窗 + 50 字重叠,
   重跑脚本 1 看检索变化([01-rag.md](01-rag.md) §7);
3. **换索引**:`pip install hnswlib`,用 HNSW 替换脚本 2 的 IVF 再测
   ([02-vector-index.md](02-vector-index.md) §4),对比同召回下的扫描代价;
4. **换稠密向量**:在项目 venv 内装 CPU 版 torch(安装方式见 AGENTS.md
   环境节,勿动全局 GPU 版)+ sentence-transformers,
   用 bge-small 编码后对比稀疏/稠密的 top-$k$ 差异([01-rag.md](01-rag.md) §5)。
