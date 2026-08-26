# 稠密检索模型:双塔、InfoNCE 与训练技术深挖

[01-rag.md](01-rag.md) §3 讲了 embedding 在 RAG 里的角色,
本篇讲这个模型**怎么训出来**:双塔结构、对比学习的细节、
以及工业界把检索模型练好的一整套技术。

## 1. 为什么需要专门的检索模型

直接拿 LLM 编码文本行不行?有三个障碍:

- **因果掩码**:decoder 的第 $i$ 个 token 看不到 $i$ 之后的内容,
  单向注意力得到的表征先天不全(对策见 §7);
- **没为相似度优化**:LLM 的训练目标是下一个 token,
  其隐空间的"近"不等于语义的"相关";
- **工程结构约束**:召回侧要求文档向量**离线预计算**、在线只编码 query
  ([01-rag.md](01-rag.md) §6),这逼出了双塔结构。

## 2. 双塔结构与 pooling

query 和 doc 各过一个 encoder(通常**共享权重**,否则两塔空间不对齐),
输出单向量。两个细节:

- **pooling**:取 [CLS] 还是 mean pooling?检索实践偏爱 mean——
  对所有 token 平均,对长度变化更稳,且缓解 [CLS] 的信息瓶颈;
- **L2 归一化**:输出向量归一后,点积 = 余弦([01-rag.md](01-rag.md) §3.2),
  检索退化为一次矩阵乘,且天然适配 [02-vector-index.md](02-vector-index.md) 的索引。

## 3. InfoNCE 深挖

回顾 [01-rag.md](01-rag.md) §3.1 的损失。这里展开两个工程上最关键的性质。

### 3.1 温度 $\tau$:梯度自动聚焦难负例

对某个负例的分数 $s^{-}$ 求导:

$$\frac{\partial \mathcal{L}}{\partial s^{-}}
= \frac{1}{\tau} \cdot
\frac{\exp(s^{-}/\tau)}{\exp(s^{+}/\tau) + \sum_j \exp(s_j^{-}/\tau)}
= \frac{p^{-}}{\tau}$$

其中 $p^{-}$ 是该负例在 softmax 中的概率。含义:
**越难分的负例(分数越高)梯度越大**——模型自动做 hard negative mining。
$\tau$ 控制聚焦程度:

- $\tau$ 小(如 0.01):分布锐化,只学最难的几个负例——
  收敛快,但容易把**假阴性**(实际相关却没标注)当成硬负例打;
- $\tau$ 大:所有负例均匀学,信号被稀释;
- 常见区间 $\tau \approx 0.01 \sim 0.05$。

### 3.2 in-batch negatives 与批大小

训练时,batch 内其他样本的正例直接充当负例:负例数 = $B - 1$。
为什么检索特别依赖**大 batch**?因为真实场景的分母是百万级文档,
batch 里只有几十个负例时,训练分布与推理分布严重失配。
DPR 用 128,BGE/E5 系用到数千至数万
(靠 GradCache、跨卡 gather 负例等工程手段撑大批量)。

### 3.3 难负例挖掘与假阴性

in-batch 负例是随机文档,太简单,训到后期几乎不提供梯度。
对策是**主动挖难负例**:用当前模型对训练 query 检索 top-$k$,
把"像但不是正例"的段落拿出来当负例(ANCE 的思路:用最新索引挖)。
风险正是假阴性——挖出来的"难负例"里混着没标注的相关文档,
需要按分数过滤或降权,否则模型被迫推开正确答案,越训越差。

## 4. 训练数据从哪来

- **弱监督对**:搜索引擎的(query, 点击文档)、问答社区的(问题, 答案)、
  天然的(标题, 正文)——规模巨大,噪声也不小;
- **人工标注**:MS MARCO(50 万 query + 段落级标注)是英文检索的公共底座;
- **合成数据**:让 LLM 给文档**反向生成问题**,构造(合成问题, 文档)对
  ——与 HyDE([01-rag.md](01-rag.md) §10)方向相反、原理相同。

## 5. 蒸馏:cross-encoder 当老师

双塔的精度有结构上限(整篇文档压成一个点,细粒度交互丢失)。
提升路径不是堆数据,而是**蒸馏**:

1. 先用 cross-encoder(query 与文档拼接输入,准但慢,见
   [05-rerank.md](05-rerank.md))对训练 query 的一批候选打分;
2. 学生(双塔)用 MarginMSE 拟合老师的分数差:
   $\mathcal{L} = \mathrm{MSE}\big(s_{\text{学生}}(q,d_i) - s_{\text{学生}}(q,d_j),\;
   s_{\text{老师}}(q,d_i) - s_{\text{老师}}(q,d_j)\big)$——
   只要求**保序**,不要求绝对分一致。

收益:把 cross-encoder 的排序能力压进可全库预计算的双塔,
且全程不需要新增人工标注。

## 6. 指令式与可变维度

- **指令式 embedding**:在输入前加任务前缀
  ("为这个句子检索相关段落" vs "为这个句子找相似句"),
  一个模型服务检索/聚类/分类多任务(E5-instruct、BGE 的多任务微调);
- **MRL(Matryoshka)**:训练时对向量的**每个前缀**都算一遍损失,
  推理时可直接截断到 256/512 维——召回略降,存储与算力省 2~4 倍。
  库大但精度要求一般时是极划算的交易。

## 7. LLM 改造的 embedding

把 decoder LLM 变成编码器的两条路线:

- **双向化 + 对比微调**(LLM2Vec):去掉 causal mask 换成双向注意力,
  用对比损失继续训练;
- **Echo embedding**:把输入重复两遍,取第二遍的表征
  (第一遍充当了"上文",规避因果掩码的信息不全)。

收益在长文本理解与指令跟随;代价是维度大(4096 维)、编码慢一个数量级。
现实建议:中小规模库,BGE/E5 系小模型(百万~亿级参数)通常足够。

## 8. 评测:BEIR 与 MTEB 怎么读

- **BEIR**:18 个零样本检索数据集,主指标 nDCG@10
  (指标定义见 [07-rag-evaluation.md](07-rag-evaluation.md));
  "零样本"意味着训测不同域,专门考验泛化;
- **MTEB**:覆盖检索/分类/聚类等 8 类任务的综合榜单,看 Retrieval 子榜即可。

读榜的坑:**榜单分数高 ≠ 你的领域好**。检索模型对领域分布敏感
(法律、代码、医学各有各的词法),选型后务必用自己的语料抽样验证
——造评测集的方法见 [07-rag-evaluation.md](07-rag-evaluation.md) §3。

## 参考

- Karpukhin et al., 2020. *DPR: Dense Passage Retrieval*
- Xiong et al., 2021. *ANCE: Approximate Nearest Neighbor Negative
  Contrastive Learning*(难负例挖掘)
- Reimers & Gurevych, 2019. *Sentence-BERT*(双塔范式确立)
- Wang et al., 2022/2024. *E5: Text Embeddings by Weakly-Supervised
  Contrastive Pre-training / Multilingual E5*
- Xiao et al., 2023. *C-Pack: BGE 系列中文检索模型*
- Kusupati et al., 2022. *Matryoshka Representation Learning*
- BehnamGhader et al., 2024. *LLM2Vec*
- Hofstätter et al., 2021. *Improving Efficient Neural Ranking Models
  with Cross-Architecture Knowledge Distillation*(MarginMSE 蒸馏)
