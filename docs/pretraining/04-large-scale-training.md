# 大规模训练工程:千卡集群上让 loss 稳定下降

前置:[01-pretraining-overview.md](01-pretraining-overview.md)、
[../finetune/02-training-memory.md](../finetune/02-training-memory.md)(单卡
显存账)、[../inference/06-parallelism.md](../inference/06-parallelism.md)
(并行策略的推理视角)。
目标:理解把训练从单卡推到千卡时,工程上必须解决的五件事——并行
切分、混合精度、稳定性、调度、容错。个人不会操作千卡集群,但这些
知识是理解技术报告与"训练事故"新闻的钥匙,其中稳定性与调度的经验
在单卡微调里同样适用。

## 1. 并行的训练视角:从"切模型"到"切状态"

inference 06 篇讲过 TP/PP/DP/EP 各切什么。训练侧的新问题是
**优化器状态与梯度也要切**(finetune 02 篇:它们才是显存大头):

**数据并行(DP)的梯度同步**:每张卡用不同数据 shard 前向/反向,
反向结束时对梯度做 **all-reduce**(每卡得到全平均梯度)再各自
step。通信量 = 梯度量,与模型大小成正比,与 batch 无关——这是
大 batch 训练通信占比下降、更容易扩展的原因。

**ZeRO / FSDP:把"每个参数 16 字节"切到多卡**。ZeRO 三阶段:

| 阶段 | 切分对象 | 每卡显存(单卡账的倍数,P=卡数) |
|---|---|---|
| ZeRO-1 | 优化器状态 | 4 + 12/P 字节/参数 |
| ZeRO-2 | + 梯度 | 4 + (2+12)/P(近似) |
| ZeRO-3 | + 权重本身 | 16/P + 通信换显存 |

FSDP(PyTorch 原生实现)思想同 ZeRO-3:**权重也分片存储,用到时
all-gather 临时拼出完整层,算完即弃**——显存换通信。70B 全量微调
(~1.1 TB)在 8×80GB 上靠的就是它。

**3D 组合实例**(Llama-3-405B 量级):TP(8 卡内,NVLink 域)
× PP(跨节点流水)× DP(最外层扩卡数),通信模式各归各位:
TP 要高频低延迟(NVLink 内),DP 只要带宽(跨机 IB)。

## 2. 混合精度的细节

finetune 02 篇给了"16 字节/参数"的账,这里补**为什么是 bf16 而不是
fp16**:

- fp16 动态范围小(指数 5 bit,最大 ~65K):梯度容易**下溢**(变成 0),
  大模型深层梯度常在 1e-6 量级;
- fp16 时代的补救是 **loss scaling**:把 loss 乘一个大常数 $S$ 再反向
  (梯度同比例放大),更新前除回来。 $S$ 要动态调:太小防不住下溢,
  太大上溢(NaN);
- bf16 指数与 fp32 相同(8 bit),范围不是问题了,于是 loss scaling
  整个机制被淘汰——**bf16 用尾数精度换范围,恰好匹配训练的需求**
  (范围敏感、精度不敏感,因为更新在 fp32 master 上做)。

这也是为什么硬件从 Ampere 起原生支持 bf16 后,它迅速成为训练标配。

## 3. 稳定性:loss spike 的战争

**现象**:数千亿 token 的训练中,loss 会突然跳升一个量级甚至 NaN,
有时自己恢复,有时再也回不来。OPT、GLM-130B、BLOOM 的报告都
记录了与 spike 的斗争。

**成因**(按频率):

1. 数据坏 batch(异常分布、损坏样本);
2. 数值溢出(attention logits 或中间激活爆炸);
3. lr 相对当前 loss landscape 过大;
4. 硬件故障(静默错误,算错不报错)。

**对策工具箱**:

- **gradient clipping**(按全局范数裁剪,典型阈值 1.0):最便宜也
  最有效的第一道防线;
- **z-loss**(PaLM 提出):额外惩罚 $\log^2 Z$(logits 的归一化项),
  抑制 logits 无界增长;
- **spike 处理 SOP**:检测到 spike → 回滚到 spike 前 checkpoint →
  **跳过可疑数据段**继续训(OPT 的做法);
- 架构级:QK-Norm(给 Q/K 加归一化防 attention logit 爆炸,现代
  模型如 Olmo2 标配)、更保守的初始化。

## 4. 学习率与 batch size 的调度

**lr schedule**:主流是 warmup → cosine 衰减到峰值的 ~10%。

- warmup(前几千步 lr 线性爬升)为什么必须:训练初期梯度方向嘈杂
  且权重离好解远,大 lr 会直接冲进坏区域;warmup 让优化器状态
  (Adam 的 $m, v$)先"预热"到合理尺度;
- 新变体 **WSD(warmup-stable-decay)**:warmup 后长期恒定 lr,
  需要收尾时快速 decay。好处:stable 阶段任意时刻 decay 都能切出
  一个不错的模型,不必预先承诺训练总长(MiniCPM、Llama 系采用),
  工程灵活性远大于 cosine。

**batch size**:不是越大越好,存在 **critical batch size**——低于它,
增大 batch 近似线性减少步数;高于它,收益急剧衰减(梯度噪声已被
充分平均)。且 critical batch size 随训练进行**增大**(后期梯度更
嘈杂),所以前沿实践会动态增 batch。Llama-3-405B 的 batch 从
4M tokens 逐步升到 16M tokens 就是这个原理。

## 5. 容错与监控

千卡集群的现实:**平均每天都有硬件故障**(GPU 掉卡、IB 链路抖动、
ECC 错误)。一个 55 天的 run 实际上由数百段拼接而成:

- **checkpoint**:按步数周期性保存(权重+优化器状态,数 TB 级),
  异步写入避免阻塞训练;
- **自动重启**:故障检测 → 隔离坏节点 → 从最近 checkpoint 热恢复,
  全程无人值守;
- **监控三件套**:loss 曲线(看趋势与 spike)、**grad norm**(最敏感
  的先兆指标,spike 前常先出现 grad norm 异常)、吞吐(tokens/s/GPU,
  掉卡或链路降级会立刻反映)。

## 6. 对个人的意义

- 单卡微调里的对应物:gradient clipping、bf16、grad norm 监控在
  单卡同样值得开(finetune 03 篇的训练配置里它们默认存在);
- 读技术报告时,训练工程章(MFU、checkpoint 策略、spike 次数)
  是评估团队工程水平的窗口;
- 遇到"训练崩了"的新闻,现在能猜到八成是 §3 的哪一类。

## 7. 小结

- 训练并行 = 切模型(TP/PP)+ 切数据(DP)+ 切状态(ZeRO/FSDP),
  通信与显存的互换贯穿始终;
- bf16 取代 fp16 是"范围换精度"匹配训练需求的经典案例;
- loss spike 不可避免,工程上靠 clip、z-loss、回滚+跳数据的 SOP
  生存;
- WSD 让"训多久"从承诺变成可随时决定的自由度;
- 千卡训练的本质是**与故障共存的系统工程**。
