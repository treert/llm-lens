# 单机部署实操:推理引擎与选型

前置:[01-inference-pipeline.md](01-inference-pipeline.md)、
[02-resource-estimation.md](02-resource-estimation.md)(放不放得下、能跑多快)。
目标:理解一个推理引擎(inference engine)内部有哪些部件,主流框架各自的
定位,以及在自己的机器上把第一个模型跑起来的最短路径。

> 命令行细节随版本演进较快,本文以机制讲解为主;具体 flag 以各框架当下
> 文档为准。

## 1. 推理引擎内部:前五篇笔记的零件都装在哪

一个 serving 框架做的事情,恰好是我们前面学过概念的组装:

```
请求 → tokenizer → 调度器 ──batch──→ model runner ──→ 采样 → detokenize → 流式返回
                     │                    │
                     │   连续 batching    │  FlashAttention / RMSNorm / Gumbel-Max …
                     ▼                    ▼
              KV Cache 池(分页管理,← 权重 + 量化格式(GGUF/AWQ/FP8…)
              PagedAttention)
```

- **调度器**:决定哪些请求凑成一个 batch(continuous batching,吞吐篇详谈);
- **model runner**:执行各算子 kernel(算子篇的内容全在这里);
- **KV Cache 池**:vLLM 的 PagedAttention 把 KV 按页(block)管理,像 OS
  管理内存页一样消除碎片,池大小直接决定能服务多少并发与多长上下文
  (资源估算篇 §2);
- **量化格式**:权重以什么形式躺在磁盘/显存里(量化篇详谈)。

选框架 = 选这些部件的实现质量与取舍。

## 2. 主流框架速览

| 框架 | 定位 | 模型格式 | 并发/serving | 上手难度 | Windows |
|---|---|---|---|---|---|
| **llama.cpp** | 轻量本地推理,C/C++ 单文件哲学 | GGUF(量化为主) | 有 server,适合单人~小团队 | 极低(下载即用) | **原生支持** |
| **vLLM** | 生产级 serving,吞吐标杆 | HF 原始权重 + FP8/AWQ/GPTQ | 强(continuous batching 发源地) | 中 | 走 WSL2 |
| **SGLang** | 生产级 serving,前缀缓存与结构化输出见长 | 同 vLLM | 强 | 中 | 走 WSL2 |
| **Ollama / LM Studio** | llama.cpp 的包装:模型仓库 + API / GUI | GGUF | 单人为佳 | 极低 | 原生支持 |
| **TensorRT-LLM** | NVIDIA 官方极致性能(编译式) | 需编译 engine | 强 | 高 | 支持但折腾 |
| HF Transformers | 教学/研究基线 | HF 原始权重 | 弱(无真正的 serving 调度) | 低 | 原生 |

关键区分是两条轴:

- **格式轴**:GGUF(llama.cpp 生态)vs HF 原始权重 + 各框架量化(vLLM
  生态)。同一模型通常两边都有现成文件,不用自己转换。
- **场景轴**:单人对话 vs 多人并发 API。单人场景 llama.cpp 与 vLLM 的
  体验差距不大;并发一上来,vLLM/SGLang 的调度优势是数量级的。

各自一句话:

- **llama.cpp**:GPU 层数可调(`-ngl`,放不下的层自动落 CPU 内存——
  资源估算篇案例 D 的 offload 就是它),Vulkan/CUDA 双后端,零依赖单文件。
- **vLLM**:PagedAttention + continuous batching 的提出者,`vllm serve`
  直接起 OpenAI 兼容 API;生态最大,新模型支持最快。
- **SGLang**:RadixAttention 用基数树缓存共享前缀(多轮对话、Agent
  场景重复前缀极多,命中率收益大);结构化输出(JSON 约束解码)最快。
- **Ollama/LM Studio**:适合"先跑起来再说",前者命令行友好,后者全 GUI;
  底层都是 llama.cpp,性能一致。

## 3. GGUF 文件名怎么读

GGUF 是 llama.cpp 的模型文件格式,文件名里编码了量化方案:

```
Qwen3-8B-Q4_K_M.gguf
           │  │  └─ 混合精度档位:S/M/L(小/中/大,大=关键层用更高位)
           │  └──── K-quant(现代主流量化族,按 block 分组量化)
           └─────── 位宽:Q2~Q8
```

- 日常使用 **Q4_K_M**(质量/体积平衡点,资源估算篇的甜点位就是它);
- 质量优先选 Q6_K / Q8_0,极限压缩选 Q3/IQ 系列(损失明显);
- 量化原理(分组、scale、为什么 Q4 几乎不掉点)留给量化篇。

模型去哪下:HuggingFace(国内可用镜像站)或 ModelScope,搜索
"模型名 + GGUF"或"模型名 + AWQ/FP8"即可,官方与社区量化版都很全。

## 4. 离线批跑 vs 在线 serving

同一个引擎有两种用法,按需求选:

| | 离线批跑(offline batch) | 在线 serving |
|---|---|---|
| 形态 | 脚本里一次喂一批 prompt,等全部完成 | 常驻 HTTP 服务(OpenAI 兼容 API),逐请求流式返回 |
| 优化目标 | 总吞吐 | 单请求延迟(TTFT/TPOT)+ 并发 |
| 典型入口 | `llama-cli`;vLLM 的 `LLM.generate()` | `llama-server`;`vllm serve`;`sglang.launch_server` |
| 适用 | 数据标注、评测、批量摘要 | 聊天应用、Agent、给其他程序当后端 |

实践中几乎都值得起 **server 模式**:就算只有自己用,OpenAI 兼容 API
意味着任何前端(聊天界面、编辑器插件、自己写的脚本)都能直接接入,
一次部署处处可用。

## 5. 我们的两台机器怎么走(Windows 实操路径)

工作机 RTX 5070 Ti(16 GB)+ 家里机 RTX 4080(16 GB),都是 Windows:

1. **第一步(10 分钟):llama.cpp 原生跑通**
   - 下载 Windows 预编译包(CUDA 版;嫌装 CUDA 麻烦可用 Vulkan 版,
     性能略低但零依赖);
   - 下载一个 GGUF,如 `Qwen3-8B-Q4_K_M.gguf`(~5 GB,资源估算篇案例 A);
   - `llama-server -m <模型> -ngl 99 -c 32768` 起服务,浏览器打开自带
     页面即可对话;`-ngl 99` = 层全部放 GPU。
2. **第二步:对账**。生成时看日志里的 tok/s,与资源估算篇 §4 的理论上界
   对比(4080 上 8B Q4 预期 60~110 tok/s);`nvidia-smi` 看显存占用
   是否符合"~5 GB 权重 + KV + 开销"的预期。
3. **第三步(可选,需要折腾):WSL2 里跑 vLLM**
   - 动机:体验 PagedAttention + continuous batching 的生产级 serving,
     以及 FP8/AWQ 等 GGUF 之外的量化生态;
   - WSL2 通过 NVIDIA 驱动的 GPU 半虚拟化使用显卡,性能损失通常
     仅几个百分点;vLLM 官方以 Linux 为主,Windows 原生支持在演进,
     以官方文档为准;
   - 注意 WSL2 默认只划一半物理内存,模型下载与权重加载阶段可能
     需要在 `.wslconfig` 里调大。
4. **家里 4080 同理**。两台卡都算力同档,5070 Ti 带宽高约 25%,
   decode 体感更快一点;4080 在 Ada 架构上 FP8 也可用。

### 决策树

- 只想本地对话/给编辑器接个后端 → llama.cpp(或 Ollama/LM Studio);
- 要多并发 API、做吞吐实验 → vLLM(WSL2);
- 大量共享前缀(多轮对话/Agent)或强结构化输出 → SGLang(WSL2);
- 显存放不下想硬扛 → llama.cpp 的 `-ngl` 部分 offload(速度换容量);
- 要榨干 NVIDIA 的生产性能 → TensorRT-LLM(进阶篇再谈)。

## 6. 常见坑(第一台机器部署前读一遍)

- **模型下错格式**:GGUF 与 HF 权重不通用,下前确认框架匹配;
- **上下文开太长**:`-c` / `max_model_len` 直接决定 KV 预算,16 GB 卡上
  8B Q4 开 32K 很从容,开 128K 就会挤占权重空间甚至 OOM(回资源估算篇
  §2 算账);
- **Windows 显存被桌面占用**:接了显示器的卡会被 WDDM 与桌面合成器
  吃掉 0.5~1 GB,规划预算时留余量;
- **offload 错觉**:`-ngl` 没设或设小了,部分层在 CPU 上跑,速度掉一个
  量级——先检查日志确认所有层都在 GPU;
- **长上下文越用越慢是正常现象**:decode 每步都要读全部 KV Cache,
  上下文越长单步越慢(见 01-inference-pipeline.md §5),不是出了故障。

## 7. 小结

- 推理引擎 = 调度器 + model runner + KV 池 + 量化权重,前几篇的概念
  各有其位;
- 单人场景 GGUF + llama.cpp 十分钟跑通;并发场景 vLLM/SGLang(WSL2);
- 起手式:Qwen3-8B-Q4_K_M + llama-server + `-ngl 99 -c 32768`,
  然后对着资源估算篇的账实测验证;
- 下一篇:量化——GGUF 的 Q4_K_M 到底对权重做了什么、为什么 4 bit
  几乎不掉点、FP8 与 AWQ/GPTQ 的区别。
