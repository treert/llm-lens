# 实操手册:在 16 GB Windows 机器上跑起本地大模型

前置阅读:[03-deployment-basics.md](03-deployment-basics.md)(选型与原理)、
[02-resource-estimation.md](02-resource-estimation.md)(本文的对账基准)。
适用机器:RTX 5070 Ti 16 GB(工作机)/ RTX 4080 16 GB(家里机),
Windows + 较新驱动。

> 版本与下载地址随时间变化,以各项目官网为准;本文给出流程与预期数字,
> 实操后把实测值填进 §7 的表。

## 0. 准备检查

```powershell
nvidia-smi   # 确认驱动正常、显存空闲 ~15 GB 以上
```

关掉占显存的程序(浏览器硬件加速、游戏等);桌面与 WDDM 会常驻
0.5~1 GB,属正常(见 02-resource-estimation.md §0)。

## 1. 安装 llama.cpp(5 分钟)

1. 到 GitHub `ggml-org/llama.cpp` 的 Releases,下载 Windows 预编译包:
   - 优先 **CUDA 版**(`llama-bXXXX-bin-win-cuda-x64.zip`);若启动报缺
     `cudart64_*.dll`,再下同页面的 cudart 包解压到一起;
   - 备选 **Vulkan 版**:零依赖解压即用,性能略低;
2. 解压到固定目录,如 `D:\tools\llama.cpp\`;
3. 验证:

```powershell
cd D:\tools\llama.cpp
.\llama-cli.exe --version
```

## 2. 下载模型(约 5 GB)

推荐起手模型:**Qwen3-8B 的 GGUF Q4_K_M**(~5 GB,资源估算篇案例 A)。

- HuggingFace 搜 `Qwen3-8B-GGUF`(官方仓库,文件
  `Qwen3-8B-Q4_K_M.gguf`);国内网络慢可换镜像站 `hf-mirror.com`
  或 ModelScope 搜同名;
- 存放位置自定,例如 `G:\llm-models\gguf\`
  (注意:仓库的只读规则针对分析脚本,下载新模型不受其约束)。

## 3. 起服务,发第一条消息

```powershell
.\llama-server.exe -m G:\llm-models\gguf\Qwen3-8B-Q4_K_M.gguf -ngl 99 -c 32768 --port 8080
```

- `-ngl 99`:所有层放 GPU(8B Q4 在 16 GB 上毫无压力);
- `-c 32768`:上下文 32K(对应 KV 预算见 02-resource-estimation.md §2,
  GQA 模型 128~144 KB/token,32K 约 4~4.6 GB);
- 启动日志确认两点:`llm_load_tensors: offloaded 37/37 layers to GPU`
  (层数全在 GPU)与 KV buffer 大小;
- 浏览器打开 `http://localhost:8080`(自带 webui)即可对话;
  这就是 OpenAI 兼容服务(03-deployment-basics.md §4),任何支持
  OpenAI API 的前端都能接。

## 4. 对账:实测 vs 理论

```powershell
.\llama-bench.exe -m G:\llm-models\gguf\Qwen3-8B-Q4_K_M.gguf -ngl 99
```

输出 `pp512`(prefill,tok/s)与 `tg128`(decode,tok/s)。预期区间
(推算见 02-resource-estimation.md §4):

| 指标 | 理论上限 | 预期实测 | 若明显偏低 |
|---|---|---|---|
| tg(decode)5070 Ti | ~179 | 60~110 | 检查 offload、上下文长度、其他占显存程序 |
| tg(decode)4080 | ~143 | 50~100 | 同上 |
| pp(prefill)两卡 | ~12K | 4~7K | CUDA 版是否装对(Vulkan 会低) |

同时开另一个终端 `nvidia-smi` 看显存:预期 ≈ 5 GB(权重)+ 4.5 GB(KV)
+ 1~1.5 GB(开销)≈ 10~11 GB。

## 5. 进阶尝试(按兴趣选做)

1. **14B Q4**:下载 `Qwen3-14B-Q4_K_M.gguf`(~9 GB),`-c 16384`;
   对比回答质量与速度(预期 tg 35~70 tok/s);
2. **API 调用**(确认 server 在跑):

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="none")
resp = client.chat.completions.create(
    model="local",
    messages=[{"role": "user", "content": "用三句话解释什么是 KV Cache"}],
    temperature=0.7,
)
print(resp.choices[0].message.content)
```

3. **投机解码**:llama.cpp 支持 `--model-draft`(小模型起草),观察
   tg 提升幅度(原理见 05-throughput.md §6);
4. **长上下文压测**:`-c` 提到 65536,观察显存与 tg 随上下文增长
   的下降(decode 每步要读全部 KV,变慢是正常物理现象)。

## 6. 可选:WSL2 里跑 vLLM(折腾向,1~2 小时)

动机:体验生产级 serving(PagedAttention、continuous batching、FP8)。

1. `wsl --install` 装 Ubuntu;**不要**在 WSL 里装 NVIDIA 驱动——
   Windows 驱动自动透传;
2. 用户目录建 `.wslconfig`,把内存调大(默认只给一半),如
   `memory=24GB`;
3. WSL 内装 Python venv,`pip install vllm`;
4. `vllm serve Qwen/Qwen3-8B-FP8 --max-model-len 16384`
   (FP8 原生支持,权重 ~8 GB,见 04-quantization.md §7);
5. 启动日志找两行对账:`GPU KV cache size: ... tokens`(KV 池容量)
   与显存占用——与 02-resource-estimation.md 的总账对照;
6. 浏览器/客户端接 `http://localhost:8000/v1`。

## 7. 记录模板(实测后填写)

| 项目 | 5070 Ti 实测 | 4080 实测 | 理论上限 | 达成率 |
|---|---|---|---|---|
| 8B Q4 tg(tok/s) | | | 179 / 143 | |
| 8B Q4 pp(tok/s) | | | ~12K | |
| 显存占用(GB) | | | ~10~11 | |
| 14B Q4 tg(tok/s) | | | 100 / 80 | |

把数字带回来,补进 [02-resource-estimation.md](02-resource-estimation.md)
作为实测基线。

## 8. 故障排查

| 症状 | 处理 |
|---|---|
| 缺 `cudart64_*.dll` | 下同版本 cudart 包,或改用 Vulkan 版 |
| 启动 OOM | 降低 `-c`;确认无其他程序占显存 |
| 速度只有个位数 tok/s | 大概率层没全在 GPU:检查 `-ngl` 与启动日志的 offloaded 行 |
| 下载慢 | 换 hf-mirror.com 或 ModelScope |
| 端口被占 | 换 `--port` |
| webui 打不开 | 确认启动无报错,直接测 `/v1/models` 接口 |

## 9. 小结

最短路径一句话:**llama.cpp CUDA 版 + Qwen3-8B Q4_K_M +
`-ngl 99 -c 32768`,浏览器开 8080**。之后所有折腾(14B、FP8/vLLM、
投机解码、长上下文)都记得回到资源估算篇的账本上对数字——
能对上账,才算真的懂了。
