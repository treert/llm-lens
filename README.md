# llm-lens

从**原理层面 + 开源权重实证**两个角度理解 LLM 的个人学习项目。

本机跑不动大模型,但可以下载权重做"解剖":直接分析其中的矩阵和向量——
词嵌入、LM Head、注意力投影矩阵、MoE 路由器与专家等,用数值证据验证(或推翻)
对 LLM 工作原理的各种直观理解。

## 分析视角

LLM 的前向计算大部分是三类操作:**矩阵投影、向量加和、向量点乘**。本项目的分析
围绕这些基本操作展开,例如一些待验证的直观假设:

- 高维空间中,随机两个向量近乎正交(夹角 ≈ 90°);语义相关的向量夹角才会显著偏离 90°。
- 残差流上的向量加和,可能起到"公共成分增强、差异成分相消"的效果。
- 矩阵投影(如注意力中的低秩分解)的实际有效秩,可能与设计秩有差距。

这些假设是否成立,用权重和数学工具说话。

## 当前分析对象

**Kimi-K3**(混合线性注意力 + MoE 的多模态模型,权重为 safetensors + mxfp4 量化),
结构笔记见 [docs/kimi-k3.md](docs/kimi-k3.md)。

## 目录结构

```
llm-lens/
├── pyproject.toml              # 打包声明 + 依赖列表(pip install -e . 一键安装)
├── config/
│   └── models.example.yaml   # 模型路径配置模板(复制为 models.local.yaml 使用,不入库)
├── docs/                     # 模型结构笔记、原理层面的学习笔记
│   └── kimi-k3.md
├── src/llm_lens/             # 模型无关的通用分析工具库(配置加载、向量指标等)
├── analysis/                 # 依赖具体模型结构的分析脚本,一个模型一个子目录
│   └── kimi_k3/
├── tmp/                      # 临时文件(git 忽略)
└── output/                   # 分析输出(git 忽略)
```

**约定:依赖某个模型结构的脚本,必须放在 `analysis/<模型名>/` 下;通用逻辑下沉到
`src/llm_lens/`。**

## 快速开始

```powershell
# 1. 创建虚拟环境,安装依赖 + llm_lens 本地包(可编辑安装,改代码立即生效)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .

# 2. 配置本机模型路径(复制模板后修改,models.local.yaml 不会提交)
copy config\models.example.yaml config\models.local.yaml

# 3. 运行示例:查看 Kimi-K3 权重总览(只读)
python analysis/kimi_k3/inspect_weights.py
# 或者显式指定权重目录(优先级高于配置文件):
python analysis/kimi_k3/inspect_weights.py --model-dir G:/llm-models/Kimi-K3
```

路径解析优先级:命令行 `--model-dir` > `config/models.local.yaml` 中对应模型。

补充:

- 想装进全局环境(免激活 venv):不激活,直接执行 `pip install -e .` 即可,依赖会一并装入全局。
- 可选:需要 torch 时(反量化等场景更方便)安装 CPU 版:
  `pip install torch --index-url https://download.pytorch.org/whl/cpu`

## 硬性规则

1. **模型权重目录一律只读**,任何脚本不得向权重目录写入文件。
2. `tmp/`、`output/`、`config/*.local.yaml` 不入库(见 `.gitignore`)。
3. 分析脚本应说明输入(哪些权重)、输出(结论/图),尽量小步验证。
