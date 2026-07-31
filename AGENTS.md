# AGENTS.md

面向在本仓库中工作的 AI 编码助手:开始任何任务前,先读完本文件并遵守其中的约定。

## 项目定位

个人学习项目:从**原理层面 + 分析开源权重**理解 LLM。本机**不运行大模型**,
只做权重的静态分析(读取 safetensors 中的矩阵/向量,用 numpy 等做数值计算)。

## 环境与依赖

- Python 3.10+;依赖与打包统一由根目录 `pyproject.toml` 管理(没有 requirements.txt)
- 安装:`pip install -e .`(venv 或全局环境均可)
- 新增/修改依赖:改 `pyproject.toml` 的 `dependencies`,然后重跑 `pip install -e .`
- torch 不在默认依赖中,按需安装,但**必须注意环境隔离**:
  - 全局环境已有 GPU 版 torch(2.9.0+cu128,ctranslate2 依赖自动带入),**禁止在全局环境重装/替换/卸载 torch**(CPU 版与 GPU 版同包名,会互相覆盖);GPU 版不调用 CUDA 时与 CPU 版用法等价,可直接用于分析
  - 项目 venv(`.venv/`)与全局隔离,如需 torch 可在 venv 内装 CPU 版:`pip install torch --index-url https://download.pytorch.org/whl/cpu`

## 代码放置规则

- `py-src/llm_lens/`:模型**无关**的通用工具库(配置加载、向量指标等)
- `analysis/<模型名>/`:依赖具体模型结构的脚本;当前只有 `kimi_k3`,分析新模型时新建对应子目录
- 其他形态的工具(notebook、其他语言):在根目录新建与 `py-src/` 平级的目录
- `docs/`:模型结构笔记、原理笔记;文档与代码注释一律用中文

## 硬性规则

1. **模型权重目录一律只读**(如 `G:\llm-models\Kimi-K3`):不得在其中创建、修改、
   删除任何文件;读取时也不得加锁/占用导致其他程序无法访问
2. 模型路径**不得硬编码**在脚本里:一律通过 `llm_lens.get_model_dir()` 解析,
   优先级为 CLI `--model-dir` > `config/models.local.yaml`(本地配置,不入库)
3. `config/*.local.yaml`、`tmp/`、`output/`、`*.egg-info/` 不入库
4. 分析产生的图和中间结果写到 `output/`,不要散落在仓库各处

## 技术要点(Kimi-K3)

- 权重为 safetensors,96 个分片;索引:`model.safetensors.index.json`
- 路由专家是 mxfp4 量化(`weight_packed`/`weight_scale`,group_size=32),分析前需反量化;
  其余权重(词嵌入、注意力投影、LM Head、路由器、共享专家、视觉塔)是 bf16,
  用 safetensors + numpy(ml_dtypes)可直接读取
- 模型结构细节与权重命名规则见 `docs/kimi-k3.md`,写 K3 脚本前先读它

## 协作风格

- 提交信息用中文,格式 `<type>: <摘要>`(如 `init:`、`refactor:`、`feat:`)
- 脚本要有 docstring:一句话功能 + 用法示例
- 分析脚本遵循"小步验证":先打印形状/范围确认读对了,再做正式计算
