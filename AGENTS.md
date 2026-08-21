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
- `tools/`:仓库自身的维护/检查小工具(如 `fix_md_math_spacing.py` 修复行内公式空格)
- `docs/`:模型结构笔记、原理笔记;文档与代码注释一律用中文

## 硬性规则

1. **模型权重目录一律只读**(如 `G:\llm-models\Kimi-K3`):不得在其中创建、修改、
   删除任何文件;读取时也不得加锁/占用导致其他程序无法访问
2. 模型路径**不得硬编码**在脚本里:一律通过 `llm_lens.get_model_dir()` 解析,
   优先级为 CLI `--model-dir` > `config/models.local.yaml`(本地配置,不入库)
3. `config/*.local.yaml`、`tmp/`、`output/`、`*.egg-info/` 不入库
4. 分析产生的图和中间结果写到 `output/`,不要散落在仓库各处

## 文档索引

- 各模型的结构笔记(架构参数、权重命名、量化格式):`docs/<模型名>.md`,如 `docs/kimi-k3.md`
- 通用数学背景(自由概率、随机矩阵等):`docs/math/`
- 推理部署/推理 Infra 笔记(kernel、serving、量化等):`docs/inference/`,索引见其 README.md
- 模型专属脚本的说明:`analysis/<模型名>/README.md`
- 写某个模型的分析脚本前,先读上面两份对应文档

## 协作风格

- 提交信息用中文,格式 `<type>: <摘要>`(如 `init:`、`refactor:`、`feat:`)
- 脚本要有 docstring:一句话功能 + 用法示例
- 分析脚本遵循"小步验证":先打印形状/范围确认读对了,再做正式计算
- markdown 中的行内公式:开头的 `$` 前若是文字或全角标点(如 `（$`),必须加空格,
  否则 GitHub 网页不渲染公式;检查/批量修复用 `python tools/fix_md_math_spacing.py`
  (默认只报告,加 `--apply` 写回)
