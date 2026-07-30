# analysis/ — 模型专属分析脚本

本目录存放**依赖具体模型结构**的分析脚本,每个模型一个子目录(目录名与配置中的模型名一致):

```
analysis/
└── kimi_k3/        # 依赖 Kimi-K3 结构(命名规则、层类型、量化格式)的脚本
```

## 约定

- **模型无关的通用逻辑**(配置加载、向量指标、绘图等)一律下沉到 `src/llm_lens/`,
  本目录的脚本只做"该模型特有"的部分:权重命名解析、层类型判断、反量化等。
- 脚本对模型权重目录**只读**,不得在权重目录内写入任何文件。
- 分析输出(图、中间结果)默认写到仓库根目录的 `output/`(已被 git 忽略)。
- 脚本开头需要把 `src/` 加入模块搜索路径,模板如下:

```python
"""脚本功能一句话说明。

用法(在仓库根目录下):
    python analysis/<model_name>/xxx.py [--model <配置中的模型名> | --model-dir <路径>]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import argparse

from llm_lens import get_model_dir  # noqa: E402
from llm_lens.cli import add_model_args  # noqa: E402
```
