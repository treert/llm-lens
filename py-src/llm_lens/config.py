"""模型路径配置加载。

优先级:命令行参数 --model-dir > config/models.local.yaml > config/models.example.yaml
"""

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

LOCAL_CONFIG = REPO_ROOT / "config" / "models.local.yaml"
EXAMPLE_CONFIG = REPO_ROOT / "config" / "models.example.yaml"


def load_config(config_path: str | Path | None = None) -> dict:
    """加载配置文件。默认优先本地配置,不存在则回退到模板。"""
    if config_path is not None:
        path = Path(config_path)
    elif LOCAL_CONFIG.exists():
        path = LOCAL_CONFIG
    else:
        path = EXAMPLE_CONFIG

    if not path.exists():
        raise FileNotFoundError(
            f"配置文件不存在: {path}\n"
            f"请复制 {EXAMPLE_CONFIG.name} 为 {LOCAL_CONFIG.name} 并填写本机模型路径。"
        )

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def list_models(config_path: str | Path | None = None) -> dict[str, str]:
    """返回 {模型名: 权重目录路径} 的字典。"""
    cfg = load_config(config_path)
    return {name: m["path"] for name, m in (cfg.get("models") or {}).items()}


def get_model_dir(
    model: str | None = None,
    model_dir: str | Path | None = None,
    config_path: str | Path | None = None,
) -> Path:
    """解析模型权重目录。

    Args:
        model: 配置文件中的模型名;为 None 时使用配置里的 default_model。
        model_dir: 命令行直接指定的目录,优先级最高。
        config_path: 指定其他配置文件。

    Returns:
        模型权重目录路径(调用方应只读使用)。
    """
    if model_dir is not None:
        path = Path(model_dir)
    else:
        cfg = load_config(config_path)
        name = model or cfg.get("default_model")
        models = cfg.get("models") or {}
        if name not in models:
            raise KeyError(
                f"配置中找不到模型 {name!r},可用模型: {list(models)}"
            )
        path = Path(models[name]["path"])

    if not path.is_dir():
        raise FileNotFoundError(f"模型目录不存在: {path}")
    return path
