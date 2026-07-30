"""分析脚本共用的命令行参数。"""

import argparse


def add_model_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """为脚本添加通用的模型选择参数。

    用法::

        parser = argparse.ArgumentParser(description=__doc__)
        add_model_args(parser)
        args = parser.parse_args()
        model_dir = get_model_dir(args.model, args.model_dir)
    """
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--model",
        help="配置文件中的模型名(默认使用配置里的 default_model)",
    )
    group.add_argument(
        "--model-dir",
        help="直接指定模型权重目录,优先级高于配置文件",
    )
    parser.add_argument(
        "--config",
        help="指定其他配置文件(默认自动查找 config/models.local.yaml)",
    )
    return parser
