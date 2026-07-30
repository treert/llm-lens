"""只读解析 Kimi-K3 的 safetensors 索引,输出权重分组统计总览。

不读取任何 .safetensors 分片本体,只读 model.safetensors.index.json,
用于快速了解权重的命名、数量、分布和量化情况。

用法(在仓库根目录下):
    python analysis/kimi_k3/inspect_weights.py
    python analysis/kimi_k3/inspect_weights.py --model-dir G:/llm-models/Kimi-K3
    python analysis/kimi_k3/inspect_weights.py --top 60
"""

import argparse
import json
import re
from collections import Counter

from llm_lens import get_model_dir
from llm_lens.cli import add_model_args


def normalize_name(name: str) -> str:
    """把 layers.12. / experts.895. 这类序号归一为 *.,便于按结构分组。"""
    return re.sub(r"\.\d+\.", ".*.", name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_args(parser)
    parser.add_argument(
        "--top", type=int, default=40, help="显示出现次数最多的前 N 类权重(默认 40)"
    )
    args = parser.parse_args()

    model_dir = get_model_dir(args.model, args.model_dir, args.config)
    index_path = model_dir / "model.safetensors.index.json"
    print(f"模型目录: {model_dir}")
    print(f"索引文件: {index_path}")

    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    weight_map: dict[str, str] = index["weight_map"]
    total_size = index.get("metadata", {}).get("total_size", 0)
    shards = Counter(weight_map.values())

    print(f"\n权重张量数: {len(weight_map)}")
    print(f"分片数量:   {len(shards)}")
    print(f"索引记录总大小: {total_size / 2**30:.2f} GiB")

    groups = Counter(normalize_name(n) for n in weight_map)
    print(f"\n按结构分组(序号归一化后共 {len(groups)} 类,前 {args.top} 类):")
    for name, count in groups.most_common(args.top):
        quantized = "量化" if name.endswith(("weight_packed", "weight_scale")) else ""
        print(f"  {count:>6}  {name}  {quantized}")


if __name__ == "__main__":
    main()
