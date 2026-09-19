"""只读解析 DeepSeek-V4.1-Flash 的 safetensors 索引与各分片头部,输出权重分组统计总览。

不读取张量数据本体,只读 model.safetensors.index.json 和各分片的 JSON 头部,
用于快速了解权重的命名、数量、dtype 分布和量化情况。

用法(在仓库根目录下):
    python analysis/ds_v4_1_flash/inspect_weights.py
    python analysis/ds_v4_1_flash/inspect_weights.py --top 60
"""

import argparse
import json
import re
import struct
from collections import Counter

from llm_lens import get_model_dir
from llm_lens.cli import add_model_args

QUANT_DTYPES = {"F8_E4M3": "fp8权重", "F8_E8M0": "ue8m0缩放", "I8": "fp4打包(int8)"}


def normalize_name(name: str) -> str:
    """把 layers.12. / experts.383. 这类序号归一为 *.,便于按结构分组。"""
    return re.sub(r"\.\d+\.", ".*.", name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_args(parser)
    parser.add_argument(
        "--top", type=int, default=60, help="显示出现次数最多的前 N 类权重(默认 60)"
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
        suffix = name.rsplit(".", 1)[-1]
        quantized = {"scale": "量化scale", "weight_packed": "量化"}.get(suffix, "")
        print(f"  {count:>6}  {name}  {quantized}")

    # 逐分片读头部,统计 dtype 分布与每类模式的 dtype
    print("\n读取各分片头部,统计 dtype ...")
    dtype_counter: Counter[str] = Counter()
    pattern_dtype: dict[str, Counter[str]] = {}
    for shard in sorted(shards):
        with open(model_dir / shard, "rb") as f:
            (header_len,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(header_len))
        for name, info in header.items():
            if name == "__metadata__":
                continue
            dt = info["dtype"]
            dtype_counter[dt] += 1
            pattern_dtype.setdefault(normalize_name(name), Counter())[dt] += 1

    print(f"\ndtype 分布: {dict(dtype_counter)}")
    print("\n各模式的 dtype(仅含量化 dtype 的模式):")
    for pat in sorted(pattern_dtype):
        dts = pattern_dtype[pat]
        if set(dts) & set(QUANT_DTYPES):
            desc = ", ".join(f"{dt}:{n}" for dt, n in dts.most_common())
            print(f"  {pat:55s}  {desc}")


if __name__ == "__main__":
    main()
