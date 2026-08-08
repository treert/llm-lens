"""safetensors 读取工具:不依赖 torch,支持 bf16。

safetensors 的 numpy 接口不支持 bfloat16,这里直接解析文件头 + 原始字节,
bf16 -> float32 用位移实现(把 uint16 放进 float32 高 16 位,精确无损)。
只读操作,不修改权重目录中任何文件。
"""

import json
import struct
from pathlib import Path

import numpy as np


def read_shard_header(path: str | Path) -> tuple[dict, int]:
    """读取 safetensors 文件头,返回 (header dict, 数据区起始偏移)。"""
    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len))
    return header, 8 + header_len


def read_tensor(path: str | Path, name: str, dtype=np.float64) -> np.ndarray:
    """从 safetensors 分片中读取单个张量(支持 BF16/F16/F32),返回 numpy 数组。

    Args:
        path: 分片文件路径(只读打开)。
        name: 张量名。
        dtype: 输出数值类型,默认 float64。
    """
    header, data_start = read_shard_header(path)
    if name not in header:
        raise KeyError(f"{path} 中不存在张量 {name!r}")
    info = header[name]
    begin, end = info["data_offsets"]
    with open(path, "rb") as f:
        f.seek(data_start + begin)
        raw = f.read(end - begin)

    st_dtype = info["dtype"]
    if st_dtype == "BF16":
        u16 = np.frombuffer(raw, dtype="<u2")
        arr = (u16.astype(np.uint32) << 16).view(np.float32)
    elif st_dtype == "F16":
        arr = np.frombuffer(raw, dtype=np.float16)
    elif st_dtype == "F32":
        arr = np.frombuffer(raw, dtype="<f4")
    else:
        raise ValueError(f"暂不支持的张量类型 {st_dtype}(张量 {name!r})")
    return arr.reshape(info["shape"]).astype(dtype)
