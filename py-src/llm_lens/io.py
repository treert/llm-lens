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


def _decode_buffer(raw: bytes, st_dtype: str, name: str) -> np.ndarray:
    """把 safetensors 原始字节解码为一维 numpy 数组(BF16 用位移转 float32,无损)。"""
    if st_dtype == "BF16":
        u16 = np.frombuffer(raw, dtype="<u2")
        return (u16.astype(np.uint32) << 16).view(np.float32)
    if st_dtype == "F16":
        return np.frombuffer(raw, dtype=np.float16)
    if st_dtype == "F32":
        return np.frombuffer(raw, dtype="<f4")
    raise ValueError(f"暂不支持的张量类型 {st_dtype}(张量 {name!r})")


_ST_ITEMSIZE = {"BF16": 2, "F16": 2, "F32": 4}


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

    arr = _decode_buffer(raw, info["dtype"], name)
    return arr.reshape(info["shape"]).astype(dtype)


def iter_row_chunks(
    path: str | Path,
    name: str,
    chunk_rows: int = 4096,
    dtype=np.float32,
):
    """按行分块迭代读取 2D 张量,避免大表(如词嵌入)整表载入内存。

    Args:
        path: 分片文件路径(只读打开)。
        name: 张量名,必须是 2D(C 行主序)。
        chunk_rows: 每块的行数。
        dtype: 输出数值类型,默认 float32(统计累加请自行转 float64)。

    Yields:
        (chunk, row_start):chunk 形状 (n_rows, n_cols),row_start 为起始行号。
    """
    header, data_start = read_shard_header(path)
    if name not in header:
        raise KeyError(f"{path} 中不存在张量 {name!r}")
    info = header[name]
    shape = info["shape"]
    if len(shape) != 2:
        raise ValueError(f"iter_row_chunks 只支持 2D 张量,{name!r} 形状为 {shape}")
    st_dtype = info["dtype"]
    if st_dtype not in _ST_ITEMSIZE:
        raise ValueError(f"暂不支持的张量类型 {st_dtype}(张量 {name!r})")
    rows, cols = shape
    row_bytes = cols * _ST_ITEMSIZE[st_dtype]
    begin, _ = info["data_offsets"]

    with open(path, "rb") as f:
        for r0 in range(0, rows, chunk_rows):
            r1 = min(r0 + chunk_rows, rows)
            f.seek(data_start + begin + r0 * row_bytes)
            raw = f.read((r1 - r0) * row_bytes)
            chunk = _decode_buffer(raw, st_dtype, name).reshape(r1 - r0, cols)
            yield chunk.astype(dtype, copy=False), r0
