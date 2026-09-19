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


def _fp8_e4m3_table() -> np.ndarray:
    """FP8 E4M3(fn) 的 256 项解码表:1 符号位 + 4 指数位(偏置 7)+ 3 尾数位。

    无无穷大,指数全 1 且尾数全 1 为 NaN;正规格化最小值 2^-6,最大 448。
    """
    bits = np.arange(256, dtype=np.uint8)
    sign = np.where(bits & 0x80, -1.0, 1.0)
    exp = (bits >> 3) & 0x0F
    mant = (bits & 0x07).astype(np.float32) / 8.0
    val = np.where(exp == 0, mant * 2.0**-6, (1.0 + mant) * 2.0 ** (exp.astype(np.int32) - 7))
    val = sign * val
    val[(bits & 0x7F) == 0x7F] = np.nan  # 0x7F / 0xFF
    return val.astype(np.float32)


_FP8_E4M3_TABLE = _fp8_e4m3_table()

# FP4 E2M1(fn) 的 16 项解码表(低 3 位:1 指数位偏置 1 + 2... 实际为 OCP MX 规格):
# 索引低 4 位 = 数值编码,bit3 为符号。
FP4_E2M1_TABLE = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32,
)


def e8m0_to_float(bits: np.ndarray) -> np.ndarray:
    """UE8M0 缩放系数解码:无符号 8 位纯指数,值 = 2^(bits-127);0xFF 为 NaN。"""
    u8 = np.ascontiguousarray(bits).view(np.uint8)
    out = np.exp2(u8.astype(np.int32) - 127).astype(np.float32)
    out[u8 == 0xFF] = np.nan
    return out


def unpack_fp4_e2m1(packed: np.ndarray) -> np.ndarray:
    """解包 FP4 E2M1:每字节低半字节在前(偶数列)、高半字节在后,最后一维宽度翻倍。"""
    u8 = np.ascontiguousarray(packed).view(np.uint8)
    low = FP4_E2M1_TABLE[u8 & 0x0F]
    high = FP4_E2M1_TABLE[u8 >> 4]
    out = np.stack([low, high], axis=-1)
    return out.reshape(*u8.shape[:-1], 2 * u8.shape[-1])


def dequant_block(weight: np.ndarray, scale: np.ndarray, block: tuple[int, int] = (32, 32)) -> np.ndarray:
    """块量化反量化:weight 为解码后的浮点矩阵,scale 为逐块缩放,返回 weight * scale(逐块广播)。

    Args:
        weight: 形状 (R, C) 的浮点数组(如 FP8 解码结果)。
        scale: 形状 (R//br, C//bc) 的浮点数组(如 UE8M0 解码结果)。
        block: 块大小 (br, bc),默认 32x32。
    """
    br, bc = block
    r, c = weight.shape
    if scale.shape != (r // br, c // bc) or r % br or c % bc:
        raise ValueError(f"weight {weight.shape} 与 scale {scale.shape} 不满足块大小 {block}")
    return weight * np.kron(scale, np.ones(block, dtype=np.float32))


def _decode_buffer(raw: bytes, st_dtype: str, name: str) -> np.ndarray:
    """把 safetensors 原始字节解码为一维 numpy 数组(BF16 用位移转 float32,无损)。

    FP8(E4M3/E8M0)解码为 float32;I8 保留原始整型(打包的 FP4 需要原始字节,
    由调用方用 unpack_fp4_e2m1 解包)。
    """
    if st_dtype == "BF16":
        u16 = np.frombuffer(raw, dtype="<u2")
        return (u16.astype(np.uint32) << 16).view(np.float32)
    if st_dtype == "F16":
        return np.frombuffer(raw, dtype=np.float16)
    if st_dtype == "F32":
        return np.frombuffer(raw, dtype="<f4")
    if st_dtype == "F8_E4M3":
        return _FP8_E4M3_TABLE[np.frombuffer(raw, dtype=np.uint8)]
    if st_dtype == "F8_E8M0":
        return e8m0_to_float(np.frombuffer(raw, dtype=np.uint8))
    if st_dtype == "I8":
        return np.frombuffer(raw, dtype=np.int8)
    raise ValueError(f"暂不支持的张量类型 {st_dtype}(张量 {name!r})")


_ST_ITEMSIZE = {"BF16": 2, "F16": 2, "F32": 4, "F8_E4M3": 1, "F8_E8M0": 1, "I8": 1}


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
