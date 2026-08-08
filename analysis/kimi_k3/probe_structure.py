"""探测 Kimi-K3 的完整结构,输出 JSON 描述文件到 output/kimi_k3/。

只读 config.json、model.safetensors.index.json 和各 safetensors 分片的头部
(不加载任何权重数据本体),产出三份描述文件:

- structure_overview.json : 整体结构总览(config 关键参数、层布局、量化方式、张量统计)
- layers.json             : 逐层结构(注意力/MLP 类型、各权重形状与 dtype;路由专家汇总)
- weight_patterns.json    : 权重命名模式统计(序号归一化分组,含数量/形状/元素总数)

用法(在仓库根目录下):
    python analysis/kimi_k3/probe_structure.py
    python analysis/kimi_k3/probe_structure.py --model-dir G:/llm-models/Kimi-K3
    python analysis/kimi_k3/probe_structure.py --out-dir output/kimi_k3
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from safetensors import safe_open

from llm_lens import get_model_dir
from llm_lens.cli import add_model_args

REPO_ROOT = Path(__file__).resolve().parents[2]

# dtype 字符串 -> 每元素字节数(用于估算存储量;U8 的 packed 权重实际含 2 个 4bit 权重)
DTYPE_ITEMSIZE = {
    "BF16": 2, "F16": 2, "F32": 4, "F64": 8,
    "U8": 1, "I8": 1, "U16": 2, "I16": 2, "I32": 4, "I64": 8,
    "BOOL": 1,
}

LAYER_RE = re.compile(r"^language_model\.model\.layers\.(\d+)\.(.*)$")
EXPERT_RE = re.compile(r"^block_sparse_moe\.experts\.(\d+)\.(.*)$")


def normalize_name(name: str) -> str:
    """把 layers.12. / experts.895. 这类序号归一为 *.,便于按结构分组。"""
    return re.sub(r"\.\d+\.", ".*.", name)


def numel(shape: list[int]) -> int:
    n = 1
    for s in shape:
        n *= s
    return n


def collect_tensor_meta(model_dir: Path) -> dict[str, dict]:
    """读 index + 各分片头部,返回 {张量名: {shape, dtype, shard}}。不加载数据。"""
    index_path = model_dir / "model.safetensors.index.json"
    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)
    weight_map: dict[str, str] = index["weight_map"]
    total_size = index.get("metadata", {}).get("total_size", 0)

    by_shard: dict[str, list[str]] = defaultdict(list)
    for name, shard in weight_map.items():
        by_shard[shard].append(name)

    meta: dict[str, dict] = {}
    for shard in sorted(by_shard):
        with safe_open(model_dir / shard, framework="numpy") as f:
            for name in by_shard[shard]:
                sl = f.get_slice(name)
                meta[name] = {
                    "shape": list(sl.get_shape()),
                    "dtype": sl.get_dtype(),
                    "shard": shard,
                }
    return meta, total_size


def build_layer_records(meta: dict[str, dict], text_cfg: dict) -> tuple[list[dict], dict]:
    """按层分组,识别每层注意力/MLP 类型,路由专家汇总。返回 (layers, leftovers)。"""
    layers_map: dict[int, dict[str, dict]] = defaultdict(dict)
    leftovers: dict[str, dict] = {}
    for name, info in meta.items():
        m = LAYER_RE.match(name)
        if m:
            layers_map[int(m.group(1))][m.group(2)] = info
        else:
            leftovers[name] = info

    la_cfg = text_cfg.get("linear_attn_config", {})
    # config 层号是 1-based(1~93),权重名 layers.N 是 0-based(0~92),统一转成 0-based 再比对
    cfg_full = {i - 1 for i in la_cfg.get("full_attn_layers", [])}
    cfg_kda = {i - 1 for i in la_cfg.get("kda_layers", [])}
    first_dense = text_cfg.get("first_k_dense_replace", 0)

    records = []
    warnings = []
    for idx in sorted(layers_map):
        weights = layers_map[idx]
        # 注意力类型:以权重名为准(q_a_proj 存在 <-> MLA),再与 config 交叉验证
        attn_type = "mla" if any(w.startswith("self_attn.q_a_proj") for w in weights) else "kda"
        if (idx in cfg_full) != (attn_type == "mla"):
            warnings.append(
                f"层 {idx}(0-based): config 标为 "
                f"{'full_attn' if idx in cfg_full else 'kda'} 但权重为 {attn_type}"
            )
        # MLP 类型:以权重名为准(有 block_sparse_moe <-> MoE),再与 first_k_dense_replace 比对
        mlp_type = "moe" if any(w.startswith("block_sparse_moe.") for w in weights) else "dense"
        if (idx < first_dense) != (mlp_type == "dense"):
            warnings.append(
                f"层 {idx}(0-based): first_k_dense_replace={first_dense} 与权重 {mlp_type} 不符"
            )

        attn_w, mlp_w, other_w = {}, {}, {}
        expert_shapes: dict[str, set] = defaultdict(set)
        expert_ids = set()
        for suffix, info in weights.items():
            em = EXPERT_RE.match(suffix)
            if em:
                expert_ids.add(int(em.group(1)))
                expert_shapes[em.group(2)].add(tuple(info["shape"]) + (info["dtype"],))
                continue
            entry = {"shape": info["shape"], "dtype": info["dtype"]}
            if suffix.startswith("self_attn."):
                attn_w[suffix] = entry
            elif suffix.startswith(("block_sparse_moe.", "mlp.")):
                mlp_w[suffix] = entry
            else:
                other_w[suffix] = entry

        experts_summary = None
        if expert_ids:
            patterns = {}
            for pat, shapes in sorted(expert_shapes.items()):
                # 所有专家同形状才有意义;不一致时记录全部出现过的 (shape, dtype)
                patterns[pat] = (
                    {"shape": list(next(iter(shapes))[:-1]), "dtype": next(iter(shapes))[-1]}
                    if len(shapes) == 1
                    else {"inconsistent": sorted(list(s) for s in shapes)}
                )
            experts_summary = {"count": len(expert_ids), "patterns": patterns}

        records.append({
            "index": idx,
            "attn_type": attn_type,
            "mlp_type": mlp_type,
            "attn_weights": attn_w,
            "mlp_weights": mlp_w,
            "other_weights": other_w,
            "routed_experts": experts_summary,
        })
    return records, {"items": leftovers, "warnings": warnings}


def summarize_patterns(meta: dict[str, dict]) -> list[dict]:
    """序号归一化后的权重模式统计,按出现次数降序。"""
    groups: dict[str, list[dict]] = defaultdict(list)
    for name, info in meta.items():
        groups[normalize_name(name)].append(info)
    rows = []
    for pat, infos in groups.items():
        shapes = {tuple(i["shape"]) + (i["dtype"],) for i in infos}
        shape, dtype = (list(next(iter(shapes))[:-1]), next(iter(shapes))[-1]) if len(shapes) == 1 else (None, None)
        rows.append({
            "pattern": pat,
            "count": len(infos),
            "shape": shape,
            "dtype": dtype,
            "total_numel": sum(numel(i["shape"]) for i in infos),
            "quantized": pat.endswith(("weight_packed", "weight_scale")),
        })
    rows.sort(key=lambda r: -r["count"])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_args(parser)
    parser.add_argument(
        "--out-dir", default=str(REPO_ROOT / "output" / "kimi_k3"),
        help="JSON 输出目录(默认 output/kimi_k3)",
    )
    args = parser.parse_args()

    model_dir = get_model_dir(args.model, args.model_dir, args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"模型目录: {model_dir}")
    print(f"输出目录: {out_dir}")

    with open(model_dir / "config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    text_cfg = config["text_config"]
    vision_cfg = config.get("vision_config", {})
    la_cfg = text_cfg.get("linear_attn_config", {})

    print("读取各分片头部(不加载权重数据)...")
    meta, total_size = collect_tensor_meta(model_dir)
    print(f"  张量总数: {len(meta)}")

    layers, extra = build_layer_records(meta, text_cfg)
    patterns = summarize_patterns(meta)

    # ---- 整体总览 ----
    dtype_stats = Counter()
    for info in meta.values():
        dtype_stats[info["dtype"]] += 1
    overview = {
        "model_dir": str(model_dir),
        "architectures": config.get("architectures"),
        "model_type": config.get("model_type"),
        "text": {
            "model_type": text_cfg.get("model_type"),
            "num_hidden_layers": text_cfg.get("num_hidden_layers"),
            "hidden_size": text_cfg.get("hidden_size"),
            "vocab_size": text_cfg.get("vocab_size"),
            "tie_word_embeddings": config.get("tie_word_embeddings"),
            "max_position_embeddings": text_cfg.get("max_position_embeddings"),
            "attn_res_block_size": text_cfg.get("attn_res_block_size"),
            "full_attn_layers": la_cfg.get("full_attn_layers"),
            "kda_layers": la_cfg.get("kda_layers"),
            "mla": {
                "num_heads": text_cfg.get("num_attention_heads"),
                "q_lora_rank": text_cfg.get("q_lora_rank"),
                "kv_lora_rank": text_cfg.get("kv_lora_rank"),
                "qk_nope_head_dim": text_cfg.get("qk_nope_head_dim"),
                "qk_rope_head_dim": text_cfg.get("qk_rope_head_dim"),
                "v_head_dim": text_cfg.get("v_head_dim"),
                "mla_use_nope": text_cfg.get("mla_use_nope"),
                "mla_use_output_gate": text_cfg.get("mla_use_output_gate"),
            },
            "kda": {
                "num_heads": la_cfg.get("num_heads"),
                "head_dim": la_cfg.get("head_dim"),
                "short_conv_kernel_size": la_cfg.get("short_conv_kernel_size"),
                "use_full_rank_gate": la_cfg.get("use_full_rank_gate"),
                "gate_lower_bound": la_cfg.get("gate_lower_bound"),
            },
            "moe": {
                "num_experts": text_cfg.get("num_experts"),
                "num_experts_per_token": text_cfg.get("num_experts_per_token"),
                "num_shared_experts": text_cfg.get("num_shared_experts"),
                "moe_intermediate_size": text_cfg.get("moe_intermediate_size"),
                "routed_expert_hidden_size": text_cfg.get("routed_expert_hidden_size"),
                "first_k_dense_replace": text_cfg.get("first_k_dense_replace"),
                "intermediate_size": text_cfg.get("intermediate_size"),
                "router_activation": text_cfg.get("moe_router_activation_func"),
                "topk_method": text_cfg.get("topk_method"),
            },
            "quantization": text_cfg.get("quantization_config", {}).get("format"),
            "position_encoding_note": (
                "语言模型无 RoPE:mla_use_nope=true 且建模代码中 rotary_emb=None,"
                "qk_rope_head_dim=64 的维度作为普通维度保留;KDA 层靠 short_conv 提供局部位置信息"
            ),
        },
        "vision": {
            "vt_num_hidden_layers": vision_cfg.get("vt_num_hidden_layers"),
            "vt_hidden_size": vision_cfg.get("vt_hidden_size"),
            "vt_num_attention_heads": vision_cfg.get("vt_num_attention_heads"),
            "patch_size": vision_cfg.get("patch_size"),
            "pos_emb_type": vision_cfg.get("pos_emb_type"),
            "text_hidden_size": vision_cfg.get("text_hidden_size"),
        },
        "tensors": {
            "total_count": len(meta),
            "total_size_gib": round(total_size / 2**30, 2),
            "dtype_counts": dict(dtype_stats.most_common()),
        },
    }

    layers_doc = {
        "layer_indexing": (
            "layers[].index 为 0-based(权重名 layers.N 的 N);"
            "config.json 的 full_attn_layers/kda_layers 为 1-based,config 层号 = index + 1"
        ),
        "layer_index_range": [layers[0]["index"], layers[-1]["index"]] if layers else None,
        "attn_type_counts": dict(Counter(r["attn_type"] for r in layers)),
        "mlp_type_counts": dict(Counter(r["mlp_type"] for r in layers)),
        "cross_check_warnings": extra["warnings"],
        "layers": layers,
        "non_layer_weights": {k: {"shape": v["shape"], "dtype": v["dtype"]}
                              for k, v in sorted(extra["items"].items())},
    }

    files = {
        "structure_overview.json": overview,
        "layers.json": layers_doc,
        "weight_patterns.json": {"patterns": patterns},
    }
    for fname, doc in files.items():
        path = out_dir / fname
        with open(path, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)
        print(f"已写出: {path} ({path.stat().st_size / 1024:.1f} KiB)")

    # ---- 小步验证:打印关键结论 ----
    print("\n== 结构核验 ==")
    print(f"层号范围: {layers_doc['layer_index_range']}, 共 {len(layers)} 层")
    print(f"注意力类型: {layers_doc['attn_type_counts']}")
    print(f"MLP 类型:  {layers_doc['mlp_type_counts']}")
    if extra["warnings"]:
        print("config 与权重交叉验证警告:")
        for w in extra["warnings"]:
            print(f"  ! {w}")
    else:
        print("config 层布局与权重名交叉验证: 一致")
    print(f"dtype 分布: {overview['tensors']['dtype_counts']}")


if __name__ == "__main__":
    main()
