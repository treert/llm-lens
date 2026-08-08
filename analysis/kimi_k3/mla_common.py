"""Kimi-K3 MLA 层谱分析的公共函数(本目录各脚本共用)。

方法见 docs/qk-spectrum.md §2/§6.1:不构造 D×D 的 M = W_Q^T W_K;
MLA 的 96 头共享 q_a / kv_a 潜空间,每层只算一次潜空间 Gram 矩阵。

K3 的 MLA 是 NoPE(config: mla_use_nope=true):q 的 192 维 = nope 128 + "rot" 64
(不施加旋转);K 侧 rot 段由 kv_a 直出、96 头共享。有效权重(RMSNorm 增益已折入):
    W_Qh = q_b[h] @ diag(g_q) @ q_a            (192×7168)
    W_Kh = [kv_b_k[h] @ diag(g_kv) @ kv_a_c]   (128×7168, nope 段)
           [kv_a_rot]                          (64×7168,  rot 段, 全头共享)
"""

from pathlib import Path

import numpy as np

from llm_lens import read_tensor

# K3 MLA 结构常量(见 output/kimi_k3/structure_overview.json)
NUM_HEADS = 96
QK_NOPE_DIM = 128   # 每头 nope 维度
QK_ROT_DIM = 64     # 每头 "rot" 维度(NoPE,不旋转)
V_HEAD_DIM = 128
Q_HEAD_DIM = QK_NOPE_DIM + QK_ROT_DIM            # 192
KV_B_HEAD_DIM = QK_NOPE_DIM + V_HEAD_DIM         # 256
Q_LORA_RANK = 1536
KV_LORA_RANK = 512

EIG_TOL = 1e-12  # Gram 特征值截断(相对最大值),防止 bf16 噪声出现负特征值


def load_mla_layer(model_dir: Path, weight_map: dict[str, str], idx: int) -> dict[str, np.ndarray]:
    """读取一个 MLA 层的 6 个相关权重,返回 float64 numpy 数组。"""
    prefix = f"language_model.model.layers.{idx}.self_attn."
    names = {
        "q_a": prefix + "q_a_proj.weight",
        "q_a_norm": prefix + "q_a_layernorm.weight",
        "q_b": prefix + "q_b_proj.weight",
        "kv_a": prefix + "kv_a_proj_with_mqa.weight",
        "kv_a_norm": prefix + "kv_a_layernorm.weight",
        "kv_b": prefix + "kv_b_proj.weight",
    }
    return {
        key: read_tensor(model_dir / weight_map[name], name)
        for key, name in names.items()
    }


def prepare_latents(w: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """折叠 RMSNorm 增益,拆分 kv_a 的压缩段/rot 段,计算潜空间 Gram(每层一次)。"""
    q_a = w["q_a_norm"][:, None] * w["q_a"]                  # 1536×7168
    kv_a = w["kv_a_norm"][:, None] * w["kv_a"][:KV_LORA_RANK]  # 512×7168
    a_rot = w["kv_a"][KV_LORA_RANK:]                         # 64×7168
    return {
        "q_a": q_a, "kv_a": kv_a, "a_rot": a_rot,
        "G_qq": q_a @ q_a.T,      # 1536×1536
        "G_qk": q_a @ kv_a.T,     # 1536×512
        "G_kk": kv_a @ kv_a.T,    # 512×512
        "G_qr": q_a @ a_rot.T,    # 1536×64
        "G_kr": kv_a @ a_rot.T,   # 512×64
        "G_rr": a_rot @ a_rot.T,  # 64×64
    }


def head_qk_slices(w: dict[str, np.ndarray], h: int) -> tuple[np.ndarray, np.ndarray]:
    """取第 h 头的 q_b 切片(192×1536)与 kv_b 的 k_nope 切片(128×512)。"""
    qb = w["q_b"][h * Q_HEAD_DIM:(h + 1) * Q_HEAD_DIM]
    kb = w["kv_b"][h * KV_B_HEAD_DIM: h * KV_B_HEAD_DIM + QK_NOPE_DIM]
    return qb, kb


def head_grams(lat: dict[str, np.ndarray], qb: np.ndarray, kb: np.ndarray):
    """由潜空间 Gram 组装逐头的三个 H×H Gram:G_Q、G_K、G_QK。"""
    qn, qr_ = qb[:QK_NOPE_DIM], qb[QK_NOPE_DIM:]
    G_Q = qb @ lat["G_qq"] @ qb.T
    G_K = np.block([[kb @ lat["G_kk"] @ kb.T, kb @ lat["G_kr"]],
                    [lat["G_kr"].T @ kb.T, lat["G_rr"]]])
    G_QK = np.block([[qn @ lat["G_qk"] @ kb.T, qn @ lat["G_qr"]],
                     [qr_ @ lat["G_qk"] @ kb.T, qr_ @ lat["G_qr"]]])
    return G_Q, G_K, G_QK


def gram_eigh(G: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """对称 PSD 矩阵特征分解,返回 (U, sqrt(特征值)),按奇异值降序、截断数值噪声。"""
    evals, U = np.linalg.eigh(G)
    order = np.argsort(evals)[::-1]
    evals, U = evals[order], U[:, order]
    keep = evals > evals[0] * EIG_TOL
    return U[:, keep], np.sqrt(evals[keep])


def head_spectrum(G_Q: np.ndarray, G_K: np.ndarray, G_QK: np.ndarray) -> dict:
    """由三个 H×H Gram 计算 M 的谱指标,不显式构造 D×D 的 M。

    G_Q = W_Q W_Q^T, G_K = W_K W_K^T, G_QK = W_Q W_K^T (均为 H×H)。
    返回的 U_K/s_K/svd_Q 供需要右奇异向量的分析(如共享潜库指纹)使用。
    """
    U_Q, s_Q = gram_eigh(G_Q)  # W_Q = U_Q diag(s_Q) V_Q^T 的 (U_Q, s_Q)
    U_K, s_K = gram_eigh(G_K)
    C = U_Q.T @ U_K                        # 对齐矩阵(H×H)
    S = (s_Q[:, None] * C) * s_K[None, :]  # Sigma_Q C Sigma_K
    P, sv, Qt = np.linalg.svd(S)           # M 的奇异值 = S 的奇异值
    Q = Qt.T

    # V_Q^T V_K = Sigma_Q^-1 U_Q^T (W_Q W_K^T) U_K Sigma_K^-1
    VQK = (U_Q.T @ G_QK @ U_K) / s_Q[:, None] / s_K[None, :]
    D = P.T @ VQK @ Q                      # D[i,j] = u_i·v_j
    uv = np.diag(D).copy()

    fro2 = float(sv @ sv)
    # ||M-M^T||_F^2 = 2 sum(sv^2) - 2 sum_{ij} sv_i sv_j (u_i·v_j)^2
    sym2 = 2.0 * fro2 - 2.0 * float((sv[:, None] * sv[None, :] * D**2).sum())
    absC = np.abs(C)
    return {
        "sv": sv,
        "uv": uv,
        "sigma1": float(sv[0]),
        "fro2": fro2,
        "r_eff": float(sv.sum() ** 2 / fro2),
        "sym": float(np.sqrt(max(sym2, 0.0) / fro2)),
        "align_max": float(absC.max()),
        "align_frac_05": float((absC > 0.5).mean()),
        "align_frac_01": float((absC > 0.1).mean()),
        "align_r_eff": float(absC.sum() ** 2 / (C**2).sum()),
        # 供下游使用的中间量
        "U_K": U_K, "s_K": s_K, "svd_Q": Q,
    }
