"""演示 DeepSeek-V4.1-Flash Engram 的 n-gram 哈希查表全过程。

对照模型目录 inference/engram.py 中 NgramHashState 的逻辑,
去掉 tokenizer 压缩映射(直接给定压缩后的 id)与分布式部分,纯 numpy 实现。
不读取模型权重,无需配置模型路径。

用法: python analysis/ds_v4_1_flash/engram_hash_demo.py
"""

import numpy as np

# ---- 配置(取自 config.json) ----
MAX_NGRAM = 4                    # engram_max_ngram_size
N_HEADS = 8                      # engram_n_heads
LAYER_IDS = (1, 14)              # engram_layer_ids
ENGRAM_VOCAB_SIZE = 16_000_000   # engram_vocab_size: 质数桶从这里开始找
COMPRESSED_VOCAB = 99092         # engram_compressed_vocab_size
PAD_ID = 2                       # 占位值; 真实实现是 pad token 经压缩映射后的 id(token_map[engram_pad_token_id])


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    r = int(n**0.5)
    for i in range(3, r + 1, 2):
        if n % i == 0:
            return False
    return True


def find_next_prime(start: int, seen: set[int]) -> int:
    """大于 start 且未分配过的最小质数。"""
    c = start + 1
    while not is_prime(c) or c in seen:
        c += 1
    return c


def build_layout():
    """为每层分配桶区间: 3 种阶数(2/3/4-gram) x 8 头, 各占一个质数大小的不相交区间。

    为什么用质数:
    - 单桶: 模奇质数会把哈希值的高低位信息都混入桶号; 若模数是 2^k 这类合数,
      等价于只看低 k 位, 而 XOR 滚动哈希的低位未必均匀, 会造成聚类;
    - 8 个不同质数(中国剩余定理): 两两互质的模数下, x -> (x mod p1, ..., x mod p8)
      在 [0, p1*...*p8) 上是双射。这里连乘约 (1.6e7)^8 ~ 1e57, 远大于 64 位哈希值域,
      即 8 元组唯一确定哈希值本身——两个不同 n-gram 要 8 头全撞, 除非哈希值本就相等,
      碰撞问题被彻底归约为单哈希值的碰撞问题;
    - 每段约 1600 万行 ≈ engram_vocab_size: 这只是单桶容量的设定值。n-gram 取值空间
      是 99092^n(2-gram 已约 1e10 种), 远大于桶数——同桶碰撞是设计内必然而非小概率,
      冲突的 n-gram 共享一行、由训练学出折中向量(哈希 embedding 的标准思路),
      8 头 × 3 阶的冗余保证整体信息仍能可靠取出。
    """
    primes, seen = [], set()
    for _ in LAYER_IDS:
        per_order = []
        for _ in range(MAX_NGRAM - 1):          # 阶数: 2-gram, 3-gram, 4-gram
            sizes, cur = [], ENGRAM_VOCAB_SIZE - 1
            for _ in range(N_HEADS):
                cur = find_next_prime(cur, seen)
                seen.add(cur)
                sizes.append(cur)
            per_order.append(sizes)
        primes.append(per_order)
    # offsets: 每层 24 个桶区间在表内的起始行号
    flat = [[p for order in layer for p in order] for layer in primes]
    offsets = [np.cumsum([0] + s[:-1]).tolist() for s in flat]
    return primes, offsets


def compute_hash_multipliers(layer_ids, vocab_size):
    """每层一组乘子: seed=10007*layer_id, 强制奇数, 上界保证 token_id*乘子 不溢出 int64。

    为什么是奇数: 在模 2^64 的整数环上, 奇数是乘法可逆元(gcd(a, 2^64)=1),
    x -> a*x 是双射, 不丢任何比特; 偶数乘子等价于左移, 高位信息丢失易碰撞。
    这是乘法哈希(Knuth; Dietzfelbinger multiply-shift 族证明 2-universal)的标准前提。
    为什么有上界: 保证 t*m 在 int64 内精确成立、不溢出回绕; 一旦溢出, 乘法变成
    隐式 mod 2^64, 比特模式被截断, 后续 XOR 混合的均匀性就没有保证了。
    每层不同 seed: 同一 n-gram 在 L1/L14 落在完全不同的槽位, 层间记忆分工不冗余。
    """
    bound = max(1, (np.iinfo(np.int64).max // vocab_size) // 2)
    rows = []
    for layer_id in layer_ids:
        g = np.random.default_rng(10007 * layer_id)
        rows.append(g.integers(0, bound, size=(MAX_NGRAM,), dtype=np.int64) * 2 + 1)
    return np.stack(rows)


def hash_sequence(seq, layer_idx, primes, offsets, multipliers):
    """对压缩 token 序列求每个位置的 24 个行号(3 阶 x 8 头)。"""
    L = len(seq)
    # 1) 回看窗口: tokens[pos] = [当前, 前1个, 前2个, 前3个], 越过序列开头填 pad
    tokens = np.full((L, MAX_NGRAM), PAD_ID, dtype=np.int64)
    for shift in range(MAX_NGRAM):
        for pos in range(shift, L):
            tokens[pos, shift] = seq[pos - shift]

    # 2) 滚动 XOR: 第 i 步结束时的值就是 (i+1)-gram 的哈希
    #    - 乘奇数常数是双射但非线性(带进位传播), XOR 在 GF(2)^64 上是线性运算,
    #      两者交替是典型的混淆-扩散组合(类似分组密码);
    #    - XOR 虽可交换, 但每个位置的乘子不同: t1*m0 ^ t0*m1 != t0*m0 ^ t1*m1,
    #      顺序信息得以保留;
    #    - 滚动结构让 2/3/4-gram 共享一条计算链, O(n) 一次扫完三种阶数
    products = tokens * multipliers[layer_idx]           # [L, 4]
    rolling = products[:, 0].copy()
    hashes = []                                          # 每阶: [L, 8]
    for i in range(1, MAX_NGRAM):
        rolling = np.bitwise_xor(rolling, products[:, i])
        hashes.append(rolling[:, None] % np.array(primes[layer_idx][i - 1]))

    # 3) 加上各桶区间的 offset -> 表内行号 [L, 24]
    return np.concatenate(hashes, axis=-1) + np.array(offsets[layer_idx])


if __name__ == "__main__":
    primes, offsets = build_layout()
    multipliers = compute_hash_multipliers(LAYER_IDS, COMPRESSED_VOCAB)

    print(f"层 1 的 24 个质数桶: {primes[0][0][:3]} ... (每阶 8 个, 共 3 阶)")
    print(f"层 1 乘子: {multipliers[0].tolist()}")
    print()

    # 假装 tokenizer 压缩后得到的 id 序列("The cat sat" 之类)
    seq = [5231, 887, 40211, 96, 7712, 330]
    row_ids = hash_sequence(seq, 0, primes, offsets, multipliers)

    # 单步展开位置 4 的计算
    pos = 4
    window = [seq[pos - s] if pos >= s else PAD_ID for s in range(MAX_NGRAM)]
    print(f"位置 {pos} 的回看窗口 [当前,前1,前2,前3] = {window}")
    prods = [w * m for w, m in zip(window, multipliers[0])]
    print(f"乘以奇数乘子后 = {prods}")
    rolling = prods[0]
    for i in range(1, MAX_NGRAM):
        rolling = np.bitwise_xor(np.int64(rolling), np.int64(prods[i]))
        print(f"  XOR 到第 {i} 步 -> {i+1}-gram 哈希 = {rolling}")
    print()

    print("各位置 24 个行号(层 1, 列为 [2-gram x8, 3-gram x8, 4-gram x8]):")
    for pos in range(len(seq)):
        print(f"  pos {pos}: {row_ids[pos].tolist()}")
