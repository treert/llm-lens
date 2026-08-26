"""纯 numpy 实现 TF-IDF 检索,在本仓库 docs/ 语料上跑通稀疏检索全流程。

不下载任何模型权重(遵守仓库"本机不跑大模型"原则)。TF-IDF 可看作
无饱和、无长度归一的 BM25 近亲,配套文档见 docs/retrieval/09-hands-on.md。

用法(在仓库根目录下):
    python demos/retrieval/tfidf_retrieval.py
    python demos/retrieval/tfidf_retrieval.py --query "KV Cache 为什么省显存" --top 5
"""

import argparse
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

# 英文/数字按词切;中文按连续汉字段切,再取相邻 bigram(无分词器时的折中)
RE_TOKEN = re.compile(r"[a-z0-9_]+|[一-鿿]+")

ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = ROOT / "docs"


def tokenize(text: str) -> list[str]:
    """切词:英文小写单词 + 中文相邻二字组。"""
    tokens: list[str] = []
    for seg in RE_TOKEN.findall(text.lower()):
        if "一" <= seg[0] <= "鿿":
            if len(seg) == 1:
                tokens.append(seg)
            else:
                tokens.extend(seg[i : i + 2] for i in range(len(seg) - 1))
        else:
            tokens.append(seg)
    return tokens


def load_chunks(min_len: int = 30, max_len: int = 300) -> list[tuple[str, str]]:
    """把 docs/**/*.md 按空行切段,过滤过短/过长段,返回 (来源文件, 文本) 列表。"""
    chunks: list[tuple[str, str]] = []
    for path in sorted(DOCS_DIR.rglob("*.md")):
        rel = str(path.relative_to(ROOT))
        text = path.read_text(encoding="utf-8")
        for para in re.split(r"\n\s*\n", text):
            para = " ".join(para.split())
            if min_len <= len(para) <= max_len:
                chunks.append((rel, para))
    return chunks


def build_tfidf(chunks: list[tuple[str, str]], max_vocab: int = 12000):
    """构建 TF-IDF 稀疏检索矩阵(稠密 float32 存储,行向量 L2 归一)。

    tf 取 1+log(f),idf 取 log((N+1)/(n+0.5))。
    返回 (矩阵 [n_chunks, n_vocab], 词表, idf 向量, chunks)。
    """
    doc_tokens = [tokenize(text) for _, text in chunks]
    df: Counter = Counter()
    for toks in doc_tokens:
        df.update(set(toks))
    vocab = {tok: i for i, (tok, _) in enumerate(df.most_common(max_vocab))}

    n_docs = len(chunks)
    mat = np.zeros((n_docs, len(vocab)), dtype=np.float32)
    for row, toks in enumerate(doc_tokens):
        for tok, f in Counter(toks).items():
            col = vocab.get(tok)
            if col is not None:
                mat[row, col] = 1.0 + math.log(f)

    idf = np.log(
        (n_docs + 1.0)
        / (np.array([df[tok] for tok in vocab], dtype=np.float32) + 0.5)
    )
    mat *= idf
    mat /= np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat, vocab, idf.astype(np.float32), chunks


def encode_query(query: str, vocab: dict, idf: np.ndarray) -> np.ndarray:
    """用与建库相同的管线把 query 编码成归一向量。"""
    q = np.zeros(len(vocab), dtype=np.float32)
    for tok, f in Counter(tokenize(query)).items():
        col = vocab.get(tok)
        if col is not None:
            q[col] = 1.0 + math.log(f)
    q *= idf
    q /= np.linalg.norm(q) + 1e-12
    return q


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default="KV Cache 为什么能省显存")
    parser.add_argument("--top", type=int, default=5, help="返回前 N 条(默认 5)")
    args = parser.parse_args()

    chunks = load_chunks()
    mat, vocab, idf, chunks = build_tfidf(chunks)
    print(f"语料: {len(chunks)} 个 chunk,词表 {len(vocab)} 词")
    print(f"查询: {args.query}\n")

    q = encode_query(args.query, vocab, idf)
    sims = mat @ q  # 行向量均已归一,点积即余弦相似度
    top_idx = sims.argsort()[::-1][: args.top]
    for rank, i in enumerate(top_idx, 1):
        src, text = chunks[i]
        print(f"[{rank}] score={sims[i]:.4f}  {src}")
        print(f"    {text[:80]}{'...' if len(text) > 80 else ''}")


if __name__ == "__main__":
    main()
