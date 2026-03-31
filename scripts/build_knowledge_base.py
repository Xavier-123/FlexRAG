#!/usr/bin/env python
"""
Build a FAISS vector knowledge base from local documents.

This standalone script loads documents from a directory (or individual files),
chunks them, embeds them via a remote embedding endpoint, and persists the
resulting FAISS index to disk.

The persisted index can later be loaded by
:class:`~flexrag.retrievers.LlamaIndexRetriever` for retrieval.

Usage
-----
::

    # Build from a directory of .txt / .md / .pdf files
    python scripts/build_knowledge_base.py --input-dir ./my_docs

    # Build from specific files
    python scripts/build_knowledge_base.py --files doc1.txt doc2.pdf

    # Override the output directory and chunk settings
    python scripts/build_knowledge_base.py --input-dir ./my_docs \\
        --output-dir ./my_index --chunk-size 256 --chunk-overlap 32

    # Force overwrite an existing index
    python scripts/build_knowledge_base.py --input-dir ./my_docs --force
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
import pickle
from pathlib import Path

# [新增] 引入分词和BM25库 (需 pip install rank_bm25 jieba)
try:
    from rank_bm25 import BM25Okapi
    import jieba
except ImportError:
    print("[WARNING] Missing sparse retrieval dependencies. Run: pip install rank_bm25 jieba")

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from flexrag.indexing.knowledge import FaissKnowledgeBuilder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 稀疏检索辅助函数
# ---------------------------------------------------------------------------
def tokenize_text(text: str) -> list[str]:
    """对文本进行分词。针对中文使用 jieba，英文按空格切分。"""
    # 过滤掉空格和空字符
    return [word for word in jieba.lcut(text) if word.strip()]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build a FAISS vector knowledge base from local documents.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--input-dir",
        type=str,
        help="Path to a directory of documents (.txt, .md, .pdf).",
    )
    source.add_argument(
        "--files",
        nargs="+",
        type=str,
        help="One or more individual file paths to index.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Directory where the FAISS index will be saved. "
            "Defaults to the KNOWLEDGE_PERSIST_DIR setting."
        ),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Max tokens per chunk (default: from KNOWLEDGE_CHUNK_SIZE setting).",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help="Token overlap between chunks (default: from KNOWLEDGE_CHUNK_OVERLAP setting).",
    )
    parser.add_argument(
        "--embedding-base-url",
        type=str,
        default="http://127.0.0.1:8018/v1/embeddings",
        help="Base URL for the embedding API.",
    )
    parser.add_argument(
        "--embedding-api-key",
        type=str,
        default="sk-xxxx",
        help="API key for the embedding service.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="Qwen3-Embedding-0.6B",
        help="Name of the embedding model to use.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing index at the output directory.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    # 允许用户选择是否开启稀疏检索
    parser.add_argument("--enable-sparse", action="store_true", help="Build a BM25 sparse index alongside the FAISS dense index.")

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


async def build(args: argparse.Namespace) -> None:
    """Execute the build pipeline according to parsed CLI *args*."""

    output_dir = args.output_dir
    chunk_size = args.chunk_size
    chunk_overlap = args.chunk_overlap

    # ---- safety check ----
    if FaissKnowledgeBuilder.index_exists(output_dir) and not args.force:
        print(
            f"[ERROR] An index already exists at '{output_dir}'.\n"
            "       Use --force to overwrite it.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ---- builder ----
    builder = FaissKnowledgeBuilder(
        embed_base_url=args.embedding_base_url,
        embed_model_name=args.embedding_model,
        embed_api_key=args.embedding_api_key,
    )

    # ---- load ----
    source = args.input_dir if args.input_dir else args.files
    t0 = time.perf_counter()
    doc_count = await builder.load_files(source)
    elapsed_load = time.perf_counter() - t0
    print(f"[INFO] Loaded {doc_count} document(s) in {elapsed_load:.1f}s.")

    # ---- build dense index ----
    print(
        f"[INFO] Building index (chunk_size={chunk_size}, "
        f"chunk_overlap={chunk_overlap}) ..."
    )
    t1 = time.perf_counter()
    await builder.build_index(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elapsed_build = time.perf_counter() - t1
    print(f"[INFO] Index built in {elapsed_build:.1f}s.")

    # ---- save dense index ----
    await builder.save(output_dir)
    print(f"[INFO] Knowledge base saved to '{output_dir}'.")

    # ---- build & save sparse index (BM25) ----
    if args.enable_sparse:
        print("[INFO] Building Sparse BM25 index ...")
        t2 = time.perf_counter()

        # ⚠️ 注意: 这里需要获取分块后的文本。
        # 你需要根据 flexrag 的具体 API 获取 document chunks。
        # 假设 builder 有一个属性 `.chunks` 或方法 `.get_all_texts()`:
        try:
            # 请根据实际情况修改下方代码获取 chunks 的纯文本列表
            if hasattr(builder, 'nodes'):  # LlamaIndex 风格
                texts = [node.text for node in builder.nodes]
            elif hasattr(builder, 'chunks'):  # 常见封装风格
                texts = [chunk.text for chunk in builder.chunks]
            elif hasattr(builder, "_raw_docs"):  # 常见封装风格
                texts = [document.text for document in builder._raw_docs]
            else:
                # 兜底方案：如果 API 不同，请在此处适配
                raise NotImplementedError("Please implement the method to extract text chunks from builder.")

            # 1. 对所有 chunk 进行分词
            tokenized_corpus = [tokenize_text(text) for text in texts]

            # 2. 训练 BM25 模型
            bm25_model = BM25Okapi(tokenized_corpus)

            # 3. 将模型和原始文本/映射保存到本地
            sparse_index_path = Path(output_dir) / "bm25_index.pkl"
            sparse_data = {
                "bm25_model": bm25_model,
                "corpus_texts": texts  # 保存文本用于后续检索时返回内容
            }
            with open(sparse_index_path, "wb") as f:
                pickle.dump(sparse_data, f)

            elapsed_sparse = time.perf_counter() - t2
            print(f"[INFO] Sparse BM25 Index built and saved to '{sparse_index_path}' in {elapsed_sparse:.1f}s.")

        except Exception as e:
            print(f"[ERROR] Failed to build sparse index: {e}")

    print(f"[INFO] Total time: {time.perf_counter() - t0:.1f}s.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """Parse arguments, configure logging, and run the async build pipeline."""
    args = _parse_args(argv)
    asyncio.run(build(args))


if __name__ == "__main__":
    main()
