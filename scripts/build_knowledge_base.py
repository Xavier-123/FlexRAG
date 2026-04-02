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
import os
import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

# 引入分词和BM25库
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
    parser.add_argument("--enable-sparse", action="store_true",
                        help="Build a BM25 sparse index alongside the FAISS dense index.")
    # 允许用户选择是否开启稀疏检索
    parser.add_argument("--enable-graph", action="store_true",
                        help="Build a BM25 sparse index alongside the FAISS dense index.")
    parser.add_argument("--top-k-graph", type=int, default=2,
                        help="检索最相关的节点/边数量 (仅在 --enable-graph 时生效)")
    parser.add_argument("--llm-model", type=str, default="Qwen/Qwen3.5-35B-A3B")
    parser.add_argument("--llm-base-url", type=str, default="https://api-inference.modelscope.cn/v1")
    parser.add_argument("--llm-api-key", type=str, default="ms-c429b084-79ba-4a00-a749-aae8681e902d")
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

        try:
            from llama_index.core.node_parser import SentenceSplitter
            from llama_index.retrievers.bm25 import BM25Retriever
            # 将文档切分为 Nodes (节点)
            # BM25Retriever 需要输入 Nodes
            splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=50)
            nodes = splitter.get_nodes_from_documents(builder._raw_docs)
            # 初始化 BM25Retriever
            retriever = BM25Retriever.from_defaults(
                nodes=nodes,
                similarity_top_k=2,  # 设置返回的 Top-K 数量
                tokenizer=tokenize_text  # 传入自定义的中文分词器
            )
            # 持久化到本地目录
            sparse_index_path = os.path.join(output_dir, "bm25_index")
            os.makedirs(sparse_index_path, exist_ok=True)
            retriever.persist(sparse_index_path)
            print(f"BM25 索引已成功保存至 {Path(output_dir) / 'bm25_index'}\n")
            elapsed_sparse = time.perf_counter() - t2
            print(f"[INFO] Sparse BM25 Index built and saved to '{sparse_index_path}' in {elapsed_sparse:.1f}s.")

        except Exception as e:
            print(f"[ERROR] Failed to build sparse index: {e}")

    # ---- build & save graph index ----
    if args.enable_graph:
        from flexrag.components.retrieval import GraphRetriever
        from flexrag.components.retrieval.neo4j_graph_retriever import Neo4jGraphRetriever
        graph_persist_dir = os.path.join(output_dir, "graph_index")
        if not os.path.exists(graph_persist_dir):
            os.makedirs(graph_persist_dir)

        graph_retriever = GraphRetriever(
            # graph_retriever = Neo4jGraphRetriever(
            llm_model_name=args.llm_model,
            llm_base_url=args.llm_base_url,
            llm_api_key=args.llm_api_key,
            embed_model_name=args.embedding_model,
            embed_base_url=args.embedding_base_url,
            embed_api_key=args.embedding_api_key,
            top_k=args.top_k_graph,
            persist_dir=graph_persist_dir,
        )
        print("正在抽取实体和关系，构建图谱 (这需要调用 LLM，请稍候)...")
        t3 = time.perf_counter()
        await graph_retriever.build_graph(builder._raw_docs[:5])  # 直接使用原始文档构建图谱，GraphRetriever 内部会处理切分和嵌入
        elapsed_graph = time.perf_counter() - t3
        print(f"[INFO] Graph Index built and saved in {elapsed_graph:.1f}s.")

    print(f"[INFO] Total time: {time.perf_counter() - t0:.1f}s.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """Parse arguments, configure logging, and run the async build pipeline."""
    import nest_asyncio
    nest_asyncio.apply()

    args = _parse_args(argv)
    asyncio.run(build(args))


if __name__ == "__main__":
    main()
