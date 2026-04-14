#!/usr/bin/env python
"""
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
import numpy as np
from pathlib import Path
from langchain_openai import ChatOpenAI
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter

# 引入分词和BM25库
try:
    import jieba
except ImportError:
    print("[WARNING] Missing sparse retrieval dependencies. Run: pip install jieba")

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from flexrag.components.retrieval import OpenAILikeEmbedding, MultiVectorRetriever, _CustomReader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 稀疏检索辅助函数
# ---------------------------------------------------------------------------
def tokenize_text(text: str) -> list[str]:
    """对文本进行分词。针对中文使用 jieba，英文按空格切分。"""
    # 过滤掉空格和空字符
    return [word for word in jieba.lcut(text) if word.strip()]


def build_dense_index(
        builder,
        embed_model,
        output_dir: str,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        vector_db: str = "faiss",  # "faiss" 或 "milvus"
        metric_type: str = "l2",  # "l2" (欧氏距离) 或 "cosine" (余弦相似度)
        milvus_uri: str = "./milvus_demo.db",  # 仅当 vector_db="milvus" 时生效(Milvus Lite本地文件)
        milvus_token: str = ""  # 服务端认证 Token 或 "user:password"
):
    """
    通用构建稠密向量索引函数，支持 FAISS 和 Milvus。
    包含 Exact (精确) 和 Approximate (近似/ANN) 两种索引。
    支持 L2 距离和 Cosine 余弦相似度。
    """
    if vector_db not in ["faiss", "milvus"]:
        raise ValueError("vector_db 必须是 'faiss' 或 'milvus'")
    if metric_type not in ["l2", "cosine"]:
        raise ValueError("metric_type 必须是 'l2' 或 'cosine'")

    # 1. 统一文本切分
    print(f"[INFO] Splitting documents (chunk_size={chunk_size}, chunk_overlap={chunk_overlap})...")
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = splitter.get_nodes_from_documents(builder._raw_docs, show_progress=True)
    print(f"[INFO] Splitted into {len(nodes)} nodes.")

    # 2. 预计算 Embeddings (核心优化：避免重复调用API扣费)
    print("[INFO] Pre-computing embeddings...")
    t_emb = time.perf_counter()
    _PROBE_TEXT = "dimension probe"
    embed_dim = len(embed_model.get_text_embedding(_PROBE_TEXT))
    texts_to_embed = [node.get_content(metadata_mode="all") for node in nodes]
    # embeddings = embed_model.get_text_embedding_batch(texts_to_embed)

    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm
    batch_size = 8
    batches = [texts_to_embed[i:i + batch_size] for i in range(0, len(texts_to_embed), batch_size)]
    embeddings = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(tqdm(executor.map(embed_model.get_text_embedding_batch, batches), total=len(batches)))
    for r in results:
        embeddings.extend(r)

    # ====== 新增核心逻辑：如果是 Cosine 相似度，需对向量进行 L2 归一化 ======
    if metric_type == "cosine":
        print("[INFO] Normalizing embeddings for Cosine Similarity...")
        emb_arr = np.array(embeddings, dtype=np.float32)
        norms = np.linalg.norm(emb_arr, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10  # 防止除以 0
        emb_arr = emb_arr / norms
        embeddings = emb_arr.tolist()
    # =================================================================

    # 将获取到的向量强行绑定到 nodes 上
    for node, emb in zip(nodes, embeddings):
        node.embedding = emb
    print(f"[INFO] Embeddings pre-calculated. Dimension: {embed_dim}, Time: {time.perf_counter() - t_emb:.1f}s.")

    # 3. 循环构建 Exact 和 Approximate 索引
    index_types = {
        "exact": {"milvus_type": "FLAT"},
        "approx": {"milvus_type": "HNSW"}
    }
    # index_types = {
    #     "exact": {"faiss_type": "FlatL2", "milvus_type": "FLAT"},
    #     "approx": {"faiss_type": "HNSWFlat", "milvus_type": "HNSW"}
    # }

    for idx_mode, config in index_types.items():
        print(f"\n[INFO] Building {vector_db.upper()} [{idx_mode.capitalize()}] index...")
        t_build = time.perf_counter()

        persist_dir = os.path.join(output_dir, f"{vector_db}_{idx_mode}_{metric_type}")
        os.makedirs(persist_dir, exist_ok=True)

        # ====== 根据类型初始化 Vector Store ======
        if vector_db == "faiss":
            import faiss
            from llama_index.vector_stores.faiss import FaissVectorStore

            # 修改点：根据度量选择不同的 FAISS Index
            if metric_type == "cosine":
                if idx_mode == "exact":
                    # Cosine = 内积 (Inner Product)，前提是向量已归一化
                    faiss_index = faiss.IndexFlatIP(embed_dim)
                else:
                    faiss_index = faiss.IndexHNSWFlat(embed_dim, 32, faiss.METRIC_INNER_PRODUCT)
            else:  # L2 距离
                if idx_mode == "exact":
                    faiss_index = faiss.IndexFlatL2(embed_dim)
                else:
                    faiss_index = faiss.IndexHNSWFlat(embed_dim, 32, faiss.METRIC_L2)

            vector_store = FaissVectorStore(faiss_index=faiss_index)

        elif vector_db == "milvus":
            from llama_index.vector_stores.milvus import MilvusVectorStore

            # Milvus 中的集合名称，用来区分 exact 和 approx
            collection_name = f"dense_{idx_mode}"

            # 配置 Milvus 索引类型
            milvus_metric = "COSINE" if metric_type == "cosine" else "L2"
            index_config = {
                "index_type": config["milvus_type"],
                "metric_type": milvus_metric
            }

            vector_store = MilvusVectorStore(
                uri=milvus_uri,
                token=milvus_token,  # 传入认证信息
                collection_name=collection_name,
                dim=embed_dim,
                overwrite=True,  # 如果重新运行，覆盖旧数据
                index_config=index_config
            )

        # ====== 构建并持久化 Index ======
        ctx = StorageContext.from_defaults(vector_store=vector_store)

        # 传入已带有 embedding 的 nodes，LlamaIndex 会自动跳过调用 embedding 模型的步骤
        index = VectorStoreIndex(nodes, storage_context=ctx)

        # 保存 Metadata 和 LlamaIndex 相关配置
        index.storage_context.persist(persist_dir=persist_dir)

        print(f"[INFO] {idx_mode.capitalize()} index saved to '{persist_dir}' in {time.perf_counter() - t_build:.1f}s.")
        if vector_db == "milvus":
            print(f"[INFO] Milvus collection '{collection_name}' stored in '{milvus_uri}'.")

    print("\n[INFO] All indexes built successfully!")


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
    source.add_argument("--input-dir", type=str,
                        help="Path to a directory of documents (.txt, .md, .pdf).")
    source.add_argument("--files", nargs="+", type=str, help="One or more individual file paths to index.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory where the FAISS index will be saved. Defaults to the KNOWLEDGE_PERSIST_DIR setting.")
    parser.add_argument("--chunk-size", type=int, default=None,
                        help="Max tokens per chunk (default: from KNOWLEDGE_CHUNK_SIZE setting).")
    parser.add_argument("--chunk-overlap", type=int, default=None,
                        help="Token overlap between chunks (default: from KNOWLEDGE_CHUNK_OVERLAP setting).")

    parser.add_argument("--embedding-base-url", type=str, default="http://127.0.0.1:8018/v1/embeddings",
                        help="Base URL for the embedding API.")
    parser.add_argument("--embedding-api-key", type=str, default="sk-xxxx", help="API key for the embedding service.")
    parser.add_argument("--embedding-model", type=str, default="Qwen3-Embedding-0.6B",
                        help="Name of the embedding model to use.")

    parser.add_argument("--force", action="store_true", help="Overwrite an existing index at the output directory.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG-level logging.")

    # 是否开启稠密检索
    parser.add_argument("--enable-dense", action="store_true", help="Build a FAISS dense index.")
    parser.add_argument("--vector-store-type", type=str, default="faiss", help="faiss or milvus")
    parser.add_argument("--metric-type", type=str, default="l2", help="l2 or cosine")
    parser.add_argument("--milvus-uri", type=str, default="http://localhost:19530", help="milvus url")
    parser.add_argument("--milvus-token", type=str, default="", help="milvus token")

    # 允许用户选择是否开启稀疏检索
    parser.add_argument("--enable-sparse", action="store_true",
                        help="Build a BM25 sparse index alongside the FAISS dense index.")

    # 允许用户选择是否开启图检索
    parser.add_argument("--enable-graph", action="store_true",
                        help="Build a Graph index.")
    parser.add_argument("--top-k-graph", type=int, default=2,
                        help="检索最相关的节点/边数量 (仅在 --enable-graph 时生效)")
    parser.add_argument("--llm-model", type=str, default="Qwen/Qwen3.5-35B-A3B")
    parser.add_argument("--llm-base-url", type=str, default="https://api-inference.modelscope.cn/v1")
    parser.add_argument("--llm-api-key", type=str, default=None)
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
    if (MultiVectorRetriever.index_exists(output_dir) or MultiVectorRetriever.index_exists(output_dir)) and not args.force:
        print(
            f"[ERROR] An index already exists at '{output_dir}'.\n"
            "       Use --force to overwrite it.",
            file=sys.stderr,
        )
        sys.exit(1)


    # ---- builder ----
    llm = ChatOpenAI(
        model=args.llm_model,
        api_key=args.llm_api_key,
        base_url=args.llm_base_url,
        temperature=0.0,
    )

    embed_model = OpenAILikeEmbedding(
        model_name=args.embedding_model,
        base_url=args.embedding_base_url,
        api_key=args.embedding_api_key
    )

    # builder = FAISSRetriever(
    builder = MultiVectorRetriever(
        embed_model=embed_model,
        vector_store_type=args.vector_store_type,
    )

    # ---- load ----
    source = args.input_dir if args.input_dir else args.files
    reader = SimpleDirectoryReader(
        input_dir=source,
        file_extractor={".json": _CustomReader(chunk_size=chunk_size, chunk_overlap=chunk_overlap)}
    )

    t0 = time.perf_counter()
    doc_count = await builder.load_files(reader)
    elapsed_load = time.perf_counter() - t0
    print(f"[INFO] Loaded {doc_count} document(s) in {elapsed_load:.1f}s.")

    # ---- build dense index ----
    if args.enable_dense:
        build_dense_index(
            builder=builder,
            embed_model=embed_model,
            output_dir=output_dir,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            vector_db=args.vector_store_type,  # "faiss" 或 "milvus"
            metric_type=args.metric_type,  # "cosine" 或 "l2"
            milvus_uri=args.milvus_uri,
            milvus_token=args.milvus_token,
        )

    # ---- build & save sparse index (BM25) ----
    if args.enable_sparse:
        print("[INFO] Building Sparse BM25 index ...")
        t2 = time.perf_counter()

        try:
            from llama_index.retrievers.bm25 import BM25Retriever
            # 将文档切分为 Nodes (节点)
            # BM25Retriever 需要输入 Nodes
            splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
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
            llm=llm,
            embed_model=embed_model,
            top_k=args.top_k_graph,
            persist_dir=graph_persist_dir,
        )
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
