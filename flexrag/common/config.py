"""
Configuration module for FlexRAG.

All settings are loaded from environment variables (or a .env file) and
validated via Pydantic Settings.
"""

from __future__ import annotations

from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings loaded from the environment.

    Each of the three model types (LLM, embedding, reranker) has its own
    ``base_url`` and ``api_key`` so they can be served from different
    endpoints or authenticated independently.

    Attributes:
        llm_base_url: Base URL of the LLM serving endpoint.
        llm_api_key: API key for the LLM endpoint.
        embedding_base_url: Base URL of the embedding model serving endpoint.
        embedding_api_key: API key for the embedding endpoint.
        reranker_base_url: Base URL of the reranker model serving endpoint.
        reranker_api_key: API key for the reranker endpoint.
        llm_model: Name / path of the chat LLM model served by vLLM.
        embedding_model: Name / path of the embedding model served by vLLM.
        reranker_model: Name / path of the reranker model served by vLLM.
        top_k_retrieval: Number of documents to retrieve before reranking.
        top_k_rerank: Number of top documents to keep after reranking.
        context_max_tokens: Approximate token budget for the optimised context
            window passed to the generator.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM 相关 ---
    llm_model: str = Field(
        "Qwen/Qwen2.5-7B-Instruct",
        validation_alias="LLM_MODEL",
        description="Chat LLM model name served by vLLM",
    )
    llm_base_url: str = Field(
        default="http://localhost:8018/v1",
        validation_alias="LLM_BASE_URL",
        description="Base URL of the LLM serving endpoint",
    )
    llm_api_key: str = Field(
        default="sk-xxxx",
        validation_alias="LLM_API_KEY",
        description="API key for the LLM endpoint",
    )
    context_max_tokens: int = Field(
        8000,
        validation_alias="CONTEXT_MAX_TOKENS",
        description="Token budget for the optimised context",
    )

    # --- Embedding 相关 ---
    embedding_model: str = Field(
        "BAAI/bge-large-en-v1.5",
        validation_alias="EMBEDDING_MODEL",
        description="Embedding model name served by vLLM",
    )
    embedding_base_url: str = Field(
        default="http://localhost:8018/v1",
        validation_alias="EMBEDDING_BASE_URL",
        description="Base URL of the embedding model serving endpoint",
    )
    embedding_api_key: str = Field(
        default="sk-xxxx",
        validation_alias="EMBEDDING_API_KEY",
        description="API key for the embedding endpoint",
    )

    # --- 知识库相关 ---
    knowledge_persist_dir: str = Field(
        default="./data/knowledge_persist_dir",
        validation_alias="KNOWLEDGE_PERSIST_DIR",
        description="Directory where the FAISS knowledge base is stored",
    )
    knowledge_chunk_size: int = Field(
        default=512,
        validation_alias="KNOWLEDGE_CHUNK_SIZE",
        description="Maximum token count per document chunk",
    )
    knowledge_chunk_overlap: int = Field(
        default=50,
        validation_alias="KNOWLEDGE_CHUNK_OVERLAP",
        description="Token overlap between consecutive document chunks",
    )

    # --- Reranker ---
    reranker_model: str = Field(
        "BAAI/bge-reranker-v2-m3",
        validation_alias="RERANKER_MODEL",
        description="Reranker model name served by vLLM",
    )
    reranker_base_url: str = Field(
        default="http://localhost:8018/v1",
        validation_alias="RERANKER_BASE_URL",
        description="Base URL of the reranker model serving endpoint",
    )
    reranker_api_key: str = Field(
        default="sk-xxxx",
        validation_alias="RERANKER_API_KEY",
        description="API key for the reranker endpoint",
    )
    top_k_rerank: int = Field(
        5,
        validation_alias="TOP_K_RERANK",
        description="Number of docs kept after reranking"
    )

    # --- 执行控制与文件 IO ---
    max_iterations: int = Field(
        3,
        validation_alias="MAX_ITERATIONS",
        description="Maximum Agentic RAG iterations",
    )
    log_level: str = Field(
        "INFO",
        validation_alias="LOG_LEVEL",
        description="Global logging level (DEBUG/INFO/WARNING/ERROR/CRITICAL)",
    )
    log_format: str = Field(
        "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        validation_alias="LOG_FORMAT",
        description="Global logging format string",
    )
    max_concurrent_tasks: int = Field(
        default=5,
        validation_alias="MAX_CONCURRENT_TASKS",
        description="最大并发任务数",
    )
    input_file: Optional[str] = Field(
        default=None,
        validation_alias="INPUT_FILE",
        description="输入的 QA 数据 JSON 文件路径",
    )
    output_file: str = Field(
        default="./eval_results.json",
        validation_alias="OUTPUT_FILE",
        description="输出的测试结果文件路径",
    )

    # --- Pipeline component feature flags ---
    # Pre-retrieval query optimizers
    use_query_rewriter: bool = Field(
        True,
        validation_alias="USE_QUERY_REWRITER",
        description="Whether to enable QueryRewriter in the pre-retrieval optimizer",
    )
    use_query_expander: bool = Field(
        True,
        validation_alias="USE_QUERY_EXPANDER",
        description="Whether to enable QueryExpander in the pre-retrieval optimizer",
    )
    use_task_splitter: bool = Field(
        True,
        validation_alias="USE_TASK_SPLITTER",
        description="Whether to enable TaskSplitter in the pre-retrieval optimizer",
    )
    use_terminology_enricher: bool = Field(
        False,
        validation_alias="USE_TERMINOLOGY_ENRICHER",
        description="Whether to enable TerminologyEnricher in the pre-retrieval optimizer",
    )

    # Retrievers
    top_k_retrieval: int = Field(
        10,
        validation_alias="TOP_K_RETRIEVAL",
        description="检索阶段 (Retrieval) 召回的 Top-K 文档数量"
    )
    # 稠密检索
    vector_store_type: str = Field(
        "faiss",
        validation_alias="VECTOR_STORE_TYPE",
        description="向量检索使用的存储类型(faiss|milvus|chroma)"
    )
    dense_mode: str = Field(
        "exact_l2",
        validation_alias="DENSE_MODE",
        description="exact_l2 | exact_cosine | approx_l2 | approx_cosine，计算距离的方式和索引类型"
    )
    use_multi_vector_retriever: bool = Field(
        True,
        validation_alias="USE_MULTI_VECTOR_RETRIEVER",
        description="Whether to enable MultiVectorRetriever (dense/vector search)",
    )
    # 稀疏检索
    use_bm25_retriever: bool = Field(
        True,
        validation_alias="USE_BM25_RETRIEVER",
        description="Whether to enable BM25Retriever (sparse/keyword search)",
    )
    # 图检索
    use_graph_retriever: bool = Field(
        False,
        validation_alias="USE_GRAPH_RETRIEVER",
        description="Whether to enable GraphRetriever (knowledge-graph search)",
    )

    # Post-retrieval processors
    use_reranker: bool = Field(
        True,
        validation_alias="USE_RERANKER",
        description="Whether to enable OpenAILikeReranker in the post-retrieval optimizer",
    )
    use_llm_context_optimizer: bool = Field(
        True,
        validation_alias="USE_LLM_CONTEXT_OPTIMIZER",
        description="Whether to enable LLMContextOptimizer in the post-retrieval optimizer",
    )
    use_copy_paste_retrieval: bool = Field(
        False,
        validation_alias="USE_COPY_PASTE_RETRIEVAL",
        description="Whether to enable CopyPasteRetrieval in the post-retrieval optimizer",
    )

    # --- Graph architecture diagram ---
    draw_image_path: Optional[str] = Field(
        default="./AgenticRAG-Architecture.png",
        validation_alias="DRAW_IMAGE_PATH",
        description="If set, saves the LangGraph architecture diagram (PNG) to this path",
    )

    # --- Tracing & Persistence ---
    checkpoint_db_path: Optional[str] = Field(
        default=None,
        validation_alias="CHECKPOINT_DB_PATH",
        description=(
            "Path to the SQLite database used for LangGraph checkpoint persistence "
            "(e.g. './data/checkpoints.db'). When None, checkpointing is disabled."
        ),
    )


# Rebuild the model so that forward-references in Optional[str] fields
# (introduced by ``from __future__ import annotations``) are resolved.
Settings.model_rebuild()
settings = Settings()
