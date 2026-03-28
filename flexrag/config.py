"""
Configuration module for FlexRAG.

All settings are loaded from environment variables (or a .env file) and
validated via Pydantic Settings.
"""

from __future__ import annotations

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
        vllm_llm_model: Name / path of the chat LLM model served by vLLM.
        vllm_embedding_model: Name / path of the embedding model served by vLLM.
        vllm_reranker_model: Name / path of the reranker model served by vLLM.
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

    # --- LLM ---
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

    # --- Embedding ---
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

    # --- Reranker ---
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

    # --- Model names ---
    vllm_llm_model: str = Field(
        "Qwen/Qwen2.5-7B-Instruct",
        validation_alias="VLLM_LLM_MODEL",
        description="Chat LLM model name served by vLLM",
    )
    vllm_embedding_model: str = Field(
        "BAAI/bge-large-en-v1.5",
        validation_alias="VLLM_EMBEDDING_MODEL",
        description="Embedding model name served by vLLM",
    )
    vllm_reranker_model: str = Field(
        "BAAI/bge-reranker-v2-m3",
        validation_alias="VLLM_RERANKER_MODEL",
        description="Reranker model name served by vLLM",
    )

    top_k_retrieval: int = Field(
        10,
        validation_alias="TOP_K_RETRIEVAL",
        description="Number of docs retrieved initially"
    )
    top_k_rerank: int = Field(
        5,
        validation_alias="TOP_K_RERANK",
        description="Number of docs kept after reranking"
    )
    context_max_tokens: int = Field(
        3000,
        validation_alias="CONTEXT_MAX_TOKENS",
        description="Token budget for the optimised context",
    )
    max_iterations: int = Field(
        3,
        validation_alias="MAX_ITERATIONS",
        description="Maximum Agentic RAG reflection/retrieval iterations",
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

    # --- Knowledge base ---
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
