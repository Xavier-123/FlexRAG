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

    Attributes:
        vllm_base_url: Base URL of the vLLM server that exposes all model
            endpoints (LLM, embedding, and reranker).
        vllm_api_key: API key used to authenticate against the vLLM server.
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

    vllm_base_url: str = Field(
        "http://localhost:8000",
        description="Base URL of the vLLM serving endpoint",
    )
    vllm_api_key: str = Field(..., description="API key for the vLLM server")
    vllm_llm_model: str = Field(
        "Qwen/Qwen2.5-7B-Instruct",
        description="Chat LLM model name served by vLLM",
    )
    vllm_embedding_model: str = Field(
        "BAAI/bge-large-en-v1.5",
        description="Embedding model name served by vLLM",
    )
    vllm_reranker_model: str = Field(
        "BAAI/bge-reranker-v2-m3",
        description="Reranker model name served by vLLM",
    )

    top_k_retrieval: int = Field(10, description="Number of docs retrieved initially")
    top_k_rerank: int = Field(5, description="Number of docs kept after reranking")
    context_max_tokens: int = Field(
        3000,
        description="Token budget for the optimised context",
    )
