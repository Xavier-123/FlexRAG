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
        openai_api_key: OpenAI API key used by the generator and (optionally)
            the LlamaIndex embedding model.
        openai_model: The OpenAI chat model name (e.g. ``"gpt-4o"``).
        vllm_base_url: Base URL of the vLLM server that exposes the
            embedding and reranker endpoints.
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

    openai_api_key: str = Field(..., description="OpenAI API key")
    openai_model: str = Field("gpt-4o", description="OpenAI chat model")

    vllm_base_url: str = Field(
        "http://localhost:8000",
        description="Base URL of the vLLM serving endpoint",
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
