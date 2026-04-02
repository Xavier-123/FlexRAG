"""
Custom exception types for FlexRAG.
"""

from __future__ import annotations


class FlexRAGError(Exception):
    """Base exception for all FlexRAG errors."""


class KnowledgeBaseError(FlexRAGError):
    """Raised when a knowledge base operation fails (load, build, save)."""


class RetrievalError(FlexRAGError):
    """Raised when document retrieval fails."""


class RerankerError(FlexRAGError):
    """Raised when the reranker service returns an unexpected response."""


class GenerationError(FlexRAGError):
    """Raised when the LLM generator fails to produce a valid response."""


class ContextOptimizationError(FlexRAGError):
    """Raised when context optimization fails."""


class PipelineError(FlexRAGError):
    """Raised when the RAG pipeline encounters an unrecoverable error."""


__all__ = [
    "FlexRAGError",
    "KnowledgeBaseError",
    "RetrievalError",
    "RerankerError",
    "GenerationError",
    "ContextOptimizationError",
    "PipelineError",
]
