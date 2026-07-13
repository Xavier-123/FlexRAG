"""
Core foundation layer: abstractions, schema, config, and exceptions.
"""

from flexrag.common.config import Settings
from flexrag.common.schema import (
    ContextEvaluation,
    Document,
    PostRetrievalResult,
    RAGOutput,
    RAGState,
    ScoreDetails,
)
from flexrag.common.logging import setup_logging

__all__ = [
    "Settings",
    "Document",
    "ScoreDetails",
    "PostRetrievalResult",
    "RAGState",
    "RAGOutput",
    "ContextEvaluation",
    "setup_logging",
]
