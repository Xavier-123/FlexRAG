"""
Core foundation layer: abstractions, schema, config, and exceptions.
"""

from flexrag.common.config import Settings
from flexrag.common.schema import ContextEvaluation, Document, RAGOutput, RAGState
from flexrag.common.logging import setup_logging

__all__ = [
    "Settings",
    "Document",
    "RAGState",
    "RAGOutput",
    "ContextEvaluation",
    "setup_logging",
]
