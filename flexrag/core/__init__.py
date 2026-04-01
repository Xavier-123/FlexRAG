"""
Core foundation layer: abstractions, schema, config, and exceptions.
"""

from flexrag.core.abstractions import BaseContextEvaluator, BaseGenerator, BaseKnowledgeBuilder
from flexrag.core.config import Settings
from flexrag.core.schema import ContextEvaluation, Document, RAGOutput, RAGState

__all__ = [
    "BaseContextEvaluator",
    "BaseGenerator",
    "BaseKnowledgeBuilder",
    "Settings",
    "Document",
    "RAGState",
    "RAGOutput",
    "ContextEvaluation",
]
