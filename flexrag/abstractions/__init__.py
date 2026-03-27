"""
Abstract base classes (Strategy Pattern) for every pluggable component in FlexRAG.

Importing from this sub-package exposes all base classes without needing
to know which submodule each lives in.
"""

from flexrag.abstractions.base_context_optimizer import BaseContextOptimizer
from flexrag.abstractions.base_context_evaluator import BaseContextEvaluator
from flexrag.abstractions.base_generator import BaseGenerator
from flexrag.abstractions.base_knowledge import BaseKnowledgeBuilder
from flexrag.abstractions.base_query_optimizer import BaseQueryOptimizer
from flexrag.abstractions.base_reranker import BaseReranker
from flexrag.abstractions.base_retriever import BaseRetriever

__all__ = [
    "BaseRetriever",
    "BaseReranker",
    "BaseContextOptimizer",
    "BaseContextEvaluator",
    "BaseGenerator",
    "BaseQueryOptimizer",
    "BaseKnowledgeBuilder",
]
