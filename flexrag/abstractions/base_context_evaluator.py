"""
Abstract base class for context evaluation strategies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from flexrag.schema import ContextEvaluation


class BaseContextEvaluator(ABC):
    """Strategy interface for deciding if context can answer the query."""

    @abstractmethod
    async def evaluate(
        self,
        original_query: str,
        optimized_context: str,
    ) -> ContextEvaluation:
        """Evaluate if current context is sufficient for final answer generation."""
