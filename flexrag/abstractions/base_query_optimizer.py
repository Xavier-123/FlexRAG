"""
Abstract base class for query optimization strategies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseQueryOptimizer(ABC):
    """Strategy interface for rewriting retrieval queries during iteration."""

    @abstractmethod
    async def optimize_query(
        self,
        original_query: str,
        accumulated_context: list[str],
        missing_info: str,
        iteration_count: int,
        previous_query: str = "",
    ) -> str:
        """Return a retrieval-ready query for the current iteration."""
