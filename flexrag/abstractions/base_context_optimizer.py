"""
Abstract base class for context optimisation strategies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from flexrag.schema import Document


class BaseContextOptimizer(ABC):
    """Strategy interface for context window optimisation.

    After reranking the context optimiser prunes, summarises, or otherwise
    transforms the selected documents into a compact string that fits within
    the generator's token budget.

    Example subclasses:
        - :class:`flexrag.context_optimizers.LLMContextOptimizer`
    """

    @abstractmethod
    async def optimize(
        self,
        query: str,
        documents: list[Document],
        accumulated_context: list[str],
        max_tokens: int,
    ) -> str:
        """Produce an optimised context string from *documents*.

        Args:
            query: The user's question (used to guide summarisation when
                the optimiser is LLM-based).
            documents: Reranked documents to be distilled.
            max_tokens: Approximate upper bound on the output length in tokens.

        Returns:
            A single string suitable for inclusion in the generator prompt.
        """
