"""
Abstract base class for answer generation strategies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from flexrag.schema import RAGOutput


class BaseGenerator(ABC):
    """Strategy interface for answer generation.

    Concrete implementations call an LLM (e.g. GPT-4o via OpenAI's Structured
    Output API) and return a validated :class:`~flexrag.schema.RAGOutput`.

    Example subclasses:
        - :class:`flexrag.generators.OpenAIGenerator`
    """

    @abstractmethod
    async def generate(
        self,
        query: str,
        context: str,
        source_documents: list[str],
    ) -> RAGOutput:
        """Generate a structured answer grounded in *context*.

        Args:
            query: The user's original question.
            context: Optimised context string produced by the context
                optimisation node.
            source_documents: Raw text of the source chunks used to build
                *context*; these are surfaced as evidence in the output.

        Returns:
            A :class:`~flexrag.schema.RAGOutput` with ``answer`` and
            ``evidence`` fields populated.
        """
