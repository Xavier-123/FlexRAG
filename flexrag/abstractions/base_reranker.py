"""
Abstract base class for reranking strategies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from flexrag.schema import Document


class BaseReranker(ABC):
    """Strategy interface for document reranking.

    Concrete implementations call a reranker model (e.g. a cross-encoder
    served via vLLM) and return a re-scored, truncated list of documents.

    Example subclasses:
        - :class:`flexrag.rerankers.VLLMReranker`
    """

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int,
    ) -> list[Document]:
        """Rerank *documents* with respect to *query* and return top *top_k*.

        Args:
            query: The user's question used as the reranking reference.
            documents: Candidate documents retrieved in the previous step.
            top_k: Number of documents to keep after reranking.

        Returns:
            A list of at most *top_k* :class:`~flexrag.schema.Document` objects
            sorted by descending rerank score.
        """
