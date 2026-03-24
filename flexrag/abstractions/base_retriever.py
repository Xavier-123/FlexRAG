"""
Abstract base class for retrieval strategies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from flexrag.schema import Document


class BaseRetriever(ABC):
    """Strategy interface for document retrieval.

    All concrete retriever implementations must subclass this ABC and
    implement :meth:`retrieve`.  This decouples the LangGraph node logic
    from any specific vector store or retrieval library.

    Example subclasses:
        - :class:`flexrag.retrievers.LlamaIndexRetriever`
    """

    @abstractmethod
    async def retrieve(self, query: str, top_k: int) -> list[Document]:
        """Retrieve the most relevant documents for *query*.

        Args:
            query: The user's question or search string.
            top_k: Maximum number of documents to return.

        Returns:
            A list of :class:`~flexrag.schema.Document` objects sorted by
            descending relevance score.
        """
