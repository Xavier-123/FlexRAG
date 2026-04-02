"""
Abstract base classes (Strategy Pattern) for every pluggable component in FlexRAG.

All base classes are defined in this single module for easy discovery.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from flexrag.core.schema import ContextEvaluation, GenOutput


# ---------------------------------------------------------------------------
# Judges (online context evaluation)
# ---------------------------------------------------------------------------
class BaseContextEvaluator(ABC):
    """Strategy interface for deciding if context can answer the query."""

    @abstractmethod
    async def evaluate(
            self,
            original_query: str,
            optimized_context: str,
            accumulated_context: list[str],
    ) -> ContextEvaluation:
        """Evaluate if current context is sufficient for final answer generation."""


class BaseGenerator(ABC):
    """Strategy interface for answer generation.

    Concrete implementations call an LLM (e.g. GPT-4o via OpenAI's Structured
    Output API) and return a validated :class:`~flexrag.core.schema.RAGOutput`.

    Example subclasses:
        - :class:`flexrag.components.generation.OpenAIGenerator`
    """

    @abstractmethod
    async def generate(
            self,
            query: str,
            context: str,
            accumulated_context: list[str],
            source_documents: list[str],
    ) -> GenOutput:
        """Generate a structured answer grounded in *context*.

        Args:
            query: The user's original question.
            context: Optimised context string produced by the context
                optimisation node.
            accumulated_context: Context collected in previous iterations.
            source_documents: Raw text of the source chunks used to build
                *context*; these are surfaced as evidence in the output.

        Returns:
            A :class:`~flexrag.core.schema.RAGOutput` with ``answer`` and
            ``evidence`` fields populated.
        """


# ---------------------------------------------------------------------------
# Knowledge building (indexing)
# ---------------------------------------------------------------------------

class BaseKnowledgeBuilder(ABC):
    """Strategy interface for building and persisting a knowledge base.

    A knowledge builder handles the construction lifecycle of a document
    corpus:

    1. **Load** – ingest files from a local directory or a list of paths.
    2. **Index** – chunk the raw documents and embed them into a vector store.
    3. **Persist** – save the index to disk so it survives restarts.

    Retrieval from a persisted index is the responsibility of a
    :class:`~flexrag.components.retrieval.BaseRetriever` implementation (e.g.
    :class:`~flexrag.components.retrieval.LlamaIndexRetriever`).
    """

    @abstractmethod
    async def load_files(self, path: str | list[str]) -> int:
        """Load documents from a directory path or an explicit list of file paths.

        Args:
            path: A directory path (``str``) **or** a list of individual file
                paths.

        Returns:
            The number of source documents loaded.
        """

    @abstractmethod
    async def build_index(
            self,
            chunk_size: int = 512,
            chunk_overlap: int = 50,
    ) -> None:
        """Chunk the loaded documents and build the vector index.

        Args:
            chunk_size: Maximum number of tokens per chunk.
            chunk_overlap: Number of tokens shared between consecutive chunks.
        """

    @abstractmethod
    async def save(self, persist_dir: str) -> None:
        """Persist the current vector index to *persist_dir* on disk.

        Args:
            persist_dir: Absolute or relative path of the storage directory.
        """

    @classmethod
    @abstractmethod
    def index_exists(cls, persist_dir: str) -> bool:
        """Return ``True`` if a valid persisted index exists at *persist_dir*.

        Args:
            persist_dir: Directory to inspect.
        """


__all__ = [
    "BaseContextEvaluator",
    "BaseGenerator",
    "BaseKnowledgeBuilder",
]
