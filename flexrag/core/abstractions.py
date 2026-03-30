"""
Abstract base classes (Strategy Pattern) for every pluggable component in FlexRAG.

All base classes are defined in this single module for easy discovery.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from flexrag.core.schema import ContextEvaluation, Document, GenOutput, RAGOutput


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


class BaseRetriever(ABC):
    """Strategy interface for document retrieval.

    All concrete retriever implementations must subclass this ABC and
    implement :meth:`retrieve`.  This decouples the LangGraph node logic
    from any specific vector store or retrieval library.

    Example subclasses:
        - :class:`flexrag.components.retrieval.LlamaIndexRetriever`
    """

    @abstractmethod
    async def retrieve(self, query: str) -> list[Document]:
        """Retrieve the most relevant documents for *query*.

        Args:
            query: The user's question or search string.

        Returns:
            A list of :class:`~flexrag.core.schema.Document` objects sorted by
            descending relevance score.
        """


# ---------------------------------------------------------------------------
# Reranking
# ---------------------------------------------------------------------------


class BaseReranker(ABC):
    """Strategy interface for document reranking.

    Concrete implementations call a reranker model (e.g. a cross-encoder
    served via vLLM) and return a re-scored, truncated list of documents.

    Example subclasses:
        - :class:`flexrag.components.post_retrieval.VLLMReranker`
    """

    @abstractmethod
    async def rerank(
        self,
        query: str,
        documents: list[Document],
    ) -> list[Document]:
        """Rerank *documents* with respect to *query* and return top *top_k*.

        Args:
            query: The user's question used as the reranking reference.
            documents: Candidate documents retrieved in the previous step.

        Returns:
            A list of at most *top_k* :class:`~flexrag.core.schema.Document`
            objects sorted by descending rerank score.
        """


# ---------------------------------------------------------------------------
# Post-retrieval: context optimisation
# ---------------------------------------------------------------------------


class BaseContextOptimizer(ABC):
    """Strategy interface for context window optimisation.

    After reranking the context optimiser prunes, summarises, or otherwise
    transforms the selected documents into a compact string that fits within
    the generator's token budget.

    Example subclasses:
        - :class:`flexrag.components.post_retrieval.LLMContextOptimizer`
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
            accumulated_context: Context collected in previous iterations.
            max_tokens: Approximate upper bound on the output length in tokens.

        Returns:
            A single string suitable for inclusion in the generator prompt.
        """


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


# ---------------------------------------------------------------------------
# Query transformation
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


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
    :class:`~flexrag.core.abstractions.BaseRetriever` implementation (e.g.
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
    "BaseRetriever",
    "BaseReranker",
    "BaseContextOptimizer",
    "BaseContextEvaluator",
    "BaseQueryOptimizer",
    "BaseGenerator",
    "BaseKnowledgeBuilder",
]
