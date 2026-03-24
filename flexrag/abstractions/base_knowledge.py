"""
Abstract base class for the knowledge base pipeline.

Defines the strategy interface for file loading, chunking, vector indexing,
and retrieval so that concrete backends (FAISS, Chroma, Weaviate …) can be
swapped without touching the rest of the codebase.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from flexrag.schema import Document


class BaseKnowledgeBase(ABC):
    """Strategy interface for a self-contained knowledge base pipeline.

    A knowledge base handles the full lifecycle of a document corpus:

    1. **Load** – ingest files from a local directory or a list of paths.
    2. **Index** – chunk the raw documents and embed them into a vector store.
    3. **Persist** – save the index to disk so it survives restarts.
    4. **Restore** – reload a previously saved index from disk.
    5. **Retrieve** – answer vector-similarity queries against the index.

    Concrete subclasses must implement every abstract method.  Optional
    helper methods (e.g. :meth:`add_documents`) provide a programmatic way
    to insert raw text without going through file loading.
    """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    def load_files(self, path: str | list[str]) -> int:
        """Load documents from a directory path or an explicit list of file paths.

        Supported file formats (at minimum): ``.txt``, ``.md``, ``.pdf``.

        Args:
            path: A directory path (``str``) **or** a list of individual file
                paths.  When a directory is given every supported file inside
                it is loaded recursively.

        Returns:
            The number of source documents loaded.
        """

    @abstractmethod
    def build_index(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ) -> None:
        """Chunk the loaded documents and build the vector index.

        Must be called after :meth:`load_files` (or :meth:`add_documents`)
        and before :meth:`save` / :meth:`retrieve`.

        Args:
            chunk_size: Maximum number of tokens per chunk.
            chunk_overlap: Number of tokens shared between consecutive chunks.
        """

    @abstractmethod
    def save(self, persist_dir: str) -> None:
        """Persist the current vector index to *persist_dir* on disk.

        The directory is created if it does not exist.  Calling this method
        a second time overwrites the previous snapshot.

        Args:
            persist_dir: Absolute or relative path of the storage directory.
        """

    @abstractmethod
    def load(self, persist_dir: str) -> None:
        """Restore a previously saved vector index from *persist_dir*.

        After this call :meth:`retrieve` is immediately usable.

        Args:
            persist_dir: Path that was previously passed to :meth:`save`.

        Raises:
            FileNotFoundError: If *persist_dir* does not contain a valid index.
        """

    @classmethod
    @abstractmethod
    def index_exists(cls, persist_dir: str) -> bool:
        """Return ``True`` if a valid persisted index exists at *persist_dir*.

        This is a lightweight check (e.g. file-existence) that does **not**
        load the full index into memory.

        Args:
            persist_dir: Directory to inspect.
        """

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        """Return the *top_k* most relevant document chunks for *query*.

        Args:
            query: The user's search string.
            top_k: Maximum number of results to return.

        Returns:
            List of :class:`~flexrag.schema.Document` objects sorted by
            descending relevance score.
        """

    # ------------------------------------------------------------------
    # Optional: programmatic document insertion
    # ------------------------------------------------------------------

    def add_documents(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """Insert raw text chunks directly into the index.

        This is an *optional* convenience method for programmatic ingestion
        without going through :meth:`load_files`.  Implementations are not
        required to support it; the default raises :class:`NotImplementedError`.

        Args:
            texts: Raw text strings to add.
            metadatas: Optional metadata dicts aligned with *texts*.

        Raises:
            NotImplementedError: If the backend does not support online insertion.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support add_documents(). "
            "Use load_files() + build_index() instead."
        )
