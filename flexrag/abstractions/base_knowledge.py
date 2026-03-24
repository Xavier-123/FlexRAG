"""
Abstract base class for the knowledge builder pipeline.

Defines the strategy interface for file loading, chunking, vector indexing,
and persistence so that concrete backends (FAISS, Chroma, Weaviate …) can be
swapped without touching the rest of the codebase.

Retrieval is handled by :class:`~flexrag.abstractions.BaseRetriever`
implementations, keeping the *build* and *query* concerns separate.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseKnowledgeBuilder(ABC):
    """Strategy interface for building and persisting a knowledge base.

    A knowledge builder handles the construction lifecycle of a document
    corpus:

    1. **Load** – ingest files from a local directory or a list of paths.
    2. **Index** – chunk the raw documents and embed them into a vector store.
    3. **Persist** – save the index to disk so it survives restarts.

    Retrieval from a persisted index is the responsibility of a
    :class:`~flexrag.abstractions.BaseRetriever` implementation (e.g.
    :class:`~flexrag.retrievers.LlamaIndexRetriever`).

    Concrete subclasses must implement every abstract method.
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

        Must be called after :meth:`load_files` and before :meth:`save`.

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

    @classmethod
    @abstractmethod
    def index_exists(cls, persist_dir: str) -> bool:
        """Return ``True`` if a valid persisted index exists at *persist_dir*.

        This is a lightweight check (e.g. file-existence) that does **not**
        load the full index into memory.

        Args:
            persist_dir: Directory to inspect.
        """
