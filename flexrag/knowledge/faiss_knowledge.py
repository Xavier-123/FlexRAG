"""
FAISS-backed knowledge builder implementation.

Implements the knowledge base *construction* pipeline using:

* **LlamaIndex** ``SimpleDirectoryReader`` for file loading (.txt / .md / .pdf)
* **LlamaIndex** ``SentenceSplitter`` for configurable text chunking
* **LlamaIndex** ``VectorStoreIndex`` + ``FaissVectorStore`` for vector storage
* The project's own ``VLLMEmbedding`` for embedding generation

This module is responsible only for **building and persisting** a FAISS index.
Retrieval from a persisted index is handled by
:class:`~flexrag.retrievers.LlamaIndexRetriever`.

Usage example::

    from flexrag.knowledge import FaissKnowledgeBuilder

    builder = FaissKnowledgeBuilder(
        embed_base_url="http://localhost:8001/v1/embeddings",
        embed_model_name="BAAI/bge-large-en-v1.5",
        embed_api_key="sk-...",
    )

    # Build from a directory of documents
    count = await builder.load_files("./my_docs")
    await builder.build_index(chunk_size=512, chunk_overlap=50)
    await builder.save("./knowledge_base")
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import faiss  # type: ignore[import]
from llama_index.core import Settings as LlamaSettings
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.faiss import FaissVectorStore  # type: ignore[import]

from flexrag.abstractions.base_knowledge import BaseKnowledgeBuilder
from flexrag.retrievers.llamaindex_retriever import VLLMEmbedding

logger = logging.getLogger(__name__)

# File name used to persist the FAISS binary inside persist_dir
_FAISS_FILE = "faiss_index.bin"
# Short probe text used to detect the embedding dimension at index-build time
_PROBE_TEXT = "dimension probe"


class FaissKnowledgeBuilder(BaseKnowledgeBuilder):
    """Knowledge builder that stores vectors in a local FAISS index.

    This class bridges the LlamaIndex ecosystem with FlexRAG's
    :class:`~flexrag.abstractions.BaseKnowledgeBuilder` strategy interface.

    It is responsible only for *building and persisting* the index.
    For retrieval, use :class:`~flexrag.retrievers.LlamaIndexRetriever`
    with :meth:`~flexrag.retrievers.LlamaIndexRetriever.load_index`.

    Args:
        embed_base_url: Base URL of the vLLM embedding endpoint
            (e.g. ``"http://localhost:8001/v1/embeddings"``).
        embed_model_name: Name of the embedding model served at that URL.
        embed_api_key: Optional bearer-token API key for the endpoint.
        http_client: Optional pre-built ``httpx.Client`` (injected in tests).
    """

    def __init__(
        self,
        embed_base_url: str,
        embed_model_name: str,
        embed_api_key: str | None = None,
        http_client: Any | None = None,
    ) -> None:
        self._embed_model = VLLMEmbedding(
            base_url=embed_base_url,
            model=embed_model_name,
            api_key=embed_api_key,
            http_client=http_client,
        )
        # Make the custom embedding model the global LlamaIndex default so
        # that VectorStoreIndex uses it for both ingestion and querying.
        LlamaSettings.embed_model = self._embed_model  # type: ignore[assignment]

        # Raw LlamaIndex Document objects populated by load_files()
        self._raw_docs: list[Any] = []

        # Built by build_index()
        self._index: VectorStoreIndex | None = None
        self._vector_store: Any | None = None  # FaissVectorStore

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _detect_embedding_dim(self) -> int:
        """Return the embedding dimension by encoding a short probe string."""
        probe = self._embed_model.get_text_embedding(_PROBE_TEXT)
        return len(probe)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def load_files(self, path: str | list[str]) -> int:
        """Load supported documents from a directory or a list of file paths.

        Uses LlamaIndex :class:`~llama_index.core.SimpleDirectoryReader` which
        handles ``.txt``, ``.md``, and ``.pdf`` files out of the box.

        Args:
            path: Directory path **or** list of individual file paths.

        Returns:
            Number of source documents successfully loaded.

        Raises:
            ValueError: If *path* is an empty list.
            FileNotFoundError: If a directory or file does not exist.
        """
        if isinstance(path, list):
            if not path:
                raise ValueError("path must be a non-empty list of file paths.")
            reader = SimpleDirectoryReader(input_files=path)
        else:
            reader = SimpleDirectoryReader(input_dir=path)

        self._raw_docs = await asyncio.to_thread(reader.load_data)
        logger.info("Loaded %d document(s) from %s", len(self._raw_docs), path)
        return len(self._raw_docs)

    async def build_index(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ) -> None:
        """Chunk all loaded documents and build a FAISS vector index.

        Calls the embedding model once with a probe string to detect the
        embedding dimension, then creates a ``faiss.IndexFlatL2`` of that
        dimension.

        Args:
            chunk_size: Maximum token count per chunk (default 512).
            chunk_overlap: Token overlap between consecutive chunks (default 50).

        Raises:
            RuntimeError: If no documents have been loaded yet.
        """
        if not self._raw_docs:
            raise RuntimeError(
                "No documents loaded. Call load_files() first."
            )

        # -- Chunk --
        splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        nodes = splitter.get_nodes_from_documents(self._raw_docs)
        logger.info(
            "Chunked %d document(s) into %d node(s) "
            "(chunk_size=%d, chunk_overlap=%d)",
            len(self._raw_docs),
            len(nodes),
            chunk_size,
            chunk_overlap,
        )

        # -- Detect embedding dimension --
        embed_dim = await asyncio.to_thread(self._detect_embedding_dim)
        logger.debug("Detected embedding dimension: %d", embed_dim)

        # -- Build FAISS index --
        faiss_idx = faiss.IndexFlatL2(embed_dim)
        self._vector_store = FaissVectorStore(faiss_index=faiss_idx)
        storage_context = StorageContext.from_defaults(
            vector_store=self._vector_store
        )
        self._index = await asyncio.to_thread(
            lambda: VectorStoreIndex(nodes, storage_context=storage_context)
        )
        logger.info("FAISS index built with %d vector(s)", faiss_idx.ntotal)

    async def save(self, persist_dir: str) -> None:
        """Persist the FAISS index and document store to *persist_dir*.

        Creates the directory (and any missing parents) if it does not exist.

        Args:
            persist_dir: Target directory for all index artefacts.

        Raises:
            RuntimeError: If the index has not been built yet.
        """
        if self._index is None or self._vector_store is None:
            raise RuntimeError(
                "Index is not built.  Call build_index() before save()."
            )

        os.makedirs(persist_dir, exist_ok=True)

        # Save FAISS binary
        faiss_path = os.path.join(persist_dir, _FAISS_FILE)
        await asyncio.to_thread(self._vector_store.persist, persist_path=faiss_path)

        # Save docstore / index_store metadata
        await asyncio.to_thread(
            self._index.storage_context.persist, persist_dir=persist_dir
        )

        logger.info("Knowledge base saved to '%s'", persist_dir)

    @classmethod
    def index_exists(cls, persist_dir: str) -> bool:
        """Return ``True`` if a FAISS binary exists at *persist_dir*.

        This is a cheap file-existence check; it does not validate the index.

        Args:
            persist_dir: Directory to inspect.
        """
        return os.path.isfile(os.path.join(persist_dir, _FAISS_FILE))
