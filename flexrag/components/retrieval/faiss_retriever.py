from __future__ import annotations

import asyncio
import logging
import os
import faiss
import json
from typing import Any

from llama_index.core import Settings, SimpleDirectoryReader
from llama_index.core import Document as LlamaDocument
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.core.readers.base import BaseReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.faiss import FaissVectorStore

from flexrag.components.retrieval import BaseFlexRetriever, OpenAILikeEmbedding
from flexrag.common.schema import Document

logger = logging.getLogger(__name__)

_FAISS_FILE = "faiss_index.bin"
_PROBE_TEXT = "dimension probe"


class _CustomReader(BaseReader):
    """自定义 JSON 读取器，专门处理 [{"idx": 0, "title": "...", "text": "..."}, ...] 格式。"""

    def load_data(self, file: str, extra_info: dict | None = None) -> list[LlamaDocument]:
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            # 如果不是列表，返回空或抛出异常视具体业务而定
            raise ValueError(f"Expected a JSON array in {file}")

        docs = []
        for item in data:
            if not isinstance(item, dict):
                continue

            title = item.get("title", "").strip()
            text = item.get("text") or item.get("context") or ""
            idx = item.get("idx")

            # 拼接 title 和 text 作为文档内容，有助于 Embedding 捕获完整语义
            content_parts = []
            if title:
                content_parts.append(f"Title: {title}")
            if text:
                content_parts.append(text)

            content = "\n\n".join(content_parts)

            if not content:
                continue

            # 将 idx 存入 metadata，这样检索时可以顺带返回原始的 id
            metadata = {}
            if idx is not None:
                metadata["idx"] = idx

            docs.append(LlamaDocument(text=content, metadata=metadata))

        return docs


class FAISSRetriever(BaseFlexRetriever):

    def __init__(
            self,
            embed_base_url: str,
            embed_model_name: str,
            index: VectorStoreIndex | None = None,
            embed_api_key: str | None = None,
            top_k: int | None = 5,
            persist_dir: str | None = None,
            http_client: Any | None = None,
            async_http_client: Any | None = None,
    ) -> None:
        self._top_k = top_k
        if http_client:
            self._embed_model = OpenAILikeEmbedding(model=embed_model_name, base_url=embed_base_url,
                                                    api_key=embed_api_key, http_client=http_client)
        elif async_http_client:
            self._embed_model = OpenAILikeEmbedding(model=embed_model_name, base_url=embed_base_url,
                                                    api_key=embed_api_key, async_http_client=async_http_client)
        else:
            self._embed_model = OpenAILikeEmbedding(model=embed_model_name, base_url=embed_base_url,
                                                    api_key=embed_api_key)
        Settings.embed_model = self._embed_model

        if index is None:
            logger.warning(
                "FAISSRetriever initialized with an empty index. Call `add_documents` before retrieving.")
            self._index: VectorStoreIndex = VectorStoreIndex([])
        else:
            self._index = index

        self._faiss_retriever: BaseRetriever | None = None

        self._knowledge_persist_dir = persist_dir
        self._load_index(persist_dir)

    # ------------------------------------------------------------------
    # Index loading
    # ------------------------------------------------------------------
    def _load_index(self, persist_dir: str) -> None:
        import faiss  # type: ignore[import]
        from llama_index.vector_stores.faiss import FaissVectorStore  # type: ignore[import]

        faiss_path = os.path.join(persist_dir, _FAISS_FILE)
        if not os.path.isfile(faiss_path):
            raise FileNotFoundError(
                f"No FAISS index found at '{faiss_path}'. "
                "Build the knowledge base first with FAISSRetriever."
            )

        faiss_idx = faiss.read_index(faiss_path)
        vector_store = FaissVectorStore(faiss_index=faiss_idx)
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
            persist_dir=persist_dir,
        )
        self._index = load_index_from_storage(storage_context)
        # Invalidate the cached retriever so it is rebuilt with the new index.
        self._faiss_retriever = None
        logger.info(
            "Index loaded from '%s' (%d vector(s))",
            persist_dir,
            faiss_idx.ntotal,
        )

    async def retrieve(self, query: str) -> list[Document]:
        """Retrieve documents relevant to *query*.

        Internally creates (or re-uses) a LlamaIndex
        :class:`~llama_index.common.VectorIndexRetriever` and converts its
        :class:`~llama_index.common.schema.NodeWithScore` results into
        framework-agnostic :class:`~flexrag.common.schema.Document` objects.

        Args:
            query: The user's search string.

        Returns:
            List of :class:`~flexrag.common.schema.Document` sorted by descending score.
        """
        if self._faiss_retriever is None or self._faiss_retriever.similarity_top_k != self._top_k:
            self._faiss_retriever = self._index.as_retriever(similarity_top_k=self._top_k)

        logger.info("Faiss Retrieving top-%d docs for query: %r", self._top_k, query)
        nodes: list[NodeWithScore] = await asyncio.to_thread(self._faiss_retriever.retrieve, query)

        documents: list[Document] = []
        for node in nodes:
            documents.append(
                Document(
                    text=node.get_content(),
                    score=node.score or 0.0,
                    metadata=node.metadata,
                )
            )
        logger.debug("Retrieved %d documents", len(documents))
        return documents

    def _detect_embedding_dim(self) -> int:
        """Return the embedding dimension by encoding a short probe string."""
        probe = self._embed_model.get_text_embedding(_PROBE_TEXT)
        return len(probe)

    async def load_files(self, path: str | list[str]) -> int:
        """Load supported documents from a directory or a list of file paths.
        Args:
            path: Directory path **or** list of individual file paths.

        Returns:
            Number of source documents successfully loaded.

        Raises:
            ValueError: If *path* is an empty list.
            FileNotFoundError: If a directory or file does not exist.
        """
        file_extractor = {".json": _CustomReader()}

        if isinstance(path, list):
            if not path:
                raise ValueError("path must be a non-empty list of file paths.")
            reader = SimpleDirectoryReader(
                input_files=path,
                file_extractor=file_extractor
            )
        else:
            reader = SimpleDirectoryReader(
                input_dir=path,
                file_extractor=file_extractor
            )

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
            VectorStoreIndex, nodes, storage_context=storage_context
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
