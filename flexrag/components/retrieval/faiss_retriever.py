from __future__ import annotations

import asyncio
import logging
import os

from llama_index.core import Settings
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.schema import NodeWithScore

from flexrag.components.retrieval import BaseFlexRetriever, OpenAILikeEmbedding
from flexrag.core.schema import Document

logger = logging.getLogger(__name__)

# File name used to locate the FAISS binary inside persist_dir
_FAISS_FILE = "faiss_index.bin"


class FAISSRetriever(BaseFlexRetriever):

    def __init__(
            self,
            index: VectorStoreIndex | None,
            embed_base_url: str,
            embed_model_name: str,
            embed_api_key: str | None = None,
            top_k: int | None = 5,
            persist_dir: str | None = None,
    ) -> None:
        self._top_k = top_k
        self._embed_model = OpenAILikeEmbedding(model=embed_model_name, base_url=embed_base_url, api_key=embed_api_key)
        Settings.embed_model = self._embed_model  # type: ignore[assignment]

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
                "Build the knowledge base first with FaissKnowledgeBuilder."
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

    # async def _load_index(self, persist_dir: str) -> None:
    #     """Restore a previously saved FAISS index from *persist_dir*.
    #
    #     After this call :meth:`retrieve` is immediately usable.
    #
    #     Args:
    #         persist_dir: Path that was previously passed to
    #             :meth:`~flexrag.indexing.knowledge.FaissKnowledgeBuilder.save`.
    #
    #     Raises:
    #         FileNotFoundError: If the expected FAISS binary is missing.
    #     """
    #     import faiss  # type: ignore[import]
    #     from llama_index.vector_stores.faiss import FaissVectorStore  # type: ignore[import]
    #
    #     faiss_path = os.path.join(persist_dir, _FAISS_FILE)
    #     if not os.path.isfile(faiss_path):
    #         raise FileNotFoundError(
    #             f"No FAISS index found at '{faiss_path}'. "
    #             "Build the knowledge base first with FaissKnowledgeBuilder."
    #         )
    #
    #     faiss_idx = await asyncio.to_thread(faiss.read_index, faiss_path)
    #     vector_store = FaissVectorStore(faiss_index=faiss_idx)
    #     storage_context = StorageContext.from_defaults(
    #         vector_store=vector_store,
    #         persist_dir=persist_dir,
    #     )
    #     self._index = await asyncio.to_thread(load_index_from_storage, storage_context)
    #     # Invalidate the cached retriever so it is rebuilt with the new index.
    #     self._faiss_retriever = None
    #     logger.info(
    #         "Index loaded from '%s' (%d vector(s))",
    #         persist_dir,
    #         faiss_idx.ntotal,
    #     )

    # ------------------------------------------------------------------
    # BaseRetriever interface
    # ------------------------------------------------------------------
    async def retrieve(self, query: str) -> list[Document]:
        """Retrieve documents relevant to *query*.

        Internally creates (or re-uses) a LlamaIndex
        :class:`~llama_index.core.VectorIndexRetriever` and converts its
        :class:`~llama_index.core.schema.NodeWithScore` results into
        framework-agnostic :class:`~flexrag.core.schema.Document` objects.

        Args:
            query: The user's search string.

        Returns:
            List of :class:`~flexrag.core.schema.Document` sorted by descending score.
        """
        if self._faiss_retriever is None or self._faiss_retriever.similarity_top_k != self._top_k:
            self._faiss_retriever = self._index.as_retriever(similarity_top_k=self._top_k)

        logger.debug("Retrieving top-%d docs for query: %r", self._top_k, query)
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
