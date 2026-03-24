"""
LlamaIndex-based retriever implementation.

This module shows how to embed a LlamaIndex retriever inside a LangGraph node
while keeping the two frameworks loosely coupled through the
:class:`~flexrag.abstractions.BaseRetriever` strategy interface.

Architecture overview::

    ┌─────────────────────────────────────────────┐
    │  LangGraph Node  (flexrag/graph/nodes.py)   │
    │                                             │
    │  state.retrieved_docs = retriever.retrieve( │
    │      state.query, cfg.top_k_retrieval       │
    │  )                                          │
    └───────────────┬─────────────────────────────┘
                    │ calls
    ┌───────────────▼─────────────────────────────┐
    │  LlamaIndexRetriever  (this file)            │
    │                                             │
    │  Wraps a llama_index VectorStoreIndex +     │
    │  a vLLM-backed embedding model.             │
    └─────────────────────────────────────────────┘
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional
from pydantic import PrivateAttr

from llama_index.core import Settings as LlamaSettings
from llama_index.core import VectorStoreIndex
from llama_index.core.base.base_retriever import BaseRetriever as LlamaBaseRetriever
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.embeddings import BaseEmbedding

from flexrag.abstractions.base_retriever import BaseRetriever
from flexrag.schema import Document

logger = logging.getLogger(__name__)


class VLLMEmbedding(BaseEmbedding):
    """Minimal OpenAI-compatible embedding wrapper for a vLLM embedding endpoint.

    Now properly inherits from LlamaIndex's BaseEmbedding to pass type validation.
    """

    # Pydantic 字段定义
    base_url: str
    model_name: str  # LlamaIndex 规范中通常使用 model_name
    api_key: Optional[str] = None

    # 非序列化的内部私有变量，需要用 PrivateAttr 声明
    _client: Any = PrivateAttr()
    _endpoint: str = PrivateAttr()

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        http_client: Any | None = None,
        **kwargs: Any,
    ) -> None:
        # 1. 初始化父类 (Pydantic BaseModel)
        super().__init__(
            base_url=base_url,
            model_name=model,
            api_key=api_key,
            **kwargs
        )

        # 2. 初始化内部变量
        import httpx
        # self._endpoint = base_url.rstrip("/") + "/v1/embeddings"
        self._endpoint = base_url
        self._model = model
        self._api_key = api_key
        self._client = http_client or httpx.Client(timeout=60.0)

    # ------------------------------------------------------------------
    # 同步方法 (Synchronous Methods)
    # ------------------------------------------------------------------
    def _get_query_embedding(self, query: str) -> List[float]:
        """为检索的问题生成单条向量"""
        return self._get_text_embeddings([query])[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        """为单本文档生成向量"""
        return self._get_text_embeddings([text])[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """批量生成文档向量 (重写此方法以利用 vLLM 的批量推理加速)"""
        payload = {"model": self.model_name, "input": texts}
        headers: dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = self._client.post(self._endpoint, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()["data"]

        # Sort by index to preserve order (OpenAI spec)
        data.sort(key=lambda item: item["index"])
        return [item["embedding"] for item in data]

    # ------------------------------------------------------------------
    # 异步方法 (Asynchronous Methods - 解决当前报错)
    # ------------------------------------------------------------------

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """异步获取查询问题的向量"""
        res = await self._aget_text_embeddings([query])
        return res[0]

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """异步获取单本文档的向量"""
        res = await self._aget_text_embeddings([text])
        return res[0]

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """异步批量获取文档向量"""
        payload = {"model": self.model_name, "input": texts}
        headers: dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = await self._aclient.post(self._endpoint, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()["data"]
        data.sort(key=lambda item: item["index"])
        return [item["embedding"] for item in data]


class LlamaIndexRetriever(BaseRetriever):
    """Retriever that delegates to a LlamaIndex :class:`VectorStoreIndex`.

    This class bridges LlamaIndex's retrieval ecosystem with FlexRAG's
    strategy interface so that the LangGraph node code stays clean and
    framework-agnostic.

    Args:
        index: A pre-built :class:`~llama_index.core.VectorStoreIndex`.  Pass
            ``None`` to create an empty in-memory index (useful for testing).
        embed_base_url: Base URL of the vLLM embedding endpoint.
        embed_model_name: Name of the embedding model served at that URL.
        embed_api_key: Optional API key for the vLLM embedding endpoint.

    Example::

        from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

        documents = SimpleDirectoryReader("./data").load_data()
        index = VectorStoreIndex.from_documents(documents)
        retriever = LlamaIndexRetriever(
            index=index,
            embed_base_url="http://localhost:8000",
            embed_model_name="BAAI/bge-large-en-v1.5",
            embed_api_key="my-secret-key",
        )
        docs = retriever.retrieve("What is RAG?", top_k=5)
    """

    def __init__(
        self,
        index: VectorStoreIndex | None,
        embed_base_url: str,
        embed_model_name: str,
        embed_api_key: str | None = None,
    ) -> None:
        self._embed_model = VLLMEmbedding(
            base_url=embed_base_url,
            model=embed_model_name,
            api_key=embed_api_key,
        )
        # Inject our custom embedding into the global LlamaIndex settings so
        # that the VectorStoreIndex uses it for both ingestion and querying.
        LlamaSettings.embed_model = self._embed_model  # type: ignore[assignment]

        if index is None:
            logger.warning(
                "LlamaIndexRetriever initialized with an empty index. "
                "Call `add_documents` before retrieving."
            )
            self._index: VectorStoreIndex = VectorStoreIndex([])
        else:
            self._index = index

        self._llama_retriever: LlamaBaseRetriever | None = None

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def add_documents(self, texts: list[str], metadatas: list[dict] | None = None) -> None:
        """Index a list of raw text strings.

        Useful when you want to build the index programmatically rather than
        passing a pre-built :class:`VectorStoreIndex`.

        Args:
            texts: Raw text chunks to add to the index.
            metadatas: Optional metadata dicts (one per chunk).
        """
        metadatas = metadatas or [{}] * len(texts)
        nodes = [
            NodeWithScore(node=TextNode(text=t, metadata=m), score=1.0)
            for t, m in zip(texts, metadatas)
        ]
        self._index.insert_nodes([n.node for n in nodes])
        # Invalidate the cached retriever so it is rebuilt with the new nodes.
        self._llama_retriever = None

    # ------------------------------------------------------------------
    # BaseRetriever interface
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int) -> list[Document]:
        """Retrieve *top_k* documents relevant to *query*.

        Internally creates (or re-uses) a LlamaIndex
        :class:`~llama_index.core.VectorIndexRetriever` and converts its
        :class:`~llama_index.core.schema.NodeWithScore` results into
        framework-agnostic :class:`~flexrag.schema.Document` objects.

        Args:
            query: The user's search string.
            top_k: Number of documents to return.

        Returns:
            List of :class:`~flexrag.schema.Document` sorted by descending score.
        """
        if self._llama_retriever is None or self._llama_retriever.similarity_top_k != top_k:
            self._llama_retriever = self._index.as_retriever(similarity_top_k=top_k)

        logger.debug("Retrieving top-%d docs for query: %r", top_k, query)
        nodes: list[NodeWithScore] = self._llama_retriever.retrieve(query)

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
