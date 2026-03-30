"""
vLLM-backed reranker implementation.

Calls the ``/v1/rerank`` endpoint exposed by a vLLM server that serves a
cross-encoder reranker model (e.g. ``BAAI/bge-reranker-v2-m3``).

The endpoint follows the Cohere-compatible rerank API:

.. code-block:: json

    POST /v1/rerank
    {
        "model": "BAAI/bge-reranker-v2-m3",
        "query": "What is RAG?",
        "documents": ["doc text 1", "doc text 2", ...]
    }

    Response:
    {
        "results": [
            {"index": 0, "relevance_score": 0.95},
            ...
        ]
    }
"""

from __future__ import annotations

import logging
import httpx
from typing import Any

from flexrag.core.abstractions import BaseReranker
from flexrag.core.schema import Document

logger = logging.getLogger(__name__)


class VLLMReranker(BaseReranker):
    """Reranker that calls a vLLM cross-encoder endpoint.

    Sends all candidate documents to the vLLM rerank API in a single batch
    request and returns the top *top_k* documents sorted by descending
    relevance score.

    Args:
        base_url: Base URL of the vLLM server.
        model: Name of the reranker model deployed on the server.
        api_key: Optional API key sent as ``Authorization: Bearer <api_key>``
            for servers that require authentication.
        http_client: Optional pre-built :class:`httpx.AsyncClient` (useful for
            testing / dependency injection).

    Example::

        reranker = VLLMReranker(
            base_url="http://localhost:8000",
            model="BAAI/bge-reranker-v2-m3",
            api_key="my-secret-key",
        )
        top_docs = await reranker.rerank(query="What is RAG?", documents=docs, top_k=3)
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        top_k: int | None = 5,
        http_client: Any | None = None,
    ) -> None:
        # self._endpoint = base_url.rstrip("/") + "/v1/rerank"
        self._endpoint = base_url
        self._model = model
        self._api_key = api_key
        self._top_k = top_k
        self._client: httpx.AsyncClient = http_client or httpx.AsyncClient(timeout=60.0)

    # ------------------------------------------------------------------
    # BaseReranker interface
    # ------------------------------------------------------------------

    async def rerank(
        self,
        query: str,
        documents: list[Document],
    ) -> list[Document]:
        """Rerank *documents* against *query* via the vLLM rerank endpoint.

        Args:
            query: The user's question.
            documents: Candidate documents to score.

        Returns:
            Up to *top_k* :class:`~flexrag.core.schema.Document` objects sorted by
            descending relevance score.

        Raises:
            httpx.HTTPStatusError: If the vLLM server returns a non-2xx status.
        """
        if not documents:
            return []

        texts = [doc.text for doc in documents]
        payload: dict[str, Any] = {
            "model": self._model,
            "query": query,
            "documents": texts,
        }

        headers: dict[str, str] = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        logger.info(
            "Sending %d documents to reranker endpoint %s",
            len(texts),
            self._endpoint,
        )
        response = await self._client.post(self._endpoint, json=payload, headers=headers)
        response.raise_for_status()

        results: list[dict[str, Any]] = response.json()["results"]

        # Merge rerank scores back into Document objects
        reranked: list[Document] = []
        for item in results:
            idx: int = item["index"]
            score: float = float(item["relevance_score"])
            doc = documents[idx]
            reranked.append(
                Document(text=doc.text, score=score, metadata=doc.metadata)
            )

        # Sort descending by score and truncate to top_k
        reranked.sort(key=lambda d: d.score, reverse=True)
        selected = reranked[:self._top_k]
        logger.info("Reranked: kept %d / %d documents", len(selected), len(documents))
        return selected
