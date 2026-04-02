from __future__ import annotations

import logging
import httpx
from typing import Any

from flexrag.common.schema import Document
from flexrag.components.post_retrieval.base import BasePostRetrieval

logger = logging.getLogger(__name__)


class OpenAILikeReranker(BasePostRetrieval):
    def __init__(
            self,
            base_url: str,
            model: str,
            api_key: str | None = None,
            top_k: int | None = 5,
            http_client: Any | None = None,
    ) -> None:
        base_url = base_url.rstrip("/")
        if base_url.endswith("/v1/rerank"):
            self._endpoint = base_url
        elif base_url.endswith("/v1"):
            self._endpoint = base_url + "/rerank"
        else:
            self._endpoint = base_url + "/v1/rerank"
        self._model = model
        self._api_key = api_key
        self._top_k = top_k
        self._client: httpx.AsyncClient = http_client or httpx.AsyncClient(timeout=60.0)

    async def optimize(
            self,
            query: str,
            documents: list[Document],
            accumulated_context: list[str],
            max_tokens: int,
    ) -> list[Document]:
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
