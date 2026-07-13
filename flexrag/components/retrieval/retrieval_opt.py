import logging
from typing import List

from flexrag.common.schema import Document
from flexrag.components.retrieval import BaseFlexRetriever

logger = logging.getLogger(__name__)


class HybridRetriever(BaseFlexRetriever):
    def __init__(
            self,
            retrievers: List[BaseFlexRetriever],
            weights: List[float] = None,
    ) -> None:
        self.retrievers = retrievers
        self.weights = weights or [1.0 / len(retrievers)] * len(retrievers)

    async def retrieve(self, query: str, filters=None) -> list[Document]:
        """Multi-retriever fusion retrieval."""
        logger.debug("Multi-retriever retrieving for query: %r", query)

        all_docs: list[Document] = []

        # 1️⃣ 多路召回 + 加权
        for i, retriever in enumerate(self.retrievers):
            docs = await retriever.retrieve(query, filters=filters)
            weight = self.weights[i]

            for doc in docs:
                doc.score = (doc.score or 0.0) * weight
                all_docs.append(doc)

        logger.debug("Total retrieved docs before dedup: %d", len(all_docs))

        # 2️⃣ 去重（关键！！）
        # 用 text 或唯一ID作为 key（推荐 metadata["id"]）
        dedup: dict[str, Document] = {}

        for doc in all_docs:
            key = doc.metadata.get("id") if doc.metadata else None
            if key is None:
                key = doc.text  # fallback（不推荐但可用）

            if key not in dedup:
                dedup[key] = doc
            else:
                # 保留分数更高的（或做 score 融合）
                if doc.score > dedup[key].score:
                    dedup[key] = doc

        merged_docs = list(dedup.values())

        logger.debug("Docs after dedup: %d", len(merged_docs))

        # 3️⃣ 排序（按 score）
        merged_docs.sort(key=lambda x: x.score, reverse=True)

        # 4️⃣ 截断 top_k（可选）
        if hasattr(self, "_top_k") and self._top_k:
            merged_docs = merged_docs[: self._top_k]

        logger.debug("Final returned docs: %d", len(merged_docs))

        return merged_docs
