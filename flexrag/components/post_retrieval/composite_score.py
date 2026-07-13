"""Final ranking based on relevance, recency, and business importance."""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from flexrag.common.metadata import parse_importance, parse_utc_timestamp
from flexrag.common.schema import Document, ScoreDetails
from flexrag.components.post_retrieval.base import BasePostRetrieval

logger = logging.getLogger(__name__)


class CompositeScoreReranker(BasePostRetrieval):
    """Apply a weighted final score and return the highest-ranked documents."""

    def __init__(
        self,
        alpha: float = 0.7,
        beta: float = 0.2,
        gamma: float = 0.1,
        half_life_days: float = 30.0,
        top_k: int | None = 5,
        timestamp_key: str = "timestamp",
        importance_key: str = "importance_score",
        neutral_score: float = 0.5,
        now_provider: Callable[[], datetime] | None = None,
    ) -> None:
        weights = (alpha, beta, gamma)
        if any(weight < 0.0 for weight in weights):
            raise ValueError("Composite score weights must be non-negative")
        if abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError("Composite score weights must sum to 1.0")
        if half_life_days <= 0.0:
            raise ValueError("half_life_days must be greater than zero")
        if not 0.0 <= neutral_score <= 1.0:
            raise ValueError("neutral_score must be in [0, 1]")

        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        self._half_life_days = half_life_days
        self._top_k = top_k
        self._timestamp_key = timestamp_key
        self._importance_key = importance_key
        self._neutral_score = neutral_score
        self._now_provider = now_provider or (lambda: datetime.now(timezone.utc))

    @classmethod
    def from_settings(cls, settings: Any) -> "CompositeScoreReranker":
        """Build the scorer from the shared application settings object."""
        return cls(
            alpha=settings.score_alpha,
            beta=settings.score_beta,
            gamma=settings.score_gamma,
            half_life_days=settings.recency_half_life_days,
            top_k=settings.top_k_rerank,
            timestamp_key=settings.timestamp_metadata_key,
            importance_key=settings.importance_metadata_key,
        )

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(1.0, value))

    def _relevance_scores(self, documents: list[Document]) -> list[float]:
        if documents and all(
            doc.score_details is not None for doc in documents
        ):
            return [self._clamp(doc.score_details.relevance) for doc in documents]  # type: ignore[union-attr]

        raw_scores = [float(doc.score or 0.0) for doc in documents]
        if not raw_scores:
            return []
        low, high = min(raw_scores), max(raw_scores)
        if high == low:
            return [self._neutral_score] * len(raw_scores)
        return [(score - low) / (high - low) for score in raw_scores]

    def _recency_score(self, value: object, now: datetime) -> float | None:
        timestamp = parse_utc_timestamp(value)
        if timestamp is None:
            return None
        age_days = max(0.0, (now - timestamp).total_seconds() / 86400.0)
        return 2.0 ** (-age_days / self._half_life_days)

    async def optimize(
        self,
        query: str,
        documents: list[Document],
        accumulated_context: list[str],
        max_tokens: int,
    ) -> list[Document]:
        if not documents:
            return []

        now = self._now_provider()
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("now_provider must return a timezone-aware datetime")
        now = now.astimezone(timezone.utc)

        relevance_scores = self._relevance_scores(documents)
        missing_timestamp = 0
        missing_importance = 0
        ranked: list[Document] = []

        for doc, relevance in zip(documents, relevance_scores):
            recency = self._recency_score(doc.metadata.get(self._timestamp_key), now)
            if recency is None:
                missing_timestamp += 1
                recency = self._neutral_score

            importance = parse_importance(doc.metadata.get(self._importance_key))
            if importance is None:
                missing_importance += 1
                importance = self._neutral_score

            final_score = self._clamp(
                self._alpha * relevance
                + self._beta * recency
                + self._gamma * importance
            )
            ranked.append(
                doc.model_copy(
                    update={
                        "score": final_score,
                        "score_details": ScoreDetails(
                            relevance=relevance,
                            recency=recency,
                            importance=importance,
                            final_score=final_score,
                        ),
                    }
                )
            )

        if missing_timestamp or missing_importance:
            logger.warning(
                "Composite scoring used neutral metadata values: timestamp=%d, importance=%d, documents=%d",
                missing_timestamp,
                missing_importance,
                len(documents),
            )

        ranked.sort(key=lambda document: document.score, reverse=True)
        return ranked if self._top_k is None else ranked[: self._top_k]
