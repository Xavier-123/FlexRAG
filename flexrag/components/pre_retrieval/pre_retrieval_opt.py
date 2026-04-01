import asyncio
import logging
import re
from typing import List
from .base import BaseQueryOptimizer

logger = logging.getLogger(__name__)


class PreQueryOptimizer(BaseQueryOptimizer):
    def __init__(self, optimizers: List[BaseQueryOptimizer]):
        self.optimizers = optimizers

    async def run(
            self,
            original_query: str,
            accumulated_context: list[str],
            missing_info: str,
            previous_query=None,
            previous_queries=None,
    ) -> tuple[list[str], dict]:

        if previous_queries is None:
            previous_queries = {}
        tasks = [
            optimizer.run(
                original_query=original_query,
                accumulated_context=accumulated_context,
                missing_info=missing_info,
                previous_query=previous_queries.get(optimizer.type, ""),
            )
            for optimizer in self.optimizers
        ]
        results = await asyncio.gather(*tasks)

        # 2. Parse outputs, aggregate, and deduplicate
        all_queries: list[str] = [original_query]
        current_queries: dict = {
            "original_query": original_query,
        }
        seen: set[str] = {original_query}

        for optimized_result in results:
            current_queries[optimized_result["type"]] = optimized_result["optimized_query"]
            queries = self.parse_optimized_query(original_query=optimized_result["original_query"], optimized_query=optimized_result["optimized_query"], type=optimized_result["type"])
            for q in queries:
                if q and q not in seen:
                    seen.add(q)
                    all_queries.append(q)

        return all_queries, current_queries

    def parse_optimized_query(
            self,
            original_query: str,
            optimized_query: str,
            type: str,
    ) -> list[str]:
        """Return a list of search queries derived from *optimized_query*."""
        if type == "split":
            queries = self._parse_sub_questions(optimized_query)
            if queries:
                logger.debug(
                    "Parsed %d sub-questions for query.",
                    len(queries),
                )
                return queries

        query = optimized_query.strip()
        result = [query] if query else [original_query]
        logger.debug("Parsed 1 query (type=%s).", type)
        return result

    @staticmethod
    def _parse_sub_questions(text: str) -> list[str]:
        """Split a multi-line decomposition output into individual queries."""
        lines = re.split(r"\n+", text.strip())
        cleaned: list[str] = []
        for line in lines:
            # Strip leading numbering / bullets: "1. ", "1) ", "- ", "• ", "（1）"
            line = re.sub(r"^[\d]+[\.\)、）]\s+|^[\-•]\s+", "", line).strip()
            if len(line) > 2:
                cleaned.append(line)
        return cleaned