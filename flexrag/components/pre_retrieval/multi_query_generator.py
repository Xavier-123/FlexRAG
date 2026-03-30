"""
Multi-query generator that converts the query optimizer output into a list
of search-ready query strings.

Parsing rules per query type:
- simple      / vague / professional → wrap the single optimized query in a list.
- complex     → split the multi-line decomposition output into individual
                sub-questions and return them as separate queries.
"""

from __future__ import annotations

import logging
import re

from flexrag.core.abstractions import BaseMultiQueryGenerator

logger = logging.getLogger(__name__)


class LLMMultiQueryGenerator(BaseMultiQueryGenerator):
    """Parse the query optimizer output into a list of retrieval queries.

    For *complex* queries the optimizer produces several sub-questions
    (one per line).  This generator parses and normalises them.  For all
    other query types the optimized query is returned as-is in a list.
    """

    async def generate_queries(
        self,
        original_query: str,
        optimized_query: str,
        query_type: str,
    ) -> list[str]:
        """Return a list of search queries derived from *optimized_query*.

        Args:
            original_query: The user's original question used as fallback.
            optimized_query: Output from the query optimizer (may be multi-line
                for decomposed sub-questions).
            query_type: Classification label from the query router.

        Returns:
            A non-empty list of query strings for the retriever.
        """
        if query_type == "complex":
            queries = self._parse_sub_questions(optimized_query)
            if queries:
                logger.debug(
                    "MultiQueryGenerator produced %d sub-questions for complex query.",
                    len(queries),
                )
                return queries

        # simple / vague / professional: use the (single) optimized query
        query = optimized_query.strip()
        result = [query] if query else [original_query]
        logger.debug("MultiQueryGenerator produced 1 query (type=%s).", query_type)
        return result

    @staticmethod
    def _parse_sub_questions(text: str) -> list[str]:
        """Split a multi-line decomposition output into individual queries."""
        lines = re.split(r"\n+", text.strip())
        cleaned: list[str] = []
        for line in lines:
            # Strip leading numbering / bullets: "1. ", "1) ", "- ", "• ", "（1）"
            # Use a pattern that only matches list markers, not numeric content.
            line = re.sub(r"^[\d]+[\.\)、）]\s+|^[\-•]\s+", "", line).strip()
            if len(line) > 2:
                cleaned.append(line)
        return cleaned
