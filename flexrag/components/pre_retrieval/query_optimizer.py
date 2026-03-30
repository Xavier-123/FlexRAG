"""
LLM-based query optimizer for iterative retrieval.

Supports four optimisation strategies selected by query type:
- simple      → rewrite: rephrase for clearer retrieval.
- vague       → expansion: generate a hypothetical answer/document (HyDE).
- complex     → decomposition: break into sub-questions (one per line).
- professional → term enhancement: augment with domain terminology.
"""

from __future__ import annotations

import re
import logging
from typing import Tuple

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from flexrag.core.abstractions import BaseQueryOptimizer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Strategy prompts
# ---------------------------------------------------------------------------

_SYSTEM_REWRITE = (
    "你是一个检索查询优化器，负责将原始问题重写为更清晰、更适合向量检索的查询文本。"
    "保持与原问题语义一致，不要回答问题本身。"
    "若提供了缺失信息，优先补足缺失点。"
    "输出仅包含一行优化后的检索查询，不要附加解释。"
)

_SYSTEM_EXPANSION = (
    "你是一个假设性文档生成器（HyDE）。"
    "根据用户的模糊问题，生成一段简短的假设性回答或文档片段，"
    "该片段将用于向量检索以找到相关真实文档。"
    "直接输出假设性文档内容，不要附加解释或标注。"
)

_SYSTEM_DECOMPOSITION = (
    "你是一个问题分解器，负责将复杂问题拆解为若干个独立的子问题。"
    "每个子问题单独占一行，无需编号或前缀，不要回答问题，不要附加解释。"
    "子问题数量控制在 2-4 个。"
)

_SYSTEM_TERM_ENHANCEMENT = (
    "你是一个领域术语增强器，负责在原始问题中补充相关专业术语和同义词，"
    "使检索能覆盖更多专业文档。"
    "输出仅包含一行增强后的检索查询，不要附加解释。"
)

_STRATEGY_PROMPTS: dict[str, str] = {
    "simple": _SYSTEM_REWRITE,
    "vague": _SYSTEM_EXPANSION,
    "complex": _SYSTEM_DECOMPOSITION,
    "professional": _SYSTEM_TERM_ENHANCEMENT,
}


class LLMQueryOptimizer(BaseQueryOptimizer):
    """Optimize retrieval query using a strategy matched to the query type."""

    def __init__(self, llm: BaseChatModel) -> None:
        self._llm = llm

    async def optimize_query(
        self,
        original_query: str,
        missing_info: str,
        accumulated_context: list[str],
        iteration_count: int,
        previous_query: str = "",
        query_type: str = "simple",
    ) -> str:
        system_prompt = _STRATEGY_PROMPTS.get(query_type, _SYSTEM_REWRITE)

        human_prompt = (
            f"原始问题: {original_query}\n"
            f"当前迭代: {iteration_count}\n"
            # f"上轮检索查询: {previous_query or '无'}\n"
            f"已有信息: {accumulated_context or '无'}\n\n"
            f"缺失信息: {missing_info or '无'}\n\n"
            "请根据策略输出优化后的检索查询："
        )
        prompt_string = f"【System】:\n{system_prompt}\n\n【Human】:\n{human_prompt}"

        try:
            response = await self._llm.ainvoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
            )
            content = str(response.content).strip()  # type: ignore[union-attr]
            logger.debug("Query optimization (%s) response: %s", query_type, content)

            # For single-query strategies, normalise to a single line.
            if query_type != "complex":
                content = " ".join(content.splitlines()).strip()
            return content
        except Exception as exc:  # noqa: BLE001
            logger.warning("Query optimization failed (%s); fallback to original query.", exc)
            return original_query

    def parse_optimized_query(
            self,
            original_query: str,
            optimized_query: str,
            query_type: str,
    ) -> list[str]:
        """Return a list of search queries derived from *optimized_query*."""
        if query_type == "complex":
            queries = self._parse_sub_questions(optimized_query)
            if queries:
                logger.debug(
                    "Parsed %d sub-questions for complex query.",
                    len(queries),
                )
                return queries

        # simple / vague / professional: use the (single) optimized query
        query = optimized_query.strip()
        result = [query] if query else [original_query]
        logger.debug("Parsed 1 query (type=%s).", query_type)
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