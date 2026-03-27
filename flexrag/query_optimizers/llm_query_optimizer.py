"""
LLM-based query optimizer for iterative retrieval.
"""

from __future__ import annotations

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from flexrag.abstractions.base_query_optimizer import BaseQueryOptimizer

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT_ZH = (
    "你是一个检索查询优化器，只负责生成用于检索的查询文本。"
    "保持与原问题语义一致，不要回答问题本身。"
    "若提供了缺失信息，优先补足缺失点。"
    "输出仅包含一行优化后的检索查询，不要附加解释。"
)


class LLMQueryOptimizer(BaseQueryOptimizer):
    """Optimize retrieval query from original question and missing-info feedback."""

    def __init__(self, llm: BaseChatModel) -> None:
        self._llm = llm

    async def optimize_query(
        self,
        original_query: str,
        missing_info: str,
        iteration_count: int,
        previous_query: str = "",
    ) -> str:
        human_prompt = (
            f"原始问题: {original_query}\n"
            f"当前迭代: {iteration_count}\n"
            f"上轮检索查询: {previous_query or '无'}\n"
            f"缺失信息: {missing_info or '无'}\n\n"
            "请输出新的检索查询："
        )
        try:
            response = await self._llm.ainvoke(
                [SystemMessage(content=_SYSTEM_PROMPT_ZH), HumanMessage(content=human_prompt)]
            )
            content = str(response.content).strip()  # type: ignore[union-attr]
            return " ".join(content.splitlines()).strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Query optimization failed (%s); fallback to original query.", exc)
            return original_query
