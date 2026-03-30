"""
LLM-based query router that classifies queries into processing strategies.
"""

from __future__ import annotations

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from flexrag.core.abstractions import BaseQueryRouter

logger = logging.getLogger(__name__)

_VALID_TYPES = ("simple", "vague", "complex", "professional")

_SYSTEM_PROMPT_ZH = (
    "你是一个查询类型分类器，需要判断用户问题属于以下哪种类型：\n"
    "- simple（简单问题）：表述清晰、有明确直接答案的问题\n"
    "- vague（模糊问题）：表述不清晰、范围模糊或意图不明确的问题\n"
    "- complex（复杂问题）：需要多步推理或涉及多个子问题的问题\n"
    "- professional（专业问题）：涉及特定领域术语或深度专业知识的问题\n\n"
    "只输出一个单词：simple、vague、complex 或 professional，不要附加任何解释。"
)


class LLMQueryRouter(BaseQueryRouter):
    """Classify a user query into one of four processing strategy types."""

    def __init__(self, llm: BaseChatModel) -> None:
        self._llm = llm

    async def route(self, query: str) -> str:
        """Return query type: 'simple', 'vague', 'complex', or 'professional'."""
        try:
            response = await self._llm.ainvoke(
                [
                    SystemMessage(content=_SYSTEM_PROMPT_ZH),
                    HumanMessage(content=f"问题：{query}"),
                ]
            )
            result = str(response.content).strip().lower()
            if result not in _VALID_TYPES:
                logger.warning(
                    "Unexpected query type %r from router; defaulting to 'simple'.", result
                )
                return "simple"
            logger.debug("Query routed as type: %s", result)
            return result
        except Exception as exc:  # noqa: BLE001
            logger.warning("Query routing failed (%s); defaulting to 'simple'.", exc)
            return "simple"
