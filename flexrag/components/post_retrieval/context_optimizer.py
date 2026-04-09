"""
LLM-based context optimization strategy.

This optimizer uses an LLM to:
1. Selectively extract the most relevant passages from each document.
2. Concatenate the extractions into a single coherent context string.
3. Ensure the result fits within the configured token budget.

If the LLM call fails or the result exceeds the budget the module falls back
to a simple truncation strategy so that the pipeline never stalls.
"""

from __future__ import annotations

import asyncio
import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from flexrag.common.schema import Document
from flexrag.components.post_retrieval.base import BasePostRetrieval

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a precise context extractor.  Given a user question and several "
    "document passages, extract and return ONLY the sentences or paragraphs that "
    "are directly relevant to answering the question.  "
    "Do NOT add commentary, introductions, or conclusions.  "
    "Separate extracted passages with a blank line."
)

# _SYSTEM_PROMPT_ZH = (
#     "你是一个精准的上下文提取专家。给定一个用户问题和若干文档段落，请仅提取并返回与回答该问题直接相关的句子或段落。"
#     "请勿添加任何评论、引言或结语。"
#     "不同提取片段之间请用空行分隔。"
# )

_SYSTEM_PROMPT_ZH = (
    "你是一个严格的上下文抽取模型（Context Extractor）。"
    "任务是：根据用户问题，从提供的文档中提取可以用于回答问题的相关的原文句子或片段。"

    "抽取规则："
    "1. 优先提取与问题直接相关的内容。"
    "2. 如果问题涉及多个实体关系（例如：电影 -> 演员 -> 成就），需要补充相关实体的关键信息（如演员的代表作或知名原因）。"
    "3. 允许跨文档组合信息（multi-hop）。"
    "4. 仅提取原文，不得改写或总结。"
    "5. 按回答问题的完整性排序，而不仅是表面相关性。"
    "6. 若无相关内容，返回空数组 []。"

    "输出要求："
    "1. 仅输出 JSON，不得包含任何解释或额外文本。"
    "2. JSON 必须合法可解析。"
    "3. 使用如下结构："
    "{"
    '  "results": ["...", "..."]'
    "}"
)


class LLMContextOptimizer(BasePostRetrieval):
    def __init__(
            self,
            llm: BaseChatModel,
            chars_per_token: int = 2,
            max_input_tokens: int = 16384,
    ) -> None:
        self._llm = llm
        self._chars_per_token = chars_per_token
        self._max_input_tokens = max_input_tokens

    def _estimate_tokens(self, text: str) -> int:
        """根据字符数粗略估算 token 数量"""
        return len(text) // self._chars_per_token

    # ------------------------------------------------------------------
    # BaseContextOptimizer interface
    # ------------------------------------------------------------------

    async def optimize(
            self,
            query: str,
            documents: list[Document],
            accumulated_context: list[str],
            max_tokens: int = 8192,
    ) -> (str, str):
        if not documents:
            return ""

        # Build a numbered document listing for the prompt
        doc_listing = "\n\n".join(
            f"[Doc {i + 1}]\n{doc.text}" for i, doc in enumerate(documents)
        )
        human_prompt = (
            f"Question: {query}\n\n"
            f"Documents:\n{doc_listing}\n\n"
            f"history context:\n{accumulated_context[-2:]}\n\n"
            "提取相关段落。"
        )
        prompt_string = f"【System】:\n{_SYSTEM_PROMPT_ZH}\n\n【Human】:\n{human_prompt}"

        try:
            response = await self._llm.ainvoke(
                [
                    SystemMessage(content=_SYSTEM_PROMPT_ZH),
                    HumanMessage(content=human_prompt),
                ]
            )
            optimized: str = response.content  # type: ignore[union-attr]
            logger.info(f"LLM context optimization response:\n{optimized}")

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "LLM context optimisation failed (%s); falling back to truncation.",
                exc,
            )
            optimized = doc_listing

        # Enforce token budget via simple character truncation
        max_chars = max_tokens * self._chars_per_token
        if len(optimized) > max_chars:
            logger.info(
                "Truncating optimised context from %d to %d chars",
                len(optimized),
                max_chars,
            )
            optimized = optimized[:max_chars]

        logger.info("Context Optimized: %s", optimized)
        return optimized, prompt_string