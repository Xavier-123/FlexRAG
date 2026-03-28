"""
LLM-based context evaluator (judge) for iterative RAG routing.
"""

from __future__ import annotations

import json
import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from flexrag.core.abstractions import BaseContextEvaluator
from flexrag.core.schema import ContextEvaluation

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT_ZH = (
    "你是RAG上下文评估器。判断当前上下文是否足以回答原始问题。"
    "严格输出JSON对象，字段必须为："
    "context_sufficient(boolean), missing_info(string), judge_reason(string)。"
    "若信息不足，missing_info必须给出下一轮检索可执行的缺失点。"
    "禁止输出任何JSON以外内容。"
)


class LLMContextEvaluator(BaseContextEvaluator):
    """Evaluate if optimized context can support a reliable final answer."""

    def __init__(self, llm: BaseChatModel) -> None:
        self._llm = llm

    async def evaluate(
        self,
        original_query: str,
        optimized_context: str,
        accumulated_context: list[str],
    ) -> ContextEvaluation:
        human_prompt = (
            f"原始问题:\n{original_query}\n\n"
            f"当前上下文:\n{optimized_context}\n\n"
            f"历史上下文:\n{accumulated_context}\n\n"
            "请输出评估JSON："
        )
        try:
            response = await self._llm.ainvoke(
                [SystemMessage(content=_SYSTEM_PROMPT_ZH), HumanMessage(content=human_prompt)]
            )
            payload = str(response.content).strip()  # type: ignore[union-attr]
            data = json.loads(payload)
            context_sufficient = data.get("context_sufficient", False)
            if context_sufficient is False:
                return ContextEvaluation(
                    context_sufficient=bool(data.get("context_sufficient", False)),
                    missing_info=str(data.get("missing_info", "") or ""),
                    judge_reason=str(data.get("judge_reason", "") or ""),
                    accumulated_context=[optimized_context]
                )
            else:
                return ContextEvaluation(
                    context_sufficient=bool(data.get("context_sufficient", False)),
                    missing_info=str(data.get("missing_info", "") or ""),
                    judge_reason=str(data.get("judge_reason", "") or ""),
                    accumulated_context=[]
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Context evaluation parse failed (%s); fallback to insufficient.", exc)
            return ContextEvaluation(
                context_sufficient=False,
                missing_info="评估输出不可解析，需要补充关键事实并重试检索",
                judge_reason=f"Evaluator fallback: {exc}",
                accumulated_context=[]
            )
