import logging

from .base import BaseQueryOptimizer
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

_SYSTEM_TERM_ENHANCEMENT = (
    "你是一个领域术语增强器，负责在原始问题中补充相关专业术语和同义词，"
    "使检索能覆盖更多专业文档。"
    "输出仅包含一行增强后的检索查询，不要附加解释。"
)


class TerminologyEnricher(BaseQueryOptimizer):
    def __init__(self, llm: BaseChatModel):
        self._llm = llm
        self.type = "enriched"

    async def run(
            self,
            original_query: str,
            accumulated_context: list[str],
            missing_info: str,
            previous_query: str = "",
            previous_queries=None,
    ) -> dict:

        human_prompt = (
            f"原始问题: {original_query}\n"
            f"上一轮优化后的Query: {previous_query}\n"
            f"已有信息: {accumulated_context or '无'}\n\n"
            f"缺失信息: {missing_info or '无'}\n\n"
            "请根据策略输出优化后的检索查询："
        )
        prompt_string = f"【System】:\n{_SYSTEM_TERM_ENHANCEMENT}\n\n【Human】:\n{human_prompt}"

        try:
            response = await self._llm.ainvoke(
                [SystemMessage(content=_SYSTEM_TERM_ENHANCEMENT), HumanMessage(content=human_prompt)]
            )
            optimized_query = str(response.content).strip()  # type: ignore[union-attr]
            logger.debug("Query Optimization Rewrite response: %s", optimized_query)

            return {
                "original_query": original_query,
                "optimized_query": optimized_query,
                "prompt": prompt_string,
                "type": "enriched",
                "success": True,
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("Query optimization failed (%s); fallback to original query.", exc)
            return {
                "original_query": original_query,
                "optimized_query": original_query,
                "prompt": prompt_string,
                "type": "enriched",
                "success": False,
            }