import logging

from .base import BaseQueryOptimizer
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

_SYSTEM_DECOMPOSITION = (
    "你是一个问题分解器，负责将复杂问题拆解为若干个独立的子问题。"
    "每个子问题单独占一行，无需编号或前缀，不要回答问题，不要附加解释。"
    "子问题数量控制在 2-4 个。"
)


class TaskSplitter(BaseQueryOptimizer):
    def __init__(self, llm: BaseChatModel):
        self._llm = llm

    async def run(
            self,
            original_query: str,
            accumulated_context: list[str],
            missing_info: str,
            previous_query: str = "",
    ) -> dict:

        human_prompt = (
            f"原始问题: {original_query}\n"
            f"已有信息: {accumulated_context or '无'}\n\n"
            f"缺失信息: {missing_info or '无'}\n\n"
            "请根据策略输出优化后的检索查询："
        )
        prompt_string = f"【System】:\n{_SYSTEM_DECOMPOSITION}\n\n【Human】:\n{human_prompt}"

        try:
            response = await self._llm.ainvoke(
                [SystemMessage(content=_SYSTEM_DECOMPOSITION), HumanMessage(content=human_prompt)]
            )
            optimized_query = str(response.content).strip()  # type: ignore[union-attr]
            logger.debug("Query Optimization Rewrite response: %s", optimized_query)

            return {
                "original_query": original_query,
                "optimized_query": optimized_query,
                "prompt": prompt_string,
                "type": "split",
                "success": True,
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("Query optimization failed (%s); fallback to original query.", exc)
            return {
                "original_query": original_query,
                "optimized_query": original_query,
                "prompt": prompt_string,
                "type": "split",
                "success": False,
            }