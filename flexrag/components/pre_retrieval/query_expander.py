import logging

from typing import Dict, Any
from .base import BaseQueryOptimizer
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

_SYSTEM_EXPANSION = (
    "你是一个假设性文档生成器（HyDE）。"
    "根据用户的模糊问题，生成一段简短的假设性回答或文档片段，"
    "该片段将用于向量检索以找到相关真实文档。"
    "直接输出假设性文档内容，不要附加解释或标注。"
)


class QueryExpander(BaseQueryOptimizer):
    def __init__(self, llm: BaseChatModel):
        self._llm = llm
        self.type = "expanded"

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
        prompt_string = f"【System】:\n{_SYSTEM_EXPANSION}\n\n【Human】:\n{human_prompt}"

        try:
            response = await self._llm.ainvoke(
                [SystemMessage(content=_SYSTEM_EXPANSION), HumanMessage(content=human_prompt)]
            )
            optimized_query = str(response.content).strip()  # type: ignore[union-attr]
            logger.debug("Query Optimization Rewrite response: %s", optimized_query)

            return {
                "original_query": original_query,
                "optimized_query": optimized_query,
                "prompt": prompt_string,
                "type": "expanded",
                "success": True,
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("Query optimization failed (%s); fallback to original query.", exc)
            return {
                "original_query": original_query,
                "optimized_query": original_query,
                "prompt": prompt_string,
                "type": "expanded",
                "success": False,
            }
