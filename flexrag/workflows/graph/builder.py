"""
LangGraph graph builder for FlexRAG.

This module assembles the five-node StateGraph from the individual node
factories defined in :mod:`flexrag.workflows.graph.nodes`.

Conditional error routing::

    Any node may write ``state["error"]`` on failure.  The next node in the
    chain detects the flag and short-circuits to ``END``.
"""

from __future__ import annotations

import logging
import operator
from typing import Any, Literal, Optional, Annotated
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from flexrag.core.abstractions import BaseContextEvaluator, BaseGenerator
from flexrag.components.pre_retrieval import PreQueryOptimizer
from flexrag.components.retrieval import BaseRetriever
from flexrag.components.post_retrieval import PostRetrieval
from flexrag.workflows.graph.nodes import (
    make_context_evaluator_node,
    make_generate_node,
    make_post_retrieval_optimizer_node,
    make_pre_retrieval_optimizer_node,
    make_retrieve_node,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LangGraph state schema
# ---------------------------------------------------------------------------
# We use a plain TypedDict-style annotation dict so that LangGraph can
# introspect the state shape.  All fields mirror flexrag.schema.RAGState.
# ---------------------------------------------------------------------------

from typing import TypedDict


class _GraphState(TypedDict, total=False):
    """Internal LangGraph state schema.

    Mirrors :class:`~flexrag.core.schema.RAGState` but uses Python primitives
    (lists of dicts) instead of Pydantic models so that LangGraph can
    serialise / deserialise the state without extra configuration.
    """

    query: str
    original_query: str
    current_queries: dict
    optimized_queries: list[str]
    iteration_count: int
    max_iterations: int
    context_sufficient: bool
    missing_info: str
    missing_info_history: list[str]
    judge_reason: str
    retrieved_docs: list[dict[str, Any]]
    reranked_docs: list[dict[str, Any]]
    optimized_context: str  # 当前这一轮提取的上下文
    accumulated_context: Annotated[list[str], operator.add]  # 记录所有迭代轮次中积累的“已有 Context”
    answer: str
    evidence: list[str]
    error: str | None
    node_trace: Annotated[list[dict[str, Any]], operator.add]


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_rag_graph(
        pre_retrieval_optimizer: PreQueryOptimizer,
        retriever: BaseRetriever,
        post_retrieval_optimizer: PostRetrieval,
        context_evaluator: BaseContextEvaluator,
        generator: BaseGenerator,
        context_max_tokens: int = 16000,
        draw_image_path: Optional[str] = None,
        checkpointer: Optional[BaseCheckpointSaver] = None,
) -> Any:
    """Assemble and compile the FlexRAG LangGraph StateGraph.

    Each component is injected as a dependency so that the graph stays fully
    decoupled from any specific implementation.  You can swap in a different
    retriever, reranker, or generator without touching this function.

    Args:
        pre_retrieval_optimizer:
        post_retrieval_optimizer:
        context_evaluator:
        retriever: Concrete retriever (e.g.
            :class:`~flexrag.components.retrieval.LlamaIndexRetriever`).
        generator: Concrete generator (e.g.
            :class:`~flexrag.components.generation.OpenAIGenerator`).
        context_max_tokens: Token budget for the context window.
        draw_image_path: 如果提供此路径（如 'architecture.png'），则将架构图保存到本地。
        checkpointer: Optional LangGraph checkpoint saver.  When provided,
            the graph persists a state snapshot after every node execution,
            enabling full execution-trace replay and audit.  Pass an instance
            of :class:`~langgraph.checkpoint.sqlite.SqliteSaver` (or any other
            :class:`~langgraph.checkpoint.base.BaseCheckpointSaver`) to enable
            persistence.  ``None`` disables checkpointing (default behaviour).

    Returns:
        A compiled LangGraph :class:`~langgraph.graph.CompiledGraph` ready
        to be invoked with ``graph.invoke({"query": "..."})`` .

    Example::

        graph = build_rag_graph(
            retriever=LlamaIndexRetriever(...),
            reranker=VLLMReranker(...),
            context_optimizer=LLMContextOptimizer(...),
            generator=OpenAIGenerator(...),
        )
        result = graph.invoke({"query": "What is Retrieval-Augmented Generation?"})
        print(result["answer"])
        print(result["evidence"])
    """
    # ---- Create node callables ----
    pre_retrieval_optimizer_node = make_pre_retrieval_optimizer_node(pre_retrieval_optimizer)
    retrieve_node = make_retrieve_node(retriever)
    post_retrieval_optimizer_node = make_post_retrieval_optimizer_node(post_retrieval_optimizer, context_max_tokens)
    context_evaluator_node = make_context_evaluator_node(context_evaluator)
    generate_node = make_generate_node(generator)

    # ---- Build the graph ----
    graph = StateGraph(_GraphState)

    graph.add_node("pre_retrieval_optimizer", pre_retrieval_optimizer_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("post_retrieval_optimizer", post_retrieval_optimizer_node)
    graph.add_node("context_evaluator", context_evaluator_node)
    graph.add_node("generate", generate_node)
    # graph.add_node("reflector", reflect_node)    # 反思与评估

    # ---- Wire up edges ----
    graph.add_edge(START, "pre_retrieval_optimizer")
    graph.add_edge("pre_retrieval_optimizer", "retrieve")
    graph.add_edge("retrieve", "post_retrieval_optimizer")
    graph.add_edge("post_retrieval_optimizer", "context_evaluator")
    graph.add_conditional_edges(
        "context_evaluator",
        _route_after_context_evaluator,
        {
            "generate": "generate",
            "pre_retrieval_optimizer": "pre_retrieval_optimizer",
        },
    )
    graph.add_edge("generate", END)

    # graph.add_edge("generate", "reflector")
    #
    # # 反思节点决定是结束还是重新回炉
    # graph.add_conditional_edges(
    #     "reflector",
    #     check_quality,
    #     {
    #         "rewrite_and_retry": "agent",  # 方案不佳，带着反馈回到 Agent 重新思考和检索
    #         "finish": END  # 方案完美，流程结束
    #     }
    # )

    logger.debug("Agentic RAG graph compiled with %d nodes", len(graph.nodes))
    compiled_graph = graph.compile(checkpointer=checkpointer)

    # 绘制并保存架构图
    if draw_image_path:
        try:
            # 获取图的 Mermaid PNG 二进制数据
            img_bytes = compiled_graph.get_graph().draw_mermaid_png()
            with open(draw_image_path, "wb") as f:
                f.write(img_bytes)
            logger.info(f"Architecture diagram successfully saved to {draw_image_path}")
        except Exception as e:
            logger.error(f"Failed to draw architecture diagram: {e}")
            logger.warning(
                "Please ensure you have network access (Mermaid ink API is used) or required dependencies installed.")

    return compiled_graph


def _route_after_context_evaluator(
        state: _GraphState,
) -> Literal["generate", "pre_retrieval_optimizer"]:
    if state.get("context_sufficient", False):
        return "generate"

    iteration_count = int(state.get("iteration_count", 0))
    max_iterations = int(state.get("max_iterations", 3))
    if iteration_count < max_iterations:
        return "pre_retrieval_optimizer"
    return "generate"

# def check_quality(state: _GraphState) -> str:
#     """判断反思节点的结果，决定是否重做"""
#     # 如果反思节点追加了带有反馈意见的消息，说明需要重做
#     last_message = state["messages"][-1]
#     if isinstance(last_message, HumanMessage) and "调整搜索策略" in last_message.content:
#         return "rewrite_and_retry"
#     return END
