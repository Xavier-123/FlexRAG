"""
LangGraph node functions for the FlexRAG pipeline.

Each function in this module corresponds to one node in the StateGraph.
Nodes follow a simple contract:

- Accept a :class:`~flexrag.core.schema.RAGState` dict (LangGraph passes the state
  as a plain ``dict`` that matches the TypedDict / Pydantic schema).
- Return a **partial** ``dict`` with only the keys they update.  LangGraph
  merges the returned dict into the running state automatically.

Node execution order::

    user_input  ──►  retrieve  ──►  rerank  ──►  optimize_context  ──►  generate

Errors are caught per-node and written to ``state["error"]`` so that the graph
can route to a graceful-error node if desired (see :mod:`flexrag.workflows.graph.builder`).
"""

from __future__ import annotations
import logging
from typing import Any

from flexrag.core.abstractions import BaseContextEvaluator, BaseGenerator
from flexrag.components.pre_retrieval import PreQueryOptimizer
from flexrag.components.retrieval import BaseFlexRetriever
from flexrag.components.post_retrieval import PostRetrieval
from flexrag.core.schema import Document
from flexrag.core.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type alias – LangGraph passes / expects plain dicts keyed by field name.
# ---------------------------------------------------------------------------
StateDict = dict[str, Any]


# ---------------------------------------------------------------------------
# Node factories
# ---------------------------------------------------------------------------

def make_pre_retrieval_optimizer_node(
    optimizer: PreQueryOptimizer,
) -> Any:
    """Create the query-optimisation node function."""

    async def pre_retrieval_optimizer_node(state: StateDict) -> StateDict:
        logger.info("-------- pre retrieval optimizer node --------")
        if state.get("error"):
            return {}
        original_query: str = state.get("original_query") or state["query"]
        accumulated_context: list[str] = state.get("accumulated_context", [""])
        missing_info: str = state.get("missing_info", "")
        iteration_count: int = int(state.get("iteration_count", 0))
        logger.info("[query_optimizer] iteration=%d", iteration_count)
        try:
            optimized_queries, current_queries = await optimizer.run(
                original_query=original_query,
                accumulated_context=accumulated_context,
                missing_info=missing_info,
                previous_queries=state.get("current_queries", {}),
            )
            return {
                "iteration_count": state["iteration_count"],
                "original_query": original_query,
                "current_queries": current_queries,
                "optimized_queries": optimized_queries,
                "node_trace": [
                    {
                        "node": "query_optimizer",
                        "iteration": iteration_count,
                        "optimized_queries": optimized_queries,
                    }
                ],
            }
        except Exception as exc:  # noqa: BLE001
            logger.exception("[query_optimizer] failed: %s", exc)
            return {"error": f"Query optimization failed: {exc}"}

    return pre_retrieval_optimizer_node


def make_retrieve_node(
    retriever: BaseFlexRetriever,
) -> Any:
    """Create the retrieval node function.

    Args:
        retriever: A concrete :class:`~flexrag.components.retrieval.BaseRetriever`.

    Returns:
        A callable compatible with :meth:`StateGraph.add_node`.
    """

    async def retrieve_node(state: StateDict) -> StateDict:
        """Retrieve relevant documents for the user's query.

        Reads:
            ``state["optimized_queries"]`` or ``state["current_queries"]``

        Writes:
            ``state["retrieved_docs"]``
        """
        logger.info("-------- retrieve node --------")
        optimized_queries: list[str] = state.get("optimized_queries") or []
        queries = optimized_queries if optimized_queries else [state["original_query"]]
        logger.info("[retrieve] %d queries  top_k=%d", len(queries), settings.top_k_retrieval)
        try:
            seen: set[str] = set()
            all_docs: list[Document] = []
            for query in queries:
                docs: list[Document] = await retriever.retrieve(query)
                for d in docs:
                    if d.text not in seen:
                        seen.add(d.text)
                        all_docs.append(d)
            logger.info("[retrieve] %d unique docs from %d queries", len(all_docs), len(queries))
            retrieved_docs = [d.model_dump() for d in all_docs]
            return {
                "retrieved_docs": retrieved_docs,
                "node_trace": [{"iteration_count": state["iteration_count"], "node": "retrieve", "queries": queries, "retrieved_docs": retrieved_docs}],
            }
        except Exception as exc:  # noqa: BLE001
            logger.exception("[retrieve] failed: %s", exc)
            return {"error": f"Retrieval failed: {exc}"}

    return retrieve_node


def make_post_retrieval_optimizer_node(
    optimizer: PostRetrieval,
    max_tokens: int,
) -> Any:

    async def post_retrieval_optimizer_node(state: StateDict) -> StateDict:
        logger.info("-------- post retrieval optimizer node --------")
        if state.get("error"):
            return {}
        query: str = state.get("current_query") or state.get("original_query") or state["query"]
        accumulated_context: list[str] = state.get("accumulated_context")
        raw_docs: list[dict] = state.get("retrieved_docs", [])
        documents = [Document(**d) for d in raw_docs]
        logger.info("[optimize_context] %d docs  max_tokens=%d", len(documents), max_tokens)

        try:
            optimized_result = await optimizer.optimize(
                query=query, documents=documents, accumulated_context=accumulated_context, max_tokens=max_tokens
            )

            if isinstance(optimized_result, tuple) and len(optimized_result) == 2:
                # LLMContextOptimizer case
                optimized_query, prompt_string = optimized_result
                return {
                    "current_query": optimized_query,
                    "node_trace": [{"iteration_count": state["iteration_count"], "node": "optimize_context", "prompt": prompt_string, "optimized_query": optimized_query}],
                }
            elif isinstance(optimized_result, list) and all(isinstance(doc, Document) for doc in optimized_result):
                # OpenAILikeReranker case
                reranked_docs = [d.model_dump() for d in optimized_result]
                return {
                    "reranked_docs": reranked_docs,
                    "node_trace": [{"iteration_count": state["iteration_count"], "node": "rerank", "reranked_docs": reranked_docs}],
                }

        except Exception as exc:  # noqa: BLE001
            logger.exception("[rerank] failed: %s", exc)
            return {"error": f"Reranking failed: {exc}"}

        return state
    return post_retrieval_optimizer_node


def make_context_evaluator_node(evaluator: BaseContextEvaluator) -> Any:
    """Create the context-evaluator (judge) node function."""

    async def context_evaluator_node(state: StateDict) -> StateDict:
        logger.info("-------- context evaluator node --------")
        if state.get("error"):
            return {}
        original_query: str = state.get("original_query") or state["query"]
        context: str = state.get("optimized_context", "")
        accumulated_context: list[str] = state.get("accumulated_context", [""])
        logger.info("[context_evaluator] context_len=%d", len(context))
        iteration_count = int(state.get("iteration_count", 0)) + 1
        try:
            result = await evaluator.evaluate(original_query=original_query, optimized_context=context, accumulated_context=accumulated_context)
            return {
                "iteration_count": iteration_count,
                "context_sufficient": result.context_sufficient,
                "missing_info": result.missing_info,
                "judge_reason": result.judge_reason,
                "accumulated_context": result.accumulated_context,
                "node_trace": [{"iteration_count": state["iteration_count"], "node": "context_evaluator",
                                "context_sufficient": result.context_sufficient, "judge_reason": result.judge_reason,
                                "prompt": result.prompt_string
                                }],
            }
        except Exception as exc:  # noqa: BLE001
            logger.exception("[context_evaluator] failed: %s", exc)
            return {"error": f"Context evaluation failed: {exc}"}

    return context_evaluator_node


def make_generate_node(generator: BaseGenerator) -> Any:
    """Create the answer-generation node function.

    Args:
        generator: A concrete :class:`~flexrag.core.abstractions.BaseGenerator`.

    Returns:
        A callable compatible with :meth:`StateGraph.add_node`.
    """

    async def generate_node(state: StateDict) -> StateDict:
        """Generate a structured answer grounded in the optimised context.

        Reads:
            ``state["query"]``, ``state["optimized_context"]``,
            ``state["reranked_docs"]``

        Writes:
            ``state["answer"]``, ``state["evidence"]``
        """
        logger.info("-------- generate node --------")
        if state.get("error"):
            return {}
        query: str = state.get("original_query") or state["query"]
        context: str = state.get("optimized_context", "")
        raw_docs: list[dict] = state.get("reranked_docs", [])
        source_texts = [d["text"] for d in raw_docs]
        accumulated_context: list[str] = state.get("accumulated_context", [])
        logger.info("[generate] query=%r  context_len=%d", query, len(context))
        try:
            output = await generator.generate(query, context, accumulated_context, source_texts)
            return {
                "answer": output.answer,
                "evidence": output.evidence,
                "node_trace": [{"node": "generate", "query": query, "answer": output.answer, "evidence": output.evidence}],
            }
        except Exception as exc:  # noqa: BLE001
            logger.exception("[generate] failed: %s", exc)
            return {"error": f"Generation failed: {exc}"}

    return generate_node
