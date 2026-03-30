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

from flexrag.core.abstractions import BaseContextOptimizer
from flexrag.core.abstractions import BaseContextEvaluator
from flexrag.core.abstractions import BaseGenerator
from flexrag.core.abstractions import BaseMultiQueryGenerator
from flexrag.core.abstractions import BaseQueryOptimizer
from flexrag.core.abstractions import BaseQueryRouter
from flexrag.core.abstractions import BaseReranker
from flexrag.core.abstractions import BaseRetriever
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

def make_query_router_node(router: BaseQueryRouter) -> Any:
    """Create the query-routing node function."""

    async def query_router_node(state: StateDict) -> StateDict:
        logger.info("-------- query router node --------")
        if state.get("error"):
            return {}
        original_query: str = state.get("original_query") or state["query"]
        try:
            query_type = await router.route(original_query)
            logger.info("[query_router] query_type=%s", query_type)
            return {
                "query_type": query_type,
                "node_trace": [{"node": "query_router", "query": original_query, "query_type": query_type}],
            }
        except Exception as exc:  # noqa: BLE001
            logger.exception("[query_router] failed: %s", exc)
            return {"error": f"Query routing failed: {exc}"}

    return query_router_node


def make_query_optimizer_node(
    optimizer: BaseQueryOptimizer,
) -> Any:
    """Create the query-optimisation node function."""

    async def query_optimizer_node(state: StateDict) -> StateDict:
        logger.info("-------- query optimizer node --------")
        if state.get("error"):
            return {}
        original_query: str = state.get("original_query") or state["query"]
        accumulated_context: list[str] = state.get("accumulated_context", [""])
        missing_info: str = state.get("missing_info", "")
        iteration_count: int = int(state.get("iteration_count", 0))
        previous_query: str = state.get("current_query", "")
        query_type: str = state.get("query_type", "simple")
        logger.info("[query_optimizer] iteration=%d  query_type=%s", iteration_count, query_type)
        try:
            current_query = await optimizer.optimize_query(
                original_query=original_query,
                accumulated_context=accumulated_context,
                missing_info=missing_info,
                iteration_count=iteration_count,
                previous_query=previous_query,
                query_type=query_type,
            )
            optimized = current_query or original_query
            return {
                "original_query": original_query,
                "current_query": optimized,
                "node_trace": [{"node": "query_optimizer", "iteration": iteration_count, "query_type": query_type, "input_query": original_query, "output_query": optimized}],
            }
        except Exception as exc:  # noqa: BLE001
            logger.exception("[query_optimizer] failed: %s", exc)
            return {"error": f"Query optimization failed: {exc}"}

    return query_optimizer_node


def make_multi_query_generator_node(generator: BaseMultiQueryGenerator) -> Any:
    """Create the multi-query-generation node function."""

    async def multi_query_generator_node(state: StateDict) -> StateDict:
        logger.info("-------- multi query generator node --------")
        if state.get("error"):
            return {}
        original_query: str = state.get("original_query") or state["query"]
        optimized_query: str = state.get("current_query") or original_query
        query_type: str = state.get("query_type", "simple")
        try:
            queries = await generator.generate_queries(
                original_query=original_query,
                optimized_query=optimized_query,
                query_type=query_type,
            )
            logger.info("[multi_query_generator] produced %d queries", len(queries))
            return {
                "optimized_queries": queries,
                "node_trace": [{"node": "multi_query_generator", "query_type": query_type, "queries": queries}],
            }
        except Exception as exc:  # noqa: BLE001
            logger.exception("[multi_query_generator] failed: %s", exc)
            return {"error": f"Multi-query generation failed: {exc}"}

    return multi_query_generator_node


def make_retrieve_node(
    retriever: BaseRetriever,
) -> Any:
    """Create the retrieval node function.

    Args:
        retriever: A concrete :class:`~flexrag.core.abstractions.BaseRetriever`.

    Returns:
        A callable compatible with :meth:`StateGraph.add_node`.
    """

    async def retrieve_node(state: StateDict) -> StateDict:
        """Retrieve relevant documents for the user's query.

        Reads:
            ``state["optimized_queries"]`` or ``state["current_query"]``

        Writes:
            ``state["retrieved_docs"]``
        """
        logger.info("-------- retrieve node --------")
        optimized_queries: list[str] = state.get("optimized_queries") or []
        fallback_query: str = state.get("current_query") or state.get("original_query") or state["query"]
        queries = optimized_queries if optimized_queries else [fallback_query]
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
            return {
                "retrieved_docs": [d.model_dump() for d in all_docs],
                "node_trace": [{"node": "retrieve", "queries": queries, "docs_retrieved": len(all_docs)}],
            }
        except Exception as exc:  # noqa: BLE001
            logger.exception("[retrieve] failed: %s", exc)
            return {"error": f"Retrieval failed: {exc}"}

    return retrieve_node


def make_rerank_node(
    reranker: BaseReranker,
) -> Any:
    """Create the reranking node function.

    Args:
        reranker: A concrete :class:`~flexrag.core.abstractions.BaseReranker`.

    Returns:
        A callable compatible with :meth:`StateGraph.add_node`.
    """

    async def rerank_node(state: StateDict) -> StateDict:
        """Rerank retrieved documents and keep the top results.

        Reads:
            ``state["query"]``, ``state["retrieved_docs"]``

        Writes:
            ``state["reranked_docs"]``
        """
        logger.info("-------- rerank node --------")
        if state.get("error"):
            return {}
        query: str = state.get("current_query") or state.get("original_query") or state["query"]
        raw_docs: list[dict] = state.get("retrieved_docs", [])
        documents = [Document(**d) for d in raw_docs]
        logger.info("[rerank] %d docs in → top_k=%d", len(documents), settings.top_k_rerank)
        try:
            reranked: list[Document] = await reranker.rerank(
                query, documents
            )
            return {
                "reranked_docs": [d.model_dump() for d in reranked],
                "node_trace": [{"node": "rerank", "docs_in": len(documents), "docs_out": len(reranked)}],
            }
        except Exception as exc:  # noqa: BLE001
            logger.exception("[rerank] failed: %s", exc)
            return {"error": f"Reranking failed: {exc}"}

    return rerank_node


def make_optimize_context_node(
    optimizer: BaseContextOptimizer,
    max_tokens: int,
) -> Any:
    """Create the context-optimisation node function.

    Args:
        optimizer: A concrete :class:`~flexrag.core.abstractions.BaseContextOptimizer`.
        max_tokens: Token budget for the optimised context.

    Returns:
        A callable compatible with :meth:`StateGraph.add_node`.
    """

    async def optimize_context_node(state: StateDict) -> StateDict:
        """Distil reranked documents into a compact context string.

        Reads:
            ``state["query"]``, ``state["reranked_docs"]``

        Writes:
            ``state["optimized_context"]``
        """
        logger.info("-------- optimize context node --------")
        if state.get("error"):
            return {}
        query: str = state.get("current_query") or state.get("original_query") or state["query"]
        accumulated_context: list[str] = state.get("accumulated_context")
        raw_docs: list[dict] = state.get("reranked_docs", [])
        documents = [Document(**d) for d in raw_docs]
        logger.info("[optimize_context] %d docs  max_tokens=%d", len(documents), max_tokens)
        try:
            context: str = await optimizer.optimize(query, documents, accumulated_context, max_tokens=max_tokens)
            return {
                "optimized_context": context,
                "node_trace": [{"node": "optimize_context", "docs_in": len(documents), "context_len": len(context)}],
            }
        except Exception as exc:  # noqa: BLE001
            logger.exception("[optimize_context] failed: %s", exc)
            return {"error": f"Context optimisation failed: {exc}"}

    return optimize_context_node


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
                "node_trace": [{"node": "generate", "query": query, "answer_len": len(output.answer), "evidence_count": len(output.evidence)}],
            }
        except Exception as exc:  # noqa: BLE001
            logger.exception("[generate] failed: %s", exc)
            return {"error": f"Generation failed: {exc}"}

    return generate_node


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
        try:
            result = await evaluator.evaluate(original_query=original_query, optimized_context=context, accumulated_context=accumulated_context)
            return {
                "context_sufficient": result.context_sufficient,
                "missing_info": result.missing_info,
                "judge_reason": result.judge_reason,
                "accumulated_context": result.accumulated_context,
                "node_trace": [{"node": "context_evaluator", "context_sufficient": result.context_sufficient, "judge_reason": result.judge_reason}],
            }
        except Exception as exc:  # noqa: BLE001
            logger.exception("[context_evaluator] failed: %s", exc)
            return {"error": f"Context evaluation failed: {exc}"}

    return context_evaluator_node


def make_analyze_missing_info_node() -> Any:
    """Create the node that records feedback and increments iteration counter."""

    async def analyze_missing_info_node(state: StateDict) -> StateDict:
        logger.info("-------- analyze missing info node --------")
        if state.get("error"):
            return {}
        missing_info: str = state.get("missing_info", "")
        history: list[str] = list(state.get("missing_info_history", []))
        if missing_info:
            history.append(missing_info)
        iteration_count = int(state.get("iteration_count", 0)) + 1
        return {
            "iteration_count": iteration_count,
            "missing_info_history": history,
            "node_trace": [{"node": "analyze_missing_info", "iteration_count": iteration_count, "missing_info": missing_info}],
        }

    return analyze_missing_info_node
