"""
LangGraph node functions for the FlexRAG pipeline.

Each function in this module corresponds to one node in the StateGraph.
Nodes follow a simple contract:

- Accept a :class:`~flexrag.schema.RAGState` dict (LangGraph passes the state
  as a plain ``dict`` that matches the TypedDict / Pydantic schema).
- Return a **partial** ``dict`` with only the keys they update.  LangGraph
  merges the returned dict into the running state automatically.

Node execution order::

    user_input  ──►  retrieve  ──►  rerank  ──►  optimize_context  ──►  generate

Errors are caught per-node and written to ``state["error"]`` so that the graph
can route to a graceful-error node if desired (see :mod:`flexrag.graph.builder`).
"""

from __future__ import annotations
import logging
from typing import Any

from flexrag.abstractions.base_context_optimizer import BaseContextOptimizer
from flexrag.abstractions.base_context_evaluator import BaseContextEvaluator
from flexrag.abstractions.base_generator import BaseGenerator
from flexrag.abstractions.base_query_optimizer import BaseQueryOptimizer
from flexrag.abstractions.base_reranker import BaseReranker
from flexrag.abstractions.base_retriever import BaseRetriever
from flexrag.schema import Document

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type alias – LangGraph passes / expects plain dicts keyed by field name.
# ---------------------------------------------------------------------------
StateDict = dict[str, Any]


# ---------------------------------------------------------------------------
# Node factories
# ---------------------------------------------------------------------------

def make_query_optimizer_node(
    optimizer: BaseQueryOptimizer,
) -> Any:
    """Create the query-optimisation node function."""

    async def query_optimizer_node(state: StateDict) -> StateDict:
        logger.debug("-------- query optimizer node --------")
        if state.get("error"):
            return {}
        original_query: str = state.get("original_query") or state["query"]
        accumulated_context: list[str] = state.get("accumulated_context", [""])
        missing_info: str = state.get("missing_info", "")
        iteration_count: int = int(state.get("iteration_count", 0))
        previous_query: str = state.get("current_query", "")
        logger.info("[query_optimizer] iteration=%d", iteration_count)
        try:
            current_query = await optimizer.optimize_query(
                original_query=original_query,
                accumulated_context=accumulated_context,
                missing_info=missing_info,
                iteration_count=iteration_count,
                previous_query=previous_query,
            )
            return {
                "original_query": original_query,
                "current_query": current_query or original_query,
            }
        except Exception as exc:  # noqa: BLE001
            logger.exception("[query_optimizer] failed: %s", exc)
            return {"error": f"Query optimization failed: {exc}"}

    return query_optimizer_node


def make_retrieve_node(
    retriever: BaseRetriever,
    top_k: int,
) -> Any:
    """Create the retrieval node function.

    Args:
        retriever: A concrete :class:`~flexrag.abstractions.BaseRetriever`.
        top_k: Number of documents to retrieve.

    Returns:
        A callable compatible with :meth:`StateGraph.add_node`.
    """

    async def retrieve_node(state: StateDict) -> StateDict:
        """Retrieve relevant documents for the user's query.

        Reads:
            ``state["query"]``

        Writes:
            ``state["retrieved_docs"]``
        """
        logger.debug("-------- retrieve node --------")
        query: str = state.get("current_query") or state.get("original_query") or state["query"]
        logger.info("[retrieve] query=%r  top_k=%d", query, top_k)
        try:
            docs: list[Document] = await retriever.retrieve(query, top_k=top_k)
            return {"retrieved_docs": [d.model_dump() for d in docs]}
        except Exception as exc:  # noqa: BLE001
            logger.exception("[retrieve] failed: %s", exc)
            return {"error": f"Retrieval failed: {exc}"}

    return retrieve_node


def make_rerank_node(
    reranker: BaseReranker,
    top_k: int,
) -> Any:
    """Create the reranking node function.

    Args:
        reranker: A concrete :class:`~flexrag.abstractions.BaseReranker`.
        top_k: Number of documents to keep after reranking.

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
        logger.debug("-------- rerank node --------")
        if state.get("error"):
            return {}
        query: str = state.get("current_query") or state.get("original_query") or state["query"]
        raw_docs: list[dict] = state.get("retrieved_docs", [])
        documents = [Document(**d) for d in raw_docs]
        logger.info("[rerank] %d docs in → top_k=%d", len(documents), top_k)
        try:
            reranked: list[Document] = await reranker.rerank(
                query, documents, top_k=top_k
            )
            return {"reranked_docs": [d.model_dump() for d in reranked]}
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
        optimizer: A concrete :class:`~flexrag.abstractions.BaseContextOptimizer`.
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
        logger.debug("-------- optimize context node --------")
        if state.get("error"):
            return {}
        query: str = state.get("current_query") or state.get("original_query") or state["query"]
        accumulated_context: list[str] = state.get("accumulated_context")
        raw_docs: list[dict] = state.get("reranked_docs", [])
        documents = [Document(**d) for d in raw_docs]
        logger.info("[optimize_context] %d docs  max_tokens=%d", len(documents), max_tokens)
        try:
            context: str = await optimizer.optimize(query, documents, accumulated_context, max_tokens=max_tokens)
            return {"optimized_context": context}
        except Exception as exc:  # noqa: BLE001
            logger.exception("[optimize_context] failed: %s", exc)
            return {"error": f"Context optimisation failed: {exc}"}

    return optimize_context_node


def make_generate_node(generator: BaseGenerator) -> Any:
    """Create the answer-generation node function.

    Args:
        generator: A concrete :class:`~flexrag.abstractions.BaseGenerator`.

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
        logger.debug("-------- generate node --------")
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
            return {"answer": output.answer, "evidence": output.evidence}
        except Exception as exc:  # noqa: BLE001
            logger.exception("[generate] failed: %s", exc)
            return {"error": f"Generation failed: {exc}"}

    return generate_node


def make_context_evaluator_node(evaluator: BaseContextEvaluator) -> Any:
    """Create the context-evaluator (judge) node function."""

    async def context_evaluator_node(state: StateDict) -> StateDict:
        logger.debug("-------- context evaluator node --------")
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
            }
        except Exception as exc:  # noqa: BLE001
            logger.exception("[context_evaluator] failed: %s", exc)
            return {"error": f"Context evaluation failed: {exc}"}

    return context_evaluator_node


def make_analyze_missing_info_node() -> Any:
    """Create the node that records feedback and increments iteration counter."""

    async def analyze_missing_info_node(state: StateDict) -> StateDict:
        logger.debug("-------- analyze missing info node --------")
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
        }

    return analyze_missing_info_node
