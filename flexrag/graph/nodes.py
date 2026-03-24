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
from flexrag.abstractions.base_generator import BaseGenerator
from flexrag.abstractions.base_reranker import BaseReranker
from flexrag.abstractions.base_retriever import BaseRetriever
from flexrag.schema import Document, RAGState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type alias – LangGraph passes / expects plain dicts keyed by field name.
# ---------------------------------------------------------------------------
StateDict = dict[str, Any]


# ---------------------------------------------------------------------------
# Node factories
# ---------------------------------------------------------------------------


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

    def retrieve_node(state: StateDict) -> StateDict:
        """Retrieve relevant documents for the user's query.

        Reads:
            ``state["query"]``

        Writes:
            ``state["retrieved_docs"]``
        """
        query: str = state["query"]
        logger.info("[retrieve] query=%r  top_k=%d", query, top_k)
        try:
            docs: list[Document] = retriever.retrieve(query, top_k=top_k)
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

    def rerank_node(state: StateDict) -> StateDict:
        """Rerank retrieved documents and keep the top results.

        Reads:
            ``state["query"]``, ``state["retrieved_docs"]``

        Writes:
            ``state["reranked_docs"]``
        """
        if state.get("error"):
            return {}
        query: str = state["query"]
        raw_docs: list[dict] = state.get("retrieved_docs", [])
        documents = [Document(**d) for d in raw_docs]
        logger.info("[rerank] %d docs in → top_k=%d", len(documents), top_k)
        try:
            reranked: list[Document] = reranker.rerank(
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

    def optimize_context_node(state: StateDict) -> StateDict:
        """Distil reranked documents into a compact context string.

        Reads:
            ``state["query"]``, ``state["reranked_docs"]``

        Writes:
            ``state["optimized_context"]``
        """
        if state.get("error"):
            return {}
        query: str = state["query"]
        raw_docs: list[dict] = state.get("reranked_docs", [])
        documents = [Document(**d) for d in raw_docs]
        logger.info("[optimize_context] %d docs  max_tokens=%d", len(documents), max_tokens)
        try:
            context: str = optimizer.optimize(query, documents, max_tokens=max_tokens)
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

    def generate_node(state: StateDict) -> StateDict:
        """Generate a structured answer grounded in the optimised context.

        Reads:
            ``state["query"]``, ``state["optimized_context"]``,
            ``state["reranked_docs"]``

        Writes:
            ``state["answer"]``, ``state["evidence"]``
        """
        if state.get("error"):
            return {}
        query: str = state["query"]
        context: str = state.get("optimized_context", "")
        raw_docs: list[dict] = state.get("reranked_docs", [])
        source_texts = [d["text"] for d in raw_docs]
        logger.info("[generate] query=%r  context_len=%d", query, len(context))
        try:
            output = generator.generate(query, context, source_texts)
            return {"answer": output.answer, "evidence": output.evidence}
        except Exception as exc:  # noqa: BLE001
            logger.exception("[generate] failed: %s", exc)
            return {"error": f"Generation failed: {exc}"}

    return generate_node
