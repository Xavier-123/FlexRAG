"""
LangGraph graph builder for FlexRAG.

This module assembles the five-node StateGraph from the individual node
factories defined in :mod:`flexrag.graph.nodes`.

Graph topology::

    ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────────────┐     ┌──────────┐
    │  START   │────►│ retrieve │────►│  rerank  │────►│ optimize_context │────►│ generate │
    └──────────┘     └──────────┘     └──────────┘     └──────────────────┘     └────┬─────┘
                                                                                       │
                                                                                  ┌────▼─────┐
                                                                                  │   END    │
                                                                                  └──────────┘

Conditional error routing::

    Any node may write ``state["error"]`` on failure.  The next node in the
    chain detects the flag and short-circuits to ``END``.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from flexrag.abstractions.base_context_optimizer import BaseContextOptimizer
from flexrag.abstractions.base_generator import BaseGenerator
from flexrag.abstractions.base_reranker import BaseReranker
from flexrag.abstractions.base_retriever import BaseRetriever
from flexrag.graph.nodes import (
    make_generate_node,
    make_optimize_context_node,
    make_rerank_node,
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

    Mirrors :class:`~flexrag.schema.RAGState` but uses Python primitives
    (lists of dicts) instead of Pydantic models so that LangGraph can
    serialise / deserialise the state without extra configuration.
    """

    query: str
    retrieved_docs: list[dict[str, Any]]
    reranked_docs: list[dict[str, Any]]
    optimized_context: str
    answer: str
    evidence: list[str]
    error: str | None


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_rag_graph(
    retriever: BaseRetriever,
    reranker: BaseReranker,
    context_optimizer: BaseContextOptimizer,
    generator: BaseGenerator,
    top_k_retrieval: int = 10,
    top_k_rerank: int = 5,
    context_max_tokens: int = 3000,
) -> Any:
    """Assemble and compile the FlexRAG LangGraph StateGraph.

    Each component is injected as a dependency so that the graph stays fully
    decoupled from any specific implementation.  You can swap in a different
    retriever, reranker, or generator without touching this function.

    Args:
        retriever: Concrete retriever (e.g.
            :class:`~flexrag.retrievers.LlamaIndexRetriever`).
        reranker: Concrete reranker (e.g.
            :class:`~flexrag.rerankers.VLLMReranker`).
        context_optimizer: Concrete context optimiser (e.g.
            :class:`~flexrag.context_optimizers.LLMContextOptimizer`).
        generator: Concrete generator (e.g.
            :class:`~flexrag.generators.OpenAIGenerator`).
        top_k_retrieval: Number of documents to retrieve.
        top_k_rerank: Number of documents to keep after reranking.
        context_max_tokens: Token budget for the context window.

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
    retrieve_node = make_retrieve_node(retriever, top_k=top_k_retrieval)
    rerank_node = make_rerank_node(reranker, top_k=top_k_rerank)
    optimize_context_node = make_optimize_context_node(
        context_optimizer, max_tokens=context_max_tokens
    )
    generate_node = make_generate_node(generator)

    # ---- Build the graph ----
    graph = StateGraph(_GraphState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("optimize_context", optimize_context_node)
    graph.add_node("generate", generate_node)

    # ---- Wire up edges ----
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "optimize_context")
    graph.add_edge("optimize_context", "generate")
    graph.add_edge("generate", END)

    logger.debug("RAG graph compiled with %d nodes", 4)
    return graph.compile()
