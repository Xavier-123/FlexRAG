"""
Pydantic data models shared across the FlexRAG system.

This module defines:
- :class:`Document` – a single retrieved text chunk.
- :class:`RAGState` – the mutable state object threaded through LangGraph nodes.
- :class:`RAGOutput` – the structured final output returned to the caller.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Document(BaseModel):
    """A single document chunk returned by the retriever or reranker.

    Attributes:
        text: Raw text content of the chunk.
        score: Relevance score assigned during retrieval or reranking.
        metadata: Arbitrary key-value metadata (source URL, page number, etc.).
    """

    text: str = Field(..., description="Raw text content of the chunk")
    score: float = Field(0.0, description="Relevance / rerank score")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary chunk metadata"
    )


class RAGState(BaseModel):
    """Mutable state object that flows through every LangGraph node.

    Each node reads from and writes to this state so that the graph remains
    stateless between requests.

    Attributes:
        query: The original user question.
        retrieved_docs: Documents returned by the retrieval agent.
        reranked_docs: Subset of documents after reranking.
        optimized_context: Final context string passed to the generator.  
        answer: The generated answer (populated by the generator node).
        evidence: Source snippets cited in the answer.
        error: Optional error message if a node fails gracefully.
    """

    query: str = Field(..., description="Original user question (backward compatible)")
    original_query: str = Field("", description="Original user question (fixed across iterations)")
    current_query: str = Field("", description="Current iteration query used for retrieval")
    query_type: str = Field("simple", description="Query type classified by the router: simple/vague/complex/professional")
    strategy_queries: dict[str, str] = Field(
        default_factory=dict,
        description="Per-strategy optimized queries produced by the query optimizer (strategy → query)",
    )
    optimized_queries: list[str] = Field(
        default_factory=list,
        description="List of search queries produced by the multi-query generator",
    )
    iteration_count: int = Field(0, description="Current iteration count")
    max_iterations: int = Field(3, description="Maximum number of iterative retries")
    context_sufficient: bool = Field(
        False, description="Whether current context is sufficient to answer"
    )
    missing_info: str = Field(
        "", description="What key information is still missing for answering"
    )
    missing_info_history: list[str] = Field(
        default_factory=list,
        description="History of missing information feedback across iterations",
    )
    judge_reason: str = Field(
        "", description="Reasoning from context evaluator for traceability"
    )
    retrieved_docs: list[Document] = Field(
        default_factory=list, description="Documents from the retrieval agent"
    )
    reranked_docs: list[Document] = Field(
        default_factory=list, description="Documents after reranking"
    )
    optimized_context: str = Field(
        "", description="Pruned / summarised context for the generator"
    )
    answer: str = Field("", description="Final generated answer")
    evidence: list[str] = Field(
        default_factory=list,
        description="Source snippets used to generate the answer",
    )
    error: str | None = Field(None, description="Error message if a node failed")
    node_trace: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Ordered list of per-node execution records for audit and replay",
    )


class GenOutput(BaseModel):
    answer: str = Field(..., description="The final generated answer")
    evidence: list[str] = Field(..., description="Source document excerpts used to produce the answer")
    thread_id: str = Field("", description="Checkpoint thread ID; empty when checkpointing is disabled")


class RAGOutput(BaseModel):
    """Structured final output returned to the caller.

    Attributes:
        answer: The final answer to the user's query.
        evidence: List of source document excerpts that support the answer.
        thread_id: The checkpoint thread identifier for this run.  Use it to
            retrieve the full execution trace from the checkpoint store via
            :class:`~flexrag.observability.tracing.CheckpointReader`.  Empty string when
            checkpointing is disabled.
    """

    answer: str = Field(..., description="The final generated answer")
    evidence: list[str] = Field(..., description="Source document excerpts used to produce the answer")
    trace: list[dict] = Field(..., description="轨迹信息，包含每个节点的输入输出等详细信息")
    thread_id: str = Field("", description="Checkpoint thread ID; empty when checkpointing is disabled")


class ContextEvaluation(BaseModel):
    """Structured output returned by the context evaluator."""

    context_sufficient: bool = Field(
        ..., description="True when current context is enough to answer"
    )
    missing_info: str = Field(
        "", description="Missing information summary when context is insufficient"
    )
    judge_reason: str = Field("", description="Short rationale for the judgement")
    accumulated_context: list[str] = Field([""], description="迭代中积累的上下文信息")
