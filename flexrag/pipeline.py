"""
High-level RAG pipeline entry point.

:class:`RAGPipeline` wires together all components, builds the LangGraph
graph, and exposes a single :meth:`~RAGPipeline.run` method for end-users.

Typical usage::

    from flexrag import RAGPipeline
    from flexrag.config import Settings

    cfg = Settings()
    pipeline = RAGPipeline.from_settings(cfg)
    pipeline.add_documents(["RAG is ...", "LangGraph is ..."])
    output = pipeline.run("What is RAG?")
    print(output.answer)
    print(output.evidence)
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_openai import ChatOpenAI

from flexrag.abstractions.base_retriever import BaseRetriever
from flexrag.config import Settings
from flexrag.context_optimizers.llm_context_optimizer import LLMContextOptimizer
from flexrag.evaluators.llm_context_evaluator import LLMContextEvaluator
from flexrag.generators.openai_generator import OpenAIGenerator
from flexrag.graph.builder import build_rag_graph
from flexrag.query_optimizers.llm_query_optimizer import LLMQueryOptimizer
from flexrag.rerankers.vllm_reranker import VLLMReranker
from flexrag.retrievers.llamaindex_retriever import LlamaIndexRetriever
from flexrag.schema import RAGOutput

logger = logging.getLogger(__name__)


class RAGPipeline:
    """End-to-end modular RAG pipeline.

    Orchestrates the full retrieval-augmented generation workflow:

    1. **Retrieve** – find relevant document chunks via LlamaIndex.
    2. **Rerank** – score candidates with a vLLM cross-encoder.
    3. **Optimize Context** – extract key passages using a vLLM-hosted LLM.
    4. **Generate** – produce a structured answer via a vLLM-hosted LLM.

    All components are decoupled through abstract base classes (strategy
    pattern) so they can be swapped individually.

    Args:
        retriever: :class:`~flexrag.retrievers.LlamaIndexRetriever` instance.
        reranker: :class:`~flexrag.rerankers.VLLMReranker` instance.
        context_optimizer: :class:`~flexrag.context_optimizers.LLMContextOptimizer`.        generator: :class:`~flexrag.generators.OpenAIGenerator` instance.
        settings: :class:`~flexrag.config.Settings` driving numeric hyper-params.

    See Also:
        :meth:`from_settings` – convenience factory that reads everything from
        environment variables / a ``.env`` file.
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        reranker: VLLMReranker,
        context_optimizer: LLMContextOptimizer,
        query_optimizer: LLMQueryOptimizer,
        context_evaluator: LLMContextEvaluator,
        generator: OpenAIGenerator,
        settings: Settings,
    ) -> None:
        self._retriever = retriever
        self._settings = settings

        self._graph = build_rag_graph(
            retriever=retriever,
            reranker=reranker,
            context_optimizer=context_optimizer,
            query_optimizer=query_optimizer,
            context_evaluator=context_evaluator,
            generator=generator,
            top_k_retrieval=settings.top_k_retrieval,
            top_k_rerank=settings.top_k_rerank,
            context_max_tokens=settings.context_max_tokens,
            draw_image_path=settings.draw_image_path,
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "RAGPipeline":
        """Construct a :class:`RAGPipeline` from :class:`~flexrag.config.Settings`.

        All components are built using the values in *settings* (which default
        to environment variables / ``.env`` file).

        Args:
            settings: Optional pre-built settings object.  When ``None`` a
                new :class:`~flexrag.config.Settings` instance is created,
                reading from the environment.

        Returns:
            A fully initialised :class:`RAGPipeline`.
        """
        if settings is None:
            settings = Settings()

        # -- Retriever (LlamaIndex + vLLM embeddings) --
        retriever = LlamaIndexRetriever(
            index=None,
            embed_base_url=settings.embedding_base_url,
            embed_model_name=settings.vllm_embedding_model,
            embed_api_key=settings.embedding_api_key,
        )

        # -- Reranker (vLLM cross-encoder) --
        reranker = VLLMReranker(
            base_url=settings.reranker_base_url,
            model=settings.vllm_reranker_model,
            api_key=settings.reranker_api_key,
        )

        # -- Context Optimiser (LLM via vLLM) --
        llm = ChatOpenAI(
            model=settings.vllm_llm_model,
            api_key=settings.llm_api_key,  # type: ignore[arg-type]
            base_url=settings.llm_base_url,
            temperature=0.0,
        )
        context_optimizer = LLMContextOptimizer(llm=llm)
        query_optimizer = LLMQueryOptimizer(llm=llm)
        context_evaluator = LLMContextEvaluator(llm=llm)

        # -- Generator (vLLM Structured Output) --
        generator = OpenAIGenerator(
            model=settings.vllm_llm_model,
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
        )

        return cls(
            retriever=retriever,
            reranker=reranker,
            context_optimizer=context_optimizer,
            query_optimizer=query_optimizer,
            context_evaluator=context_evaluator,
            generator=generator,
            settings=settings,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_documents(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """Index a list of text chunks so they can be retrieved later.

        Args:
            texts: Raw text chunks to add to the retriever's index.
            metadatas: Optional metadata dicts aligned with *texts*.
        """
        self._retriever.add_documents(texts, metadatas=metadatas)
        logger.info("Indexed %d document(s)", len(texts))

    def run(self, query: str) -> RAGOutput:
        """Execute the full RAG pipeline for *query* (synchronous wrapper).

        Prefer :meth:`arun` in async code. This convenience wrapper calls
        :func:`asyncio.run` and therefore **cannot** be used inside an
        already-running event loop.

        Args:
            query: The user's question.

        Returns:
            A :class:`~flexrag.schema.RAGOutput` containing ``answer`` and
            ``evidence``.

        Raises:
            RuntimeError: If any pipeline node reports an unrecoverable error.
        """
        import asyncio
        return asyncio.run(self.arun(query))

    async def arun(self, query: str) -> RAGOutput:
        """Execute the full RAG pipeline for *query* asynchronously.

        Args:
            query: The user's question.

        Returns:
            A :class:`~flexrag.schema.RAGOutput` containing ``answer`` and
            ``evidence``.

        Raises:
            RuntimeError: If any pipeline node reports an unrecoverable error.
        """
        logger.info("Pipeline started – query: %r", query)
        result: dict[str, Any] = await self._graph.ainvoke(
            {
                "query": query,
                "original_query": query,
                "current_query": "",
                "iteration_count": 0,
                "max_iterations": self._settings.max_iterations,
                "missing_info": "",
                "missing_info_history": [],
                "accumulated_context": [],
            }
        )

        if error := result.get("error"):
            raise RuntimeError(f"RAG pipeline error: {error}")

        return RAGOutput(
            answer=result.get("answer", ""),
            evidence=result.get("evidence", []),
        )
