"""
High-level RAG pipeline entry point.

:class:`RAGPipeline` wires together all components, builds the LangGraph
graph, and exposes a single :meth:`~RAGPipeline.run` method for end-users.

Typical usage::

    from flexrag import RAGPipeline
    from flexrag.core.config import Settings

    cfg = Settings()
    pipeline = RAGPipeline.from_settings(cfg)
    pipeline.add_documents(["RAG is ...", "LangGraph is ..."])
    output = pipeline.run("What is RAG?")
    print(output.answer)
    print(output.evidence)
    # Retrieve the full execution trace using output.thread_id:
    #   from flexrag.observability.tracing import CheckpointReader
    #   reader = CheckpointReader(cfg.checkpoint_db_path)
    #   reader.print_run(output.thread_id)
"""

from __future__ import annotations

import logging
import sqlite3
import uuid
from typing import Any, Optional

from langchain_openai import ChatOpenAI

from flexrag.core.abstractions import BaseRetriever
from flexrag.core.config import Settings
from flexrag.components.post_retrieval.llm_context_optimizer import LLMContextOptimizer
from flexrag.components.judges.llm_context_evaluator import LLMContextEvaluator
from flexrag.components.generation.openai_generator import OpenAIGenerator
from flexrag.workflows.graph.builder import build_rag_graph
from flexrag.components.query_transform.llm_query_optimizer import LLMQueryOptimizer
from flexrag.components.post_retrieval.vllm_reranker import VLLMReranker
from flexrag.components.retrieval.llamaindex_retriever import LlamaIndexRetriever
from flexrag.core.schema import RAGOutput

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
        retriever: :class:`~flexrag.components.retrieval.LlamaIndexRetriever` instance.
        reranker: :class:`~flexrag.components.post_retrieval.VLLMReranker` instance.
        context_optimizer: :class:`~flexrag.components.post_retrieval.LLMContextOptimizer`.        generator: :class:`~flexrag.components.generation.OpenAIGenerator` instance.
        settings: :class:`~flexrag.core.config.Settings` driving numeric hyper-params.
        checkpoint_db_path: Optional path to a SQLite database file used to
            persist LangGraph state checkpoints after every node.  When set,
            every :meth:`arun` call saves a full per-node trace that can be
            replayed with :class:`~flexrag.observability.tracing.CheckpointReader`.

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
        checkpoint_db_path: Optional[str] = None,
    ) -> None:
        self._retriever = retriever
        self._settings = settings
        self._checkpoint_db_path = checkpoint_db_path

        # --- Checkpoint saver (optional) ---
        # SqliteSaver is used because it works correctly with both the
        # synchronous run() wrapper and the async arun() entry point.
        # check_same_thread=False is required so the connection can be reused
        # across asyncio tasks.
        self._checkpoint_conn: Optional[sqlite3.Connection] = None
        checkpointer = None
        if checkpoint_db_path:
            try:
                from langgraph.checkpoint.sqlite import SqliteSaver

                self._checkpoint_conn = sqlite3.connect(
                    checkpoint_db_path, check_same_thread=False
                )
                # check_same_thread=False is intentional: SqliteSaver
                # serialises its own writes internally, and asyncio requires
                # sharing the connection across coroutines on the same thread.
                checkpointer = SqliteSaver(self._checkpoint_conn)
                logger.info(
                    "Checkpoint persistence enabled – database: %r", checkpoint_db_path
                )
            except ImportError:
                logger.warning(
                    "langgraph-checkpoint-sqlite is not installed; "
                    "checkpointing is disabled. "
                    "Install it with: pip install langgraph-checkpoint-sqlite"
                )

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
            checkpointer=checkpointer,
        )

    def close(self) -> None:
        """Close the underlying checkpoint database connection, if open."""
        if self._checkpoint_conn is not None:
            self._checkpoint_conn.close()
            self._checkpoint_conn = None

    def __del__(self) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "RAGPipeline":
        """Construct a :class:`RAGPipeline` from :class:`~flexrag.core.config.Settings`.

        All components are built using the values in *settings* (which default
        to environment variables / ``.env`` file).

        Args:
            settings: Optional pre-built settings object.  When ``None`` a
                new :class:`~flexrag.core.config.Settings` instance is created,
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
            checkpoint_db_path=settings.checkpoint_db_path,
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

    def run(self, query: str, thread_id: Optional[str] = None):
        """Execute the full RAG pipeline for *query* (synchronous wrapper).

        Prefer :meth:`arun` in async code. This convenience wrapper calls
        :func:`asyncio.run` and therefore **cannot** be used inside an
        already-running event loop.

        Args:
            query: The user's question.
            thread_id: Optional checkpoint thread identifier.  When
                checkpointing is enabled, pass a stable ID (e.g. a session ID)
                to group multiple queries into one conversation thread.  When
                ``None`` a fresh UUID is generated for each call.

        Returns:
            A :class:`~flexrag.core.schema.RAGOutput` containing ``answer``,
            ``evidence``, and ``thread_id``.

        Raises:
            RuntimeError: If any pipeline node reports an unrecoverable error.
        """
        import asyncio
        return asyncio.run(self.arun(query, thread_id=thread_id))

    async def arun(self, query: str, thread_id: Optional[str] = None):
        """Execute the full RAG pipeline for *query* asynchronously.

        Args:
            query: The user's question.
            thread_id: Optional checkpoint thread identifier.  When
                checkpointing is enabled, pass a stable ID (e.g. a session ID)
                to group multiple queries into one conversation thread.  When
                ``None`` a fresh UUID is generated for each call so that every
                invocation is stored independently.

        Returns:
            A :class:`~flexrag.core.schema.RAGOutput` containing ``answer``,
            ``evidence``, and ``thread_id`` (the ID used for this run,
            regardless of whether checkpointing is active).

        Raises:
            RuntimeError: If any pipeline node reports an unrecoverable error.
        """
        # Each run gets a unique thread_id so checkpoints are stored per query.
        # Callers may supply a stable ID to create a multi-turn conversation
        # thread in the checkpoint store.
        run_thread_id = thread_id or str(uuid.uuid4())

        logger.info("Pipeline started – query: %r  thread_id: %s", query, run_thread_id)

        # LangGraph reads the thread_id from config["configurable"]["thread_id"]
        # to namespace all checkpoints for this invocation.
        config: dict[str, Any] = {"configurable": {"thread_id": run_thread_id}}

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
                "node_trace": [],
            },
            config=config,
        )

        if error := result.get("error"):
            raise RuntimeError(f"RAG pipeline error: {error}")

        return RAGOutput(
            answer=result.get("answer", ""),
            evidence=result.get("evidence", []),
            trace=result.get("node_trace", []),
            thread_id=run_thread_id,
        )
