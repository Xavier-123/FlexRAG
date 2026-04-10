from __future__ import annotations

import logging
import sqlite3
import os
import uuid
from typing import Any, Optional

from langchain_openai import ChatOpenAI

from flexrag.common import RAGOutput, Settings
from flexrag.workflows.builder import build_rag_graph
from flexrag.components.pre_retrieval import PreQueryOptimizer, QueryRewriter, QueryExpander, TaskSplitter, TerminologyEnricher
from flexrag.components.retrieval import BaseFlexRetriever, HybridRetriever, BM25Retriever, GraphRetriever, MultiVectorRetriever, OpenAILikeEmbedding
from flexrag.components.post_retrieval import PostRetrieval, LLMContextOptimizer, OpenAILikeReranker
from flexrag.components.reasoning import OpenAIGenerator, LLMContextEvaluator


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# State keys to extract per node for input display in the UI.
# Only a curated subset of the full LangGraph state is forwarded so that the
# streaming events stay lightweight.
# ---------------------------------------------------------------------------
_NODE_INPUT_KEYS: dict[str, list[str]] = {
    "pre_retrieval_optimizer": [
        "original_query", "missing_info", "iteration_count", "missing_info_history",
    ],
    "retrieve": ["optimized_queries", "original_query"],
    "post_retrieval_optimizer": ["retrieved_docs", "original_query"],
    "context_evaluator": ["original_query", "optimized_context", "accumulated_context"],
    "generate": ["original_query", "accumulated_context", "optimized_context"],
}


class RAGPipeline:
    """End-to-end modular RAG pipeline. Orchestrates the full retrieval-augmented generation workflow:

    Args:
        retriever: :class:`~flexrag.components.retrieval.LlamaIndexRetriever` instance.
        settings: :class:`~flexrag.common.config.Settings` driving numeric hyper-params.
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
            pre_retrieval_optimizer: PreQueryOptimizer,
            retriever: BaseFlexRetriever,
            post_retrieval_optimizer: PostRetrieval,
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
            pre_retrieval_optimizer=pre_retrieval_optimizer,
            retriever=retriever,
            post_retrieval_optimizer=post_retrieval_optimizer,
            context_evaluator=context_evaluator,
            generator=generator,
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
        """Construct a :class:`RAGPipeline` from :class:`~flexrag.common.config.Settings`.

        All components are built using the values in *settings* (which default
        to environment variables / ``.env`` file).

        Args:
            settings: Optional pre-built settings object.  When ``None`` a
                new :class:`~flexrag.common.config.Settings` instance is created,
                reading from the environment.

        Returns:
            A fully initialised :class:`RAGPipeline`.
        """
        if settings is None:
            settings = Settings()

        llm = ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.llm_api_key,  # type: ignore[arg-type]
            base_url=settings.llm_base_url,
            temperature=0.0,
        )

        embed_model = OpenAILikeEmbedding(
            model_name=settings.embedding_model,
            base_url=settings.embedding_base_url,
            api_key=settings.embedding_api_key
        )

        pre_retrieval_optimizer = PreQueryOptimizer([
            # QueryRewriter(llm=llm),
            # QueryExpander(llm=llm),
            # TaskSplitter(llm=llm),
        ])

        persist_dir = settings.knowledge_persist_dir
        bm25_dir = os.path.join(settings.knowledge_persist_dir, "bm25_index")
        retriever = HybridRetriever(
            retrievers=[
                # MultiVectorRetriever(
                #     index=None,
                #     embed_model=embed_model,
                #     top_k=5,
                #     persist_dir=persist_dir,
                # ),
                BM25Retriever(
                    top_k=5,
                    persist_dir=bm25_dir,
                )
            ]
        )

        post_retrieval_optimizer = PostRetrieval([
            OpenAILikeReranker(
                base_url=settings.reranker_base_url,
                model=settings.reranker_model,
                api_key=settings.reranker_api_key,
                top_k=5
            ),
            LLMContextOptimizer(llm=llm)
        ])

        context_evaluator = LLMContextEvaluator(llm=llm)
        generator = OpenAIGenerator(
            model=settings.llm_model,
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
        )

        return cls(
            retriever=retriever,
            pre_retrieval_optimizer=pre_retrieval_optimizer,
            post_retrieval_optimizer=post_retrieval_optimizer,
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
            A :class:`~flexrag.common.schema.RAGOutput` containing ``answer``,
            ``evidence``, and ``thread_id``.

        Raises:
            RuntimeError: If any pipeline node reports an unrecoverable error.
        """
        import asyncio
        return asyncio.run(self.arun(query, thread_id=thread_id))

    async def astream_run(self, query: str, thread_id: Optional[str] = None):
        """Execute the RAG pipeline with real-time node-level streaming events.

        Uses LangGraph's ``astream_events`` to emit a progress event each time a
        pipeline node starts or finishes, enabling live UI updates without waiting
        for the entire workflow to complete.

        Yields:
            dict: One of the following event shapes:

            * ``{"type": "node_start", "node": str}``
              Emitted when a LangGraph node begins execution.
            * ``{"type": "node_end", "node": str}``
              Emitted when a LangGraph node finishes execution.
            * ``{"type": "result", "answer": str, "evidence": list[str], "thread_id": str}``
              Emitted once after all nodes complete – carries the final answer.
            * ``{"type": "error", "message": str}``
              Emitted on an unrecoverable error; no ``result`` event follows.

        Args:
            query: The user's question.
            thread_id: Optional checkpoint thread identifier.  A fresh UUID is
                generated per call when ``None``.
        """
        run_thread_id = thread_id or str(uuid.uuid4())
        config: dict[str, Any] = {"configurable": {"thread_id": run_thread_id}}

        initial_state: dict[str, Any] = {
            "query": query,
            "original_query": query,
            "current_queries": {},
            "optimized_queries": [],
            "iteration_count": 0,
            "max_iterations": self._settings.max_iterations,
            "missing_info": "",
            "missing_info_history": [],
            "accumulated_context": [],
            "node_trace": [],
        }

        answer: str = ""
        evidence: list[str] = []
        error_msg: Optional[str] = None

        try:
            async for event in self._graph.astream_events(
                initial_state, config=config, version="v2"
            ):
                event_type: str = event.get("event", "")
                metadata: dict = event.get("metadata", {})
                node_name: Optional[str] = metadata.get("langgraph_node")

                # Only process events that belong to a named graph node.
                if not node_name:
                    continue

                if event_type == "on_chain_start":
                    raw_input = event.get("data", {}).get("input", {})
                    keys = _NODE_INPUT_KEYS.get(node_name, [])
                    node_input = (
                        {k: raw_input[k] for k in keys if k in raw_input}
                        if isinstance(raw_input, dict)
                        else {}
                    )
                    yield {"type": "node_start", "node": node_name, "input": node_input}

                elif event_type == "on_chain_end":
                    output = event.get("data", {}).get("output", {})
                    if isinstance(output, dict):
                        if output.get("error"):
                            error_msg = output["error"]
                        if "answer" in output:
                            answer = output["answer"]
                        if "evidence" in output:
                            evidence = output["evidence"]
                    # Strip the internal node_trace list to keep events compact.
                    node_output: dict = {}
                    if isinstance(output, dict):
                        node_output = {k: v for k, v in output.items() if k != "node_trace"}
                    yield {"type": "node_end", "node": node_name, "output": node_output}

        except Exception as exc:  # noqa: BLE001
            logger.exception("Streaming pipeline error: %s", exc)
            yield {"type": "error", "message": str(exc)}
            return

        if error_msg:
            yield {"type": "error", "message": f"RAG pipeline error: {error_msg}"}
            return

        yield {
            "type": "result",
            "answer": answer,
            "evidence": evidence,
            "thread_id": run_thread_id,
        }

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
            A :class:`~flexrag.common.schema.RAGOutput` containing ``answer``,
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
                "current_queries": {},
                "optimized_queries": [],
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
