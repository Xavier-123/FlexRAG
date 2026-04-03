"""
FlexRAG – interactive question-answering entry point.

This script demonstrates how to wire the full pipeline together with a
persistent knowledge base and provides an interactive Q&A loop.

Start-up logic
--------------
1. Check whether a FAISS index already exists in ``KNOWLEDGE_PERSIST_DIR``
   (default: ``./knowledge_base``).
2. If it **exists** → load automatically into a retriever.
3. If it does **not** exist → offer two choices:

   * **Build** – point to a local directory of documents (``.txt`` / ``.md``
     / ``.pdf``), chunk them, embed them, and save the index to disk.
   * **Demo** – index five short in-memory paragraphs so you can try the
     pipeline without any external documents.

Prerequisites
-------------
1. Set environment variables (or create a ``.env`` file)
2. Install dependencies::

       pip install -r requirements.txt

Usage
-----
::

    python main.py
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)

from langchain_openai import ChatOpenAI

from flexrag.workflows.pipeline import RAGPipeline
from flexrag.common import Settings, setup_logging
from flexrag.components.pre_retrieval import PreQueryOptimizer, QueryExpander, QueryRewriter, TaskSplitter, \
    TerminologyEnricher
from flexrag.components.retrieval import HybridRetriever, BM25Retriever, MultiVectorRetriever, OpenAILikeEmbedding
from flexrag.components.post_retrieval import PostRetrieval, LLMContextOptimizer, OpenAILikeReranker
from flexrag.components.reasoning import OpenAIGenerator, LLMContextEvaluator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Demo corpus (used when no external document directory is provided)
# ---------------------------------------------------------------------------

_DEMO_CORPUS = [
    (
        "Retrieval-Augmented Generation (RAG) is a technique that combines "
        "information retrieval with large language models.  A retriever first "
        "fetches relevant documents from a knowledge base; the LLM then "
        "generates an answer conditioned on those documents."
    ),
    (
        "LangGraph is an open-source library built on top of LangChain that "
        "lets you model complex LLM workflows as directed graphs (StateGraph).  "
        "Each node in the graph is a Python function that reads from and writes "
        "to a shared state object."
    ),
    (
        "LlamaIndex (formerly GPT Index) is a data framework for LLM "
        "applications.  It provides connectors, indexes, and query engines for "
        "ingesting, structuring, and retrieving data from diverse sources."
    ),
    (
        "vLLM is a high-throughput LLM inference engine that supports "
        "continuous batching and paged attention.  It exposes an OpenAI-"
        "compatible REST API, making it easy to swap in as the backend for "
        "embedding and reranker models."
    ),
    (
        "The strategy design pattern defines a family of algorithms, "
        "encapsulates each one, and makes them interchangeable.  In Python this "
        "is typically implemented with abstract base classes (abc.ABC)."
    ),
]


# ---------------------------------------------------------------------------
# Knowledge-base helpers
# ---------------------------------------------------------------------------


async def build_knowledge_base(directory: str, settings: Settings) -> None:
    """Load files from *directory*, build the FAISS index, and save it."""
    embed_model = OpenAILikeEmbedding(
        model_name=settings.embedding_model,
        base_url=settings.embedding_base_url,
        api_key=settings.embedding_api_key
    )

    builder = MultiVectorRetriever(
        embed_model=embed_model,
        vector_store_type="faiss",
        top_k=settings.top_k_retrieval
    )
    count = await builder.load_files(directory)
    print(f"  Loaded {count} file(s) from '{directory}'.")

    print(
        f"  Building index "
        f"(chunk_size={settings.knowledge_chunk_size}, "
        f"chunk_overlap={settings.knowledge_chunk_overlap}) ..."
    )
    await builder.build_index(
        chunk_size=settings.knowledge_chunk_size,
        chunk_overlap=settings.knowledge_chunk_overlap,
    )

    await builder.save(settings.knowledge_persist_dir)
    print(f"  Knowledge base saved to '{settings.knowledge_persist_dir}'.")


# ---------------------------------------------------------------------------
# Pipeline assembly
# ---------------------------------------------------------------------------


def _build_pipeline(settings: Settings, is_demo: bool = False) -> RAGPipeline:
    """Wire up all pipeline components around *retriever*."""
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
        QueryRewriter(llm=llm),
        QueryExpander(llm=llm),
        TaskSplitter(llm=llm),
    ])

    persist_dir = None if is_demo else settings.knowledge_persist_dir
    bm25_dir = None if is_demo else os.path.join(settings.knowledge_persist_dir, "bm25_index")
    retriever = HybridRetriever(
        retrievers=[
            MultiVectorRetriever(
                embed_model=embed_model,
                vector_store_type=settings.vector_store_type,
                index=None,
                top_k=5,
                persist_dir=persist_dir,
            ),
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

    return RAGPipeline(
        pre_retrieval_optimizer=pre_retrieval_optimizer,
        retriever=retriever,
        post_retrieval_optimizer=post_retrieval_optimizer,
        context_evaluator=context_evaluator,
        generator=generator,
        settings=settings,
    )


# ---------------------------------------------------------------------------
# Interactive Q&A loop
# ---------------------------------------------------------------------------


async def interactive_qa(pipeline: RAGPipeline) -> None:
    """Run an interactive question-answering loop."""
    print("\n" + "=" * 60)
    print("FlexRAG -- Interactive Q&A  (type 'quit' to exit)")
    print("=" * 60)

    while True:
        try:
            query = input("\nQuestion: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not query:
            continue
        if query.lower() in {"quit", "exit", "q"}:
            print("Bye!")
            break

        try:
            output = await pipeline.arun(query)
        except RuntimeError as exc:
            print(f"[ERROR] {exc}")
            continue

        print()
        print(f"Answer  : {output.answer}")
        if output.evidence:
            print("Evidence:")
            for i, snippet in enumerate(output.evidence, 1):
                preview = snippet[:120] + ("..." if len(snippet) > 120 else "")
                print(f"  [{i}] {preview}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def main() -> None:
    """Start FlexRAG: check for a FAISS knowledge base and begin Q&A."""
    settings = Settings()

    # 统一使用内置的 setup_logging
    setup_logging(settings.log_level, settings.log_format)
    logger = logging.getLogger(__name__)

    persist_dir = settings.knowledge_persist_dir

    # ------------------------------------------------------------------ #
    # 1. Check whether a persisted knowledge base already exists         #
    # ------------------------------------------------------------------ #
    if MultiVectorRetriever.index_exists(persist_dir):
        print(f"[INFO] Found existing knowledge base at '{persist_dir}'. Loading ...")
        pipeline = _build_pipeline(settings, is_demo=False)

    else:
        # ---------------------------------------------------------------- #
        # 2. No knowledge base found – offer build or demo                 #
        # ---------------------------------------------------------------- #
        print(f"[INFO] No knowledge base found at '{persist_dir}'.")
        print()
        print("Choose an option:")
        print("  b  -- Build from a local directory of documents")
        print("  d  -- Use built-in demo data (no files needed)")
        print("  q  -- Quit")
        print()

        choice = input("Option [b/d/q]: ").strip().lower()

        if choice == "b":
            directory = input("Enter the path to your documents directory: ").strip()
            if not directory:
                print("[ERROR] No directory specified. Exiting.")
                sys.exit(1)
            print()
            await build_knowledge_base(directory, settings)
            pipeline = _build_pipeline(settings, is_demo=False)

        elif choice == "d":
            print("[INFO] Indexing demo corpus ...")
            pipeline = RAGPipeline.from_settings(settings)
            pipeline.add_documents(
                texts=_DEMO_CORPUS,
                metadatas=[{"source": f"demo_{i}"} for i in range(len(_DEMO_CORPUS))],
            )
            print("[INFO] Demo corpus indexed.")

        else:
            print("Exiting.")
            sys.exit(0)

    # ------------------------------------------------------------------ #
    # 3. Interactive Q&A                                                   #
    # ------------------------------------------------------------------ #
    await interactive_qa(pipeline)


if __name__ == "__main__":
    asyncio.run(main())
