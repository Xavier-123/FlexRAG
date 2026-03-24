"""
FlexRAG – runnable example / quick-start.

This script demonstrates how to wire the pipeline together, index a small
corpus, and ask a question.

Prerequisites
-------------
1. Set environment variables (or create a ``.env`` file)::

       OPENAI_API_KEY=sk-...
       VLLM_BASE_URL=http://localhost:8000
       VLLM_EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
       VLLM_RERANKER_MODEL=BAAI/bge-reranker-v2-m3

2. Install dependencies::

       pip install -r requirements.txt

Usage
-----
::

    python main.py
"""

from __future__ import annotations

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)

from flexrag import RAGPipeline
from flexrag.config import Settings


def main() -> None:
    """Demonstrate the FlexRAG pipeline end-to-end."""
    # --------------------------------------------------------------------
    # 1. Load settings (from .env or environment variables)
    # --------------------------------------------------------------------
    settings = Settings()  # reads OPENAI_API_KEY, VLLM_BASE_URL, etc.

    # --------------------------------------------------------------------
    # 2. Build the pipeline using the factory method
    # --------------------------------------------------------------------
    pipeline = RAGPipeline.from_settings(settings)

    # --------------------------------------------------------------------
    # 3. Index a small in-memory corpus
    # --------------------------------------------------------------------
    corpus = [
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

    pipeline.add_documents(
        texts=corpus,
        metadatas=[{"source": f"doc_{i}"} for i in range(len(corpus))],
    )

    # --------------------------------------------------------------------
    # 4. Ask a question
    # --------------------------------------------------------------------
    query = "What is Retrieval-Augmented Generation and how does it work?"
    output = pipeline.run(query)

    print("\n" + "=" * 60)
    print(f"Query   : {query}")
    print("=" * 60)
    print(f"Answer  : {output.answer}")
    print("-" * 60)
    print("Evidence:")
    for i, snippet in enumerate(output.evidence, 1):
        print(f"  [{i}] {snippet[:120]}{'...' if len(snippet) > 120 else ''}")
    print("=" * 60)


if __name__ == "__main__":
    main()
