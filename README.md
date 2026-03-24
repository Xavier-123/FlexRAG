# FlexRAG

A modular, enterprise-grade **Retrieval-Augmented Generation** (RAG) system
built on **LangGraph** and **LlamaIndex**.

FlexRAG ships with a full **knowledge-base construction pipeline** – load
local files, chunk them, embed them with a vLLM model, and persist a FAISS
vector index to disk.  On next startup, the index is restored automatically
so your documents are always ready to query.

---

## Features

| Feature | Detail |
|---|---|
| **Persistent FAISS knowledge base** | Build once from `.txt` / `.md` / `.pdf` files, reload on demand |
| **Configurable chunking** | `chunk_size` and `chunk_overlap` settable via env vars or code |
| **vLLM embeddings** | Any OpenAI-compatible embedding endpoint (local or remote) |
| **Four-stage RAG graph** | Retrieve → Rerank → Optimise Context → Generate |
| **Structured output** | Every answer is a validated `RAGOutput(answer, evidence)` Pydantic model |
| **Strategy pattern** | Every component is backed by an ABC; swap any stage without touching the rest |
| **Interactive Q&A loop** | `main.py` detects or builds the knowledge base, then drops into a REPL |

---

## Architecture

```
                ┌──────────────────────────────────┐
                │   Knowledge Base Pipeline        │
                │                                  │
                │  load_files()  ──►  build_index()│
                │       │                   │      │
                │  .txt/.md/.pdf    FAISS + vLLM   │
                │                   embeddings     │
                │                       │          │
                │                   save()         │
                └───────────────────────┬──────────┘
                                        │  persist_dir
                                        ▼
                              faiss_index.bin  +
                              docstore.json  + ...

                ┌─────────────────────────────────┐
                │   RAG Query Pipeline            │
                │                                 │
  User query ──►│  [Retrieve]  LlamaIndex FAISS   │
                │       │                         │
                │  [Rerank]    vLLM cross-encoder │
                │       │                         │
                │  [Optimise]  LLM context filter │
                │       │                         │
                │  [Generate]  LLM structured out │
                └───────┬─────────────────────────┘
                        │
                    RAGOutput
                 { answer, evidence }
```

---

## Project Structure

```
FlexRAG/
├── main.py                              # Interactive Q&A entry point
├── requirements.txt                     # Python dependencies
└── flexrag/
    ├── __init__.py                      # Package root (exports RAGPipeline)
    ├── config.py                        # Pydantic Settings (env vars / .env)
    ├── schema.py                        # Document, RAGState, RAGOutput models
    ├── pipeline.py                      # RAGPipeline – high-level orchestrator
    ├── abstractions/                    # Abstract base classes (Strategy Pattern)
    │   ├── base_retriever.py
    │   ├── base_reranker.py
    │   ├── base_context_optimizer.py
    │   ├── base_generator.py
    │   └── base_knowledge.py           ← knowledge builder ABC
    ├── knowledge/                       ← knowledge builder package
    │   ├── __init__.py
    │   └── faiss_knowledge.py          ← FAISS + LlamaIndex builder
    ├── retrievers/
    │   └── llamaindex_retriever.py      # LlamaIndex VectorStoreIndex + vLLM
    ├── rerankers/
    │   └── vllm_reranker.py             # vLLM cross-encoder reranker
    ├── context_optimizers/
    │   └── llm_context_optimizer.py     # LLM-based context extractor
    ├── generators/
    │   └── openai_generator.py          # Structured output generator
    └── graph/
        ├── nodes.py                     # LangGraph node factories
        └── builder.py                   # StateGraph assembly & compilation
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Graph orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) |
| LLM & generation | OpenAI-compatible API via [langchain-openai](https://pypi.org/project/langchain-openai/) |
| Retrieval & file loading | [LlamaIndex](https://www.llamaindex.ai/) (`SimpleDirectoryReader`, `SentenceSplitter`, `VectorStoreIndex`) |
| Vector store | [FAISS](https://github.com/facebookresearch/faiss) via `llama-index-vector-stores-faiss` |
| Embeddings & reranking | vLLM remote API (OpenAI-compatible) |
| PDF parsing | [pypdf](https://pypi.org/project/pypdf/) |
| Data validation | [Pydantic v2](https://docs.pydantic.dev/) |
| Configuration | [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Xavier-123/FlexRAG.git
cd FlexRAG
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note** – `faiss-cpu` requires a CPU build of FAISS.  If you have a GPU
> and want maximum performance, replace it with `faiss-gpu` in
> `requirements.txt` before installing.

### 3. Configure environment

Create a `.env` file (or export the variables) in the project root:

```dotenv
# --- LLM ---
LLM_BASE_URL=http://localhost:8000/v1
LLM_API_KEY=sk-...
VLLM_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct

# --- Embedding ---
EMBEDDING_BASE_URL=http://localhost:8001/v1/embeddings
EMBEDDING_API_KEY=sk-...
VLLM_EMBEDDING_MODEL=BAAI/bge-large-en-v1.5

# --- Reranker ---
RERANKER_BASE_URL=http://localhost:8002/v1
RERANKER_API_KEY=sk-...
VLLM_RERANKER_MODEL=BAAI/bge-reranker-v2-m3

# --- Knowledge base (optional – shown with defaults) ---
KNOWLEDGE_PERSIST_DIR=./knowledge_base
KNOWLEDGE_CHUNK_SIZE=512
KNOWLEDGE_CHUNK_OVERLAP=50

# --- Pipeline hyper-params (optional) ---
TOP_K_RETRIEVAL=10
TOP_K_RERANK=5
CONTEXT_MAX_TOKENS=3000
```

---

## Quick Start

### Build a knowledge base from your documents

```python
from flexrag.config import Settings
from flexrag.knowledge import FaissKnowledgeBuilder

settings = Settings()

builder = FaissKnowledgeBuilder(
    embed_base_url=settings.embedding_base_url,
    embed_model_name=settings.vllm_embedding_model,
    embed_api_key=settings.embedding_api_key,
)

# Load all .txt / .md / .pdf files from a directory
builder.load_files("./my_documents")

# Chunk and embed (uses settings for chunk_size / chunk_overlap)
builder.build_index(
    chunk_size=settings.knowledge_chunk_size,
    chunk_overlap=settings.knowledge_chunk_overlap,
)

# Persist to disk
builder.save(settings.knowledge_persist_dir)
print("Knowledge base built and saved.")
```

### Load an existing knowledge base and run Q&A

```python
from flexrag.config import Settings
from flexrag.context_optimizers.llm_context_optimizer import LLMContextOptimizer
from flexrag.generators.openai_generator import OpenAIGenerator
from flexrag.pipeline import RAGPipeline
from flexrag.rerankers.vllm_reranker import VLLMReranker
from flexrag.retrievers import LlamaIndexRetriever
from langchain_openai import ChatOpenAI

settings = Settings()

# Load persisted index into a retriever
retriever = LlamaIndexRetriever(
    index=None,
    embed_base_url=settings.embedding_base_url,
    embed_model_name=settings.vllm_embedding_model,
    embed_api_key=settings.embedding_api_key,
)
retriever.load_index(settings.knowledge_persist_dir)

# Build the full pipeline with the retriever
reranker = VLLMReranker(
    base_url=settings.reranker_base_url,
    model=settings.vllm_reranker_model,
    api_key=settings.reranker_api_key,
)
llm = ChatOpenAI(
    model=settings.vllm_llm_model,
    api_key=settings.llm_api_key,
    base_url=settings.llm_base_url,
    temperature=0.0,
)
pipeline = RAGPipeline(
    retriever=retriever,
    reranker=reranker,
    context_optimizer=LLMContextOptimizer(llm=llm),
    generator=OpenAIGenerator(
        model=settings.vllm_llm_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
    ),
    settings=settings,
)

output = pipeline.run("What is Retrieval-Augmented Generation?")
print(output.answer)
print(output.evidence)
```

### Run the interactive entry point

```bash
python main.py
```

On first run you will see:

```
[INFO] No knowledge base found at './knowledge_base'.

Choose an option:
  b  -- Build from a local directory of documents
  d  -- Use built-in demo data (no files needed)
  q  -- Quit

Option [b/d/q]:
```

* Enter **`b`** and provide a directory path to index your own documents.
* Enter **`d`** to spin up instantly with five built-in demo paragraphs.

On subsequent runs the saved index is loaded automatically and the Q&A loop
starts immediately.

---

## Extending FlexRAG

Every component exposes a thin ABC.  To swap a stage:

1. Subclass the appropriate abstract class in `flexrag/abstractions/`.
2. Implement the single required method.
3. Pass your implementation to `RAGPipeline(...)`.

```python
from flexrag.abstractions import BaseReranker
from flexrag.schema import Document

class MyReranker(BaseReranker):
    def rerank(self, query: str, documents: list[Document], top_k: int) -> list[Document]:
        # custom scoring logic
        return sorted(documents, key=lambda d: len(d.text), reverse=True)[:top_k]
```

Similarly, to build a custom knowledge backend subclass `BaseKnowledgeBuilder`
and implement `load_files`, `build_index`, `save`, and `index_exists`.

---

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

All tests run fully offline using `unittest.mock` – no running services
required.

---

## Environment Variable Reference

The table below shows the **built-in defaults** from `flexrag/config.py`.
Override any value in your `.env` file or via environment variables.

| Variable | Built-in default | Description |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:8018/v1` | LLM serving endpoint |
| `LLM_API_KEY` | `sk-xxxx` | LLM API key |
| `EMBEDDING_BASE_URL` | `http://localhost:8018/v1` | Embedding serving endpoint |
| `EMBEDDING_API_KEY` | `sk-xxxx` | Embedding API key |
| `RERANKER_BASE_URL` | `http://localhost:8018/v1` | Reranker serving endpoint |
| `RERANKER_API_KEY` | `sk-xxxx` | Reranker API key |
| `VLLM_LLM_MODEL` | `Qwen/Qwen2.5-7B-Instruct` | Chat model name |
| `VLLM_EMBEDDING_MODEL` | `BAAI/bge-large-en-v1.5` | Embedding model name |
| `VLLM_RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | Reranker model name |
| `KNOWLEDGE_PERSIST_DIR` | `./knowledge_base` | FAISS index storage directory |
| `KNOWLEDGE_CHUNK_SIZE` | `512` | Max tokens per document chunk |
| `KNOWLEDGE_CHUNK_OVERLAP` | `50` | Token overlap between chunks |
| `TOP_K_RETRIEVAL` | `10` | Docs to retrieve before reranking |
| `TOP_K_RERANK` | `5` | Docs to keep after reranking |
| `CONTEXT_MAX_TOKENS` | `3000` | Token budget for optimised context |
