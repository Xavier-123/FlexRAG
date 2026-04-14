<div align="center">

# 🤖 FlexRAG

**A Highly Decoupled Agentic RAG System | 强解耦 Agentic RAG 系统**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-≥0.2.0-orange)](https://github.com/langchain-ai/langgraph)
[![LlamaIndex](https://img.shields.io/badge/LlamaIndex-≥0.10.0-green)](https://www.llamaindex.ai/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

[English](#english) | [中文](#chinese)

</div>

---

<a name="english"></a>

## Overview

**FlexRAG** is a modular, production-oriented **Retrieval-Augmented Generation (RAG)** framework built on top of [LangGraph](https://github.com/langchain-ai/langgraph) and [LlamaIndex](https://www.llamaindex.ai/). Its defining characteristic is **strong decoupling**: every pipeline stage is governed by an abstract base class (ABC), allowing you to swap any component—retriever, reranker, context optimizer, or generator—without touching the rest of the system.

The framework ships with a complete, end-to-end workflow:
- **Knowledge-base construction** (document ingestion → chunking → embedding → FAISS / BM25 / knowledge-graph indexing)
- **Agentic iterative retrieval** via a LangGraph `StateGraph` that re-queries until context is sufficient
- **Gradio Web UI** with multi-knowledge-base switching
- **Offline batch evaluation** (Exact Match, Char-F1, Recall@k)

---

## Table of Contents

- [Architecture](#architecture)
- [Core Features](#core-features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Quick Start](#quick-start)
- [Scripts](#scripts)
- [Extending FlexRAG](#extending-flexrag)
- [Running Tests](#running-tests)
- [Environment Variables Reference](#environment-variables-reference)
- [Contributing](#contributing)
- [License](#license)

---

## Architecture

FlexRAG assembles a five-node directed graph compiled by LangGraph:

```
User Query
    │
    ▼
┌─────────────────────┐
│  PreQueryOptimizer  │  Query rewriting / expansion / task splitting
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│      Retrieve       │  Dense (FAISS) + Sparse (BM25) + Graph retrieval
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  PostRetrieval      │  Cross-encoder reranking + LLM context pruning
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  ContextEvaluator   │  LLM judge: is context sufficient?
└────┬────────────────┘
     │                  ╔══ NO (iteration < MAX_ITERATIONS) ══╗
     │                  ║                                     ║
     │ YES              ╚═════════════════════════════════════╝
     ▼                                (loop back to PreQueryOptimizer)
┌─────────────────────┐
│      Generate       │  Structured JSON output (answer + evidence)
└─────────────────────┘
```

Each node communicates through a shared `_GraphState` TypedDict. The `ContextEvaluator` node drives the agentic loop: when it finds the context insufficient and the iteration budget is not exhausted, it routes back to `PreQueryOptimizer` with a `missing_info` hint, triggering a refined retrieval cycle.

---

## Core Features

| Feature | Description |
|---|---|
| 🔌 **Strategy-pattern decoupling** | Every stage (pre-retrieval, retrieval, post-retrieval, reasoning) has an ABC. Swap any component without touching the orchestrator. |
| 🔄 **Agentic iterative loop** | LangGraph `StateGraph` with conditional routing. Retrieves again when context is insufficient, up to `MAX_ITERATIONS` rounds. |
| 🗄️ **Hybrid retrieval** | Dense (FAISS / Chroma / Milvus via LlamaIndex), Sparse (BM25 + jieba Chinese tokenizer), and Graph (LlamaIndex `PropertyGraphIndex`) — fused by `HybridRetriever`. |
| 📚 **Full KB build pipeline** | CLI script builds dense (FAISS exact & approximate, Milvus), sparse (BM25), and graph indexes from `.txt` / `.md` / `.pdf` / `.json` files. |
| 🖥️ **Gradio Web UI** | Multi-knowledge-base dropdown, live node-level streaming via `astream_events`, and per-node execution trace panel. |
| 📊 **Offline evaluation** | `batch_run.py` for async concurrent inference; `eval_rag.py` for EM, Char-F1, Recall@k metrics. |
| 🧾 **Checkpoint persistence** | Optional SQLite-backed LangGraph checkpoints for full per-run replay and audit. |

---

## Tech Stack

| Category | Library | Role |
|---|---|---|
| Graph orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) ≥ 0.2.0 | Agentic state machine, conditional routing, SQLite checkpoints |
| LLM calls | [langchain-openai](https://pypi.org/project/langchain-openai/) | OpenAI-compatible endpoints (vLLM, ModelScope, OpenAI, etc.) |
| Retrieval framework | [LlamaIndex](https://www.llamaindex.ai/) ≥ 0.10.0 | Document ingestion, chunking, vector/graph indexes |
| Sparse retrieval | llama-index-retrievers-bm25 + jieba | BM25Okapi with Chinese tokenization support |
| Vector store | [FAISS](https://github.com/facebookresearch/faiss) via llama-index-vector-stores-faiss | Exact (IndexFlatL2) and approximate (HNSWFlat) indexes |
| Data validation | [Pydantic v2](https://docs.pydantic.dev/) + pydantic-settings | Schema models and env-based config |
| Async HTTP | [httpx](https://www.python-httpx.org/) | Async calls to vLLM reranker / embedding endpoints |
| Web UI | [Gradio](https://www.gradio.app/) | Chat interface with streaming support |
| PDF parsing | pypdf | Text extraction from PDF files |

---

## Project Structure

```
FlexRAG/
├── main.py                              # CLI interactive Q&A entry point
├── web_UI.py                            # Gradio Web UI (port 7860)
├── requirements.txt                     # Python dependencies
├── .env.example                         # Template for environment variables
├── scripts/
│   ├── build_knowledge_base.py          # Knowledge-base build CLI (dense/sparse/graph)
│   ├── batch_run.py                     # Async batch inference with concurrency control
│   └── eval_rag.py                      # Offline evaluation (EM / F1 / Recall@k)
└── flexrag/
    ├── __init__.py                      # Public exports: RAGPipeline, RAGOutput, RAGState
    ├── common/
    │   ├── config.py                    # Pydantic Settings (reads .env / env vars)
    │   ├── schema.py                    # Data models: Document, RAGState, RAGOutput
    │   ├── logging.py                   # Global logging setup
    │   └── exceptions.py               # Custom exception hierarchy
    ├── components/
    │   ├── pre_retrieval/               # Pre-retrieval query optimizers (pluggable)
    │   │   ├── query_rewriter.py        # LLM-based query rewriting
    │   │   ├── query_expander.py        # HyDE-style query expansion
    │   │   ├── task_splitter.py         # Complex question decomposition
    │   │   └── terminology_enricher.py  # Domain terminology enrichment
    │   ├── retrieval/                   # Retrievers (pluggable)
    │   │   ├── multi_vector_retriever.py # Dense vector retrieval (FAISS/Chroma/Milvus)
    │   │   ├── bm25_retriever.py        # Sparse BM25 retrieval
    │   │   ├── graph_retriever.py       # Knowledge-graph retrieval
    │   │   └── retrieval_opt.py         # HybridRetriever (multi-source fusion)
    │   ├── post_retrieval/              # Post-retrieval processors (pluggable)
    │   │   ├── reranker.py              # OpenAI-compatible cross-encoder reranker
    │   │   └── context_optimizer.py     # LLM-based context pruning and extraction
    │   ├── reasoning/                   # Reasoning components (pluggable)
    │   │   ├── context_evaluator.py     # LLM judge for Agentic loop routing
    │   │   └── generator.py             # Structured JSON answer generator
    │   └── evaluate/
    │       └── metrics/                 # EM / Char-F1 / Recall@k implementations
    └── workflows/
        ├── pipeline.py                  # RAGPipeline: high-level orchestrator (arun/run/astream_run)
        ├── builder.py                   # LangGraph StateGraph assembly and compilation
        └── nodes.py                     # Node factory functions
```

---

## Installation

### Prerequisites

| Requirement | Version |
|---|---|
| Python | ≥ 3.10 |
| LLM serving endpoint | Any OpenAI-compatible chat endpoint (vLLM, ModelScope, OpenAI, etc.) |
| Embedding endpoint | Any OpenAI-compatible embedding endpoint |
| Reranker endpoint | Any OpenAI-compatible cross-encoder endpoint (optional) |

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/Xavier-123/FlexRAG.git
cd FlexRAG

# 2. Install dependencies
pip install -r requirements.txt
```

> **GPU acceleration:** Replace `faiss-cpu` with `faiss-gpu` in `requirements.txt` before installing.

---

## Configuration

Copy the template and fill in your values:

```bash
cp .env.example .env   # or create .env manually
```

**Never commit `.env` to version control.** All fields have built-in defaults; override only what you need.

```dotenv
# ── LLM ──────────────────────────────────────────────────────────────────────
LLM_BASE_URL=http://localhost:8000/v1
LLM_API_KEY=your-llm-key
LLM_MODEL=Qwen/Qwen2.5-7B-Instruct

# ── Embedding ─────────────────────────────────────────────────────────────────
EMBEDDING_BASE_URL=http://localhost:8001/v1
EMBEDDING_API_KEY=your-embedding-key
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5

# ── Reranker (optional) ───────────────────────────────────────────────────────
RERANKER_BASE_URL=http://localhost:8002/v1
RERANKER_API_KEY=your-reranker-key
RERANKER_MODEL=BAAI/bge-reranker-v2-m3

# ── Knowledge Base ────────────────────────────────────────────────────────────
KNOWLEDGE_PERSIST_DIR=./data/knowledge_persist_dir
KNOWLEDGE_CHUNK_SIZE=512
KNOWLEDGE_CHUNK_OVERLAP=50

# ── Pipeline hyperparameters ──────────────────────────────────────────────────
TOP_K_RETRIEVAL=10          # documents retrieved before reranking
TOP_K_RERANK=5              # documents kept after reranking
CONTEXT_MAX_TOKENS=8000     # token budget passed to the generator
MAX_ITERATIONS=3            # max agentic re-retrieval iterations

# ── Optional features ─────────────────────────────────────────────────────────
# CHECKPOINT_DB_PATH=./data/checkpoints.db    # SQLite checkpoint store
# DRAW_IMAGE_PATH=./architecture.png          # save LangGraph diagram
LOG_LEVEL=INFO
```

---

## Quick Start

### Option A — CLI interactive Q&A

```bash
python main.py
```

On first run you will be prompted to build a knowledge base or use the built-in demo data. Subsequent runs auto-load the persisted index.

### Option B — Python API

```python
import asyncio
from flexrag import RAGPipeline

async def main():
    # Reads .env / environment variables automatically
    pipeline = RAGPipeline.from_settings()

    result = await pipeline.arun("What is Retrieval-Augmented Generation?")
    print("Answer :", result.answer)
    for i, snippet in enumerate(result.evidence, 1):
        print(f"  [{i}] {snippet[:120]}")

asyncio.run(main())
```

> In a synchronous context use `pipeline.run(query)`, which wraps `asyncio.run()` internally. **Do not** call it from within a running event loop.

### Option C — Gradio Web UI

```bash
# Step 1: build the knowledge base
python scripts/build_knowledge_base.py \
    --input-dir ./data/docs/ \
    --output-dir ./data/knowledge_persist_dir/default/ \
    --chunk-size 512 --chunk-overlap 50 \
    --embedding-base-url http://localhost:8001/v1 \
    --embedding-model BAAI/bge-large-en-v1.5 \
    --enable-sparse          # also build BM25 index

# Step 2: launch the UI
python web_UI.py
# → http://localhost:7860
```

### Option D — Streaming (server-sent events style)

```python
import asyncio
from flexrag import RAGPipeline

async def main():
    pipeline = RAGPipeline.from_settings()
    async for event in pipeline.astream_run("What is RAG?"):
        if event["type"] == "node_start":
            print(f"▶ {event['node']} started")
        elif event["type"] == "node_end":
            print(f"✔ {event['node']} finished")
        elif event["type"] == "result":
            print("Answer:", event["answer"])
        elif event["type"] == "error":
            print("Error:", event["message"])

asyncio.run(main())
```

---

## Scripts

### Build knowledge base — `scripts/build_knowledge_base.py`

```bash
# Minimal: build dense FAISS index from a directory
python scripts/build_knowledge_base.py \
    --input-dir ./my_docs \
    --output-dir ./data/index \
    --embedding-base-url http://localhost:8001/v1 \
    --embedding-model BAAI/bge-large-en-v1.5 \
    --enable-dense

# Also build a BM25 sparse index
python scripts/build_knowledge_base.py \
    --input-dir ./my_docs --output-dir ./data/index \
    --enable-dense --enable-sparse

# Also build a local knowledge-graph index (requires LLM)
python scripts/build_knowledge_base.py \
    --input-dir ./my_docs --output-dir ./data/index \
    --enable-dense --enable-sparse --enable-graph \
    --llm-base-url http://localhost:8000/v1 \
    --llm-model Qwen/Qwen2.5-7B-Instruct

# Overwrite an existing index
python scripts/build_knowledge_base.py --input-dir ./my_docs \
    --output-dir ./data/index --force
```

### Async batch inference — `scripts/batch_run.py`

```bash
python scripts/batch_run.py \
    --embedding-base-url http://localhost:8001/v1 \
    --embedding-model BAAI/bge-large-en-v1.5 \
    --llm-base-url http://localhost:8000/v1 \
    --llm-model Qwen/Qwen2.5-7B-Instruct \
    --knowledge-persist-dir ./data/knowledge_persist_dir/hotpotqa \
    --input-file ./data/hotpotqa-100.json \
    --output-file ./eval_results/hotpotqa.json \
    --max-concurrent-tasks 5 \
    --max-iterations 3
```

Input JSON format: `[{"question": "...", "answer": "..."}, ...]`

### Offline evaluation — `scripts/eval_rag.py`

```bash
python scripts/eval_rag.py --input ./eval_results/hotpotqa.json --k_list 1,5,10,20
```

Sample output:

```
overall scores:
  EM:       {'ExactMatch': 0.42}
  F1:       {'CharF1': 0.61}
  Recall@k: {'Recall@1': 0.35, 'Recall@5': 0.58, 'Recall@10': 0.71, 'Recall@20': 0.83}
```

---

## Extending FlexRAG

Each stage has an abstract base class in `flexrag/components/*/base.py`. To add a new component:

1. Subclass the appropriate ABC.
2. Implement the required `async def` methods.
3. Pass the instance into `RAGPipeline(...)` or `PostRetrieval([...])`.

**Example: custom post-retrieval processor**

```python
from flexrag.components.post_retrieval.base import BasePostRetrieval
from flexrag.common.schema import Document

class KeywordFilter(BasePostRetrieval):
    """Keep only documents that contain at least one keyword from the query."""

    async def optimize(
        self,
        query: str,
        documents: list[Document],
        accumulated_context: list[str],
        max_tokens: int,
    ) -> tuple[str, str]:
        keywords = set(query.lower().split())
        filtered = [d for d in documents if keywords & set(d.text.lower().split())]
        context = "\n\n".join(d.text for d in filtered or documents)
        return context, ""
```

**Example: custom retriever**

```python
from flexrag.components.retrieval.base import BaseFlexRetriever
from flexrag.common.schema import Document

class ElasticRetriever(BaseFlexRetriever):
    def __init__(self, es_client, index: str, top_k: int = 10):
        self._es = es_client
        self._index = index
        self._top_k = top_k

    async def retrieve(self, query: str) -> list[Document]:
        import asyncio
        hits = await asyncio.to_thread(
            self._es.search,
            index=self._index,
            body={"query": {"match": {"text": query}}, "size": self._top_k},
        )
        return [
            Document(text=h["_source"]["text"], score=h["_score"])
            for h in hits["hits"]["hits"]
        ]
```

---

## Running Tests

```bash
pip install pytest pytest-asyncio
python -m pytest tests/ -v
```

All tests run fully offline using `unittest.mock` — no external services required.

---

## Environment Variables Reference

All variables are read from `.env` (via pydantic-settings). Shell environment variables take precedence.

| Variable | Default | Description |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:8018/v1` | LLM serving endpoint |
| `LLM_API_KEY` | `sk-xxxx` | LLM API key |
| `LLM_MODEL` | `Qwen/Qwen2.5-7B-Instruct` | Chat model name |
| `EMBEDDING_BASE_URL` | `http://localhost:8018/v1` | Embedding endpoint |
| `EMBEDDING_API_KEY` | `sk-xxxx` | Embedding API key |
| `EMBEDDING_MODEL` | `BAAI/bge-large-en-v1.5` | Embedding model name |
| `RERANKER_BASE_URL` | `http://localhost:8018/v1` | Reranker endpoint |
| `RERANKER_API_KEY` | `sk-xxxx` | Reranker API key |
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | Reranker model name |
| `KNOWLEDGE_PERSIST_DIR` | `./data/knowledge_persist_dir` | FAISS index directory |
| `KNOWLEDGE_CHUNK_SIZE` | `512` | Max tokens per chunk |
| `KNOWLEDGE_CHUNK_OVERLAP` | `50` | Token overlap between chunks |
| `TOP_K_RETRIEVAL` | `10` | Docs retrieved before reranking |
| `TOP_K_RERANK` | `5` | Docs kept after reranking |
| `CONTEXT_MAX_TOKENS` | `8000` | Generator context token budget |
| `MAX_ITERATIONS` | `3` | Max agentic re-retrieval rounds |
| `CHECKPOINT_DB_PATH` | `None` (disabled) | SQLite checkpoint path |
| `DRAW_IMAGE_PATH` | `None` (disabled) | LangGraph diagram output path |
| `LOG_LEVEL` | `INFO` | Logging level |
| `MAX_CONCURRENT_TASKS` | `5` | Concurrency for batch_run.py |

---

<a name="chinese"></a>

## 中文说明

### 项目简介

**FlexRAG** 是基于 **LangGraph** 和 **LlamaIndex** 构建的模块化企业级**检索增强生成（RAG）**系统。核心设计理念是**强解耦**——每个阶段通过抽象基类（ABC）约束，可独立替换检索器、重排序器、上下文优化器和生成器。

### 工作流程

系统由 LangGraph 编译的五节点有向图驱动：

1. **PreQueryOptimizer（查询优化）** — 查询改写、多查询扩展、复杂问题分解
2. **Retrieve（检索）** — 密集向量检索（FAISS/Chroma/Milvus）+ 稀疏 BM25 检索 + 知识图谱检索的混合融合
3. **PostRetrieval（后处理）** — Cross-Encoder 重排序 + LLM 上下文精炼抽取
4. **ContextEvaluator（上下文评估）** — LLM 判断上下文是否充分；不充分则携带 `missing_info` 提示重新触发检索循环
5. **Generate（生成）** — 结构化 JSON 输出（包含 answer 和 evidence 字段）

### 安全注意事项

- **请勿**将 `.env` 文件、API 密钥或任何凭证提交到版本控制系统
- 建议使用 `pre-commit` + `detect-secrets` 钩子防止意外泄漏
- 生产环境应使用密钥管理服务（HashiCorp Vault、AWS Secrets Manager 等）替代 `.env` 文件

### 快速上手（中文）

```bash
# 1. 克隆仓库
git clone https://github.com/Xavier-123/FlexRAG.git
cd FlexRAG

# 2. 安装依赖
pip install -r requirements.txt

# 3. 创建配置文件
cp .env.example .env
# 编辑 .env，填入 LLM、Embedding、Reranker 的服务地址和 API Key

# 4. 构建知识库（以本地文档目录为例）
python scripts/build_knowledge_base.py \
    --input-dir ./data/docs \
    --output-dir ./data/knowledge_persist_dir/default \
    --enable-dense --enable-sparse

# 5. 启动命令行交互问答
python main.py

# 或启动 Web UI
python web_UI.py   # 访问 http://localhost:7860
```

### 知识库文件格式

支持 `.txt`、`.md`、`.pdf` 直接导入。若使用 JSON 格式，请遵循以下结构：

```json
[
  {"idx": 0, "title": "文章标题", "text": "正文内容..."},
  {"idx": 1, "title": "另一篇文章", "text": "正文内容..."}
]
```

---

## Contributing

Contributions are welcome! Please open an issue to discuss your idea before submitting a pull request.

1. Fork the repo and create a feature branch: `git checkout -b feat/my-feature`
2. Make your changes with tests where applicable
3. Run `python -m pytest tests/ -v` and ensure all tests pass
4. Submit a pull request with a clear description of the change

**Areas where contributions are especially welcome:**
- Additional retriever backends (Milvus, Weaviate, Pinecone, Elasticsearch)
- Additional pre-retrieval strategies
- End-to-end integration tests
- English documentation

---

## License

This project is licensed under the [MIT License](LICENSE).
