# FlexRAG

A modular, enterprise-grade **Retrieval-Augmented Generation** (RAG) conversational
system built on **LangGraph** and **LlamaIndex**.  
Features a highly decoupled architecture (Strategy Pattern) with a four-stage
processing pipeline and strict, evidence-grounded structured outputs.

---

## Architecture

```
User Input
    │
    ▼
┌──────────────────┐
│  Retrieve Agent  │  LlamaIndex VectorStoreIndex + vLLM embeddings
└────────┬─────────┘
         │
    ▼
┌──────────────────┐
│  Rerank Agent    │  vLLM cross-encoder (e.g. BAAI/bge-reranker-v2-m3)
└────────┬─────────┘
         │
    ▼
┌──────────────────────────┐
│  Context Optimiser Agent │  GPT-4o – extracts the most relevant passages
└────────┬─────────────────┘
         │
    ▼
┌──────────────────┐
│  Generate Agent  │  GPT-4o Structured Output → { answer, evidence }
└──────────────────┘
```

### Structured output

Every response is a validated `RAGOutput` Pydantic model:

```python
class RAGOutput(BaseModel):
    answer: str          # final answer to the user's question
    evidence: list[str]  # source excerpts that ground the answer
```

---

## Directory structure

```
FlexRAG/
├── main.py                              # Quick-start example
├── requirements.txt
└── flexrag/
    ├── __init__.py
    ├── config.py                        # Pydantic Settings (env vars / .env)
    ├── schema.py                        # Document, RAGState, RAGOutput models
    ├── pipeline.py                      # RAGPipeline – high-level entry point
    ├── abstractions/                    # Abstract base classes (Strategy Pattern)
    │   ├── base_retriever.py
    │   ├── base_reranker.py
    │   ├── base_context_optimizer.py
    │   └── base_generator.py
    ├── retrievers/
    │   └── llamaindex_retriever.py      # LlamaIndex + vLLM embeddings
    ├── rerankers/
    │   └── vllm_reranker.py             # vLLM cross-encoder reranker
    ├── context_optimizers/
    │   └── llm_context_optimizer.py     # GPT-4o context extractor
    ├── generators/
    │   └── openai_generator.py          # GPT-4o Structured Output generator
    └── graph/
        ├── nodes.py                     # LangGraph node factories
        └── builder.py                   # StateGraph assembly & compilation
```

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

Create a `.env` file (or export the variables):

```dotenv
OPENAI_API_KEY=sk-...
VLLM_BASE_URL=http://localhost:8000
VLLM_EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
VLLM_RERANKER_MODEL=BAAI/bge-reranker-v2-m3
```

### 3. Run the example

```bash
python main.py
```

### 4. Use as a library

```python
from flexrag import RAGPipeline
from flexrag.config import Settings

pipeline = RAGPipeline.from_settings(Settings())
pipeline.add_documents(["RAG combines retrieval with generation ...", ...])
output = pipeline.run("What is Retrieval-Augmented Generation?")

print(output.answer)
print(output.evidence)
```

---

## Extending FlexRAG

Every component implements a thin abstract base class.  To swap in a different
vector store, reranker, or LLM:

1. Subclass the appropriate ABC in `flexrag/abstractions/`.
2. Implement the single abstract method.
3. Pass your new class to `build_rag_graph(...)` or `RAGPipeline(...)`.

```python
from flexrag.abstractions import BaseReranker
from flexrag.schema import Document

class MyCustomReranker(BaseReranker):
    def rerank(self, query: str, documents: list[Document], top_k: int) -> list[Document]:
        # your logic here
        ...
```

---

## Running tests

```bash
pip install pytest
pytest tests/ -v
```

---

## Tech stack

| Layer | Technology |
|---|---|
| Graph orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) |
| LLM & generation | OpenAI GPT-4o via [langchain-openai](https://pypi.org/project/langchain-openai/) |
| Retrieval | [LlamaIndex](https://www.llamaindex.ai/) VectorStoreIndex |
| Embeddings & reranking | [vLLM](https://github.com/vllm-project/vllm) remote API |
| Data validation | [Pydantic v2](https://docs.pydantic.dev/) |
| Configuration | [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) |
