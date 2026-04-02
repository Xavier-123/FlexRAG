<div align="center">

# 🤖 FlexRAG

**一个具有强解耦性的 Agentic RAG 系统**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-≥0.2.0-orange)](https://github.com/langchain-ai/langgraph)
[![LlamaIndex](https://img.shields.io/badge/LlamaIndex-≥0.10.0-green)](https://www.llamaindex.ai/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)]()

</div>

---

## 📖 项目简介

**FlexRAG** 是一个基于 **LangGraph** 和 **LlamaIndex** 构建的模块化、企业级**检索增强生成（RAG）**系统。

传统 RAG 系统往往将检索、重排序、生成等各阶段紧耦合在一起，难以独立替换和扩展。FlexRAG 通过**策略模式**对每个阶段进行严格解耦，每个组件均有抽象基类约束，让你可以随意替换任一环节而不影响其他模块。

系统内置完整的**知识库构建流水线**——从本地文件加载、分块、使用 vLLM 模型向量化，到持久化 FAISS 索引到磁盘。下次启动时自动恢复索引，让文档随时可查。FlexRAG 还提供了 **Gradio Web UI**，支持多知识库动态切换，以及完整的**评估模块**，涵盖 EM、F1 和 Recall@k 等指标。

---

## ✨ 核心特性

- 🔌 **强解耦策略模式**：Pre-Retrieval、Retrieval、Post-Retrieval、Reasoning 四大阶段均有 ABC 抽象基类约束，各组件独立可替换，修改任意一环节无需触及其他模块。

- 🔄 **Agentic 迭代检索循环**：LangGraph `StateGraph` 驱动五节点有向图（PreQueryOptimizer → Retrieve → PostRetrieval → ContextEvaluator → Generate），`ContextEvaluator` 判断上下文是否充分，不足时自动优化查询并循环重查，最多执行 `MAX_ITERATIONS` 轮。

- 🗄️ **多策略混合检索**：内置 `FAISSRetriever`（密集向量）、`BM25Retriever`（稀疏关键词）、`GraphRetriever`（知识图谱）三种检索器，并通过 `HybridRetriever` 融合结果，消除单一召回策略的盲区。

- 📚 **完整知识库构建流水线**：`FaissKnowledgeBuilder` 支持 `.txt` / `.md` / `.pdf` 文件批量加载、分块、调用 vLLM 嵌入端点向量化，并持久化 FAISS 索引；可选同步构建 BM25 稀疏索引和 Neo4j / 本地知识图谱索引。

- 🖥️ **Gradio Web UI + 多知识库秒切**：`web_UI.py` 提供开箱即用的聊天界面，支持 hotpotqa / 2wikimultihopqa / musique / nq 等多知识库下拉切换，首次加载后自动缓存，切换延迟接近零。

- 📊 **离线批量评估工具链**：`scripts/batch_run.py` 以 `asyncio.Semaphore` 控制并发批量推理，输出标准 JSON；`scripts/eval_rag.py` 一键计算 Exact Match、Char-F1 和 Recall@k 等指标。

---

## 🛠️ 技术栈

| 类别 | 技术 / 库 | 说明 |
|---|---|---|
| **图编排** | [LangGraph](https://github.com/langchain-ai/langgraph) ≥ 0.2.0 | Agentic 状态机编排，含条件路由与可选 SQLite 检查点 |
| **LLM 调用** | [langchain-openai](https://pypi.org/project/langchain-openai/) ≥ 0.1.0 | OpenAI 兼容 API（支持 vLLM、ModelScope 等本地/云端端点） |
| **检索框架** | [LlamaIndex](https://www.llamaindex.ai/) ≥ 0.10.0 | `SimpleDirectoryReader`、`SentenceSplitter`、`VectorStoreIndex` |
| **稀疏检索** | [rank-bm25](https://pypi.org/project/rank-bm25/) + llama-index-retrievers-bm25 | BM25Okapi + 自定义中文分词（jieba） |
| **向量存储** | [FAISS](https://github.com/facebookresearch/faiss) ≥ 1.7.0 | 通过 `llama-index-vector-stores-faiss` 集成 |
| **数据校验** | [Pydantic v2](https://docs.pydantic.dev/) + pydantic-settings | 结构化输出（`RAGOutput`、`Document`）及配置管理 |
| **HTTP 客户端** | [httpx](https://www.python-httpx.org/) ≥ 0.27.0 | 异步调用 vLLM 重排序远程接口 |
| **Web UI** | [Gradio](https://www.gradio.app/) | 多知识库聊天界面，原生支持 async |
| **PDF 解析** | [pypdf](https://pypi.org/project/pypdf/) ≥ 3.0.0 | 从 PDF 文件中提取文本 |
| **配置加载** | [python-dotenv](https://pypi.org/project/python-dotenv/) ≥ 1.0.0 | 从 `.env` 文件自动读取环境变量 |

---

## 📁 项目结构

```
FlexRAG/
├── main.py                              # 🚀 命令行交互式问答入口（自动检测/构建知识库）
├── web_UI.py                            # 🖥️  Gradio Web UI（多知识库动态切换，端口 7860）
├── requirements.txt                     # 📦 Python 依赖清单
├── scripts/
│   ├── build_knowledge_base.py          # 🔨 独立知识库构建脚本（CLI，支持稀疏/图索引）
│   ├── batch_run.py                     # ⚡ 异步批量 QA 推理脚本（并发 Semaphore 控制）
│   └── eval_rag.py                      # 📊 RAG 离线评估脚本（EM / F1 / Recall@k）
└── flexrag/                             # 核心库
    ├── __init__.py                      # 包入口（导出 RAGPipeline、RAGOutput、RAGState）
    ├── common/
    │   ├── config.py                    # Pydantic Settings（读取 .env / 环境变量）
    │   ├── schema.py                    # 数据模型：Document、RAGState、RAGOutput、ContextEvaluation
    │   ├── logging.py                   # 全局日志配置
    │   └── exceptions.py               # 自定义异常
    ├── indexing/
    │   └── knowledge.py                 # FaissKnowledgeBuilder：加载文件 → 分块 → 嵌入 → 持久化
    ├── components/
    │   ├── pre_retrieval/               # 检索前优化（策略可插拔）
    │   │   ├── query_rewriter.py        # LLM 查询改写
    │   │   ├── query_expander.py        # LLM 多查询扩展
    │   │   ├── task_splitter.py         # 复杂问题拆解为子任务
    │   │   └── terminology_enricher.py  # 专业术语增强
    │   ├── retrieval/                   # 检索器（策略可插拔）
    │   │   ├── faiss_retriever.py       # 密集向量检索（FAISS + LlamaIndex）
    │   │   ├── bm25_retriever.py        # 稀疏 BM25 检索
    │   │   ├── graph_retriever.py       # 本地知识图谱检索
    │   │   └── retrieval_opt.py         # HybridRetriever（多路融合）
    │   ├── post_retrieval/              # 检索后优化（策略可插拔）
    │   │   ├── reranker.py              # OpenAI-Like cross-encoder 重排序
    │   │   └── context_optimizer.py     # LLM 上下文裁剪优化
    │   ├── reasoning/                   # 推理组件（策略可插拔）
    │   │   ├── context_evaluator.py     # LLM 上下文充分性评估（Agentic 裁判）
    │   │   └── generator.py             # OpenAI 结构化输出生成器
    │   └── evaluate/                    # 评估指标
    │       └── metrics/                 # EM / Char-F1 / Recall@k
    └── workflows/
        ├── pipeline.py                  # RAGPipeline 高层编排器（arun / run）
        ├── builder.py                   # LangGraph StateGraph 组装与编译
        └── nodes.py                     # 节点工厂（pre_retrieval / retrieve / post_retrieval / eval / generate）
```

---

## 🚀 快速开始

### 环境要求

| 要求 | 版本 |
|---|---|
| Python | ≥ 3.10 |
| 嵌入模型服务 | 任意 OpenAI 兼容嵌入端点（如 vLLM、ModelScope） |
| 重排序模型服务 | 任意 OpenAI 兼容 cross-encoder 端点（可选） |
| 对话大模型服务 | 任意 OpenAI 兼容聊天端点 |

### 安装步骤

**1. 克隆仓库**

```bash
git clone https://github.com/Xavier-123/FlexRAG.git
cd FlexRAG
```

**2. 安装依赖**

```bash
pip install -r requirements.txt
```

> 💡 `faiss-cpu` 为默认 CPU 版本。若需 GPU 加速，请将 `requirements.txt` 中的
> `faiss-cpu` 替换为 `faiss-gpu` 后再执行上述命令。

### 配置说明

在项目根目录创建 `.env` 文件（所有配置均有内置默认值，可按需覆盖）：

```dotenv
# ============================================================
# LLM（对话模型）
# ============================================================
LLM_BASE_URL=http://localhost:8000/v1
LLM_API_KEY=sk-your-key
LLM_MODEL=Qwen/Qwen2.5-7B-Instruct

# ============================================================
# Embedding（嵌入模型）
# ============================================================
EMBEDDING_BASE_URL=http://localhost:8001/v1
EMBEDDING_API_KEY=sk-your-key
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5

# ============================================================
# Reranker（重排序模型）
# ============================================================
RERANKER_BASE_URL=http://localhost:8002/v1
RERANKER_API_KEY=sk-your-key
RERANKER_MODEL=BAAI/bge-reranker-v2-m3

# ============================================================
# 知识库（以下均为可选项，括号内为内置默认值）
# ============================================================
KNOWLEDGE_PERSIST_DIR=./data/knowledge_persist_dir   # (./data/knowledge_persist_dir)
KNOWLEDGE_CHUNK_SIZE=512                             # (512)
KNOWLEDGE_CHUNK_OVERLAP=50                           # (50)

# ============================================================
# Pipeline 超参数（可选）
# ============================================================
TOP_K_RETRIEVAL=10      # 重排序前初次检索文档数 (10)
TOP_K_RERANK=5          # 重排序后保留文档数 (5)
CONTEXT_MAX_TOKENS=3000 # 传给生成器的上下文 token 预算 (3000)
MAX_ITERATIONS=3        # Agentic 迭代重查最大轮数 (3)

# ============================================================
# 可选高级配置
# ============================================================
CHECKPOINT_DB_PATH=./data/checkpoints.db   # 启用 LangGraph SQLite 检查点（不设则禁用）
LLM_AUDIT_LOG_PATH=./data/audit_llm.jsonl  # LLM 调用审计日志（不设则禁用）
DRAW_IMAGE_PATH=./langgraph.png            # 保存架构图（不设则不生成）
LOG_LEVEL=INFO                             # 日志级别 (INFO)
```

### 运行项目

**方式一：命令行交互式问答**

```bash
python main.py
```

首次运行时会提示选择模式：

```
[INFO] No knowledge base found at './data/knowledge_persist_dir'.

Choose an option:
  b  -- Build from a local directory of documents
  d  -- Use built-in demo data (no files needed)
  q  -- Quit

Option [b/d/q]:
```

- 输入 **`b`** → 指定文档目录，自动完成分块、向量化、保存索引
- 输入 **`d`** → 使用内置 5 段演示文本，无需任何外部文件，直接体验完整流程

再次运行时自动加载已保存的索引，无需重新构建。

**方式二：Gradio Web UI**

```bash
# 第一步：构建知识库（以 hotpotqa 为例）
python scripts/build_knowledge_base.py \
    --input-dir ./data/knowledge_files_dir/hotpotqa/ \
    --output-dir ./data/knowledge_persist_dir/hotpotqa/ \
    --chunk-size 1024 \
    --chunk-overlap 50 \
    --embedding-base-url http://localhost:8001/v1 \
    --embedding-api-key sk-your-key \
    --embedding-model BAAI/bge-large-en-v1.5 \
    --force

# 第二步：启动 Web UI
python web_UI.py
```

启动后访问 `http://localhost:7860`，通过下拉框在多个知识库间无缝切换。

---

## 💡 使用指南

### 示例一：Python API 快速问答

```python
import asyncio
from flexrag import RAGPipeline
from flexrag.common import Settings

async def main():
    # 读取 .env / 环境变量，自动加载已持久化的 FAISS 知识库
    pipeline = RAGPipeline.from_settings()

    output = await pipeline.arun("什么是检索增强生成（RAG）？")

    print("Answer :", output.answer)
    print("Evidence:")
    for i, snippet in enumerate(output.evidence, 1):
        print(f"  [{i}] {snippet[:120]}...")

asyncio.run(main())
```

> 在同步环境中可使用 `pipeline.run(query)`（内部调用 `asyncio.run()`），
> **不能**在已有事件循环中使用。

---

### 示例二：手动组装 Pipeline（自定义各阶段组件）

```python
import asyncio
from langchain_openai import ChatOpenAI
from flexrag import RAGPipeline
from flexrag.common import Settings
from flexrag.components.pre_retrieval import PreQueryOptimizer, QueryRewriter, QueryExpander
from flexrag.components.retrieval import HybridRetriever, FAISSRetriever, BM25Retriever
from flexrag.components.post_retrieval import PostRetrieval, OpenAILikeReranker, LLMContextOptimizer
from flexrag.components.reasoning import LLMContextEvaluator, OpenAIGenerator

async def main():
    settings = Settings()

    llm = ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        temperature=0.0,
    )

    # --- 检索前优化：查询改写 + 多查询扩展 ---
    pre_opt = PreQueryOptimizer([
        QueryRewriter(llm=llm),
        QueryExpander(llm=llm),
    ])

    # --- 混合检索：FAISS 密集 + BM25 稀疏 ---
    retriever = HybridRetriever(retrievers=[
        FAISSRetriever(
            index=None,
            embed_base_url=settings.embedding_base_url,
            embed_model_name=settings.embedding_model,
            embed_api_key=settings.embedding_api_key,
            top_k=5,
            persist_dir=settings.knowledge_persist_dir,
        ),
        BM25Retriever(
            top_k=5,
            persist_dir=f"{settings.knowledge_persist_dir}/bm25_index",
        ),
    ])

    # --- 检索后优化：重排序 + LLM 上下文裁剪 ---
    post_opt = PostRetrieval([
        OpenAILikeReranker(
            base_url=settings.reranker_base_url,
            model=settings.reranker_model,
            api_key=settings.reranker_api_key,
            top_k=5,
        ),
        LLMContextOptimizer(llm=llm),
    ])

    pipeline = RAGPipeline(
        pre_retrieval_optimizer=pre_opt,
        retriever=retriever,
        post_retrieval_optimizer=post_opt,
        context_evaluator=LLMContextEvaluator(llm=llm),
        generator=OpenAIGenerator(
            model=settings.llm_model,
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
        ),
        settings=settings,
    )

    output = await pipeline.arun("LangGraph 与 LangChain 有什么区别？")
    print(output.answer)

asyncio.run(main())
```

---

## 🔧 脚本工具

### 知识库构建（`scripts/build_knowledge_base.py`）

```bash
# 从目录构建（.txt / .md / .pdf）
python scripts/build_knowledge_base.py --input-dir ./my_docs

# 同时构建稀疏 BM25 索引
python scripts/build_knowledge_base.py --input-dir ./my_docs --enable-sparse

# 同时构建本地知识图谱索引
python scripts/build_knowledge_base.py --input-dir ./my_docs --enable-graph \
    --llm-model Qwen/Qwen2.5-7B-Instruct \
    --llm-base-url http://localhost:8000/v1 \
    --llm-api-key sk-your-key

# 自定义输出目录、分块大小，强制覆盖已有索引
python scripts/build_knowledge_base.py --input-dir ./my_docs \
    --output-dir ./my_index --chunk-size 256 --chunk-overlap 32 --force
```

### 异步批量推理（`scripts/batch_run.py`）

```bash
python scripts/batch_run.py \
    --embedding-base-url http://localhost:8001/v1 \
    --embedding-model BAAI/bge-large-en-v1.5 \
    --reranker-base-url http://localhost:8002/v1 \
    --reranker-model BAAI/bge-reranker-v2-m3 \
    --llm-base-url http://localhost:8000/v1 \
    --llm-model Qwen/Qwen2.5-7B-Instruct \
    --knowledge-persist-dir ./data/knowledge_persist_dir \
    --input-file ./qa_data.json \
    --output-file ./eval_results.json \
    --max-concurrent-tasks 5 \
    --max-iterations 3
```

### 评估（`scripts/eval_rag.py`）

```bash
python scripts/eval_rag.py --input eval_results.json --k_list 1,5,10,20
```

输出示例：

```
overall scores:
  EM:       {'ExactMatch': 0.42}
  F1:       {'CharF1': 0.61}
  Recall@k: {'Recall@1': 0.35, 'Recall@5': 0.58, 'Recall@10': 0.71, 'Recall@20': 0.83}
```

---

## 🧩 扩展 FlexRAG

每个阶段均有抽象基类（位于各 `components/*/base.py`）。替换组件只需：

1. 继承对应的抽象基类
2. 实现所需的 `async def` 抽象方法
3. 将实例传入 `RAGPipeline(...)`

**示例：自定义检索后处理器**

```python
from flexrag.components.post_retrieval.base import BasePostRetrieval
from flexrag.common.schema import Document

class MySimpleRanker(BasePostRetrieval):
    async def optimize(self, query, documents, accumulated_context, max_tokens):
        # 按文本长度降序排列后返回前 5 条
        ranked = sorted(documents, key=lambda d: len(d.text), reverse=True)[:5]
        return "\n\n".join(d.text for d in ranked), ""
```

---

## 🧪 运行测试

```bash
pip install pytest pytest-asyncio
python -m pytest tests/ -v
```

所有测试均通过 `unittest.mock` 在完全离线环境中运行，无需任何外部服务。

---

## 📋 环境变量速查

下表列出 `flexrag/common/config.py` 中的**内置默认值**，可通过 `.env` 文件或 Shell 环境变量覆盖任意项。

> 💡 LLM、Embedding、Reranker 三者默认均指向同一端口 `8018`，这意味着你可以用一个 vLLM 实例同时托管多个模型。若使用不同主机/端口，在 `.env` 中分别配置对应的 `*_BASE_URL` 即可。

| 变量名 | 默认值 | 说明 |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:8018/v1` | LLM 服务端点 |
| `LLM_API_KEY` | `sk-xxxx` | LLM API 密钥 |
| `LLM_MODEL` | `Qwen/Qwen2.5-7B-Instruct` | 对话模型名称 |
| `EMBEDDING_BASE_URL` | `http://localhost:8018/v1` | 嵌入模型服务端点 |
| `EMBEDDING_API_KEY` | `sk-xxxx` | 嵌入模型 API 密钥 |
| `EMBEDDING_MODEL` | `BAAI/bge-large-en-v1.5` | 嵌入模型名称 |
| `RERANKER_BASE_URL` | `http://localhost:8018/v1` | 重排序模型服务端点 |
| `RERANKER_API_KEY` | `sk-xxxx` | 重排序模型 API 密钥 |
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | 重排序模型名称 |
| `KNOWLEDGE_PERSIST_DIR` | `./data/knowledge_persist_dir` | FAISS 索引存储目录 |
| `KNOWLEDGE_CHUNK_SIZE` | `512` | 每个文档块最大 token 数 |
| `KNOWLEDGE_CHUNK_OVERLAP` | `50` | 相邻块间 token 重叠量 |
| `TOP_K_RETRIEVAL` | `10` | 重排序前初次检索文档数 |
| `TOP_K_RERANK` | `5` | 重排序后保留文档数 |
| `CONTEXT_MAX_TOKENS` | `3000` | 传给生成器的上下文 token 预算 |
| `MAX_ITERATIONS` | `3` | Agentic RAG 最大迭代重查轮数 |
| `CHECKPOINT_DB_PATH` | `None`（禁用） | LangGraph SQLite 检查点路径 |
| `LLM_AUDIT_LOG_PATH` | `None`（禁用） | LLM 调用审计日志路径（JSONL） |
| `DRAW_IMAGE_PATH` | `None`（不生成） | LangGraph 架构图保存路径 |
| `LOG_LEVEL` | `INFO` | 日志级别（DEBUG/INFO/WARNING/ERROR/CRITICAL） |
