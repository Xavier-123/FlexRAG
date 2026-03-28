# FlexRAG

**FlexRAG** 是一个基于 **LangGraph** 和 **LlamaIndex** 构建的模块化、企业级**检索增强生成（RAG）**系统。

系统内置完整的**知识库构建流水线**——从本地文件加载、分块、使用 vLLM 模型向量化，到持久化 FAISS 索引到磁盘。下次启动时自动恢复索引，让文档随时可查。FlexRAG 还提供了 **Gradio Web UI**，支持多知识库动态切换，以及完整的**评估模块**，涵盖 EM、F1 和 Recall@k 等指标。

---

## 功能亮点

| 功能 | 说明 |
|---|---|
| **持久化 FAISS 知识库** | 一次构建，多次复用；支持 `.txt` / `.md` / `.pdf` 文件 |
| **可配置分块策略** | `chunk_size` 与 `chunk_overlap` 可通过环境变量或代码设定 |
| **vLLM 向量化与重排序** | 对接任意 OpenAI 兼容嵌入端点（本地或远程）及 vLLM cross-encoder |
| **七节点 Agentic RAG 图** | Query Optimizer → Retrieve → Rerank → Optimize Context → Context Evaluator → (Generate \| Analyze Missing Info → 迭代回查) |
| **迭代式检索（Agentic Loop）** | LLM 上下文评估器判断信息是否充分，不足时自动优化查询并重试，最多 `MAX_ITERATIONS` 轮 |
| **结构化输出** | 每条答案均为经 Pydantic 校验的 `RAGOutput(answer, evidence)` 对象 |
| **策略模式** | 每个组件都有 ABC 抽象基类，随意替换任一阶段而不影响其余组件 |
| **Gradio Web UI** | 支持多知识库（HotpotQA / 2WikiMultihopQA / MuSiQue / NQ）动态无缝切换 |
| **评估模块** | 内置 EM、Char-F1、Recall@k 等指标，支持离线批量评估 |
| **异步批量处理** | `batch_run.py` 支持并发控制（Semaphore）批量运行 QA 并输出 JSON 结果 |
| **交互式命令行** | `main.py` 自动检测或构建知识库，进入交互式问答循环 |

---

## 系统架构

![langgraph](E:\LLM\6-RAG\cursor\FlexRAG\tests\langgraph.png)

---

## 项目结构

```
FlexRAG/
├── main.py                              # 命令行交互式问答入口
├── web_UI.py                            # Gradio Web UI（多知识库切换）
├── requirements.txt                     # Python 依赖
├── scripts/
│   ├── build_knowledge_base.py          # 独立知识库构建脚本（含 CLI 参数）
│   ├── batch_run.py                     # 异步批量 QA 处理脚本（并发 Semaphore 控制）
│   └── eval_rag.py                      # RAG 评估脚本（读取 eval_results.json）
└── flexrag/
    ├── __init__.py                      # 包入口（导出 RAGPipeline）
    ├── config.py                        # Pydantic Settings（读取环境变量 / .env）
    ├── schema.py                        # Document、RAGState、RAGOutput、ContextEvaluation 数据模型
    ├── pipeline.py                      # RAGPipeline 高层编排器
    ├── logging_config.py                # 全局日志配置工具
    ├── utils.py                         # 通用工具函数
    ├── abstractions/                    # 抽象基类（策略模式）
    │   ├── base_retriever.py
    │   ├── base_reranker.py
    │   ├── base_context_optimizer.py
    │   ├── base_context_evaluator.py    # 上下文评估器抽象基类
    │   ├── base_query_optimizer.py      # 查询优化器抽象基类
    │   ├── base_generator.py
    │   └── base_knowledge.py
    ├── knowledge/                       # 知识库构建模块
    │   ├── __init__.py
    │   └── faiss_knowledge.py           # FAISS + LlamaIndex 知识库构建器
    ├── retrievers/
    │   └── llamaindex_retriever.py      # LlamaIndex VectorStoreIndex 检索器
    ├── rerankers/
    │   └── vllm_reranker.py             # vLLM cross-encoder 重排序器
    ├── context_optimizers/
    │   └── llm_context_optimizer.py     # 基于 LLM 的上下文提取优化器
    ├── evaluators/
    │   └── llm_context_evaluator.py     # 基于 LLM 的上下文充分性评估器（Agentic 裁判）
    ├── query_optimizers/
    │   └── llm_query_optimizer.py       # 基于 LLM 的检索查询优化器（迭代重查）
    ├── generators/
    │   └── openai_generator.py          # 结构化输出生成器
    ├── graph/
    │   ├── nodes.py                     # LangGraph 节点工厂（7 节点）
    │   └── builder.py                   # StateGraph 组装与编译（含条件路由）
    └── evaluate/                        # 评估模块
        ├── metrics/
        │   ├── base.py                  # BaseMetric 抽象类
        │   ├── em.py                    # Exact Match 指标
        │   ├── f1.py                    # Char-F1 指标
        │   └── recall_k.py             # Recall@k 指标
        └── utils/
            └── eval_utils.py           # 答案归一化等工具函数
```

---

## 技术栈

| 层次 | 技术 |
|---|---|
| 图编排 | [LangGraph](https://github.com/langchain-ai/langgraph) |
| LLM 调用 | OpenAI 兼容 API，通过 [langchain-openai](https://pypi.org/project/langchain-openai/) |
| 检索与文件读取 | [LlamaIndex](https://www.llamaindex.ai/)（`SimpleDirectoryReader`、`SentenceSplitter`、`VectorStoreIndex`） |
| 向量存储 | [FAISS](https://github.com/facebookresearch/faiss)，通过 `llama-index-vector-stores-faiss` |
| 向量化与重排序 | vLLM 远程 API（OpenAI 兼容） |
| PDF 解析 | [pypdf](https://pypi.org/project/pypdf/) |
| Web UI | [Gradio](https://www.gradio.app/) |
| 数据校验 | [Pydantic v2](https://docs.pydantic.dev/) |
| 配置管理 | [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) |
| HTTP 客户端 | [httpx](https://www.python-httpx.org/)（异步） |

---

## 安装

### 1. 克隆仓库

```bash
git clone https://github.com/Xavier-123/FlexRAG.git
cd FlexRAG
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

> **注意** — `faiss-cpu` 为 CPU 版本。若需 GPU 加速，请在 `requirements.txt`
> 中将其替换为 `faiss-gpu` 后再安装。

### 3. 配置环境变量

在项目根目录创建 `.env` 文件（或直接导出环境变量）：

```dotenv
# --- LLM ---
LLM_BASE_URL=http://localhost:8000/v1
LLM_API_KEY=sk-...
VLLM_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct

# --- Embedding ---
EMBEDDING_BASE_URL=http://localhost:8001/v1
EMBEDDING_API_KEY=sk-...
VLLM_EMBEDDING_MODEL=BAAI/bge-large-en-v1.5

# --- Reranker ---
RERANKER_BASE_URL=http://localhost:8002/v1
RERANKER_API_KEY=sk-...
VLLM_RERANKER_MODEL=BAAI/bge-reranker-v2-m3

# --- 知识库（可选，括号内为默认值）---
KNOWLEDGE_PERSIST_DIR=./data/knowledge_persist_dir
KNOWLEDGE_CHUNK_SIZE=512
KNOWLEDGE_CHUNK_OVERLAP=50

# --- Pipeline 超参数（可选）---
TOP_K_RETRIEVAL=10
TOP_K_RERANK=5
CONTEXT_MAX_TOKENS=3000
MAX_ITERATIONS=3

# --- 日志（可选）---
LOG_LEVEL="INFO"
```

---

## 快速开始

### 方式一：命令行交互式问答

```bash
python main.py
```

首次运行时会提示：

```
[INFO] No knowledge base found at './data/knowledge_persist_dir'.

Choose an option:
  b  -- Build from a local directory of documents
  d  -- Use built-in demo data (no files needed)
  q  -- Quit

Option [b/d/q]:
```

- 输入 **`b`**，然后指定文档目录路径，自动完成分块、向量化、保存。
- 输入 **`d`**，使用内置的 5 段演示文本即刻体验，无需外部文件。

后续运行会自动加载已保存的索引，直接进入问答循环。

---

### 方式二：Gradio Web UI

#### 构建知识库

```shell
python scripts/build_knowledge_base.py \
--input-dir ./data/knowledge_files_dir/hotpotqa/ \
--output-dir ./data/knowledge_persist_dir/hotpotqa/ \
--chunk-size 1024 \
--chunk-overlap 50 \
--embedding-base-url http://127.0.0.1:19002/v1/embeddings \
--embedding-api-key sk-1234567890 \
--embedding-model Qwen3-Embedding-0.6B \
--force \
--verbose
```

#### 启动页面

```shell
python web_UI.py
```

启动后访问 `http://0.0.0.0:7860`，支持从下拉框切换以下知识库：

| 知识库名称 | 本地路径 |
|---|---|
| `hotpotqa` | `./data/knowledge_persist_dir/hotpotqa` |
| `2wikimultihopqa` | `./data/knowledge_persist_dir/2wikimultihopqa` |
| `musique` | `./data/knowledge_persist_dir/musique` |
| `nq` | `./data/knowledge_persist_dir/nq` |

首次切换到某个知识库时会自动加载并缓存，后续切换实现秒级响应。每条回答均附带可展开的**参考检索片段**。

---

### 方式三：Python API

#### 构建知识库

```python
import asyncio
from flexrag.config import Settings
from flexrag.knowledge import FaissKnowledgeBuilder

settings = Settings()

async def build():
    builder = FaissKnowledgeBuilder(
        embed_base_url=settings.embedding_base_url,
        embed_model_name=settings.vllm_embedding_model,
        embed_api_key=settings.embedding_api_key,
    )
    await builder.load_files("./my_documents")   # 支持 .txt / .md / .pdf
    await builder.build_index(
        chunk_size=settings.knowledge_chunk_size,
        chunk_overlap=settings.knowledge_chunk_overlap,
    )
    await builder.save(settings.knowledge_persist_dir)
    print("知识库构建完成。")

asyncio.run(build())
```

#### 加载知识库并运行问答

```python
import asyncio
from flexrag.config import Settings
from flexrag.context_optimizers.llm_context_optimizer import LLMContextOptimizer
from flexrag.evaluators.llm_context_evaluator import LLMContextEvaluator
from flexrag.generators.openai_generator import OpenAIGenerator
from flexrag.pipeline import RAGPipeline
from flexrag.query_optimizers.llm_query_optimizer import LLMQueryOptimizer
from flexrag.rerankers.vllm_reranker import VLLMReranker
from flexrag.retrievers import LlamaIndexRetriever
from langchain_openai import ChatOpenAI

async def query():
    settings = Settings()

    retriever = LlamaIndexRetriever(
        index=None,
        embed_base_url=settings.embedding_base_url,
        embed_model_name=settings.vllm_embedding_model,
        embed_api_key=settings.embedding_api_key,
    )
    await retriever.load_index(settings.knowledge_persist_dir)

    llm = ChatOpenAI(
        model=settings.vllm_llm_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        temperature=0.0,
    )
    pipeline = RAGPipeline(
        retriever=retriever,
        reranker=VLLMReranker(
            base_url=settings.reranker_base_url,
            model=settings.vllm_reranker_model,
            api_key=settings.reranker_api_key,
        ),
        context_optimizer=LLMContextOptimizer(llm=llm),
        query_optimizer=LLMQueryOptimizer(llm=llm),
        context_evaluator=LLMContextEvaluator(llm=llm),
        generator=OpenAIGenerator(
            model=settings.vllm_llm_model,
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
        ),
        settings=settings,
    )

    output = await pipeline.arun("什么是检索增强生成？")
    print(output.answer)
    print(output.evidence)

asyncio.run(query())
```

> **提示** — 在同步环境中也可以使用 `pipeline.run(query)` 作为便捷封装，
> 其内部调用 `asyncio.run()`，**不能**在已有事件循环中使用。

---

## 脚本工具

### 异步批量处理脚本

`batch_run.py` 支持并发控制（`asyncio.Semaphore`）对大规模 QA 数据集进行批量推理，并将结果输出为 JSON 文件，供后续评估使用。

```bash
python scripts/batch_run.py \
    --embedding-base-url http://localhost:8001/v1 \
    --vllm-embedding-model BAAI/bge-large-en-v1.5 \
    --reranker-base-url http://localhost:8002/v1 \
    --vllm-reranker-model BAAI/bge-reranker-v2-m3 \
    --llm-base-url http://localhost:8000/v1 \
    --vllm-llm-model Qwen/Qwen2.5-7B-Instruct \
    --knowledge-persist-dir ./data/knowledge_persist_dir \
    --input-file ./qa_data.json \
    --output-file ./eval_results.json \
    --max-concurrent-tasks 5 \
    --max-iterations 3
```

输入 JSON 格式（每项包含 `question` 与可选的 `answer`）：

```json
[
  {"question": "问题文本", "answer": "参考答案（可选）"}
]
```

输出 JSON 格式：

```json
[
  {
    "question": "问题文本",
    "expected": "参考答案",
    "generated_answer": "模型生成的答案",
    "evidence": ["检索片段1", "检索片段2"],
    "status": "success",
    "error": null
  }
]
```

### 知识库构建脚本

```bash
# 从目录构建
python scripts/build_knowledge_base.py --input-dir ./my_docs

# 从指定文件构建
python scripts/build_knowledge_base.py --files doc1.txt doc2.pdf

# 自定义输出目录和分块参数
python scripts/build_knowledge_base.py --input-dir ./my_docs \
    --output-dir ./my_index --chunk-size 256 --chunk-overlap 32

# 强制覆盖已有索引
python scripts/build_knowledge_base.py --input-dir ./my_docs --force

# 查看所有参数
python scripts/build_knowledge_base.py --help
```

### RAG 评估脚本

先准备一个 `eval_results.json`，格式为 JSON 数组，每项包含：

```json
[
  {
    "question": "问题文本",
    "expected": "参考答案",
    "generated_answer": "模型生成的答案",
    "evidence": ["检索片段1", "检索片段2"]
  }
]
```

然后运行：

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

## 扩展 FlexRAG

每个组件均有对应的抽象基类，位于 `flexrag/abstractions/`。替换某一阶段只需：

1. 继承对应的抽象基类。
2. 实现所需的抽象方法（均为 `async def`）。
3. 将自定义实现传入 `RAGPipeline(...)`。

**示例：自定义重排序器**

```python
from flexrag.abstractions import BaseReranker
from flexrag.schema import Document

class MyReranker(BaseReranker):
    async def rerank(
        self, query: str, documents: list[Document], top_k: int
    ) -> list[Document]:
        # 自定义评分逻辑（此处以文本长度为例）
        return sorted(documents, key=lambda d: len(d.text), reverse=True)[:top_k]
```

**示例：自定义查询优化器**

继承 `BaseQueryOptimizer`，实现 `optimize_query(original_query, missing_info, accumulated_context, iteration_count, previous_query)` 方法即可，用于在 Agentic 迭代循环中生成优化后的检索查询。

**示例：自定义上下文评估器**

继承 `BaseContextEvaluator`，实现 `evaluate(original_query, optimized_context, accumulated_context)` 方法，返回 `ContextEvaluation(context_sufficient, missing_info, judge_reason, accumulated_context)`，控制 Agentic 循环是否继续迭代。

**示例：自定义知识库后端**

继承 `BaseKnowledgeBuilder`，实现 `load_files`、`build_index`、`save` 和 `index_exists` 四个方法即可。

---

## 运行测试

```bash
pip install pytest pytest-asyncio
python -m pytest tests/ -v
```

所有测试均使用 `unittest.mock` 在完全离线的环境中运行，无需任何外部服务。

---

## 环境变量速查

下表列出 `flexrag/config.py` 中的**内置默认值**。可通过 `.env` 文件或环境变量覆盖任意项。

| 变量名 | 内置默认值 | 说明 |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:8018/v1` | LLM 服务端点 |
| `LLM_API_KEY` | `sk-xxxx` | LLM API 密钥 |
| `EMBEDDING_BASE_URL` | `http://localhost:8018/v1` | 向量化服务端点 |
| `EMBEDDING_API_KEY` | `sk-xxxx` | 向量化 API 密钥 |
| `RERANKER_BASE_URL` | `http://localhost:8018/v1` | 重排序服务端点 |
| `RERANKER_API_KEY` | `sk-xxxx` | 重排序 API 密钥 |
| `VLLM_LLM_MODEL` | `Qwen/Qwen2.5-7B-Instruct` | 对话模型名称 |
| `VLLM_EMBEDDING_MODEL` | `BAAI/bge-large-en-v1.5` | 向量化模型名称 |
| `VLLM_RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | 重排序模型名称 |
| `KNOWLEDGE_PERSIST_DIR` | `./data/knowledge_persist_dir` | FAISS 索引存储目录 |
| `KNOWLEDGE_CHUNK_SIZE` | `512` | 每个文档块的最大 token 数 |
| `KNOWLEDGE_CHUNK_OVERLAP` | `50` | 相邻块之间的 token 重叠量 |
| `TOP_K_RETRIEVAL` | `10` | 重排序前检索的文档数量 |
| `TOP_K_RERANK` | `5` | 重排序后保留的文档数量 |
| `CONTEXT_MAX_TOKENS` | `3000` | 传给生成器的上下文 token 预算 |
| `MAX_ITERATIONS` | `3` | Agentic RAG 最大迭代重查轮数 |
| `LOG_LEVEL` | `INFO` | 全局日志级别（DEBUG / INFO / WARNING / ERROR / CRITICAL） |
| `LOG_FORMAT` | `%(asctime)s  %(levelname)-8s  %(name)s  %(message)s` | 全局日志格式字符串 |
