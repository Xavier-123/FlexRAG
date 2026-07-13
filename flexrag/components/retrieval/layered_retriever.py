"""
三级轻量索引检索器：关键词级 / 句子级 / 语块级分层设计，配合 LLM ReAct 编排三工具。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any, List, Optional, Set

import jieba
import numpy as np
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from llama_index.core import Document as LlamaDocument
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode
from pydantic import BaseModel, Field

from flexrag.common.schema import Document
from flexrag.components.retrieval.base import BaseFlexRetriever

logger = logging.getLogger(__name__)

_CHUNKS_FILE = "chunks.json"
_SENTENCES_FILE = "sentences.json"
_VECTORS_FILE = "sentence_vectors.npy"
_EMBED_BATCH_SIZE = 16


class AgentContext:
    """管理 Agent 运行时的状态和预算限制。"""

    def __init__(self, max_loops: int = 10, max_token_budget: int = 128000) -> None:
        self.seen_chunks: Set[str] = set()
        self.retrieved_docs: List[Document] = []
        self.loop_count: int = 0
        self.token_used: int = 0
        self.max_loops = max_loops
        self.max_token_budget = max_token_budget

    def add_tokens(self, count: int) -> None:
        self.token_used += count

    def is_budget_exceeded(self) -> bool:
        return self.loop_count >= self.max_loops or self.token_used >= self.max_token_budget


class KeywordSearchInput(BaseModel):
    query: str = Field(description="包含实体、专有名词或关键字的检索查询")


class SemanticSearchInput(BaseModel):
    query: str = Field(description="用于概念性或模糊语义匹配的检索查询")


class ChunkReadInput(BaseModel):
    chunk_ids: List[str] = Field(description="需要精读完整文本的语块 ID 列表")


class LayeredRetriever(BaseFlexRetriever):
    """三级轻量索引检索器，通过 LLM ReAct 循环编排 keyword / semantic / chunk_read 工具。"""

    REACT_SYSTEM_PROMPT = """作为基于 ReAct 框架运作的智能体，你需要自主决定如何调用以下三个核心检索工具（keyword_search, semantic_search, chunk_read）。请严格遵循以下工作模式：
1. 核心工作流：“先搜索，后精读”双步模式
   第一步（搜索）：根据用户问题，调用 keyword_search 或 semantic_search。注意：这两个工具只返回“文本缩略片段（snippet）”和对应的 chunk_id。
   第二步（精读）：评估搜索返回的片段。如果发现某个片段包含核心线索，你必须显式调用 chunk_read 工具获取完整上下文。
   第三步（完成）：只有在获取到充足的完整文本后，才停止调用工具。
2. 工具选择建议：
   - keyword_search：已知明确实体、专有名词、数字、日期时使用。
   - semantic_search：关键词检索效果不佳、需要概念性/模糊语义匹配时使用。
   - chunk_read：根据前两步返回的 chunk_id 读取完整语块全文。
3. 资源限制与安全兜底机制（强制生成）
   限制条件：你的思考与调用循环上限为 10 次，总上下文消耗上限为 128,000 tokens。
   兜底动作：一旦触碰上述任一红线，必须停止调用任何工具，基于当前已收集的信息结束检索。"""

    def __init__(
        self,
        embed_model: Any,
        llm: BaseChatModel,
        top_k: int | None = 5,
        persist_dir: str | None = None,
        max_loops: int = 10,
        max_token_budget: int = 128000,
        sentence_chunk_size: int = 128,
    ) -> None:
        self._embed_model = embed_model
        self._llm = llm
        self._similarity_top_k = top_k or 5
        self._persist_dir = persist_dir
        self._max_loops = max_loops
        self._max_token_budget = max_token_budget
        self._sentence_chunk_size = sentence_chunk_size

        self._chunks_db: dict[str, dict[str, Any]] = {}
        self._sentences: list[dict[str, Any]] = []
        self._sentence_vectors: np.ndarray | None = None

        if persist_dir and self.index_exists(persist_dir):
            self._load_index(persist_dir)

    # ------------------------------------------------------------------
    # Index persistence
    # ------------------------------------------------------------------

    @classmethod
    def index_exists(cls, persist_dir: str) -> bool:
        return all(
            os.path.exists(os.path.join(persist_dir, name))
            for name in (_CHUNKS_FILE, _SENTENCES_FILE, _VECTORS_FILE)
        )

    def _load_index(self, persist_dir: str | None = None) -> None:
        target = persist_dir or self._persist_dir
        if not target:
            raise ValueError("No persist_dir provided to load index.")

        chunks_path = os.path.join(target, _CHUNKS_FILE)
        sentences_path = os.path.join(target, _SENTENCES_FILE)
        vectors_path = os.path.join(target, _VECTORS_FILE)

        with open(chunks_path, "r", encoding="utf-8") as f:
            self._chunks_db = json.load(f)
        with open(sentences_path, "r", encoding="utf-8") as f:
            self._sentences = json.load(f)
        self._sentence_vectors = np.load(vectors_path).astype(np.float32)

        logger.info(
            "Layered index loaded from %r: %d chunks, %d sentences",
            target,
            len(self._chunks_db),
            len(self._sentences),
        )

    async def build_index(
        self,
        nodes: list[BaseNode] | None = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 50,
        **kwargs: Any,
    ) -> None:
        """从 LlamaIndex 节点构建三级索引（语块级 + 句子级向量）。"""
        if not nodes:
            raise RuntimeError("build_index requires a non-empty list of nodes")

        self._chunks_db = {}
        self._sentences = []

        sentence_splitter = SentenceSplitter(chunk_size=self._sentence_chunk_size, chunk_overlap=0)
        all_sentence_texts: list[str] = []

        for i, node in enumerate(nodes):
            chunk_id = getattr(node, "node_id", None) or f"chunk_{i}"
            text = node.get_content(metadata_mode="none") if hasattr(node, "get_content") else str(node.text)
            metadata = dict(getattr(node, "metadata", None) or {})
            metadata["chunk_id"] = chunk_id

            self._chunks_db[chunk_id] = {"text": text, "metadata": metadata}

            sub_nodes = sentence_splitter.get_nodes_from_documents(
                [LlamaDocument(text=text, metadata=metadata)]
            )
            if not sub_nodes:
                sub_nodes = self._split_sentences_fallback(text)

            for sent_id, sub in enumerate(sub_nodes):
                sent_text = sub.get_content(metadata_mode="none")
                sent_text = sent_text.strip()
                if not sent_text:
                    continue
                self._sentences.append(
                    {"chunk_id": chunk_id, "sent_id": sent_id, "text": sent_text}
                )
                all_sentence_texts.append(sent_text)

        if not all_sentence_texts:
            raise RuntimeError("No sentences extracted from nodes")

        logger.info("Embedding %d sentences for layered index...", len(all_sentence_texts))
        embeddings = await self._embed_texts_batched(all_sentence_texts)
        emb_arr = np.array(embeddings, dtype=np.float32)
        self._sentence_vectors = self._normalize_vectors(emb_arr)

        logger.info(
            "Layered index built: %d chunks, %d sentences, dim=%d",
            len(self._chunks_db),
            len(self._sentences),
            self._sentence_vectors.shape[1],
        )

    async def save(self, persist_dir: str | None = None) -> None:
        target = persist_dir or self._persist_dir
        if not target:
            raise ValueError("No persist_dir provided to save index.")
        if not self._chunks_db or self._sentence_vectors is None:
            raise RuntimeError("No index to save; call build_index first.")

        os.makedirs(target, exist_ok=True)
        with open(os.path.join(target, _CHUNKS_FILE), "w", encoding="utf-8") as f:
            json.dump(self._chunks_db, f, ensure_ascii=False, indent=2)
        with open(os.path.join(target, _SENTENCES_FILE), "w", encoding="utf-8") as f:
            json.dump(self._sentences, f, ensure_ascii=False, indent=2)
        np.save(os.path.join(target, _VECTORS_FILE), self._sentence_vectors)
        self._persist_dir = target
        logger.info("Layered index saved to %r", target)

    # ------------------------------------------------------------------
    # Three retrieval tools
    # ------------------------------------------------------------------

    async def keyword_search(self, query: str, top_k: int | None = None) -> list[Document]:
        """关键词级精确匹配，返回含关键词的句子片段。"""
        keywords = self._extract_keywords(query)
        if not keywords or not self._chunks_db:
            return []

        k = top_k or self._similarity_top_k
        scored: list[tuple[float, str, str]] = []

        for chunk_id, entry in self._chunks_db.items():
            full_text = entry["text"]
            text_lower = full_text.lower()
            score = 0.0
            for kw in keywords:
                kw_lower = kw.lower()
                score += text_lower.count(kw_lower) * len(kw)

            if score > 0:
                snippet = self._extract_keyword_snippet(full_text, keywords)
                scored.append((score, chunk_id, snippet))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            Document(
                text=snippet,
                score=score,
                metadata={"chunk_id": chunk_id, "type": "keyword_snippet"},
            )
            for score, chunk_id, snippet in scored[:k]
        ]

    async def semantic_search(self, query: str, top_k: int | None = None) -> list[Document]:
        """句子级语义匹配，按语块内最高相似度句子打分。"""
        if not self._sentences or self._sentence_vectors is None:
            return []

        k = top_k or self._similarity_top_k
        q_vec = await self._embed_model.aget_query_embedding(query)
        q_arr = np.array(q_vec, dtype=np.float32).reshape(1, -1)
        q_arr = self._normalize_vectors(q_arr)[0]

        sims = self._sentence_vectors @ q_arr
        chunk_best: dict[str, tuple[float, str]] = {}

        for idx, sent in enumerate(self._sentences):
            cid = sent["chunk_id"]
            score = float(sims[idx])
            sent_text = sent["text"]
            if cid not in chunk_best or score > chunk_best[cid][0]:
                chunk_best[cid] = (score, sent_text)

        sorted_chunks = sorted(chunk_best.items(), key=lambda x: x[1][0], reverse=True)
        return [
            Document(
                text=best_sentence,
                score=score,
                metadata={"chunk_id": cid, "type": "semantic_snippet"},
            )
            for cid, (score, best_sentence) in sorted_chunks[:k]
        ]

    async def chunk_read(
        self,
        chunk_ids: list[str],
        context: AgentContext,
    ) -> list[Document]:
        """语块级全文读取，配合上下文追踪器避免重复读取。"""
        results: list[Document] = []
        for cid in chunk_ids:
            cid = str(cid).strip()
            if not cid or cid in context.seen_chunks:
                continue
            entry = self._chunks_db.get(cid)
            if not entry:
                continue

            context.seen_chunks.add(cid)
            doc = Document(
                text=entry["text"],
                score=1.0,
                metadata={**entry.get("metadata", {}), "chunk_id": cid, "type": "full_chunk"},
            )
            results.append(doc)
            context.retrieved_docs.append(doc)

        return results

    # ------------------------------------------------------------------
    # ReAct orchestration
    # ------------------------------------------------------------------

    async def retrieve(self, query: str, filters: Any = None) -> list[Document]:
        """LLM 驱动的 ReAct 检索循环，返回已精读的完整语块。"""
        if not self._chunks_db:
            logger.warning("LayeredRetriever: index not loaded, returning empty results")
            return []

        context = AgentContext(
            max_loops=self._max_loops,
            max_token_budget=self._max_token_budget,
        )
        agent_context = context

        async def _keyword_search_tool(query: str) -> str:
            docs = await self.keyword_search(query)
            return self._format_snippet_observation(docs)

        async def _semantic_search_tool(query: str) -> str:
            docs = await self.semantic_search(query)
            return self._format_snippet_observation(docs)

        async def _chunk_read_tool(chunk_ids: List[str]) -> str:
            docs = await self.chunk_read(chunk_ids, agent_context)
            if not docs:
                return "No new chunks read (already seen or invalid ids)."
            ids = [d.metadata.get("chunk_id", "") for d in docs]
            return f"Successfully read full text for chunk(s): {', '.join(ids)}"

        tools = [
            StructuredTool.from_function(
                coroutine=_keyword_search_tool,
                name="keyword_search",
                description="基于精确词法匹配检索语块片段，适用于实体、专有名词、数字、日期。",
                args_schema=KeywordSearchInput,
            ),
            StructuredTool.from_function(
                coroutine=_semantic_search_tool,
                name="semantic_search",
                description="基于语义相似度检索语块片段，适用于概念性或模糊查询。",
                args_schema=SemanticSearchInput,
            ),
            StructuredTool.from_function(
                coroutine=_chunk_read_tool,
                name="chunk_read",
                description="根据 chunk_id 列表读取语块完整文本，用于精读高相关语块。",
                args_schema=ChunkReadInput,
            ),
        ]

        llm_with_tools = self._llm.bind_tools(tools)
        messages: list[Any] = [
            SystemMessage(content=self.REACT_SYSTEM_PROMPT),
            HumanMessage(content=f"User Query: {query}"),
        ]
        context.add_tokens(self._count_tokens(self.REACT_SYSTEM_PROMPT + query))

        tool_map = {t.name: t for t in tools}

        while context.loop_count < context.max_loops:
            context.loop_count += 1

            try:
                response: AIMessage = await llm_with_tools.ainvoke(messages)
            except Exception as exc:
                logger.warning("ReAct LLM invoke failed (%s), falling back to semantic_search", exc)
                sem_docs = await self.semantic_search(query)
                top_ids = [
                    d.metadata.get("chunk_id")
                    for d in sem_docs
                    if d.metadata.get("chunk_id")
                ]
                if top_ids:
                    await self.chunk_read(top_ids[: self._similarity_top_k], context)
                break

            context.add_tokens(self._count_tokens(str(response.content)))

            if not getattr(response, "tool_calls", None):
                break

            messages.append(response)
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call.get("args", {})
                tool_fn = tool_map.get(tool_name)
                observation = "Unknown tool."
                if tool_fn:
                    try:
                        observation = await tool_fn.ainvoke(tool_args)
                    except Exception as exc:
                        observation = f"Tool error: {exc}"
                        logger.warning("Tool %s failed: %s", tool_name, exc)

                messages.append(
                    ToolMessage(content=str(observation), tool_call_id=tool_call["id"])
                )
                context.add_tokens(self._count_tokens(str(observation)))

            if context.is_budget_exceeded():
                self._force_final_answer(context)
                break

        if not context.retrieved_docs:
            logger.info("ReAct finished without chunk_read; auto-reading top semantic hits")
            sem_docs = await self.semantic_search(query)
            chunk_ids = [
                d.metadata["chunk_id"]
                for d in sem_docs
                if d.metadata.get("chunk_id")
            ]
            if chunk_ids:
                await self.chunk_read(chunk_ids[: self._similarity_top_k], context)

        return context.retrieved_docs

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        return vectors / norms

    async def _embed_texts_batched(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for i in range(0, len(texts), _EMBED_BATCH_SIZE):
            batch = texts[i : i + _EMBED_BATCH_SIZE]
            batch_emb = await asyncio.to_thread(
                self._embed_model.get_text_embedding_batch, batch
            )
            embeddings.extend(batch_emb)
        return embeddings

    @staticmethod
    def _extract_keywords(query: str) -> list[str]:
        words = [w.strip() for w in jieba.cut(query) if w.strip()]
        if not words:
            words = [w for w in re.split(r"\s+", query) if w.strip()]
        stop = {"的", "是", "在", "和", "与", "了", "吗", "什么", "哪", "谁", "如何", "the", "a", "an", "is", "are", "what", "who", "how"}
        return [w for w in words if w.lower() not in stop and len(w) > 1]

    @staticmethod
    def _extract_keyword_snippet(text: str, keywords: list[str], window: int = 80) -> str:
        text_lower = text.lower()
        best_pos = -1
        for kw in keywords:
            idx = text_lower.find(kw.lower())
            if idx != -1 and (best_pos == -1 or idx < best_pos):
                best_pos = idx
        if best_pos == -1:
            return text[: min(120, len(text))] + ("..." if len(text) > 120 else "")
        start = max(0, best_pos - window)
        end = min(len(text), best_pos + window)
        snippet = text[start:end]
        if start > 0:
            snippet = "..." + snippet
        if end < len(text):
            snippet = snippet + "..."
        return snippet

    @staticmethod
    def _split_sentences_fallback(text: str) -> list[LlamaDocument]:
        """标点分句兜底。"""
        parts = re.split(r"(?<=[。！？.!?])\s*", text)
        return [LlamaDocument(text=p.strip()) for p in parts if p.strip()]

    @staticmethod
    def _format_snippet_observation(docs: list[Document]) -> str:
        if not docs:
            return "No matching snippets found."
        lines = []
        for d in docs:
            cid = d.metadata.get("chunk_id", "unknown")
            lines.append(f"chunk_id: {cid} | score: {d.score:.4f} | snippet: {d.text}")
        return "\n".join(lines)

    @staticmethod
    def _count_tokens(text: str) -> int:
        return max(1, len(text) // 4)

    @staticmethod
    def _force_final_answer(context: AgentContext) -> str:
        logger.warning(
            "Resource limit reached (loops=%d, tokens=%d). Forcing final answer.",
            context.loop_count,
            context.token_used,
        )
        return "RESOURCE_LIMIT_REACHED"
