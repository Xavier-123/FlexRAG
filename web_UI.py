import asyncio
import os
import logging
import gradio as gr

from flexrag.workflows import RAGPipeline
from flexrag.common import Settings, setup_logging
from flexrag.components.pre_retrieval import (
    PreQueryOptimizer,
    QueryRewriter,
    QueryExpander,
    TaskSplitter,
    TerminologyEnricher,
)
from flexrag.components.retrieval import (
    HybridRetriever,
    BM25Retriever,
    GraphRetriever,
    MultiVectorRetriever,
    OpenAILikeEmbedding,
)
from flexrag.components.post_retrieval import PostRetrieval, OpenAILikeReranker, LLMContextOptimizer
from flexrag.components.reasoning import LLMContextEvaluator, OpenAIGenerator
from langchain_openai import ChatOpenAI

# ================== 全局配置与缓存 ==================
settings = Settings()

# 知识库路径映射
KB_DICT = {
    "hotpotqa": "./data/knowledge_persist_dir/hotpotqa",
    "2wikimultihopqa": "./data/knowledge_persist_dir/2wikimultihopqa",
    "musique": "./data/knowledge_persist_dir/musique",
    "nq": "./data/knowledge_persist_dir/nq"
}

# 共享的基础组件（LLM、EmbedModel 等与知识库无关的通用组件）
base_components = {}
# Pipeline 缓存，按 (kb_name, retrievers, pre_opts, post_opts) 组合键存储
pipelines_cache = {}

# ================== Pipeline 节点元信息 ==================

# LangGraph 节点的执行顺序（用于状态面板渲染）
PIPELINE_ORDER = [
    "pre_retrieval_optimizer",
    "retrieve",
    "post_retrieval_optimizer",
    "context_evaluator",
    "generate",
]

# 节点显示名称与图标
NODE_META = {
    "pre_retrieval_optimizer": {"icon": "⚡", "label": "查询优化"},
    "retrieve":                 {"icon": "🔍", "label": "文档检索"},
    "post_retrieval_optimizer": {"icon": "🎯", "label": "后处理优化"},
    "context_evaluator":        {"icon": "🧐", "label": "上下文评估"},
    "generate":                 {"icon": "✨", "label": "答案生成"},
}

# 各节点下的子组件映射（配置选项名 → 中文描述）
_SUB_COMPONENTS = {
    "pre_retrieval_optimizer": {
        "QueryRewriter":       "查询改写",
        "QueryExpander":       "查询扩展",
        "TaskSplitter":        "问题分解",
        "TerminologyEnricher": "术语增强",
    },
    "retrieve": {
        "MultiVectorRetriever": "向量检索",
        "BM25Retriever":        "BM25检索",
        "GraphRetriever":       "图谱检索",
    },
    "post_retrieval_optimizer": {
        "OpenAILikeReranker":  "重排序",
        "LLMContextOptimizer": "上下文精炼",
    },
}


def _active_subcomponents(
    node: str,
    pre_opt_names: list,
    retriever_names: list,
    post_opt_names: list,
) -> list[str]:
    """Return human-readable labels for the active sub-components of *node*."""
    mapping = _SUB_COMPONENTS.get(node, {})
    if node == "pre_retrieval_optimizer":
        source = pre_opt_names
    elif node == "retrieve":
        source = retriever_names
    elif node == "post_retrieval_optimizer":
        source = post_opt_names
    else:
        return []
    return [mapping.get(n, n) for n in source if n in mapping]


# ================== 执行状态 HTML 渲染 ==================

_PULSE_CSS = """
<style>
@keyframes flexrag-pulse {
  0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(59,130,246,0.4); }
  50%       { opacity: 0.85; box-shadow: 0 0 0 6px rgba(59,130,246,0); }
}
</style>
"""

# ================== 多轮详情渲染 ==================

# Human-readable field names shown in the detail panels.
_FIELD_LABELS: dict[str, str] = {
    "original_query":       "原始问题",
    "optimized_queries":    "优化后查询列表",
    "current_queries":      "当前查询",
    "missing_info":         "缺失信息",
    "missing_info_history": "缺失信息历史",
    "iteration_count":      "迭代次数",
    "retrieved_docs":       "检索文档",
    "optimized_context":    "优化后上下文",
    "accumulated_context":  "累积上下文",
    "context_sufficient":   "上下文充足",
    "judge_reason":         "评估理由",
    "answer":               "最终答案",
    "evidence":             "引用来源",
}

# Fields to display for each node's input and output.
_NODE_DISPLAY_FIELDS: dict[str, dict[str, list[str]]] = {
    "pre_retrieval_optimizer": {
        "input":  ["original_query", "iteration_count", "missing_info", "missing_info_history"],
        "output": ["optimized_queries", "current_queries"],
    },
    "retrieve": {
        "input":  ["optimized_queries"],
        "output": ["retrieved_docs"],
    },
    "post_retrieval_optimizer": {
        "input":  ["original_query", "retrieved_docs"],
        "output": ["optimized_context"],
    },
    "context_evaluator": {
        "input":  ["original_query", "optimized_context"],
        "output": ["context_sufficient", "missing_info", "judge_reason", "accumulated_context"],
    },
    "generate": {
        "input":  ["original_query", "accumulated_context"],
        "output": ["answer", "evidence"],
    },
}

_ROUNDS_DETAIL_CSS = """
<style>
.rd-wrap{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;font-size:13px;margin:8px 0;}
.rd-round{border:1px solid #e5e7eb;border-radius:10px;margin-bottom:10px;overflow:hidden;background:#fff;}
.rd-round>details>summary{cursor:pointer;padding:10px 14px;background:#f0f9ff;font-weight:700;color:#1e40af;
  list-style:none;display:flex;align-items:center;gap:8px;border-bottom:1px solid transparent;}
.rd-round>details>summary::-webkit-details-marker{display:none;}
.rd-round>details[open]>summary{border-bottom:1px solid #bfdbfe;}
.rd-nodes{padding:10px;display:flex;flex-direction:column;gap:8px;}
.rd-node{border:1px solid #e5e7eb;border-radius:8px;overflow:hidden;}
.rd-node>details>summary{cursor:pointer;padding:7px 12px;font-weight:600;color:#374151;background:#f9fafb;
  list-style:none;display:flex;align-items:center;gap:6px;}
.rd-node>details>summary::-webkit-details-marker{display:none;}
.rd-node>details[open]>summary{border-bottom:1px solid #e5e7eb;}
.rd-io{padding:10px 14px;display:grid;grid-template-columns:1fr 1fr;gap:10px;}
.rd-io-section{background:#fafafa;border:1px solid #e5e7eb;border-radius:6px;padding:8px 10px;}
.rd-io-section h4{margin:0 0 6px 0;font-size:11px;text-transform:uppercase;letter-spacing:0.05em;color:#6b7280;font-weight:700;}
.rd-field{margin:4px 0;}
.rd-field-key{font-weight:600;color:#374151;font-size:12px;}
.rd-field-val{color:#1f2937;font-size:12px;margin-top:2px;word-break:break-word;white-space:pre-wrap;}
.rd-badge{font-size:11px;padding:1px 8px;border-radius:9px;margin-left:auto;}
.rd-ok{background:#d1fae5;color:#065f46;}
.rd-retry{background:#fef3c7;color:#92400e;}
.rd-running{background:#dbeafe;color:#1e40af;}
</style>
"""


def _escape(text: str) -> str:
    """HTML-escape a string."""
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
    )


# Truncation limits used when formatting state values for display.
_MAX_STR_LEN = 600
_MAX_DOC_PREVIEW_LEN = 250
_MAX_LIST_ITEM_LEN = 300
_MAX_DICT_STR_LEN = 500


def _fmt_value(key: str, value) -> str:
    """Format a single state field value as safe HTML."""
    if value is None:
        return "<em style='color:#9ca3af;'>—</em>"

    if isinstance(value, bool):
        return "✅ 是" if value else "❌ 否"

    if isinstance(value, int):
        return str(value)

    if isinstance(value, str):
        truncated = value[:_MAX_STR_LEN] + ("…" if len(value) > _MAX_STR_LEN else "")
        return _escape(truncated)

    if isinstance(value, list):
        if not value:
            return "<em style='color:#9ca3af;'>（空列表）</em>"
        if key == "retrieved_docs":
            count = len(value)
            parts = [f"<em>共 {count} 个文档</em>"]
            for i, doc in enumerate(value[:3], 1):
                text = (doc.get("text", "") if isinstance(doc, dict) else str(doc))[:_MAX_DOC_PREVIEW_LEN]
                parts.append(f"<b>[{i}]</b> {_escape(text)}{'…' if len(text) == _MAX_DOC_PREVIEW_LEN else ''}")
            if count > 3:
                parts.append(f"<em>… 另有 {count - 3} 个文档</em>")
            return "<br>".join(parts)
        # Generic list
        items = []
        for i, item in enumerate(value[:5], 1):
            s = str(item)
            items.append(f"[{i}] " + _escape(s[:_MAX_LIST_ITEM_LEN] + ("…" if len(s) > _MAX_LIST_ITEM_LEN else "")))
        if len(value) > 5:
            items.append(f"<em>… 另有 {len(value) - 5} 项</em>")
        return "<br>".join(items)

    if isinstance(value, dict):
        s = str(value)
        return _escape(s[:_MAX_DICT_STR_LEN] + ("…" if len(s) > _MAX_DICT_STR_LEN else ""))

    return _escape(str(value)[:_MAX_DICT_STR_LEN])


def _render_io_fields(node_name: str, data: dict, io_type: str) -> str:
    """Render the input or output fields for a single node as HTML rows."""
    if not data:
        return "<em style='color:#9ca3af;'>暂无数据</em>"
    fields = _NODE_DISPLAY_FIELDS.get(node_name, {}).get(io_type, list(data.keys()))
    parts = []
    for key in fields:
        if key not in data:
            continue
        label = _FIELD_LABELS.get(key, key)
        formatted = _fmt_value(key, data[key])
        parts.append(
            f'<div class="rd-field">'
            f'<div class="rd-field-key">{label}</div>'
            f'<div class="rd-field-val">{formatted}</div>'
            f"</div>"
        )
    return "\n".join(parts) if parts else "<em style='color:#9ca3af;'>暂无数据</em>"


def render_rounds_detail(rounds_data: list) -> str:
    """Render per-round, per-node collapsible detail panels as HTML.

    Args:
        rounds_data: List of round dicts, each with keys ``round`` (int),
            ``nodes`` (dict of node_name → ``{input, output}``), and
            ``status`` (``"running"`` | ``"done"``).

    Returns:
        An HTML string suitable for a ``gr.HTML`` component.
    """
    if not rounds_data or not any(r["nodes"] for r in rounds_data):
        return ""

    parts = [_ROUNDS_DETAIL_CSS, '<div class="rd-wrap">']
    parts.append(
        '<p style="font-weight:700;color:#374151;margin:0 0 8px 0;font-size:13px;">'
        '📋 各轮次详细数据</p>'
    )

    for round_data in rounds_data:
        rnum = round_data["round"]
        nodes = round_data["nodes"]
        status = round_data.get("status", "running")

        # Badge based on context_evaluator output
        eval_out = nodes.get("context_evaluator", {}).get("output", {})
        if eval_out:
            sufficient = eval_out.get("context_sufficient")
            if sufficient is True:
                badge = '<span class="rd-badge rd-ok">✅ 上下文充足</span>'
            elif sufficient is False:
                badge = '<span class="rd-badge rd-retry">🔄 需补充信息</span>'
            else:
                badge = ""
        elif status == "running":
            badge = '<span class="rd-badge rd-running">⚡ 进行中</span>'
        else:
            badge = ""

        parts.append(
            f'<div class="rd-round"><details{"" if rnum > 1 else " open"}>'
            f'<summary>🔁 第 {rnum} 轮 {badge}</summary>'
            f'<div class="rd-nodes">'
        )

        for node_name in PIPELINE_ORDER:
            if node_name not in nodes:
                continue
            node_data = nodes[node_name]
            meta = NODE_META[node_name]
            icon, label = meta["icon"], meta["label"]
            input_html = _render_io_fields(node_name, node_data.get("input", {}), "input")
            output_html = _render_io_fields(node_name, node_data.get("output", {}), "output")
            parts.append(
                f'<div class="rd-node"><details>'
                f"<summary>{icon} {label}</summary>"
                f'<div class="rd-io">'
                f'<div class="rd-io-section"><h4>📥 输入</h4>{input_html}</div>'
                f'<div class="rd-io-section"><h4>📤 输出</h4>{output_html}</div>'
                f"</div></details></div>"
            )

        parts.append("</div></details></div>")

    parts.append("</div>")
    return "\n".join(parts)


def render_execution_status(
    completed_nodes: list,
    current_node: str | None,
    pre_opt_names: list,
    retriever_names: list,
    post_opt_names: list,
    iteration: int = 1,
    done: bool = False,
    error: str | None = None,
) -> str:
    """Render the live pipeline execution status as an HTML string.

    Args:
        completed_nodes: Ordered list of node names that have already finished
            (may contain duplicates when the pipeline loops).
        current_node: Node that is currently executing, or ``None``.
        pre_opt_names / retriever_names / post_opt_names: Active component names
            used to generate the per-node sub-component hint.
        iteration: Current loop iteration number (starts at 1).
        done: ``True`` when the whole pipeline has finished successfully.
        error: Non-empty error message when the pipeline failed.
    """
    nodes_html_parts = []

    for node in PIPELINE_ORDER:
        count = completed_nodes.count(node)
        is_running = node == current_node
        is_done = count > 0
        meta = NODE_META[node]
        icon = meta["icon"]
        label = meta["label"]

        if is_running:
            subs = _active_subcomponents(node, pre_opt_names, retriever_names, post_opt_names)
            sub_line = (
                f'<div style="font-size:11px;margin-top:3px;color:#1d4ed8;">'
                + " · ".join(subs)
                + "</div>"
                if subs
                else ""
            )
            node_html = (
                f'<div style="'
                f"background:#dbeafe;border:2px solid #3b82f6;border-radius:8px;"
                f"padding:8px 14px;text-align:center;min-width:90px;"
                f'animation:flexrag-pulse 1.4s ease-in-out infinite;">'
                f'<div style="font-size:15px;">🔄</div>'
                f'<div style="font-size:12px;font-weight:700;color:#1e40af;margin-top:2px;">{label}</div>'
                f"{sub_line}"
                f"</div>"
            )
        elif is_done:
            badge = (
                f' <span style="font-size:10px;background:#6ee7b7;'
                f'border-radius:8px;padding:1px 5px;color:#065f46;">×{count}</span>'
                if count > 1
                else ""
            )
            node_html = (
                f'<div style="'
                f"background:#d1fae5;border:2px solid #10b981;border-radius:8px;"
                f'padding:8px 14px;text-align:center;min-width:90px;">'
                f'<div style="font-size:15px;">✅</div>'
                f'<div style="font-size:12px;font-weight:700;color:#065f46;margin-top:2px;">{label}{badge}</div>'
                f"</div>"
            )
        else:
            node_html = (
                f'<div style="'
                f"background:#f3f4f6;border:2px solid #e5e7eb;border-radius:8px;"
                f'padding:8px 14px;text-align:center;min-width:90px;opacity:0.55;">'
                f'<div style="font-size:15px;">{icon}</div>'
                f'<div style="font-size:12px;font-weight:600;color:#9ca3af;margin-top:2px;">{label}</div>'
                f"</div>"
            )
        nodes_html_parts.append(node_html)

    arrow = '<div style="font-size:16px;color:#9ca3af;padding:0 2px;display:flex;align-items:center;">→</div>'
    flow_html = arrow.join(nodes_html_parts)

    if error:
        header = f'<div style="color:#ef4444;font-weight:700;margin-bottom:8px;font-size:13px;">❌ 执行出错</div>'
    elif done:
        iter_info = f"  ·  共 {iteration} 轮迭代" if iteration > 1 else ""
        header = f'<div style="color:#10b981;font-weight:700;margin-bottom:8px;font-size:13px;">✅ Pipeline 执行完成{iter_info}</div>'
    elif current_node:
        iter_tag = f"第 {iteration} 轮  ·  " if iteration > 1 else ""
        header = f'<div style="color:#3b82f6;font-weight:700;margin-bottom:8px;font-size:13px;">🔄 {iter_tag}正在执行…</div>'
    elif completed_nodes:
        header = f'<div style="color:#6b7280;font-weight:700;margin-bottom:8px;font-size:13px;">⏳ 第 {iteration} 轮  ·  等待下一节点…</div>'
    else:
        header = '<div style="color:#6b7280;font-weight:700;margin-bottom:8px;font-size:13px;">🚀 Pipeline 启动中…</div>'

    return (
        _PULSE_CSS
        + f'<div style="background:#ffffff;border:1px solid #e5e7eb;border-radius:12px;'
        f'padding:12px 16px;margin:6px 0;box-shadow:0 1px 4px rgba(0,0,0,0.08);">'
        f"{header}"
        f'<div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;">'
        f"{flow_html}"
        f"</div></div>"
    )


# ================== 初始化基础组件 ==================
def init_base_components():
    """初始化大模型、Embedding 等与具体知识库和策略无关的通用组件"""
    global base_components

    llm = ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        temperature=0.0,
    )

    embed_model = OpenAILikeEmbedding(
        model_name=settings.embedding_model,
        base_url=settings.embedding_base_url,
        api_key=settings.embedding_api_key,
    )

    context_evaluator = LLMContextEvaluator(llm=llm)
    generator = OpenAIGenerator(
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
    )

    base_components = {
        "llm": llm,
        "embed_model": embed_model,
        "context_evaluator": context_evaluator,
        "generator": generator,
        "settings": settings,
    }


# ================== 动态构建并缓存 Pipeline ==================
async def get_or_load_pipeline(
    kb_name: str,
    retriever_names: list,
    pre_opt_names: list,
    post_opt_names: list,
) -> RAGPipeline:
    """根据所选知识库和策略动态构建并缓存 Pipeline"""
    cache_key = (
        kb_name,
        frozenset(retriever_names),
        frozenset(pre_opt_names),
        frozenset(post_opt_names),
    )
    if cache_key in pipelines_cache:
        return pipelines_cache[cache_key]

    print(
        f"🔄 正在构建 Pipeline ("
        f"知识库={kb_name}, "
        f"检索器={retriever_names}, "
        f"预检索={pre_opt_names}, "
        f"后检索={post_opt_names}) ..."
    )

    persist_dir = KB_DICT.get(kb_name, settings.knowledge_persist_dir)
    llm = base_components["llm"]
    embed_model = base_components["embed_model"]

    # ---- 1. 构建检索器列表 ----
    def _make_multi_vector_retriever() -> MultiVectorRetriever:
        return MultiVectorRetriever(
            embed_model=embed_model,
            index=None,
            top_k=5,
            persist_dir=persist_dir,
        )

    retrievers = []
    if "MultiVectorRetriever" in retriever_names:
        retrievers.append(_make_multi_vector_retriever())
    if "BM25Retriever" in retriever_names:
        retrievers.append(
            BM25Retriever(
                top_k=5,
                persist_dir=os.path.join(persist_dir, "bm25_index"),
            )
        )
    if "GraphRetriever" in retriever_names:
        retrievers.append(
            GraphRetriever(
                llm=llm,
                embed_model=embed_model,
                persist_dir=os.path.join(persist_dir, "graph_index"),
            )
        )
    # 至少保留一个检索器（防御性兜底，UI 侧已作验证）
    if not retrievers:
        retrievers.append(_make_multi_vector_retriever())
    retriever = HybridRetriever(retrievers=retrievers)

    # ---- 2. 构建预检索优化器列表 ----
    pre_opts = []
    if "QueryRewriter" in pre_opt_names:
        pre_opts.append(QueryRewriter(llm=llm))
    if "QueryExpander" in pre_opt_names:
        pre_opts.append(QueryExpander(llm=llm))
    if "TaskSplitter" in pre_opt_names:
        pre_opts.append(TaskSplitter(llm=llm))
    if "TerminologyEnricher" in pre_opt_names:
        pre_opts.append(TerminologyEnricher(llm=llm))
    pre_retrieval_optimizer = PreQueryOptimizer(pre_opts)

    # ---- 3. 构建后检索优化器列表 ----
    post_opts = []
    if "OpenAILikeReranker" in post_opt_names:
        post_opts.append(
            OpenAILikeReranker(
                base_url=settings.reranker_base_url,
                model=settings.reranker_model,
                api_key=settings.reranker_api_key,
                top_k=settings.top_k_rerank,
            )
        )
    if "LLMContextOptimizer" in post_opt_names:
        post_opts.append(LLMContextOptimizer(llm=llm))
    post_retrieval_optimizer = PostRetrieval(post_opts)

    # ---- 4. 组装 Pipeline ----
    pipeline = RAGPipeline(
        pre_retrieval_optimizer=pre_retrieval_optimizer,
        retriever=retriever,
        post_retrieval_optimizer=post_retrieval_optimizer,
        context_evaluator=base_components["context_evaluator"],
        generator=base_components["generator"],
        settings=base_components["settings"],
    )

    pipelines_cache[cache_key] = pipeline
    print("✅ Pipeline 构建完成！")
    return pipeline


# ================== Gradio 响应逻辑（流式 Async 生成器） ==================
async def respond(message, chat_history, kb_name, retriever_names, pre_opt_names, post_opt_names):
    """Async generator: streams node-level execution status to the UI in real-time.

    Yields a 4-tuple ``(msg_input_value, chat_history, status_html, rounds_html)`` on
    every meaningful event so that Gradio re-renders both panels without waiting
    for the whole pipeline to finish.
    """
    if not message.strip():
        yield "", chat_history, "", ""
        return

    if chat_history is None:
        chat_history = []

    if not retriever_names:
        new_chat = chat_history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "⚠️ **请至少选择一个检索器后再提问。**"},
        ]
        yield "", new_chat, "", ""
        return

    # ── 初始状态：Pipeline 启动 ──────────────────────────────────────────────
    status = render_execution_status([], None, pre_opt_names, retriever_names, post_opt_names)
    yield "", chat_history, status, ""

    try:
        pipeline = await get_or_load_pipeline(kb_name, retriever_names, pre_opt_names, post_opt_names)
    except Exception as e:
        error_chat = chat_history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": f"❌ **Pipeline 初始化失败**: {e}"},
        ]
        yield "", error_chat, "", ""
        return

    # ── 流式执行 Pipeline ────────────────────────────────────────────────────
    completed_nodes: list = []
    current_node: str | None = None
    iteration = 1
    first_pre_retrieval = True
    answer = ""
    evidence: list = []

    # Per-round detail tracking
    rounds_data: list = [{"round": 1, "nodes": {}, "status": "running"}]

    async for event in pipeline.astream_run(message):
        etype = event["type"]

        if etype == "node_start":
            node = event["node"]
            # Every time pre_retrieval_optimizer starts again → new iteration
            if node == "pre_retrieval_optimizer":
                if first_pre_retrieval:
                    first_pre_retrieval = False
                else:
                    iteration += 1
                    # Finalise the previous round and open a new one
                    rounds_data[-1]["status"] = "done"
                    rounds_data.append({"round": iteration, "nodes": {}, "status": "running"})
            current_node = node
            # Store input data for this node in the current round
            if node not in rounds_data[-1]["nodes"]:
                rounds_data[-1]["nodes"][node] = {}
            rounds_data[-1]["nodes"][node]["input"] = event.get("input", {})

            status = render_execution_status(
                completed_nodes, current_node,
                pre_opt_names, retriever_names, post_opt_names,
                iteration=iteration,
            )
            rounds_html = render_rounds_detail(rounds_data)
            yield "", chat_history, status, rounds_html

        elif etype == "node_end":
            node = event["node"]
            completed_nodes.append(node)
            current_node = None
            # Store output data for this node in the current round
            if node not in rounds_data[-1]["nodes"]:
                rounds_data[-1]["nodes"][node] = {}
            rounds_data[-1]["nodes"][node]["output"] = event.get("output", {})

            status = render_execution_status(
                completed_nodes, current_node,
                pre_opt_names, retriever_names, post_opt_names,
                iteration=iteration,
            )
            rounds_html = render_rounds_detail(rounds_data)
            yield "", chat_history, status, rounds_html

        elif etype == "result":
            answer = event.get("answer", "")
            evidence = event.get("evidence", [])

        elif etype == "error":
            error_chat = chat_history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": f"❌ **发生错误**: {event['message']}"},
            ]
            status = render_execution_status(
                completed_nodes, None,
                pre_opt_names, retriever_names, post_opt_names,
                iteration=iteration,
                error=event["message"],
            )
            rounds_data[-1]["status"] = "done"
            rounds_html = render_rounds_detail(rounds_data)
            yield "", error_chat, status, rounds_html
            return

    # ── 构造最终回复（含折叠的 evidence） ────────────────────────────────────
    evidence_html = "\n\n<details><summary><b>👉 点击展开查看参考检索片段</b></summary>\n\n"
    if evidence:
        for i, ev in enumerate(evidence, 1):
            preview = ev[:200] + ("..." if len(ev) > 200 else "")
            evidence_html += f"**[来源{i}]** {preview}\n\n"
    else:
        evidence_html += "未检索到相关片段。\n\n"
    evidence_html += "</details>"

    final_message = answer + evidence_html
    new_chat = chat_history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": final_message},
    ]
    rounds_data[-1]["status"] = "done"
    status = render_execution_status(
        completed_nodes, None,
        pre_opt_names, retriever_names, post_opt_names,
        iteration=iteration,
        done=True,
    )
    rounds_html = render_rounds_detail(rounds_data)
    yield "", new_chat, status, rounds_html


# ================== UI ==================
custom_css = """
.gr-chatbot {
    font-family: "JetBrains Mono", "Microsoft YaHei", sans-serif;
    font-size: 15px;
}

.gr-chatbot .message {
    line-height: 1.7;
}

textarea {
    font-family: "JetBrains Mono", "Microsoft YaHei";
    font-size: 15px;
}

.settings-panel {
    border-right: 1px solid #e0e0e0;
    padding-right: 12px;
}
"""

with gr.Blocks(theme=gr.themes.Soft(), title="FlexRAG 智能问答系统", css=custom_css) as demo:
    gr.Markdown("""
    <h1 style="text-align: center;">🤖 FlexRAG 智能问答系统</h1>
    <p style="text-align: center;">支持多知识库 · 动态检索器 · 灵活优化策略</p>
    """)

    with gr.Row():
        # ---- 左侧设置面板 ----
        with gr.Column(scale=2, elem_classes="settings-panel"):
            gr.Markdown("### ⚙️ 系统配置")

            kb_selector = gr.Dropdown(
                choices=list(KB_DICT.keys()),
                value="hotpotqa",
                label="📚 知识库（首次切换时加载，后续秒切）",
            )

            gr.Markdown("---")

            retriever_selector = gr.CheckboxGroup(
                choices=["MultiVectorRetriever", "BM25Retriever", "GraphRetriever"],
                value=["MultiVectorRetriever"],
                label="🔍 检索器（至少选一个）",
                info="MultiVectorRetriever: 向量检索 | BM25Retriever: 关键词检索 | GraphRetriever: 图谱检索",
            )

            gr.Markdown("---")

            pre_opt_selector = gr.CheckboxGroup(
                choices=["QueryRewriter", "QueryExpander", "TaskSplitter", "TerminologyEnricher"],
                value=[],
                label="⚡ 检索前优化策略",
                info="QueryRewriter: 查询改写 | QueryExpander: 查询扩展 | TaskSplitter: 问题分解 | TerminologyEnricher: 术语增强",
            )

            gr.Markdown("---")

            post_opt_selector = gr.CheckboxGroup(
                choices=["OpenAILikeReranker", "LLMContextOptimizer"],
                value=["LLMContextOptimizer"],
                label="🎯 检索后优化策略",
                info="OpenAILikeReranker: 重排序 | LLMContextOptimizer: LLM 上下文精炼",
            )

        # ---- 右侧对话区 ----
        with gr.Column(scale=5):
            chatbot = gr.Chatbot(
                height=420,  # reduced from 480 to leave room for the status panel below
                buttons=["copy"],
                layout="bubble",
            )

            # ---- 执行状态实时面板 ----
            status_display = gr.HTML(
                value="",
                label="Pipeline 执行状态",
            )

            # ---- 多轮详情面板（可折叠） ----
            rounds_display = gr.HTML(
                value="",
                label="各轮次详细数据",
            )

            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="请输入问题...",
                    scale=8,
                    show_label=False,
                )
                submit_btn = gr.Button("发送 ▶", variant="primary", scale=1)

    # ---- 绑定提交事件 ----
    shared_inputs = [msg_input, chatbot, kb_selector, retriever_selector, pre_opt_selector, post_opt_selector]
    shared_outputs = [msg_input, chatbot, status_display, rounds_display]

    submit_btn.click(fn=respond, inputs=shared_inputs, outputs=shared_outputs)
    msg_input.submit(fn=respond, inputs=shared_inputs, outputs=shared_outputs)


# ================== 启动 ==================
if __name__ == "__main__":
    # 解决 asyncio 嵌套运行问题
    import nest_asyncio

    nest_asyncio.apply()
    # 1. 初始化日志配置
    setup_logging(settings.log_level, settings.log_format)
    logger = logging.getLogger(__name__)

    # 2. 初始化通用大模型组件
    init_base_components()

    # 3. 可选：预热默认知识库（避免用户第一次发消息时等待太久）
    print("正在预热默认知识库（MultiVectorRetriever + LLMContextOptimizer）...")
    loop = asyncio.get_event_loop()
    loop.run_until_complete(
        get_or_load_pipeline("hotpotqa", ["MultiVectorRetriever"], [], ["LLMContextOptimizer"])
    )

    # 4. 启动 WebUI（开启 queue 以支持流式生成器推送）
    demo.queue()
    # demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
    demo.launch(server_name="127.0.0.1", server_port=7860, share=True)

