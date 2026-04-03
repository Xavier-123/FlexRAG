import asyncio
import os
import logging
import gradio as gr

from flexrag.workflows.pipeline import RAGPipeline
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
                llm_model_name=settings.llm_model,
                llm_base_url=settings.llm_base_url,
                llm_api_key=settings.llm_api_key,
                embed_model_name=settings.embedding_model,
                embed_base_url=settings.embedding_base_url,
                embed_api_key=settings.embedding_api_key,
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


# ================== Gradio 响应逻辑（原生 Async） ==================
async def respond(message, chat_history, kb_name, retriever_names, pre_opt_names, post_opt_names):
    """Gradio 原生支持 async def，不再需要手动 new_event_loop"""
    if not message.strip():
        return "", chat_history

    if chat_history is None:
        chat_history = []

    if not retriever_names:
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": "⚠️ **请至少选择一个检索器后再提问。**"})
        return "", chat_history

    try:
        # 1. 动态获取对应配置的 Pipeline
        pipeline = await get_or_load_pipeline(
            kb_name, retriever_names, pre_opt_names, post_opt_names
        )

        # 2. 运行 Pipeline
        result = await pipeline.arun(message)
        answer = result.answer
        evidences = result.evidence

        # 3. 构造折叠 evidence
        evidence_html = "\n\n<details><summary><b>👉 点击展开查看参考检索片段</b></summary>\n\n"
        if evidences:
            for i, ev in enumerate(evidences, 1):
                preview = ev[:200] + ("..." if len(ev) > 200 else "")
                evidence_html += f"**[来源{i}]** {preview}\n\n"
        else:
            evidence_html += "未检索到相关片段。\n\n"
        evidence_html += "</details>"

        final_bot_message = answer + evidence_html

    except Exception as e:
        final_bot_message = f"❌ **发生错误**: {str(e)}"

    # 4. 更新对话历史
    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": final_bot_message})

    return "", chat_history


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
                height=480,
                buttons=["copy"],
                layout="bubble",
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
    shared_outputs = [msg_input, chatbot]

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

    # 4. 启动 WebUI
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)

