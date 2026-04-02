import gradio as gr
import asyncio
import os

from flexrag import RAGPipeline
from flexrag.common.config import Settings
from flexrag.components.post_retrieval.context_optimizer import LLMContextOptimizer
from flexrag.components.judges.context_evaluator import LLMContextEvaluator
from flexrag.components.generation.generator import OpenAIGenerator
from flexrag.components.pre_retrieval.query_optimizer import LLMQueryOptimizer
from flexrag.indexing.knowledge import FaissKnowledgeBuilder
from flexrag.components.post_retrieval.reranker import VLLMReranker
from flexrag.components.retrieval import LlamaIndexRetriever
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

# 共享的基础组件（避免重复初始化 LLM、Reranker）
base_components = {}
# Pipeline 缓存，按知识库名称存储，实现切换后秒切
pipelines_cache = {}


# ================== 初始化基础组件 ==================
def init_base_components():
    """初始化大模型、Reranker等与具体知识库无关的通用组件"""
    global base_components

    reranker = VLLMReranker(
        base_url=settings.reranker_base_url,
        model=settings.reranker_model,
        api_key=settings.reranker_api_key,
    )

    llm = ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        temperature=0.0,
    )

    context_optimizer = LLMContextOptimizer(llm=llm)
    query_optimizer = LLMQueryOptimizer(llm=llm)
    context_evaluator = LLMContextEvaluator(llm=llm)

    generator = OpenAIGenerator(
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
    )

    base_components = {
        "reranker": reranker,
        "context_optimizer": context_optimizer,
        "query_optimizer": query_optimizer,
        "context_evaluator": context_evaluator,
        "generator": generator,
        "settings": settings
    }


# ================== 动态获取或加载 Pipeline ==================
async def get_or_load_pipeline(kb_name: str) -> RAGPipeline:
    """根据选择的知识库动态加载并缓存 Pipeline"""
    # 如果已经加载过，直接从缓存返回，实现秒切
    if kb_name in pipelines_cache:
        return pipelines_cache[kb_name]

    print(f"🔄 正在加载知识库: {kb_name} ...")
    persist_dir = KB_DICT.get(kb_name, settings.knowledge_persist_dir)

    retriever = LlamaIndexRetriever(
        index=None,
        embed_base_url=settings.embedding_base_url,  # 修复：移除多余的引号
        embed_model_name=settings.embedding_model,
        embed_api_key=settings.embedding_api_key,
    )

    if os.path.exists(persist_dir):
        await retriever.load_index(persist_dir)
    else:
        print(f"⚠️ 警告: 知识库路径 {persist_dir} 不存在，将使用空知识库。")

    # 组装针对该知识库的 Pipeline
    pipeline = RAGPipeline(
        retriever=retriever,
        reranker=base_components["reranker"],
        context_optimizer=base_components["context_optimizer"],
        query_optimizer=base_components["query_optimizer"],
        context_evaluator=base_components["context_evaluator"],
        generator=base_components["generator"],
        settings=base_components["settings"],
    )

    # 存入缓存
    pipelines_cache[kb_name] = pipeline
    print(f"✅ 知识库 {kb_name} 加载完成！")
    return pipeline


# ================== Gradio 响应逻辑（原生 Async） ==================
async def respond(message, chat_history, kb_name):
    """Gradio 原生支持 async def，不再需要手动 new_event_loop"""
    if not message.strip():
        return "", chat_history

    if chat_history is None:
        chat_history = []

    try:
        # 1. 动态获取对应知识库的 Pipeline
        pipeline = await get_or_load_pipeline(kb_name)

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
"""

with gr.Blocks(theme=gr.themes.Soft(), title="FlexRAG 智能问答系统", css=custom_css) as demo:
    gr.Markdown("""
    <h1 style="text-align: center;">🤖 FlexRAG 智能问答系统</h1>
    <p style="text-align: center;">支持多知识库动态无缝切换</p>
    """)

    with gr.Row():
        # 选择器选项与字典对应
        kb_selector = gr.Dropdown(
            choices=list(KB_DICT.keys()),
            value="hotpotqa",
            label="📚 选择知识库 (首次切换时会加载，后续秒切)",
        )

    # 适配 Gradio 4.x+ 的 messages 格式
    chatbot = gr.Chatbot(
        height=400,
        buttons=["copy"],
        layout="bubble",
    )

    with gr.Row():
        msg_input = gr.Textbox(placeholder="请输入问题...", scale=8)
        submit_btn = gr.Button("发送", variant="primary", scale=1)

    # 绑定提交事件
    submit_btn.click(
        fn=respond,
        inputs=[msg_input, chatbot, kb_selector],
        outputs=[msg_input, chatbot]
    )

    msg_input.submit(
        fn=respond,
        inputs=[msg_input, chatbot, kb_selector],
        outputs=[msg_input, chatbot]
    )

# ================== 启动 ==================
if __name__ == "__main__":
    # 1. 初始化通用大模型组件
    init_base_components()

    # 2. 可选：在服务启动时预先加载默认知识库（避免用户第一次发消息时等待太久）
    print("正在预热默认知识库...")
    loop = asyncio.get_event_loop()
    loop.run_until_complete(get_or_load_pipeline("hotpotqa"))

    # 3. 启动 WebUI
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
