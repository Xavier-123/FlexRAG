import argparse
import asyncio
import json
import os
import sys
import time
import logging
from typing import List, Dict
from langchain_openai import ChatOpenAI

from flexrag.core.config import Settings
from flexrag.observability import setup_logging
from flexrag.components import LLMContextOptimizer, LLMContextEvaluator, OpenAIGenerator, OpenAILikeReranker
from flexrag.workflows import RAGPipeline
from flexrag.components.pre_retrieval import CompositeQueryOptimizer, QueryExpander, QueryRewriter, TaskSplitter, \
    TerminologyEnricher

from flexrag.components.retrieval.FAISSRetriever import FAISSRetriever
from flexrag.components.retrieval.BM25Retriever import BM25Retriever
from flexrag.components.retrieval.HybridRetriever import HybridRetriever


def is_debug():
    return sys.gettrace() is not None


# 1. 组装并初始化 Pipeline
async def setup_pipeline(args: argparse.Namespace) -> RAGPipeline:
    # 加载已有的知识库
    # retriever = LlamaIndexRetriever(
    #     index=None,
    #     embed_base_url=args.embedding_base_url,
    #     embed_model_name=args.embedding_model,
    #     embed_api_key=args.embedding_api_key,
    #     top_k=args.top_k_retrieval,
    # )
    retriever = HybridRetriever(
        retrievers=[
            FAISSRetriever(
                index=None,
                embed_base_url=args.embedding_base_url,
                embed_model_name=args.embedding_model,
                embed_api_key=args.embedding_api_key,
                top_k=args.top_k_retrieval,
                knowledge_persist_dir=args.knowledge_persist_dir,
            ),
            BM25Retriever(
                # index=None,
                top_k=args.top_k_retrieval,
                persist_dir=os.path.join(args.knowledge_persist_dir, "bm25_index"),
            )
        ],
    )
    # await retriever.load_index(args.knowledge_persist_dir)

    reranker = OpenAILikeReranker(
        base_url=args.reranker_base_url,
        model=args.reranker_model,
        api_key=args.reranker_api_key,
        top_k=args.top_k_rerank,
    )
    llm = ChatOpenAI(
        model=args.llm_model,
        api_key=args.llm_api_key,
        base_url=args.llm_base_url,
        temperature=0.0,
    )

    query_optimizer = CompositeQueryOptimizer([
        QueryRewriter(llm=llm),
        # QueryExpander(llm=llm),
        # TaskSplitter(llm=llm),
        # TerminologyEnricher(llm=llm),
    ])

    context_evaluator = LLMContextEvaluator(llm=llm)
    pipeline = RAGPipeline(
        retriever=retriever,
        reranker=reranker,
        context_optimizer=LLMContextOptimizer(llm=llm),
        query_optimizer=query_optimizer,
        context_evaluator=context_evaluator,
        generator=OpenAIGenerator(
            model=args.llm_model,
            api_key=args.llm_api_key,
            base_url=args.llm_base_url,
        ),
        settings=args,
    )
    return pipeline


# 2. 带有并发控制（Semaphore）的异步单条处理函数
async def process_single_qa(
        pipeline: RAGPipeline,
        item: Dict,
        semaphore: asyncio.Semaphore
) -> Dict:
    async with semaphore:
        question = item["question"]
        try:
            # 调用内置的异步处理函数 arun
            output = await pipeline.arun(question)
            return {
                "question": question,
                "expected": item.get("answer"),
                "generated_answer": output.answer,
                "evidence": output.evidence,
                "trace": output.trace,
                "status": "success",
                "error": None
            }
        except Exception as e:
            return {
                "question": question,
                "expected": item.get("answer"),
                "generated_answer": None,
                "evidence": [],
                "trace": [],
                "status": "error",
                "error": str(e)
            }


# 3. 运行主控函数
async def run_batch_test(qa_data: List[Dict], args: argparse.Namespace):
    logger.info(f"Initializing pipeline...")
    pipeline = await setup_pipeline(args)

    # 限制最大并发数，保护 vLLM 或外部 API 服务
    max_concurrent = args.max_concurrent_tasks
    semaphore = asyncio.Semaphore(max_concurrent)

    logger.info(f"Starting batch execution of {len(qa_data)} questions with concurrency {max_concurrent}...")
    start_time = time.time()

    # 创建所有的并发任务
    tasks = [
        process_single_qa(pipeline, item, semaphore)
        for item in qa_data
    ]

    # asyncio.gather 收集所有的返回结果
    results = await asyncio.gather(*tasks)

    elapsed = time.time() - start_time
    logger.info(f"Batch execution completed in {elapsed:.2f} seconds.")

    # 将结果持久化为 JSON 文件供后续人工/程序评估
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info("Results saved to eval_results.json")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Async Batch QA Tester for RAG Pipeline")

    # --- Embedding 相关参数 ---
    parser.add_argument("--embedding-base-url", type=str, required=True, help="Embedding API Base URL")
    parser.add_argument("--embedding-model", type=str, required=True, help="Embedding 模型名称")
    parser.add_argument("--embedding-api-key", type=str, default="EMPTY", help="Embedding API Key")

    # --- 知识库存储相关参数 ---
    parser.add_argument("--knowledge-persist-dir", type=str, required=True, help="知识库持久化目录路径")

    # --- Query 优化相关参数 ---
    parser.add_argument("--pre-retrieval-strategies", type=lambda x: x.split(','), required=True, help="query优化类型")

    # --- 向量检索相关参数 ---
    parser.add_argument("--top-k-retrieval", type=int, default=5, help="初次检索阶段 (Retrieval) 召回的 Top-K 文档数量")

    # --- Reranker 相关参数 ---
    parser.add_argument("--reranker-base-url", type=str, required=True, help="Reranker API Base URL")
    parser.add_argument("--reranker-model", type=str, required=True, help="Reranker 模型名称")
    parser.add_argument("--reranker-api-key", type=str, default="EMPTY", help="Reranker API Key")
    parser.add_argument("--top-k-rerank", type=int, default=5,
                        help="重排序阶段 (Rerank) 最终保留给大模型的 Top-K 文档数量")

    # --- LLM 相关参数 ---
    parser.add_argument("--llm-base-url", type=str, required=True, help="LLM API Base URL")
    parser.add_argument("--llm-model", type=str, required=True, help="LLM 模型名称")
    parser.add_argument("--llm-api-key", type=str, default="EMPTY", help="LLM API Key")
    parser.add_argument("--context-max-tokens", type=int, default=4096,
                        help="LLM 的上下文 (Context) 最大 Token 长度限制")

    # --- Agent 参数 ---
    parser.add_argument("--max-iterations", type=int, default=3, help="最大迭代次数")
    parser.add_argument("--draw-image-path", type=str, default="./langgraph.png", help="架构图保存路径")

    # --- 执行控制参数 ---
    parser.add_argument("--max-concurrent-tasks", type=int, default=5, help="最大并发任务数 (默认: 5)")

    # --- 文件 I/O 相关参数 (额外添加，方便实战使用) ---
    parser.add_argument("--input-file", type=str, default=None, help="输入的 QA 数据 JSON 文件路径")
    parser.add_argument("--output-file", type=str, default="eval_results.json", help="输出的测试结果文件路径")

    return parser.parse_args()


if __name__ == "__main__":
    # 日志设置
    settings = Settings()
    setup_logging(settings.log_level, settings.log_format)
    logger = logging.getLogger(__name__)

    # 解析命令行参数
    args = parse_arguments()

    # 数据加载：如果有输入文件，则从文件读取；否则生成 mock 数据
    if args.input_file:
        logger.info(f"Loading data from {args.input_file}...")
        with open(args.input_file, "r", encoding="utf-8") as f:
            if is_debug():
                qa_data = json.load(f)[:1]
            else:
                qa_data = json.load(f)[:1]
    else:
        logger.info("No input file provided, using mock data...")
        # 示例的格式数据，请根据你这 1000 条真实 QA 数据做替换加载
        qa_data = [
            {"question": f"Question {i}", "answer": f"Expected Answer {i}"}
            for i in range(3)
        ]

    # 启动异步事件循环
    asyncio.run(run_batch_test(qa_data, args))
