import asyncio
import json
import os
import sys
import time
import logging
from typing import List, Dict
from langchain_openai import ChatOpenAI

from flexrag.common import setup_logging, Settings
from flexrag.workflows import RAGPipeline
from flexrag.components.pre_retrieval import PreQueryOptimizer, QueryExpander, QueryRewriter, TaskSplitter, \
    TerminologyEnricher
from flexrag.components.post_retrieval import PostRetrieval, OpenAILikeReranker, LLMContextOptimizer, CopyPasteRetrieval
from flexrag.components.retrieval import BM25Retriever, HybridRetriever, GraphRetriever, \
    MultiVectorRetriever, OpenAILikeEmbedding
from flexrag.components.reasoning import OpenAIGenerator, LLMContextEvaluator


def is_debug():
    return sys.gettrace() is not None


# 1. 组装并初始化 Pipeline
# async def setup_pipeline(args: argparse.Namespace) -> RAGPipeline:
async def setup_pipeline(settings: Settings) -> RAGPipeline:
    llm = ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        temperature=0.0,
    )

    embed_model = OpenAILikeEmbedding(
        model_name=settings.embedding_model,
        base_url=settings.embedding_base_url,
        api_key=settings.embedding_api_key
    )

    pre_retrieval_optimizer = PreQueryOptimizer([
        # QueryRewriter(llm=llm),
        # QueryExpander(llm=llm),
        # TaskSplitter(llm=llm),
        # TerminologyEnricher(llm=llm),
    ])

    retriever = HybridRetriever(
        retrievers=[
            MultiVectorRetriever(
                embed_model=embed_model,
                vector_store_type=settings.vector_store_type,
                top_k=settings.top_k_retrieval,
                persist_dir=settings.knowledge_persist_dir,
            ),
            BM25Retriever(
                top_k=settings.top_k_retrieval,
                persist_dir=os.path.join(settings.knowledge_persist_dir, "bm25_index"),
            ),
            # GraphRetriever(
            #     llm=llm,
            #     embed_model=embed_model,
            #     persist_dir=os.path.join(args.knowledge_persist_dir, "graph_index"),
            # )
        ],
    )

    post_retrieval_optimizer = PostRetrieval([
        # OpenAILikeReranker(
        #     base_url=args.reranker_base_url,
        #     model=args.reranker_model,
        #     api_key=args.reranker_api_key,
        #     top_k=args.top_k_rerank
        # ),
        LLMContextOptimizer(llm=llm),
        # CopyPasteRetrieval(
        #     model=settings.llm_model,
        #     base_url=settings.llm_base_url,
        #     api_key=settings.llm_api_key,
        #     pipeline="cp-refine"
        # )
    ])

    context_evaluator = LLMContextEvaluator(llm=llm)

    pipeline = RAGPipeline(
        pre_retrieval_optimizer=pre_retrieval_optimizer,
        retriever=retriever,
        post_retrieval_optimizer=post_retrieval_optimizer,
        context_evaluator=context_evaluator,
        generator=OpenAIGenerator(
            model=settings.llm_model,
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
        ),
        settings=settings,
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
async def run_batch_test(qa_data: List[Dict], settings: Settings):
    logger.info(f"Initializing pipeline...")
    pipeline = await setup_pipeline(settings)

    # 限制最大并发数，保护 vLLM 或外部 API 服务
    max_concurrent = settings.max_concurrent_tasks
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
    output_dir = os.path.dirname(settings.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(settings.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Results saved to {settings.output_file}")


if __name__ == "__main__":
    import nest_asyncio

    nest_asyncio.apply()

    # 读取 .env 文件中的配置
    settings = Settings()
    print("-" * 50)
    print(settings)
    print("-" * 50)

    # 日志设置
    setup_logging(settings.log_level, settings.log_format)
    logger = logging.getLogger(__name__)

    # 解析命令行参数
    # args = parse_arguments()

    # 数据加载：如果有输入文件，则从文件读取；否则生成 mock 数据
    if settings.input_file:
        logger.info(f"Loading data from {settings.input_file}...")
        with open(settings.input_file, "r", encoding="utf-8") as f:
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
    asyncio.run(run_batch_test(qa_data, settings))
