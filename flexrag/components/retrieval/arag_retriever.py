import re
import json
import numpy as np
from typing import List, Dict, Any, Set, Tuple

from flexrag.components.retrieval.base import BaseFlexRetriever
from flexrag.common.schema import Document


class AgentContext:
    """用于管理 Agent 运行时的状态和预算限制"""

    def __init__(self, max_loops: int = 10, max_token_budget: int = 128000):
        self.seen_chunks: Set[str] = set()  # 已读 chunk 记录
        self.retrieved_docs: List[Document] = []  # 最终收集到的完整文档
        self.loop_count: int = 0
        self.token_used: int = 0
        self.max_loops = max_loops
        self.max_token_budget = max_token_budget

    def add_tokens(self, count: int):
        self.token_used += count

    def is_budget_exceeded(self) -> bool:
        return self.loop_count >= self.max_loops or self.token_used >= self.max_token_budget


class AragRetriever(BaseFlexRetriever):
    # 注入给 LLM 的系统提示词（完全遵循你的要求）
    REACT_SYSTEM_PROMPT = """作为基于 ReAct 框架运作的智能体，你需要自主决定如何调用以下三个核心检索工具（key_word_search, semantic_search, read_chunk）。请严格遵循以下工作模式：
    1. 核心工作流：“先搜索，后精读”双步模式
    第一步（搜索）： 根据用户问题，调用 key_word_search 或 semantic_search。注意：这两个工具只返回“文本缩略片段（snippet）”和对应的 chunk_id。
    第二步（精读）： 评估搜索返回的片段。如果发现某个片段包含核心线索，你必须显式调用 read_chunk(chunk_ids=[...]) 工具获取完整上下文。
    第三步（回答）： 只有在获取到充足的完整文本后，才生成最终答案。
    2. 资源限制与安全兜底机制（强制生成）
    限制条件： 你的思考与调用循环（max_loops）上限为 10 次，总上下文消耗上限为 128,000 tokens。
    兜底动作： 一旦触碰上述任一红线，系统将触发 _force_final_answer() 信号。严禁继续请求调用任何工具，必须立即强行基于当前已收集的信息生成最终答案。"""

    def __init__(
            self,
            top_k: int | None = 5,
            persist_dir: str | None = None,
    ) -> None:
        self._similarity_top_k = top_k or 5
        self.persist_dir = persist_dir

        # 内部数据模拟（实际应用中应从 persist_dir 加载）
        self._mock_load_data()

    def _mock_load_data(self):
        """模拟加载预建索引、文本块和向量模型"""
        # 格式: {chunk_id: full_text}
        self.chunks_db = {"chunk_1": "Apple is a tech company...", "chunk_2": "Banana is a fruit..."}
        # 格式: [{'chunk_id': 'xxx', 'text': 'sentence', 'vector': np.array}]
        self.sentence_index = []

        # 模拟模型 (实际使用 SentenceTransformer)
        class DummyEncoder:
            def encode(self, text, normalize_embeddings=True):
                return np.random.rand(768)  # 模拟返回归一化的向量

        self.encoder = DummyEncoder()

    async def key_word_search(self, query: str) -> list[Document]:
        """词法层精确匹配"""
        # 1. 提取关键词 (简单按空格划分，实际可使用jieba等分词工具)
        keywords = [kw for kw in query.split() if kw.strip()]
        if not keywords:
            return []

        scored_chunks: List[Tuple[float, str, str]] = []

        # 2. 对每个 chunk 进行计分
        for chunk_id, full_text in self.chunks_db.items():
            score = 0.0
            text_lower = full_text.lower()

            for kw in keywords:
                kw_lower = kw.lower()
                # 计算词频 count(keyword)
                count = text_lower.count(kw_lower)
                # 评分公式：Σcount(keyword) × len(keyword)
                score += count * len(kw)

            if score > 0:
                # 3. 提取片段 (snippet)，这里简化为截取包含关键词的前后50个字符
                snippet = full_text[:100] + "..."  # 简化的snippet提取逻辑
                scored_chunks.append((score, chunk_id, snippet))

        # 4. 排序并返回 top-k (不返回全文)
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, cid, snippet in scored_chunks[:self._similarity_top_k]:
            results.append(Document(
                page_content=snippet,  # 强调：这里只返回snippet
                metadata={"chunk_id": cid, "score": score, "type": "keyword_snippet"}
            ))
        return results

    async def semantic_search(self, query: str) -> list[Document]:
        """向量语义搜索"""
        if not self.sentence_index:
            return []

        # 1. 查询编码
        q_vec = self.encoder.encode(query, normalize_embeddings=True)

        chunk_max_scores: Dict[str, Tuple[float, str]] = {}

        # 2. 与预建句子索引做余弦相似度（np.dot）
        for item in self.sentence_index:
            cid = item['chunk_id']
            sentence_text = item['text']
            doc_vec = item['vector']

            # 余弦相似度
            sim_score = float(np.dot(q_vec, doc_vec))

            # 3. 以 chunk 内最高相似度句子的分数代表该 chunk 的分数
            if cid not in chunk_max_scores or sim_score > chunk_max_scores[cid][0]:
                chunk_max_scores[cid] = (sim_score, sentence_text)

        # 4. 排序并返回 top-k (只返回最高分句子作为片段)
        sorted_chunks = sorted(chunk_max_scores.items(), key=lambda x: x[1][0], reverse=True)

        results = []
        for cid, (score, best_sentence) in sorted_chunks[:self._similarity_top_k]:
            results.append(Document(
                text=best_sentence,  # 强调：这里只返回最高分句子
                metadata={"chunk_id": cid, "score": score, "type": "semantic_snippet"}
            ))
        return results

    async def read_chunk(self, query: str, context: AgentContext) -> list[Document]:
        """精读工具：根据 chunk_id 获取完整文本"""
        # 解析 LLM 传进来的 chunk_ids（尝试解析 JSON 或逗号分隔字符串）
        try:
            chunk_ids = json.loads(query) if "[" in query else [c.strip() for c in query.split(",")]
        except Exception:
            chunk_ids = [query.strip()]

        results = []
        for cid in chunk_ids:
            # AgentContext 记录已读过的 chunk，避免重复读取
            if cid in context.seen_chunks:
                continue

            if cid in self.chunks_db:
                full_text = self.chunks_db[cid]
                context.seen_chunks.add(cid)
                doc = Document(
                    page_content=full_text,
                    metadata={"chunk_id": cid, "type": "full_chunk"}
                )
                results.append(doc)
                # 将精读的完整文档加入到最终上下文中
                context.retrieved_docs.append(doc)

        return results

    def _count_tokens(self, text: str) -> int:
        """简单的 token 估算（实际使用 tiktoken）"""
        return len(text) // 4

    def _force_final_answer(self, context: AgentContext):
        """兜底机制信号"""
        # 在真实环境中，这里会将系统指令注入给 LLM，强制要求其使用当前 context.retrieved_docs 停止思考并作答
        print("[System Alert] Resource limit reached! Triggering _force_final_answer().")
        return "RESOURCE_LIMIT_REACHED"

    async def retrieve(self, query: str) -> list[Document]:
        """ReAct 核心工作流引擎"""
        context = AgentContext(max_loops=10, max_token_budget=128000)

        # 初始化 LLM 对话历史
        chat_history = [
            {"role": "system", "content": self.REACT_SYSTEM_PROMPT},
            {"role": "user", "content": f"User Query: {query}"}
        ]
        context.add_tokens(self._count_tokens(self.REACT_SYSTEM_PROMPT + query))

        while context.loop_count < context.max_loops:
            context.loop_count += 1

            # --- 模拟 LLM 思考与动作生成 ---
            # 真实环境中这里应调用 await llm.ainvoke(chat_history)
            action, action_input, is_final_answer = self._mock_llm_reasoning(context, query)

            # Token 预算统计
            context.add_tokens(100)  # 模拟LLM生成的token消耗

            if is_final_answer:
                break

            # 执行工具
            observation = ""
            if action == "key_word_search":
                docs = await self.key_word_search(action_input)
                observation = "\n".join([f"ID: {d.metadata['chunk_id']}, Snippet: {d.page_content}" for d in docs])
            elif action == "semantic_search":
                docs = await self.semantic_search(action_input)
                observation = "\n".join([f"ID: {d.metadata['chunk_id']}, Snippet: {d.page_content}" for d in docs])
            elif action == "read_chunk":
                docs = await self.read_chunk(action_input, context)
                observation = "\n".join(
                    [f"Full Text read for chunk(s): {', '.join([d.metadata['chunk_id'] for d in docs])}"])
            else:
                observation = "Invalid Tool."

            # 将观察结果返回给 LLM
            chat_history.append({"role": "assistant", "content": f"Action: {action}\nInput: {action_input}"})
            chat_history.append({"role": "user", "content": f"Observation: {observation}"})
            context.add_tokens(self._count_tokens(observation))

            # --- 安全兜底机制检查 ---
            if context.is_budget_exceeded():
                self._force_final_answer(context)
                break

        # 最终只返回精读过的完整 chunk，作为后续 RAG 生成最终答案的 Context
        return context.retrieved_docs

    def _mock_llm_reasoning(self, context: AgentContext, query: str):
        """这是一个用于占位的模拟 LLM 推理过程，用于跑通流程"""
        if context.loop_count == 1:
            return "semantic_search", query, False
        elif context.loop_count == 2:
            return "read_chunk", "chunk_1, chunk_2", False
        else:
            return "FinalAnswer", "Done", True


if __name__ == '__main__':
    import asyncio
    import numpy as np

    # 导入你刚才编写的完整类和依赖（如果写在同一个文件，可忽略此导入）
    from arag_retriever import AragRetriever, AgentContext


    async def main():
        print("=" * 60)
        print("🚀 初始化 AragRetriever...")
        retriever = AragRetriever(top_k=2)

        # ---------------------------------------------------------
        # 1. 注入测试数据（模拟真实环境中的文档库）
        # ---------------------------------------------------------
        retriever.chunks_db = {
            "chunk_101": "ReAct 框架的核心理念是让智能体交替进行思考（Reasoning）和行动（Acting）。通过这种模式，大语言模型可以不仅生成文本，还能调用外部API或工具来获取信息。",
            "chunk_102": "苹果公司（Apple Inc.）是一家美国的跨国科技公司，成立于1976年，主要业务包括iPhone等消费电子产品。",
            "chunk_103": "在 RAG（检索增强生成）系统中，Chunking（分块）策略至关重要。合理的 Chunk 大小对最终向量检索的召回率有极大的影响。"
        }

        # 模拟句子向量索引 (随机生成768维归一化向量模拟SentenceTransformer)
        def random_normalized_vector():
            vec = np.random.rand(768)
            return vec / np.linalg.norm(vec)

        retriever.sentence_index = [
            {"chunk_id": "chunk_101", "text": "ReAct 框架的核心理念是让智能体交替进行思考和行动。",
             "vector": random_normalized_vector()},
            {"chunk_id": "chunk_102", "text": "苹果公司成立于1976年。", "vector": random_normalized_vector()},
            {"chunk_id": "chunk_103", "text": "合理的 Chunk 大小对最终向量检索的召回率有极大的影响。",
             "vector": random_normalized_vector()},
        ]

        # ---------------------------------------------------------
        # 2. 模拟真实 LLM 的 Agent 决策流 (仅用于演示工作流)
        # ---------------------------------------------------------
        def simulated_agent_llm(context: AgentContext, query: str):
            if context.loop_count == 1:
                print(f"\n🧠 [Agent 思考 {context.loop_count}] : 用户询问 ReAct 和 RAG，我需要先搜索相关片段（Snippet）。")
                print("🛠️ [Agent 行动] : 调用 semantic_search")
                return "semantic_search", query, False

            elif context.loop_count == 2:
                print(
                    f"\n🧠 [Agent 思考 {context.loop_count}] : 搜索返回了片段，我发现 chunk_101 和 chunk_103 包含核心线索，但我目前只有片段，没有全文。")
                print("🛠️ [Agent 行动] : 调用 read_chunk，传入 [chunk_101, chunk_103]")
                return "read_chunk", '["chunk_101", "chunk_103"]', False

            else:
                print(f"\n🧠 [Agent 思考 {context.loop_count}] : 我已经获取到了充足的完整文本，可以停止探索，准备作答。")
                print("🏁 [Agent 行动] : 结束检索循环")
                return "FinalAnswer", "Done", True

        # 替换原本简单的 mock 函数
        retriever._mock_llm_reasoning = simulated_agent_llm

        # ---------------------------------------------------------
        # 3. 执行检索
        # ---------------------------------------------------------
        user_query = "请解释一下什么是 ReAct 框架，以及 RAG 系统中的关键要素？"
        print(f"\n👤 [用户提问]: {user_query}")
        print("-" * 60)

        # 启动异步检索流程
        final_docs = await retriever.retrieve(user_query)

        # ---------------------------------------------------------
        # 4. 打印最终结果（这将是交给最终 LLM 生成答案的 Context）
        # ---------------------------------------------------------
        print("=" * 60)
        print("✅ 检索流程结束。最终传递给 LLM 用于生成的【完整上下文】如下：\n")

        if not final_docs:
            print("未获取到任何文档。")

        for i, doc in enumerate(final_docs, 1):
            print(f"📄 【精读文档 {i}】 (ID: {doc.metadata.get('chunk_id')})")
            print(f"内容 : {doc.page_content}\n")


    if __name__ == "__main__":
        # 运行异步主函数
        asyncio.run(main())