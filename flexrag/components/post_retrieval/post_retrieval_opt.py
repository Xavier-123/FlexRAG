from typing import Any

from flexrag.common.schema import Document
from flexrag.components.post_retrieval.reranker import OpenAILikeReranker
from flexrag.components.post_retrieval.context_optimizer import LLMContextOptimizer
from flexrag.components.post_retrieval.copy_paste import CopyPasteRetrieval


class PostRetrieval:
    def __init__(self, optimizers: list[
        LLMContextOptimizer | OpenAILikeReranker | CopyPasteRetrieval] = LLMContextOptimizer) -> None:
        self.optimizers = optimizers

    async def optimize(
            self,
            query: str,
            documents: list[Document],
            accumulated_context: list[str],
            max_tokens: int,
    ) -> Any:
        for optimizer in self.optimizers:
            if isinstance(optimizer, OpenAILikeReranker):
                documents = await optimizer.optimize(query, documents, accumulated_context, max_tokens)

        for optimizer in self.optimizers:
            if isinstance(optimizer, LLMContextOptimizer):
                optimized_query, prompt_string = await optimizer.optimize(query, documents, accumulated_context, max_tokens)

        for optimizer in self.optimizers:
            if isinstance(optimizer, CopyPasteRetrieval):
                optimized_query, prompt_string = await optimizer.optimize(query, documents, accumulated_context,
                                                                          max_tokens)
                return optimized_query, prompt_string

        # Fallback: no LLMContextOptimizer present – join document texts directly
        optimized_context = "\n\n".join(doc.text for doc in documents)
        return optimized_context, ""