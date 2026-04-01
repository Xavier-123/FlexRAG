from typing import Any

from flexrag.core.schema import Document
from flexrag.components.post_retrieval.reranker import OpenAILikeReranker
from flexrag.components.post_retrieval.context_optimizer import LLMContextOptimizer


class PostRetrieval:
    def __init__(self, optimizers: list[LLMContextOptimizer | OpenAILikeReranker] = LLMContextOptimizer) -> None:
        self.optimizers = optimizers

    async def optimize(
            self,
            query: str,
            documents: list[Document],
            accumulated_context: list[str],
            max_tokens: int,
    ) -> Any:
        for i, optimizer in enumerate(self.optimizers):
            if isinstance(optimizer, OpenAILikeReranker):
                documents = await optimizer.optimize(query, documents, accumulated_context, max_tokens)

        for i, optimizer in enumerate(self.optimizers):
            if isinstance(optimizer, LLMContextOptimizer):
                optimized_query, prompt_string = await optimizer.optimize(query, documents, accumulated_context, max_tokens)

        return optimized_query, prompt_string