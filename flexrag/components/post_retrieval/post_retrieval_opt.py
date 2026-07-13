from flexrag.common.schema import Document, PostRetrievalResult
from flexrag.components.post_retrieval.reranker import OpenAILikeReranker
from flexrag.components.post_retrieval.composite_score import CompositeScoreReranker
from flexrag.components.post_retrieval.context_optimizer import LLMContextOptimizer
from flexrag.components.post_retrieval.copy_paste import CopyPasteRetrieval


class PostRetrieval:
    def __init__(self, optimizers=None) -> None:
        self.optimizers = list(optimizers or [])

    async def optimize(
            self,
            query: str,
            documents: list[Document],
            accumulated_context: list[str],
            max_tokens: int,
    ) -> PostRetrievalResult:
        optimized_context = "\n\n".join(doc.text for doc in documents)
        prompt_string = ""

        for optimizer in self.optimizers:
            if isinstance(optimizer, OpenAILikeReranker):
                documents = await optimizer.optimize(query, documents, accumulated_context, max_tokens)
                optimized_context, prompt_string = '\n\n'.join(doc.text for doc in documents), ""

        for optimizer in self.optimizers:
            if isinstance(optimizer, CompositeScoreReranker):
                documents = await optimizer.optimize(query, documents, accumulated_context, max_tokens)
                optimized_context, prompt_string = '\n\n'.join(doc.text for doc in documents), ""

        for optimizer in self.optimizers:
            if isinstance(optimizer, LLMContextOptimizer):
                result = await optimizer.optimize(query, documents, accumulated_context, max_tokens)
                if isinstance(result, tuple):
                    optimized_context, prompt_string = result
                else:
                    optimized_context, prompt_string = result, ""

        for optimizer in self.optimizers:
            if isinstance(optimizer, CopyPasteRetrieval):
                optimized_context, prompt_string = await optimizer.optimize(
                    query, documents, accumulated_context, max_tokens
                )
        return PostRetrievalResult(
            documents=documents,
            optimized_context=optimized_context,
            prompt_string=prompt_string,
        )
