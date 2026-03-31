import logging
import jieba

from llama_index.retrievers.bm25 import BM25Retriever as LlamaIndexBM25Retriever

from flexrag.components.retrieval import BaseRetriever
from flexrag.core.schema import Document

logger = logging.getLogger(__name__)


class BM25Retriever(BaseRetriever):

    def __init__(
            self,
            top_k: int | None = 5,
            persist_dir: str | None = None,
    ) -> None:
        self._similarity_top_k = top_k or 5
        self._bm25_retriever: LlamaIndexBM25Retriever | None = None

        # 优先尝试从持久化目录加载 .pkl 文件
        self._knowledge_persist_dir = persist_dir

        if persist_dir:
            self._bm25_retriever = LlamaIndexBM25Retriever.from_persist_dir(
                persist_dir,
                # tokenizer=self.chinese_tokenizer
            )
            self._bm25_retriever.similarity_top_k = self._similarity_top_k


    async def retrieve(self, query: str) -> list[Document]:
        """Retrieve documents relevant to *query* using BM25.

        Args:
            query: The user's search string.

        Returns:
            List of :class:`~flexrag.core.schema.Document` sorted by descending score.
        """
        nodes = self._bm25_retriever.retrieve(query)

        documents: list[Document] = []
        for node in nodes:
            documents.append(
                Document(
                    text=node.get_content(),
                    score=node.score or 0.0,
                    metadata=node.metadata,
                )
            )
        logger.debug("Retrieved %d documents", len(documents))

        # for idx, node in enumerate(nodes):
        #     print(f"[{idx + 1}] 得分: {node.score:.4f} | 文本: {node.text[:40]}...")

        return documents

    def chinese_tokenizer(self, text: str):
        # 使用 jieba 进行精确模式分词
        return list(jieba.cut(text))