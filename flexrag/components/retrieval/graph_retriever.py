from typing import List
from langchain_openai import ChatOpenAI

from llama_index.llms.langchain import LangChainLLM
from llama_index.core import Settings
from llama_index.core.base import base_retriever
from llama_index.core.indices.knowledge_graph import KnowledgeGraphIndex
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core import load_index_from_storage, PropertyGraphIndex

from flexrag.core.schema import Document
from flexrag.components.retrieval import BaseFlexRetriever, OpenAILikeEmbedding
# from llama_index.embeddings.openai_like import OpenAILikeEmbedding


import logging

logger = logging.getLogger(__name__)


class GraphRetriever(BaseFlexRetriever):
    """Graph-based retriever using Knowledge Graph."""

    def __init__(
            self,
            llm_model_name: str | None = None,
            llm_base_url: str | None = None,
            llm_api_key: str | None = None,
            embed_model_name: str | None = None,
            embed_base_url: str | None = None,
            embed_api_key: str | None = None,
            top_k: int | None = 2,
            persist_dir: str | None = None,
    ) -> None:
        self._graph_retriever = None
        self._persist_dir = persist_dir
        self._similarity_top_k = top_k
        self._embed_model = OpenAILikeEmbedding(model=embed_model_name, base_url=embed_base_url, api_key=embed_api_key)
        if llm_model_name:
            llm = ChatOpenAI(model=llm_model_name, api_key=llm_api_key, base_url=llm_base_url, temperature=0.0)
            llama_index_llm = LangChainLLM(llm=llm)
            Settings.llm = llama_index_llm

        # self._llm = OpenAILike(model=llm_model_name, api_key=llm_api_key, api_base=llm_base_url, temperature=0.0)
        Settings.embed_model = self._embed_model


    async def retrieve(self, query: str) -> List[Document]:
        """Retrieve documents using graph traversal."""

        if not self._graph_retriever:
            self.load_index(self._persist_dir)
            # raise ValueError("Graph retriever is not initialized")

        nodes = self._graph_retriever.query(query)

        documents: List[Document] = []
        # for node in nodes:
        #     documents.append(
        #         Document(
        #             text=node.get_content(),
        #             score=getattr(node, "score", 0.0) or 0.0,
        #             metadata=node.metadata,
        #         )
        #     )
        #
        # logger.debug("Graph retrieved %d documents", len(documents))
        return documents

    def load_index(self, persist_dir: str) -> None:
        if persist_dir:
            # 从持久化目录加载 Graph Index
            storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
            index = load_index_from_storage(storage_context)

            # 构建 graph retriever
            self._graph_retriever = index.as_query_engine(
                include_text=True,  # 检索时包含原始文本块，提供更丰富的上下文
                similarity_top_k=self._similarity_top_k  # 检索最相关的节点/边数量
            )

    async def build_graph(self, documents: List[Document]) -> None:
        """Build the knowledge graph index from the provided documents."""



        # 1️⃣ 创建 Graph Store
        from llama_index.core.graph_stores.simple import SimpleGraphStore

        graph_store = SimpleGraphStore()

        # 2️⃣ 创建 StorageContext
        storage_context = StorageContext.from_defaults(
            graph_store=graph_store,
        )

        # 3️⃣ 构建 PropertyGraphIndex
        print("正在抽取实体和关系，构建图谱 (这需要调用 LLM，请稍候)...")
        index = PropertyGraphIndex.from_documents(
            documents,
            llm=Settings.llm,
            embed_model=Settings.embed_model,
            storage_context=storage_context,
            show_progress=True,
        )

        # 4️⃣ 持久化
        if self._persist_dir:
            index.storage_context.persist(persist_dir=self._persist_dir)
        logger.info("Knowledge graph index built and persisted to '%s'", self._persist_dir)
