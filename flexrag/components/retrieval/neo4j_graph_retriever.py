import os
import logging
from typing import List
import nest_asyncio

from llama_index.core import Settings, StorageContext, load_index_from_storage, Document
from llama_index.core.indices.property_graph import PropertyGraphIndex
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from langchain_openai import ChatOpenAI
from llama_index.llms.langchain import LangChainLLM

from flexrag.components.retrieval import BaseFlexRetriever, OpenAILikeEmbedding

logger = logging.getLogger(__name__)

# 在模块顶部执行即可，解决 Jupyter 的事件循环问题
nest_asyncio.apply()


class Neo4jGraphRetriever(BaseFlexRetriever):
    """Graph-based retriever using Knowledge Graph with Neo4j."""

    def __init__(
            self,
            llm_model_name: str | None = None,
            llm_base_url: str | None = None,
            llm_api_key: str | None = None,
            embed_model_name: str | None = None,
            embed_base_url: str | None = None,
            embed_api_key: str | None = None,
            top_k: int | None = 2,
            persist_dir: str | None = "./storage",
            neo4j_url: str = "bolt://127.0.0.1:7687",
            neo4j_user: str = "neo4j",
            neo4j_pass: str = "test1234",
            refresh_schema: bool = False,
    ) -> None:
        self._graph_retriever = None
        self._index = None
        self._persist_dir = persist_dir
        self._similarity_top_k = top_k

        # 提取 Neo4j 配置为实例属性，方便复用
        self._neo4j_url = neo4j_url
        self._neo4j_user = neo4j_user
        self._neo4j_pass = neo4j_pass
        self._refresh_schema = refresh_schema

        # 配置模型
        self._embed_model = OpenAILikeEmbedding(model=embed_model_name, base_url=embed_base_url, api_key=embed_api_key)
        if llm_model_name:
            llm = ChatOpenAI(model=llm_model_name, api_key=llm_api_key, base_url=llm_base_url, temperature=0.0)
            Settings.llm = LangChainLLM(llm=llm)
        Settings.embed_model = self._embed_model

        # 尝试加载已有的图索引
        self._load_index_if_exists()

    def _get_neo4j_store(self) -> Neo4jPropertyGraphStore:
        """抽取获取 Neo4j Store 的公共方法"""
        return Neo4jPropertyGraphStore(
            username=self._neo4j_user,
            password=self._neo4j_pass,
            url=self._neo4j_url,
            database="neo4j",
            refresh_schema=self._refresh_schema,
        )

    def _load_index_if_exists(self) -> None:
        """安全地加载索引，如果不存在则不报错，等待构建"""
        if not self._persist_dir or not os.path.exists(self._persist_dir):
            logger.warning(f"Persist dir '{self._persist_dir}' does not exist. Please call build_graph() first.")
            return

        try:
            graph_store = self._get_neo4j_store()
            storage_context = StorageContext.from_defaults(
                persist_dir=self._persist_dir,
                property_graph_store=graph_store,
            )
            self._index = load_index_from_storage(storage_context)

            # 【重要修改】：如果是单纯检索，请使用 as_retriever 而不是 as_query_engine
            self._graph_retriever = self._index.as_retriever(
                include_text=True,
                similarity_top_k=self._similarity_top_k
            )
            logger.info("✅ 索引加载成功")
        except Exception as e:
            logger.error(f"Failed to load index from storage: {e}")

    async def retrieve(self, query: str) -> List[Document]:
        """Retrieve documents using graph traversal."""
        if not self._graph_retriever:
            logger.error("Graph retriever is not initialized. Need to build_graph first.")
            return []

        # 注意：如果你之前使用了 as_query_engine，这里返回的是 Response 对象
        # 因为我们改成了 as_retriever，这里返回的是 List[NodeWithScore]
        nodes_with_score = self._graph_retriever.retrieve(query)

        documents: List[Document] = []
        for node_with_score in nodes_with_score:
            node = node_with_score.node
            documents.append(
                Document(
                    text=node.get_content(),
                    score=node_with_score.score or 0.0,
                    metadata=node.metadata,
                )
            )

        logger.debug(f"Graph retrieved {len(documents)} documents")
        return documents

    def load_index(self, persist_dir: str) -> None:
        """重新加载索引的公开方法"""
        self._persist_dir = persist_dir
        self._load_index_if_exists()

    async def build_graph(self, documents: List[Document]) -> None:
        """Build the knowledge graph index from the provided documents."""
        # 1. 创建 Graph Store (复用方法)
        property_graph_store = self._get_neo4j_store()

        # 2. 创建 StorageContext
        storage_context = StorageContext.from_defaults(
            property_graph_store=property_graph_store,
        )

        # 3. 构建 PropertyGraphIndex
        logger.info("正在抽取实体和关系，构建图谱 (这需要调用 LLM，请稍候)...")
        self._index = PropertyGraphIndex.from_documents(
            documents,
            llm=Settings.llm,
            embed_model=Settings.embed_model,
            storage_context=storage_context,
            show_progress=True,
        )

        # 4. 持久化并更新 Retriever
        if self._persist_dir:
            # 确保目录存在
            os.makedirs(self._persist_dir, exist_ok=True)
            self._index.storage_context.persist(persist_dir=self._persist_dir)

        # 构建完成后，立即初始化 retriever 以供使用
        self._graph_retriever = self._index.as_retriever(
            include_text=True,
            similarity_top_k=self._similarity_top_k
        )

        logger.info(f"Knowledge graph index built and persisted to '{self._persist_dir}'")