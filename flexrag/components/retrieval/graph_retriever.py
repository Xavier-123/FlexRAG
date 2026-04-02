import os
import nest_asyncio
from typing import List
from langchain_openai import ChatOpenAI

from llama_index.llms.langchain import LangChainLLM
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core import load_index_from_storage, PropertyGraphIndex, Settings
from llama_index.core.graph_stores import SimplePropertyGraphStore

from flexrag.core.schema import Document
from flexrag.components.retrieval import BaseFlexRetriever, OpenAILikeEmbedding
# from llama_index.embeddings.openai_like import OpenAILikeEmbedding


import logging

logger = logging.getLogger(__name__)

# 在模块顶部执行，解决 Jupyter/FastAPI 等环境的事件循环问题
nest_asyncio.apply()


# # ==========================================================
# # 针对 Windows 环境下 LlamaIndex 持久化图谱时的 GBK 编码报错修复代码
# # ==========================================================
# def _patched_persist(self, persist_path: str, fs=None) -> None:
#     """热修复 LlamaIndex 的 SimplePropertyGraphStore 在 Windows 下的保存 Bug"""
#     import fsspec
#     fs = fs or fsspec.filesystem("file")
#     dirpath = os.path.dirname(persist_path)
#     if not fs.exists(dirpath):
#         fs.makedirs(dirpath)
#
#     # 关键修复：强制加入 encoding="utf-8"
#     with fs.open(persist_path, "w", encoding="utf-8") as f:
#         f.write(self.graph.model_dump_json())
#
#
# # 将打好补丁的方法替换回原始类中
# SimplePropertyGraphStore.persist = _patched_persist


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
            persist_dir: str | None = "./storage/simple_graph",
    ) -> None:
        self._index = None
        self._graph_retriever = None
        self._persist_dir = persist_dir
        self._similarity_top_k = top_k
        self._embed_model = OpenAILikeEmbedding(model=embed_model_name, base_url=embed_base_url, api_key=embed_api_key)
        if llm_model_name:
            llm = ChatOpenAI(model=llm_model_name, api_key=llm_api_key, base_url=llm_base_url, temperature=0.0)
            llama_index_llm = LangChainLLM(llm=llm)
            Settings.llm = llama_index_llm
        Settings.embed_model = self._embed_model

        # 尝试加载本地已有的图索引
        self._load_index_if_exists()

    def _load_index_if_exists(self) -> None:
        """安全地加载索引，如果不存在则不报错，等待后续 build_graph"""
        if not self._persist_dir or not os.path.exists(self._persist_dir):
            logger.warning(f"Persist dir '{self._persist_dir}' does not exist. Please call build_graph() first.")
            return

        try:
            # 对于 SimplePropertyGraphStore，直接从 persist_dir 恢复 StorageContext 即可
            # LlamaIndex 会自动从目录中读取 graph_store.json (如果存在的话)
            storage_context = StorageContext.from_defaults(persist_dir=self._persist_dir)
            self._index = load_index_from_storage(storage_context)

            # 初始化 Retriever
            self._graph_retriever = self._index.as_retriever(
                include_text=True,
                similarity_top_k=self._similarity_top_k
            )
            logger.info("✅ 本地简单图索引加载成功")
        except Exception as e:
            logger.error(f"Failed to load simple graph index from storage: {e}")

    async def retrieve(self, query: str) -> List[Document]:
        """Retrieve documents using graph traversal."""
        if not self._graph_retriever:
            logger.info("Graph retriever is not initialized.")
            return []

        # 获取检索到的节点 (返回 List[NodeWithScore])
        nodes_with_score = self._graph_retriever.retrieve(query)

        documents: List[Document] = []
        for node_with_score in nodes_with_score:
            node = node_with_score.node
            documents.append(
                Document(
                    text=node.get_content(),
                    score=getattr(node, "score", 0.0) or 0.0,
                    metadata=node.metadata,
                )
            )

        logger.debug("Graph retrieved %d documents", len(documents))
        return documents

    def load_index(self, persist_dir: str) -> None:
        """提供给外部调用的手动加载索引方法"""
        self._persist_dir = persist_dir
        self._load_index_if_exists()

    async def build_graph(self, documents: List[Document]) -> None:
        """Build the knowledge graph index from the provided documents."""
        # 1. 创建基于内存的 SimplePropertyGraphStore
        property_graph_store = SimplePropertyGraphStore()  # 有时候会编码报错

        # 2. 创建 StorageContext
        storage_context = StorageContext.from_defaults(
            property_graph_store=property_graph_store,
        )

        # 3. 抽取实体与关系，构建 PropertyGraphIndex
        print("正在抽取实体和关系，构建图谱 (这需要调用 LLM，请稍候)...")
        self._index = PropertyGraphIndex.from_documents(
            documents,
            llm=Settings.llm,
            embed_model=Settings.embed_model,
            storage_context=storage_context,
            show_progress=True,
        )

        # 4. 持久化到本地磁盘 (包括 graph_store.json)
        if self._persist_dir:
            os.makedirs(self._persist_dir, exist_ok=True)
            self._index.storage_context.persist(persist_dir=self._persist_dir)

            # SimplePropertyGraphStore 独有的功能：直接保存可视化网页
            try:
                html_path = os.path.join(self._persist_dir, "knowledge_graph.html")
                self._index.property_graph_store.save_networkx_graph(name=html_path)
                logger.info(f"图谱可视化文件已保存至: {html_path}")
            except ImportError:
                logger.warning("可视化失败：如需生成知识图谱 HTML，请先安装 `pip install pyvis networkx`")
            except Exception as e:
                logger.warning(f"生成可视化图谱失败: {e}")

        # 5. 构建完成后立即初始化 Retriever 供后续查询使用
        self._graph_retriever = self._index.as_retriever(
            include_text=True,
            similarity_top_k=self._similarity_top_k
        )

        logger.info("Knowledge graph index built and persisted to '%s'", self._persist_dir)
