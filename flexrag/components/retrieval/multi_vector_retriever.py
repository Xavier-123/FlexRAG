import os
import asyncio
import faiss
import json
from typing import Any, List, Optional, Dict

from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage, Settings
from llama_index.core import Document as LlamaDocument
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import NodeWithScore
from llama_index.core.readers.base import BaseReader

from flexrag.common.schema import Document

# -----------------------------
# Config
# -----------------------------
_PROBE_TEXT = "dimension probe"


class _CustomReader(BaseReader):
    """自定义 JSON 读取器，专门处理 [{"idx": 0, "title": "...", "text": "..."}, ...] 格式。"""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50) -> None:
        super().__init__()
        self.CHUNK_SIZE = chunk_size
        self.CHUNK_OVERLAP = chunk_overlap

    def _split_text(self, text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
        """
        兼容中英文的智能滑动窗口切割算法。
        按字符数切割，并在重叠区寻找标点或空格进行“软截断”，防止切断句子或英文单词。
        """
        if not text:
            return []

        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = start + self.CHUNK_SIZE

            # 如果还没到文本末尾，尝试“软截断”
            if end < text_len:
                # 在重叠区内（从 end 往回找）寻找最佳截断点
                search_start = max(start, end - chunk_overlap)

                # 优先级 1：寻找句尾标点（中英文）或换行符
                found_punctuation = False
                for i in range(end - 1, search_start - 1, -1):
                    # 匹配中文句号/问号/叹号，换行符，或者英文句点(带空格，防止切断 3.14 这种小数)
                    if text[i] in ['。', '！', '？', '\n'] or (text[i] == '.' and i + 1 < text_len and text[i + 1] == ' '):
                        end = i + 1
                        found_punctuation = True
                        break

                # 优先级 2：如果没有句子标点，寻找空格（主要防止切断英文单词）
                if not found_punctuation:
                    for i in range(end - 1, search_start - 1, -1):
                        if text[i] == ' ':
                            end = i + 1
                            break

            # 提取切块并去除首尾空白
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # 计算下一个块的起点，确保步进（防止死循环）
            next_start = end - chunk_overlap
            start = max(start + 1, next_start)

        return chunks

    def load_data(self, file: str, extra_info: dict | None = None) -> list[LlamaDocument]:
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON array in {file}")

        docs = []
        for item in data:
            if not isinstance(item, dict):
                continue

            title = item.get("title", "").strip()
            text = item.get("text") or item.get("context") or ""
            idx = item.get("idx")

            if not text:
                continue

            # 对长文本进行滑动窗口切割
            text_chunks = self._split_text(text, chunk_size=self.CHUNK_SIZE, chunk_overlap=self.CHUNK_OVERLAP)

            # 遍历切出来的每一块（chunk），强制拼上 Title
            for chunk_id, chunk in enumerate(text_chunks):
                content_parts = []
                if title:
                    # 每一小块的头部都带上标题，死死锚定全局语义
                    content_parts.append(f"Title: {title}")

                content_parts.append(f"Context: {chunk}")

                # 拼接合并
                content = "\n".join(content_parts)

                # 将 idx 和 chunk_id 存入 metadata
                metadata = {}
                if idx is not None:
                    metadata["idx"] = idx
                if title:
                    metadata["title"] = title

                # 记录这是当前文章的第几个切块
                metadata["chunk_id"] = chunk_id

                # （可选高级技巧：父子文档检索）
                # 你可以把完整的原始 text 也存进 metadata 里。这样检索时虽然匹配的是这个 content，但传给大模型的是原汁原味的 metadata["parent_text"]，但有可能超长
                # metadata["parent_text"] = text

                docs.append(LlamaDocument(text=content, metadata=metadata))

        return docs


class MultiVectorRetriever:
    def __init__(
            self,
            embed_model,
            vector_store_type: str = "faiss",  # faiss | milvus | chroma
            dense_mode: str = "exact_l2",  # exact_l2 | exact_cosine | approx_l2 | approx_cosine
            collection_name: str = "default",
            host="localhost",
            port=8000,
            persist_dir: Optional[str] = None,
            top_k: int = 5,
            metadata_filters: Optional[Dict] = None,
            **kwargs,
    ):
        """
        Multi vector store retriever supporting FAISS / Milvus / Chroma
        """
        self._raw_docs = None
        self._embed_model = embed_model
        Settings.embed_model = embed_model

        self._vector_store_type = vector_store_type.lower()
        self._dense_mode = dense_mode.lower()
        self._collection_name = collection_name
        self._host = host
        self._port = port
        self._persist_dir = os.path.join(persist_dir, self._vector_store_type + "_" + dense_mode)
        self._top_k = top_k
        self._metadata_filters = metadata_filters

        self._index: Optional[VectorStoreIndex] = None
        self._vector_store = None
        self._retriever = None

        if persist_dir:
            if not os.path.exists(persist_dir):
                raise FileNotFoundError(f"Persist directory not found: {persist_dir}")
            self._load_index(**kwargs)

    def _is_index_files_exist(self) -> bool:
        """检查具体的持久化文件是否存在"""
        if not self._persist_dir:
            return False
        if self._vector_store_type == "faiss":
            return os.path.exists(os.path.join(self._persist_dir, "default__vector_store.json"))
        return True  # Chroma等直接看目录即可

    def _create_vector_store(self, embed_dim: int, **kwargs):
        if self._vector_store_type == "faiss":
            from llama_index.vector_stores.faiss import FaissVectorStore
            faiss_index = faiss.IndexFlatL2(embed_dim)
            return FaissVectorStore(faiss_index=faiss_index)

        elif self._vector_store_type == "chroma":
            import chromadb
            from llama_index.vector_stores.chroma import ChromaVectorStore

            if self._persist_dir:
                db = chromadb.PersistentClient(path=self._persist_dir)
            else:
                db = chromadb.EphemeralClient()

            chroma_collection = db.get_or_create_collection(self._collection_name)
            return ChromaVectorStore(
                chroma_collection=chroma_collection,
                collection_name=self._collection_name,
                **kwargs
            )

        elif self._vector_store_type == "milvus":
            from llama_index.vector_stores.milvus import MilvusVectorStore
            return MilvusVectorStore(dim=embed_dim, host=self._host, port=self._port, **kwargs)

        else:
            raise ValueError(f"Unsupported vector store: {self._vector_store_type}")

    # -----------------------------
    # Load index
    # -----------------------------
    def _load_index(self, **kwargs):
        if self._vector_store_type == "faiss":
            from llama_index.vector_stores.faiss import FaissVectorStore
            self._vector_store = FaissVectorStore.from_persist_dir(self._persist_dir)

        elif self._vector_store_type == "chroma":
            import chromadb
            from llama_index.vector_stores.chroma import ChromaVectorStore
            # db = chromadb.PersistentClient(path=persist_dir)
            db = chromadb.PersistentClient(path=self._persist_dir)
            chroma_collection = db.get_or_create_collection(self._collection_name)
            self._vector_store = ChromaVectorStore(
                chroma_collection=chroma_collection,
                collection_name=self._collection_name,
                **kwargs
            )

        elif self._vector_store_type == "milvus":
            from llama_index.vector_stores.milvus import MilvusVectorStore
            self._vector_store = MilvusVectorStore(host=self._host, port=self._port, **kwargs)

        storage_context = StorageContext.from_defaults(
            vector_store=self._vector_store,
            # persist_dir=persist_dir,
            persist_dir=self._persist_dir,
        )

        self._index = load_index_from_storage(storage_context)
        self._retriever = None

    async def load_files(self, reader) -> int:
        self._raw_docs = await asyncio.to_thread(reader.load_data)
        return len(self._raw_docs)

    # -----------------------------
    # Build index
    # -----------------------------
    async def build_index(self, chunk_size=512, chunk_overlap=50, **kwargs):
        if not hasattr(self, "_raw_docs"):
            raise RuntimeError("No documents loaded")

        splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        nodes = splitter.get_nodes_from_documents(self._raw_docs)

        embed_dim = await asyncio.to_thread(self._detect_embedding_dim)

        self._vector_store = self._create_vector_store(embed_dim=embed_dim, **kwargs)
        storage_context = StorageContext.from_defaults(vector_store=self._vector_store)

        self._index = await asyncio.to_thread(VectorStoreIndex, nodes, storage_context=storage_context)

    # -----------------------------
    # Retrieve with metadata filter
    # -----------------------------
    async def retrieve(self, query: str, filters: Optional[Dict] = None):
        final_filters = filters or self._metadata_filters

        llama_filters = None
        is_faiss = self._vector_store_type == "faiss"

        # 只有非 FAISS 的向量库（如 Chroma/Milvus），才使用 LlamaIndex 原生的预过滤
        if final_filters and not is_faiss:
            from llama_index.core.vector_stores.types import MetadataFilters, ExactMatchFilter
            filter_list = [ExactMatchFilter(key=k, value=v) for k, v in final_filters.items()]
            llama_filters = MetadataFilters(filters=filter_list)

        # 动态决定 fetch_k，如果是 FAISS 并且有过滤条件，我们要放大检索量 (比如放大10倍或50倍)，防止过滤后结果为空
        fetch_k = self._top_k
        if is_faiss and final_filters:
            fetch_k = self._top_k * 10  # 你可以根据数据量自行调节放大倍数

        self._retriever = self._index.as_retriever(
            similarity_top_k=fetch_k,
            filters=llama_filters
        )

        nodes: List[NodeWithScore] = await self._retriever.aretrieve(query)

        documents: list[Document] = []
        for node in nodes:
            documents.append(
                Document(
                    text=node.get_content(),
                    score=node.score or 0.0,
                    metadata=node.metadata,
                )
            )

        # 如果是 FAISS 且有过滤条件，则执行 Python 层的后置过滤 (Post-filtering)
        if is_faiss and final_filters:
            def match(meta):
                return all(meta.get(k) == v for k, v in final_filters.items())

            # 执行 Python 列表过滤
            documents = [d for d in documents if match(d.metadata)]
            # 截断，只保留最终需要的 top_k 个
            documents = documents[:self._top_k]

        return documents

    async def save(self, persist_dir: Optional[str] = None):
        target_dir = persist_dir or self._persist_dir
        if not target_dir:
            raise ValueError("No persist_dir provided to save index.")

        os.makedirs(target_dir, exist_ok=True)

        if self._vector_store_type == "faiss":
            faiss_path = os.path.join(target_dir, "default__vector_store.json")
            await asyncio.to_thread(self._vector_store.persist, persist_path=faiss_path)
            # if self._index_mode == "exact":
            #     faiss_path = os.path.join(persist_dir, "faiss_exact")
            # else:
            #     faiss_path = os.path.join(persist_dir, "faiss_approx")
            # await asyncio.to_thread(self._vector_store.persist, persist_path=faiss_path)

        await asyncio.to_thread(self._index.storage_context.persist, persist_dir=target_dir)


    def _detect_embedding_dim(self):
        return len(self._embed_model.get_text_embedding(_PROBE_TEXT))

    @staticmethod
    def index_exists(persist_dir: str):
        return os.path.exists(persist_dir)


async def faiss_test():
    # 1️⃣ 构造 Reader
    from llama_index.core import SimpleDirectoryReader
    reader = SimpleDirectoryReader(
        input_dir="../../../data/other"  # 放一些 txt/json/pdf
    )

    # 2️⃣ 初始化 embedding
    from flexrag.components.retrieval import OpenAILikeEmbedding
    embed_model = OpenAILikeEmbedding(
        model_name="Qwen3-Embedding-0.6B",
        base_url="http://127.0.0.1:19002/v1",
        api_key="sk-1234567890"
    )
    persist_dir = "../../../data/knowledge_persist_dir/hotpotqa/faiss_exact_l2"

    print("--- Testing FAISS ---")
    retriever = MultiVectorRetriever(
        embed_model=embed_model,
        vector_store_type="faiss",
        persist_dir=persist_dir
    )

    if not retriever._index:
        print("⚠️ 未检测到有效索引，开始构建...")
        num_docs = await retriever.load_files(reader)
        print(f"Loaded {num_docs} documents")
        await retriever.build_index(chunk_size=512, chunk_overlap=50)
        await retriever.save()
        print("✅ 索引构建完成并保存")
    else:
        print("✅ 检测到已有索引，直接加载成功...")

    results = await retriever.retrieve(
        "what is one of the stars of  The Newcomers known for",
        # filters={"file_name": "5.txt"}
        filters={"title": "The Newcomers (film)"}
    )

    print("\n🔍 检索结果：")
    for r in results:
        # ✅ 修复：对象属性访问
        print(f"score: {r.score:.4f}")
        print(f"text: {r.text[:100]}")
        print(f"metadata: {r.metadata}")
        print("-" * 50)


async def chroma_test():
    from llama_index.core import SimpleDirectoryReader
    from llama_index.embeddings.openai import OpenAIEmbedding
    embed_model = OpenAIEmbedding()

    persist_dir = "./storage/chroma"
    reader = SimpleDirectoryReader(input_dir="../../../data/other")

    print("--- Testing Chroma ---")
    # ✅ 修复：初始化时直接传入 persist_dir，不要事后赋值
    retriever = MultiVectorRetriever(
        embed_model=embed_model,
        vector_store_type="chroma",
        persist_dir=persist_dir,
        collection_name="test_collection"
    )

    if not retriever._index:
        print("⚠️ 构建 Chroma 索引")
        await retriever.load_files(reader)
        await retriever.build_index()
        await retriever.save()
    else:
        print("✅ 加载 Chroma 索引成功")

    results = await retriever.retrieve(
        "李雪峰是谁？",
        filters={"file_name": "5.txt"}
    )

    print("\n🔍 Chroma结果：")
    for r in results:
        print(f"score: {r.score:.4f} | text: {r.text[:50]}...")


if __name__ == '__main__':
    asyncio.run(faiss_test())
    # asyncio.run(chroma_test())
