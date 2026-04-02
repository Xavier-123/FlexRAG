from abc import ABC, abstractmethod
from pydantic import PrivateAttr
from typing import Any, List, Optional
import httpx

from llama_index.core.embeddings import BaseEmbedding

from flexrag.common.schema import Document


class OpenAILikeEmbedding(BaseEmbedding):
    """Minimal OpenAI-compatible embedding wrapper for a vLLM embedding endpoint.

    Now properly inherits from LlamaIndex's BaseEmbedding to pass type validation.
    """

    # Pydantic 字段定义
    base_url: str
    model_name: str  # LlamaIndex 规范中通常使用 model_name
    api_key: Optional[str] = None

    # 非序列化的内部私有变量，需要用 PrivateAttr 声明
    _client: Any = PrivateAttr()
    _aclient: Any = PrivateAttr()
    _endpoint: str = PrivateAttr()

    def __init__(
            self,
            base_url: str,
            model: str,
            api_key: str | None = None,
            embed_batch_size: int = 5,
            http_client: Any | None = None,
            async_http_client: Any | None = None,
            **kwargs: Any,
    ) -> None:
        # 1. 初始化父类 (Pydantic BaseModel)
        super().__init__(
            base_url=base_url,
            model_name=model,
            api_key=api_key,
            **kwargs
        )

        # 2. 初始化内部变量
        base_url = base_url.rstrip("/")
        if base_url.endswith("/v1/embeddings"):
            self._endpoint = base_url
        elif base_url.endswith("/v1"):
            self._endpoint = base_url + "/embeddings"
        else:
            self._endpoint = base_url + "/v1/embeddings"
        self._model = model
        self._api_key = api_key
        self._client = http_client or httpx.Client(timeout=60.0)
        self._aclient = async_http_client or httpx.AsyncClient(timeout=60.0)
        self.embed_batch_size = embed_batch_size

    # ------------------------------------------------------------------
    # 同步方法 (Synchronous Methods)
    # ------------------------------------------------------------------
    def _get_query_embedding(self, query: str) -> List[float]:
        """为检索的问题生成单条向量"""
        return self._get_text_embeddings([query])[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        """为单本文档生成向量"""
        return self._get_text_embeddings([text])[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """批量生成文档向量 (重写此方法以利用 vLLM 的批量推理加速)"""
        payload = {"model": self.model_name, "input": texts, "encoding_format": "float"}
        headers: dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = self._client.post(self._endpoint, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()["data"]

        # Sort by index to preserve order (OpenAI spec)
        data.sort(key=lambda item: item["index"])
        return [item["embedding"] for item in data]

    # ------------------------------------------------------------------
    # 异步方法 (Asynchronous Methods)
    # ------------------------------------------------------------------

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """异步获取查询问题的向量"""
        res = await self._aget_text_embeddings([query])
        return res[0]

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """异步获取单本文档的向量"""
        res = await self._aget_text_embeddings([text])
        return res[0]

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """异步批量获取文档向量"""
        payload = {"model": self.model_name, "input": texts}
        headers: dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = await self._aclient.post(self._endpoint, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()["data"]
        data.sort(key=lambda item: item["index"])
        return [item["embedding"] for item in data]


class BaseFlexRetriever(ABC):
    """Strategy interface for document retrieval.

    All concrete retriever implementations must subclass this ABC and
    implement :meth:`retrieve`.  This decouples the LangGraph node logic
    from any specific vector store or retrieval library.

    Example subclasses:
        - :class:`flexrag.components.retrieval.LlamaIndexRetriever`
    """
    similarity_top_k = 10

    @abstractmethod
    async def retrieve(self, query: str) -> list[Document]:
        """Retrieve the most relevant documents for *query*.

        Args:
            query: The user's question or search string.

        Returns:
            A list of :class:`~flexrag.common.schema.Document` objects sorted by
            descending relevance score.
        """


    async def load_files(self, path: str | list[str]) -> int:
        """Load documents from a directory path or an explicit list of file paths.

        Args:
            path: A directory path (``str``) **or** a list of individual file
                paths.

        Returns:
            The number of source documents loaded.
        """


    async def build_index(
            self,
            chunk_size: int = 512,
            chunk_overlap: int = 50,
    ) -> None:
        """Chunk the loaded documents and build the vector index.

        Args:
            chunk_size: Maximum number of tokens per chunk.
            chunk_overlap: Number of tokens shared between consecutive chunks.
        """


    async def save(self, persist_dir: str) -> None:
        """Persist the current vector index to *persist_dir* on disk.

        Args:
            persist_dir: Absolute or relative path of the storage directory.
        """

    @classmethod
    def index_exists(cls, persist_dir: str) -> bool:
        """Return ``True`` if a valid persisted index exists at *persist_dir*.

        Args:
            persist_dir: Directory to inspect.
        """
