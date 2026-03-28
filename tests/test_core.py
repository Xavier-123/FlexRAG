"""
Unit tests for FlexRAG core components.

External network dependencies (OpenAI, vLLM) are replaced with stubs so the
tests run fully offline.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from flexrag.schema import Document, RAGOutput, RAGState


# ---------------------------------------------------------------------------
# Settings tests
# ---------------------------------------------------------------------------


class TestSettings:
    def test_default_values(self) -> None:
        from flexrag.config import Settings

        settings = Settings()
        assert settings.llm_base_url == "http://localhost:8018/v1"
        assert settings.llm_api_key == "sk-xxxx"
        assert settings.embedding_base_url == "http://localhost:8018/v1"
        assert settings.embedding_api_key == "sk-xxxx"
        assert settings.reranker_base_url == "http://localhost:8018/v1"
        assert settings.reranker_api_key == "sk-xxxx"

    def test_independent_per_component_settings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Each component should read its own env vars independently."""
        monkeypatch.setenv("LLM_BASE_URL", "http://llm:8000/v1")
        monkeypatch.setenv("LLM_API_KEY", "llm-secret")
        monkeypatch.setenv("EMBEDDING_BASE_URL", "http://embed:8001/v1")
        monkeypatch.setenv("EMBEDDING_API_KEY", "embed-secret")
        monkeypatch.setenv("RERANKER_BASE_URL", "http://reranker:8002/v1")
        monkeypatch.setenv("RERANKER_API_KEY", "reranker-secret")

        from flexrag.config import Settings

        settings = Settings()
        assert settings.llm_base_url == "http://llm:8000/v1"
        assert settings.llm_api_key == "llm-secret"
        assert settings.embedding_base_url == "http://embed:8001/v1"
        assert settings.embedding_api_key == "embed-secret"
        assert settings.reranker_base_url == "http://reranker:8002/v1"
        assert settings.reranker_api_key == "reranker-secret"

    def test_components_can_differ(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify that different components can have different base URLs and keys."""
        monkeypatch.setenv("LLM_BASE_URL", "http://llm-host/v1")
        monkeypatch.setenv("LLM_API_KEY", "key-a")
        monkeypatch.setenv("EMBEDDING_BASE_URL", "http://embed-host/v1")
        monkeypatch.setenv("EMBEDDING_API_KEY", "key-b")
        monkeypatch.setenv("RERANKER_BASE_URL", "http://reranker-host/v1")
        monkeypatch.setenv("RERANKER_API_KEY", "key-c")

        from flexrag.config import Settings

        s = Settings()
        # All three pairs must be distinct
        urls = {s.llm_base_url, s.embedding_base_url, s.reranker_base_url}
        keys = {s.llm_api_key, s.embedding_api_key, s.reranker_api_key}
        assert len(urls) == 3
        assert len(keys) == 3


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


class TestDocument:
    def test_defaults(self) -> None:
        doc = Document(text="hello")
        assert doc.text == "hello"
        assert doc.score == 0.0
        assert doc.metadata == {}

    def test_with_metadata(self) -> None:
        doc = Document(text="hello", score=0.9, metadata={"source": "wiki"})
        assert doc.metadata["source"] == "wiki"

    def test_model_dump_round_trip(self) -> None:
        doc = Document(text="test", score=0.5, metadata={"k": "v"})
        restored = Document(**doc.model_dump())
        assert restored == doc


class TestRAGOutput:
    def test_required_fields(self) -> None:
        out = RAGOutput(answer="42", evidence=["fact 1", "fact 2"])
        assert out.answer == "42"
        assert len(out.evidence) == 2

    def test_serialisation(self) -> None:
        out = RAGOutput(answer="hello", evidence=["a"])
        payload = out.model_dump()
        assert payload == {"answer": "hello", "evidence": ["a"]}


class TestRAGState:
    def test_minimal_construction(self) -> None:
        state = RAGState(query="What is RAG?")
        assert state.query == "What is RAG?"
        assert state.retrieved_docs == []
        assert state.answer == ""
        assert state.error is None


# ---------------------------------------------------------------------------
# VLLMReranker (stub HTTP)
# ---------------------------------------------------------------------------


class TestVLLMReranker:
    """Test VLLMReranker with a stubbed httpx.AsyncClient."""

    def _make_reranker(self, stub_response: dict, api_key: str | None = None) -> Any:
        from flexrag.rerankers.vllm_reranker import VLLMReranker

        mock_client = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = stub_response
        mock_resp.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_resp
        return VLLMReranker(
            base_url="http://fake",
            model="fake-reranker",
            api_key=api_key,
            http_client=mock_client,
        )

    @pytest.mark.asyncio
    async def test_basic_rerank(self) -> None:
        docs = [Document(text="doc A"), Document(text="doc B"), Document(text="doc C")]
        stub = {
            "results": [
                {"index": 0, "relevance_score": 0.3},
                {"index": 1, "relevance_score": 0.9},
                {"index": 2, "relevance_score": 0.5},
            ]
        }
        reranker = self._make_reranker(stub)
        result = await reranker.rerank("query", docs, top_k=2)
        assert len(result) == 2
        assert result[0].text == "doc B"  # highest score
        assert result[0].score == pytest.approx(0.9)

    @pytest.mark.asyncio
    async def test_empty_documents(self) -> None:
        reranker = self._make_reranker({"results": []})
        result = await reranker.rerank("query", [], top_k=5)
        assert result == []

    @pytest.mark.asyncio
    async def test_top_k_truncation(self) -> None:
        docs = [Document(text=f"doc {i}") for i in range(5)]
        stub = {
            "results": [{"index": i, "relevance_score": float(i) / 10} for i in range(5)]
        }
        reranker = self._make_reranker(stub)
        result = await reranker.rerank("query", docs, top_k=2)
        assert len(result) == 2
        assert result[0].score >= result[1].score  # descending order

    @pytest.mark.asyncio
    async def test_api_key_sent_in_header(self) -> None:
        """Authorization header must be included when api_key is set."""
        docs = [Document(text="doc A")]
        stub = {"results": [{"index": 0, "relevance_score": 0.7}]}
        reranker = self._make_reranker(stub, api_key="test-secret")
        await reranker.rerank("query", docs, top_k=1)
        _, call_kwargs = reranker._client.post.call_args
        assert call_kwargs["headers"]["Authorization"] == "Bearer test-secret"

    @pytest.mark.asyncio
    async def test_no_api_key_omits_header(self) -> None:
        """No Authorization header should be sent when api_key is None."""
        docs = [Document(text="doc A")]
        stub = {"results": [{"index": 0, "relevance_score": 0.7}]}
        reranker = self._make_reranker(stub, api_key=None)
        await reranker.rerank("query", docs, top_k=1)
        _, call_kwargs = reranker._client.post.call_args
        assert "Authorization" not in call_kwargs["headers"]


# ---------------------------------------------------------------------------
# VLLMEmbedding (stub HTTP)
# ---------------------------------------------------------------------------


class TestVLLMEmbedding:
    def _make_embedding(self, vectors: list[list[float]], api_key: str | None = None) -> Any:
        from flexrag.retrievers.llamaindex_retriever import VLLMEmbedding

        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "data": [
                {"index": i, "embedding": v} for i, v in enumerate(vectors)
            ]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_resp
        return VLLMEmbedding(
            base_url="http://fake",
            model="fake-embed",
            api_key=api_key,
            http_client=mock_client,
        )

    def test_single_embedding(self) -> None:
        embed = self._make_embedding([[0.1, 0.2, 0.3]])
        vec = embed.get_text_embedding("hello")
        assert vec == [0.1, 0.2, 0.3]

    def test_batch_embedding(self) -> None:
        embed = self._make_embedding([[1.0], [2.0]])
        vecs = embed.get_text_embedding_batch(["a", "b"])
        assert len(vecs) == 2
        assert vecs[0] == [1.0]
        assert vecs[1] == [2.0]

    def test_api_key_sent_in_header(self) -> None:
        """Authorization header must be included when api_key is set."""
        embed = self._make_embedding([[0.1]], api_key="embed-secret")
        embed.get_text_embedding("hello")
        _, call_kwargs = embed._client.post.call_args
        assert call_kwargs["headers"]["Authorization"] == "Bearer embed-secret"

    def test_no_api_key_omits_header(self) -> None:
        """No Authorization header should be sent when api_key is None."""
        embed = self._make_embedding([[0.1]], api_key=None)
        embed.get_text_embedding("hello")
        _, call_kwargs = embed._client.post.call_args
        assert "Authorization" not in call_kwargs["headers"]


# ---------------------------------------------------------------------------
# LLMContextOptimizer
# ---------------------------------------------------------------------------


class TestLLMContextOptimizer:
    def _make_optimizer(self, llm_response: str) -> Any:
        from flexrag.context_optimizers.llm_context_optimizer import LLMContextOptimizer

        mock_llm = MagicMock()
        mock_msg = MagicMock()
        mock_msg.content = llm_response
        mock_llm.ainvoke = AsyncMock(return_value=mock_msg)
        return LLMContextOptimizer(llm=mock_llm)

    @pytest.mark.asyncio
    async def test_basic_optimize(self) -> None:
        optimizer = self._make_optimizer("Extracted passage 1.")
        docs = [Document(text="Full document text.")]
        context = await optimizer.optimize("What is RAG?", docs, max_tokens=500)
        assert context == "Extracted passage 1."

    @pytest.mark.asyncio
    async def test_empty_documents(self) -> None:
        optimizer = self._make_optimizer("")
        context = await optimizer.optimize("query", [], max_tokens=100)
        assert context == ""

    @pytest.mark.asyncio
    async def test_truncation(self) -> None:
        long_response = "x" * 1000
        optimizer = self._make_optimizer(long_response)
        docs = [Document(text="doc")]
        context = await optimizer.optimize("q", docs, max_tokens=10)  # 10 * 4 = 40 chars
        assert len(context) <= 40

    @pytest.mark.asyncio
    async def test_llm_failure_fallback(self) -> None:
        from flexrag.context_optimizers.llm_context_optimizer import LLMContextOptimizer

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=RuntimeError("LLM down"))
        optimizer = LLMContextOptimizer(llm=mock_llm)
        docs = [Document(text="fallback text")]
        # Should not raise; falls back to raw concatenation
        context = await optimizer.optimize("q", docs, max_tokens=1000)
        assert "fallback text" in context


# ---------------------------------------------------------------------------
# OpenAIGenerator (stub chain)
# ---------------------------------------------------------------------------


class TestOpenAIGenerator:
    @pytest.mark.asyncio
    async def test_generate_returns_rag_output(self) -> None:
        from flexrag.generators.openai_generator import OpenAIGenerator

        mock_chain = MagicMock()
        mock_chain.ainvoke = AsyncMock(return_value=RAGOutput(
            answer="RAG is cool", evidence=["doc snippet"]
        ))

        gen = OpenAIGenerator.__new__(OpenAIGenerator)
        gen._chain = mock_chain

        output = await gen.generate("What is RAG?", "context text", ["doc snippet"], ["doc snippet"])
        assert isinstance(output, RAGOutput)
        assert output.answer == "RAG is cool"
        assert output.evidence == ["doc snippet"]

    @pytest.mark.asyncio
    async def test_evidence_fallback(self) -> None:
        from flexrag.generators.openai_generator import OpenAIGenerator

        mock_chain = MagicMock()
        # Return output with empty evidence list
        mock_chain.ainvoke = AsyncMock(return_value=RAGOutput(answer="answer", evidence=[]))

        gen = OpenAIGenerator.__new__(OpenAIGenerator)
        gen._chain = mock_chain

        output = await gen.generate("query", "ctx", ["doc snippet"], ["source doc 1", "source doc 2"])
        # Should fall back to source_documents
        assert len(output.evidence) > 0
        assert output.evidence[0] == "source doc 1"


# ---------------------------------------------------------------------------
# Graph nodes (unit level)
# ---------------------------------------------------------------------------


class TestGraphNodes:
    @pytest.mark.asyncio
    async def test_retrieve_node(self) -> None:
        from flexrag.graph.nodes import make_retrieve_node

        mock_retriever = AsyncMock()
        mock_retriever.retrieve.return_value = [
            Document(text="result 1", score=0.8),
            Document(text="result 2", score=0.6),
        ]
        node = make_retrieve_node(mock_retriever, top_k=5)
        result = await node({"query": "test query"})
        assert "retrieved_docs" in result
        assert len(result["retrieved_docs"]) == 2
        assert result["retrieved_docs"][0]["text"] == "result 1"

    @pytest.mark.asyncio
    async def test_retrieve_node_error(self) -> None:
        from flexrag.graph.nodes import make_retrieve_node

        mock_retriever = AsyncMock()
        mock_retriever.retrieve.side_effect = RuntimeError("retrieval failed")
        node = make_retrieve_node(mock_retriever, top_k=5)
        result = await node({"query": "test"})
        assert "error" in result
        assert "retrieval failed" in result["error"]

    @pytest.mark.asyncio
    async def test_rerank_node_skips_on_error(self) -> None:
        from flexrag.graph.nodes import make_rerank_node

        mock_reranker = AsyncMock()
        node = make_rerank_node(mock_reranker, top_k=3)
        # Pre-existing error in state should cause node to skip
        result = await node({"query": "q", "error": "upstream failure", "retrieved_docs": []})
        assert result == {}
        mock_reranker.rerank.assert_not_called()

    @pytest.mark.asyncio
    async def test_generate_node_skips_on_error(self) -> None:
        from flexrag.graph.nodes import make_generate_node

        mock_generator = AsyncMock()
        node = make_generate_node(mock_generator)
        result = await node({"query": "q", "error": "something went wrong"})
        assert result == {}
        mock_generator.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_optimize_context_node(self) -> None:
        from flexrag.graph.nodes import make_optimize_context_node

        mock_optimizer = AsyncMock()
        mock_optimizer.optimize.return_value = "optimized context"
        node = make_optimize_context_node(mock_optimizer, max_tokens=500)
        doc = Document(text="raw text", score=0.9)
        state = {
            "query": "query",
            "reranked_docs": [doc.model_dump()],
        }
        result = await node(state)
        assert result["optimized_context"] == "optimized context"


# ---------------------------------------------------------------------------
# Settings – knowledge base fields
# ---------------------------------------------------------------------------


class TestSettingsKnowledge:
    def test_knowledge_defaults(self) -> None:
        from flexrag.config import Settings

        s = Settings()
        assert s.knowledge_persist_dir == "./knowledge_base"
        assert s.knowledge_chunk_size == 512
        assert s.knowledge_chunk_overlap == 50

    def test_knowledge_env_overrides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KNOWLEDGE_PERSIST_DIR", "/tmp/my_kb")
        monkeypatch.setenv("KNOWLEDGE_CHUNK_SIZE", "256")
        monkeypatch.setenv("KNOWLEDGE_CHUNK_OVERLAP", "25")

        from flexrag.config import Settings

        s = Settings()
        assert s.knowledge_persist_dir == "/tmp/my_kb"
        assert s.knowledge_chunk_size == 256
        assert s.knowledge_chunk_overlap == 25


# ---------------------------------------------------------------------------
# BaseKnowledgeBuilder – abstract interface enforcement
# ---------------------------------------------------------------------------


class TestBaseKnowledgeBuilder:
    def test_cannot_instantiate_directly(self) -> None:
        """BaseKnowledgeBuilder must not be directly instantiatable."""
        from flexrag.abstractions.base_knowledge import BaseKnowledgeBuilder

        with pytest.raises(TypeError):
            BaseKnowledgeBuilder()  # type: ignore[abstract]

    def test_concrete_subclass_must_implement_all_methods(self) -> None:
        """A partial subclass that omits abstract methods must raise TypeError."""
        from flexrag.abstractions.base_knowledge import BaseKnowledgeBuilder

        class Partial(BaseKnowledgeBuilder):
            pass  # implements nothing

        with pytest.raises(TypeError):
            Partial()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# FaissKnowledgeBuilder (stub embedding + FAISS)
# ---------------------------------------------------------------------------


class TestFaissKnowledgeBuilder:
    """Tests for FaissKnowledgeBuilder using a fully mocked embedding model."""

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _make_embed_mock(self, dim: int = 4) -> Any:
        """Return a MagicMock that behaves like VLLMEmbedding."""
        mock_embed = MagicMock()
        # get_text_embedding returns a single vector
        mock_embed.get_text_embedding.return_value = [0.1] * dim
        # get_text_embedding_batch returns a list of vectors
        mock_embed.get_text_embedding_batch.return_value = [[0.1] * dim]
        return mock_embed

    def _make_builder(
        self,
        embed_mock: Any | None = None,
        dim: int = 4,
    ) -> Any:
        """Create a FaissKnowledgeBuilder with injected mocks."""
        from flexrag.knowledge.faiss_knowledge import FaissKnowledgeBuilder

        mock_http = MagicMock()
        # The HTTP response for the embedding probe used in build_index
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "data": [{"index": 0, "embedding": [0.1] * dim}]
        }
        mock_http.post.return_value = mock_resp

        builder = FaissKnowledgeBuilder(
            embed_base_url="http://fake/v1/embeddings",
            embed_model_name="fake-embed",
            embed_api_key=None,
            http_client=mock_http,
        )
        # Replace the embed model with our mock so tests work without HTTP
        if embed_mock is not None:
            builder._embed_model = embed_mock
        return builder

    # ------------------------------------------------------------------ #
    # index_exists                                                         #
    # ------------------------------------------------------------------ #

    def test_index_exists_false_when_no_file(self, tmp_path: Any) -> None:
        from flexrag.knowledge.faiss_knowledge import FaissKnowledgeBuilder

        assert FaissKnowledgeBuilder.index_exists(str(tmp_path)) is False

    def test_index_exists_true_when_file_present(self, tmp_path: Any) -> None:
        from flexrag.knowledge.faiss_knowledge import FaissKnowledgeBuilder

        (tmp_path / "faiss_index.bin").touch()
        assert FaissKnowledgeBuilder.index_exists(str(tmp_path)) is True

    # ------------------------------------------------------------------ #
    # load_files                                                           #
    # ------------------------------------------------------------------ #

    @pytest.mark.asyncio
    async def test_load_files_empty_list_raises(self) -> None:
        builder = self._make_builder()
        with pytest.raises(ValueError, match="non-empty"):
            await builder.load_files([])

    @pytest.mark.asyncio
    async def test_load_files_from_directory(self, tmp_path: Any) -> None:
        """load_files() should call SimpleDirectoryReader and return count."""
        # Write a small .txt file
        (tmp_path / "doc.txt").write_text("Hello world", encoding="utf-8")

        builder = self._make_builder()
        with patch(
            "flexrag.knowledge.faiss_knowledge.SimpleDirectoryReader"
        ) as mock_reader_cls:
            mock_reader = MagicMock()
            mock_reader.load_data.return_value = [MagicMock(), MagicMock()]
            mock_reader_cls.return_value = mock_reader

            count = await builder.load_files(str(tmp_path))

        assert count == 2
        mock_reader_cls.assert_called_once_with(input_dir=str(tmp_path))

    @pytest.mark.asyncio
    async def test_load_files_from_list(self, tmp_path: Any) -> None:
        """load_files() should pass input_files= when given a list."""
        files = [str(tmp_path / "a.txt"), str(tmp_path / "b.md")]
        builder = self._make_builder()

        with patch(
            "flexrag.knowledge.faiss_knowledge.SimpleDirectoryReader"
        ) as mock_reader_cls:
            mock_reader = MagicMock()
            mock_reader.load_data.return_value = [MagicMock()]
            mock_reader_cls.return_value = mock_reader

            count = await builder.load_files(files)

        assert count == 1
        mock_reader_cls.assert_called_once_with(input_files=files)

    # ------------------------------------------------------------------ #
    # build_index                                                          #
    # ------------------------------------------------------------------ #

    @pytest.mark.asyncio
    async def test_build_index_raises_without_docs(self) -> None:
        builder = self._make_builder()
        with pytest.raises(RuntimeError, match="No documents loaded"):
            await builder.build_index()

    @pytest.mark.asyncio
    async def test_build_index_creates_faiss_index(self) -> None:
        """build_index() should produce a non-None _index and _vector_store."""
        import faiss  # type: ignore[import]
        from llama_index.core.schema import Document as LlamaDoc

        builder = self._make_builder(dim=4)
        # Inject a minimal LlamaIndex Document as raw doc
        builder._raw_docs = [LlamaDoc(text="some content about RAG")]

        with (
            patch("flexrag.knowledge.faiss_knowledge.SentenceSplitter") as mock_split,
            patch("flexrag.knowledge.faiss_knowledge.FaissVectorStore") as mock_fvs,
            patch("flexrag.knowledge.faiss_knowledge.VectorStoreIndex") as mock_vsi,
            patch("flexrag.knowledge.faiss_knowledge.StorageContext") as mock_sc,
            patch("flexrag.knowledge.faiss_knowledge.faiss") as mock_faiss,
        ):
            mock_split.return_value.get_nodes_from_documents.return_value = [
                MagicMock()
            ]
            mock_faiss.IndexFlatL2.return_value = MagicMock()
            mock_fvs.return_value = MagicMock()
            mock_sc.from_defaults.return_value = MagicMock()
            mock_vsi.return_value = MagicMock()

            await builder.build_index(chunk_size=128, chunk_overlap=16)

        assert builder._index is not None
        assert builder._vector_store is not None

    # ------------------------------------------------------------------ #
    # save                                                                 #
    # ------------------------------------------------------------------ #

    @pytest.mark.asyncio
    async def test_save_raises_when_index_not_built(self, tmp_path: Any) -> None:
        builder = self._make_builder()
        with pytest.raises(RuntimeError, match="not built"):
            await builder.save(str(tmp_path))


# ---------------------------------------------------------------------------
# LlamaIndexRetriever – load_index
# ---------------------------------------------------------------------------


class TestLlamaIndexRetrieverLoadIndex:
    """Tests for LlamaIndexRetriever.load_index()."""

    @pytest.mark.asyncio
    async def test_load_index_raises_when_file_missing(self, tmp_path: Any) -> None:
        from flexrag.retrievers.llamaindex_retriever import LlamaIndexRetriever

        retriever = LlamaIndexRetriever(
            index=None,
            embed_base_url="http://fake",
            embed_model_name="fake-embed",
        )
        with pytest.raises(FileNotFoundError):
            await retriever.load_index(str(tmp_path))
