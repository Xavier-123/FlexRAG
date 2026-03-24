"""
Unit tests for FlexRAG core components.

External network dependencies (OpenAI, vLLM) are replaced with stubs so the
tests run fully offline.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

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
    """Test VLLMReranker with a stubbed httpx.Client."""

    def _make_reranker(self, stub_response: dict, api_key: str | None = None) -> Any:
        from flexrag.rerankers.vllm_reranker import VLLMReranker

        mock_client = MagicMock()
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

    def test_basic_rerank(self) -> None:
        docs = [Document(text="doc A"), Document(text="doc B"), Document(text="doc C")]
        stub = {
            "results": [
                {"index": 0, "relevance_score": 0.3},
                {"index": 1, "relevance_score": 0.9},
                {"index": 2, "relevance_score": 0.5},
            ]
        }
        reranker = self._make_reranker(stub)
        result = reranker.rerank("query", docs, top_k=2)
        assert len(result) == 2
        assert result[0].text == "doc B"  # highest score
        assert result[0].score == pytest.approx(0.9)

    def test_empty_documents(self) -> None:
        reranker = self._make_reranker({"results": []})
        result = reranker.rerank("query", [], top_k=5)
        assert result == []

    def test_top_k_truncation(self) -> None:
        docs = [Document(text=f"doc {i}") for i in range(5)]
        stub = {
            "results": [{"index": i, "relevance_score": float(i) / 10} for i in range(5)]
        }
        reranker = self._make_reranker(stub)
        result = reranker.rerank("query", docs, top_k=2)
        assert len(result) == 2
        assert result[0].score >= result[1].score  # descending order

    def test_api_key_sent_in_header(self) -> None:
        """Authorization header must be included when api_key is set."""
        docs = [Document(text="doc A")]
        stub = {"results": [{"index": 0, "relevance_score": 0.7}]}
        reranker = self._make_reranker(stub, api_key="test-secret")
        reranker.rerank("query", docs, top_k=1)
        _, call_kwargs = reranker._client.post.call_args
        assert call_kwargs["headers"]["Authorization"] == "Bearer test-secret"

    def test_no_api_key_omits_header(self) -> None:
        """No Authorization header should be sent when api_key is None."""
        docs = [Document(text="doc A")]
        stub = {"results": [{"index": 0, "relevance_score": 0.7}]}
        reranker = self._make_reranker(stub, api_key=None)
        reranker.rerank("query", docs, top_k=1)
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
        mock_llm.invoke.return_value = mock_msg
        return LLMContextOptimizer(llm=mock_llm)

    def test_basic_optimize(self) -> None:
        optimizer = self._make_optimizer("Extracted passage 1.")
        docs = [Document(text="Full document text.")]
        context = optimizer.optimize("What is RAG?", docs, max_tokens=500)
        assert context == "Extracted passage 1."

    def test_empty_documents(self) -> None:
        optimizer = self._make_optimizer("")
        context = optimizer.optimize("query", [], max_tokens=100)
        assert context == ""

    def test_truncation(self) -> None:
        long_response = "x" * 1000
        optimizer = self._make_optimizer(long_response)
        docs = [Document(text="doc")]
        context = optimizer.optimize("q", docs, max_tokens=10)  # 10 * 4 = 40 chars
        assert len(context) <= 40

    def test_llm_failure_fallback(self) -> None:
        from flexrag.context_optimizers.llm_context_optimizer import LLMContextOptimizer

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("LLM down")
        optimizer = LLMContextOptimizer(llm=mock_llm)
        docs = [Document(text="fallback text")]
        # Should not raise; falls back to raw concatenation
        context = optimizer.optimize("q", docs, max_tokens=1000)
        assert "fallback text" in context


# ---------------------------------------------------------------------------
# OpenAIGenerator (stub chain)
# ---------------------------------------------------------------------------


class TestOpenAIGenerator:
    def test_generate_returns_rag_output(self) -> None:
        from flexrag.generators.openai_generator import OpenAIGenerator

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = RAGOutput(
            answer="RAG is cool", evidence=["doc snippet"]
        )

        gen = OpenAIGenerator.__new__(OpenAIGenerator)
        gen._chain = mock_chain

        output = gen.generate("What is RAG?", "context text", ["doc snippet"])
        assert isinstance(output, RAGOutput)
        assert output.answer == "RAG is cool"
        assert output.evidence == ["doc snippet"]

    def test_evidence_fallback(self) -> None:
        from flexrag.generators.openai_generator import OpenAIGenerator

        mock_chain = MagicMock()
        # Return output with empty evidence list
        mock_chain.invoke.return_value = RAGOutput(answer="answer", evidence=[])

        gen = OpenAIGenerator.__new__(OpenAIGenerator)
        gen._chain = mock_chain

        output = gen.generate("query", "ctx", ["source doc 1", "source doc 2"])
        # Should fall back to source_documents
        assert len(output.evidence) > 0
        assert output.evidence[0] == "source doc 1"


# ---------------------------------------------------------------------------
# Graph nodes (unit level)
# ---------------------------------------------------------------------------


class TestGraphNodes:
    def test_retrieve_node(self) -> None:
        from flexrag.graph.nodes import make_retrieve_node

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [
            Document(text="result 1", score=0.8),
            Document(text="result 2", score=0.6),
        ]
        node = make_retrieve_node(mock_retriever, top_k=5)
        result = node({"query": "test query"})
        assert "retrieved_docs" in result
        assert len(result["retrieved_docs"]) == 2
        assert result["retrieved_docs"][0]["text"] == "result 1"

    def test_retrieve_node_error(self) -> None:
        from flexrag.graph.nodes import make_retrieve_node

        mock_retriever = MagicMock()
        mock_retriever.retrieve.side_effect = RuntimeError("retrieval failed")
        node = make_retrieve_node(mock_retriever, top_k=5)
        result = node({"query": "test"})
        assert "error" in result
        assert "retrieval failed" in result["error"]

    def test_rerank_node_skips_on_error(self) -> None:
        from flexrag.graph.nodes import make_rerank_node

        mock_reranker = MagicMock()
        node = make_rerank_node(mock_reranker, top_k=3)
        # Pre-existing error in state should cause node to skip
        result = node({"query": "q", "error": "upstream failure", "retrieved_docs": []})
        assert result == {}
        mock_reranker.rerank.assert_not_called()

    def test_generate_node_skips_on_error(self) -> None:
        from flexrag.graph.nodes import make_generate_node

        mock_generator = MagicMock()
        node = make_generate_node(mock_generator)
        result = node({"query": "q", "error": "something went wrong"})
        assert result == {}
        mock_generator.generate.assert_not_called()

    def test_optimize_context_node(self) -> None:
        from flexrag.graph.nodes import make_optimize_context_node

        mock_optimizer = MagicMock()
        mock_optimizer.optimize.return_value = "optimized context"
        node = make_optimize_context_node(mock_optimizer, max_tokens=500)
        doc = Document(text="raw text", score=0.9)
        state = {
            "query": "query",
            "reranked_docs": [doc.model_dump()],
        }
        result = node(state)
        assert result["optimized_context"] == "optimized context"
