"""
vLLM-compatible structured output generator.

Uses the ``with_structured_output`` feature of LangChain's ``ChatOpenAI``
against an OpenAI-compatible endpoint (e.g. a vLLM server) to return a JSON
object that is automatically validated against
:class:`~flexrag.schema.RAGOutput`.

Reference:
    https://python.langchain.com/docs/how_to/structured_output/
"""

from __future__ import annotations

import logging

from langchain_openai import ChatOpenAI

from flexrag.abstractions.base_generator import BaseGenerator
from flexrag.schema import RAGOutput

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a helpful, factual assistant.  "
    "Answer the user's question using ONLY the information in the provided context.  "
    "If the context does not contain enough information, say so honestly.  "
    "Always cite the exact passages from the context that support your answer "
    "in the 'evidence' field."
)

_SYSTEM_PROMPT_ZH = '''你是一个严谨的基于事实的问答助手。请严格遵循以下规则：
严禁脱稿：仅使用提供的上下文（Context）信息回答问题，绝不能使用外部知识或自行推测。
诚实作答：如果上下文信息不足以回答该问题，请直接回答‘抱歉，提供的上下文中没有足够的信息’。
提供证据：务必在 evidence 字段中，evidence 必须是 list，里面可能包含多个string，一字不差地引用支持你回答的原文段落。'''


class OpenAIGenerator(BaseGenerator):
    """Answer generator that calls an LLM with structured output.

    Uses LangChain's ``with_structured_output`` against any OpenAI-compatible
    endpoint (vLLM, OpenAI, etc.) to bind
    :class:`~flexrag.schema.RAGOutput` as the expected response schema so that
    the raw JSON returned by the model is automatically parsed and validated.

    Args:
        model: Model name (e.g. ``"Qwen/Qwen2.5-7B-Instruct"``).
        api_key: API key for the endpoint.  When ``None`` the key is read from
            the ``OPENAI_API_KEY`` environment variable.
        base_url: Base URL of an OpenAI-compatible API endpoint
            (e.g. ``"http://localhost:8000/v1"`` for a vLLM server).
            When ``None``, the official OpenAI endpoint is used.
        temperature: Sampling temperature.  Defaults to ``0`` for deterministic,
            faithful answers.

    Example::

        gen = OpenAIGenerator(
            model="Qwen/Qwen2.5-7B-Instruct",
            api_key="my-secret-key",
            base_url="http://localhost:8000/v1",
        )
        output = gen.generate(
            query="What is RAG?",
            context="RAG stands for Retrieval-Augmented Generation ...",
            source_documents=["RAG stands for Retrieval-Augmented Generation ..."],
        )
        print(output.answer)
        print(output.evidence)
    """

    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-7B-Instruct",
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        llm = ChatOpenAI(
            model=model,
            api_key=api_key,  # type: ignore[arg-type]
            base_url=base_url,
            temperature=temperature,
        )
        # Bind the Pydantic schema – LangChain will enforce the JSON shape.
        self._chain = llm.with_structured_output(RAGOutput)

    # ------------------------------------------------------------------
    # BaseGenerator interface
    # ------------------------------------------------------------------

    async def generate(
        self,
        query: str,
        context: str,
        source_documents: list[str],
    ) -> RAGOutput:
        """Call GPT-4o and return a structured :class:`~flexrag.schema.RAGOutput`.

        Args:
            query: The user's question.
            context: Optimised context produced by the context optimisation node.
            source_documents: Raw document texts (used to populate
                ``evidence`` when the model cannot cite specific passages).

        Returns:
            A :class:`~flexrag.schema.RAGOutput` with ``answer`` and
            ``evidence`` populated.

        Raises:
            Exception: Re-raises any LLM / network exception after logging.
        """
        human_prompt = (
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Return a JSON object with 'answer' and 'evidence' fields."
        )

        from langchain_core.messages import HumanMessage, SystemMessage

        logger.debug("Calling OpenAI structured output for query: %r", query)
        try:
            result: RAGOutput = await self._chain.ainvoke(  # type: ignore[assignment]
                [
                    SystemMessage(content=_SYSTEM_PROMPT_ZH),
                    HumanMessage(content=human_prompt),
                ]
            )
        except Exception as e:
            logger.exception(f"OpenAI generator failed. \n  {e}")
            raise

        # Safety net: if model returns no evidence fall back to source docs
        if not result.evidence and source_documents:
            result = RAGOutput(
                answer=result.answer,
                evidence=source_documents[:3],
            )

        logger.debug("Generated answer (len=%d chars)", len(result.answer))
        return result
