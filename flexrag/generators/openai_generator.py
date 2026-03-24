"""
OpenAI Structured Output generator implementation.

Uses the ``with_structured_output`` feature of LangChain's ``ChatOpenAI``
to make GPT-4o return a JSON object that is automatically validated against
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


class OpenAIGenerator(BaseGenerator):
    """Answer generator that calls GPT-4o with OpenAI Structured Output.

    The LangChain ``with_structured_output`` method binds
    :class:`~flexrag.schema.RAGOutput` as the expected response schema so that
    the raw JSON returned by the model is automatically parsed and validated.

    Args:
        model: OpenAI model name (e.g. ``"gpt-4o"``).
        api_key: OpenAI API key.  When ``None`` the key is read from the
            ``OPENAI_API_KEY`` environment variable.
        temperature: Sampling temperature.  Defaults to ``0`` for deterministic,
            faithful answers.

    Example::

        gen = OpenAIGenerator(model="gpt-4o", api_key="sk-...")
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
        model: str = "gpt-4o",
        api_key: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        llm = ChatOpenAI(
            model=model,
            api_key=api_key,  # type: ignore[arg-type]
            temperature=temperature,
        )
        # Bind the Pydantic schema – LangChain will enforce the JSON shape.
        self._chain = llm.with_structured_output(RAGOutput)

    # ------------------------------------------------------------------
    # BaseGenerator interface
    # ------------------------------------------------------------------

    def generate(
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
            result: RAGOutput = self._chain.invoke(  # type: ignore[assignment]
                [
                    SystemMessage(content=_SYSTEM_PROMPT),
                    HumanMessage(content=human_prompt),
                ]
            )
        except Exception:
            logger.exception("OpenAI generator failed")
            raise

        # Safety net: if model returns no evidence fall back to source docs
        if not result.evidence and source_documents:
            result = RAGOutput(
                answer=result.answer,
                evidence=source_documents[:3],
            )

        logger.debug("Generated answer (len=%d chars)", len(result.answer))
        return result
