"""
LLM-based context optimization strategy.

This optimizer uses an LLM to:
1. Selectively extract the most relevant passages from each document.
2. Concatenate the extractions into a single coherent context string.
3. Ensure the result fits within the configured token budget.

If the LLM call fails or the result exceeds the budget the module falls back
to a simple truncation strategy so that the pipeline never stalls.
"""

from __future__ import annotations

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from flexrag.abstractions.base_context_optimizer import BaseContextOptimizer
from flexrag.schema import Document

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a precise context extractor.  Given a user question and several "
    "document passages, extract and return ONLY the sentences or paragraphs that "
    "are directly relevant to answering the question.  "
    "Do NOT add commentary, introductions, or conclusions.  "
    "Separate extracted passages with a blank line."
)


class LLMContextOptimizer(BaseContextOptimizer):
    """Context optimiser powered by an LLM (e.g. GPT-4o).

    Uses the LLM to selectively extract the most relevant portions of each
    document so that the generator receives a dense, high-signal context
    window.  Falls back to plain truncation when the LLM is unavailable or
    returns an overly long response.

    Args:
        llm: A LangChain :class:`~langchain_core.language_models.BaseChatModel`
            instance (e.g. ``ChatOpenAI(model="gpt-4o")``).
        chars_per_token: Approximate character-to-token ratio used when
            estimating token counts without a tokeniser.  Defaults to ``4``.

    Example::

        from langchain_openai import ChatOpenAI
        optimizer = LLMContextOptimizer(llm=ChatOpenAI(model="gpt-4o"))
        context = optimizer.optimize(
            query="What is RAG?",
            documents=reranked_docs,
            max_tokens=3000,
        )
    """

    def __init__(
        self,
        llm: BaseChatModel,
        chars_per_token: int = 4,
    ) -> None:
        self._llm = llm
        self._chars_per_token = chars_per_token

    # ------------------------------------------------------------------
    # BaseContextOptimizer interface
    # ------------------------------------------------------------------

    def optimize(
        self,
        query: str,
        documents: list[Document],
        max_tokens: int,
    ) -> str:
        """Extract relevant passages from *documents* using the LLM.

        Args:
            query: The user's original question.
            documents: Reranked documents to distil.
            max_tokens: Approximate token budget for the output context.

        Returns:
            A compact, relevant context string ready to be included in the
            generator prompt.
        """
        if not documents:
            return ""

        # Build a numbered document listing for the prompt
        doc_listing = "\n\n".join(
            f"[Doc {i + 1}]\n{doc.text}" for i, doc in enumerate(documents)
        )
        human_msg = (
            f"Question: {query}\n\n"
            f"Documents:\n{doc_listing}\n\n"
            "Extract the relevant passages."
        )

        try:
            response = self._llm.invoke(
                [
                    SystemMessage(content=_SYSTEM_PROMPT),
                    HumanMessage(content=human_msg),
                ]
            )
            optimized: str = response.content  # type: ignore[union-attr]
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "LLM context optimisation failed (%s); falling back to truncation.",
                exc,
            )
            optimized = doc_listing

        # Enforce token budget via simple character truncation
        max_chars = max_tokens * self._chars_per_token
        if len(optimized) > max_chars:
            logger.debug(
                "Truncating optimised context from %d to %d chars",
                len(optimized),
                max_chars,
            )
            optimized = optimized[:max_chars]

        return optimized
