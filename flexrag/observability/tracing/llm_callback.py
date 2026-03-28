"""
LangChain callback handler that audits every LLM prompt and response.

Usage::

    from flexrag.observability.tracing import PromptAuditCallback
    from langchain_openai import ChatOpenAI

    callback = PromptAuditCallback(log_path="./data/audit_llm.jsonl")
    llm = ChatOpenAI(model="...", callbacks=[callback])

Each line in the output file is a JSON object with the fields::

    {
      "timestamp": "2026-03-28T08:00:00.000000",
      "run_id": "uuid-string",
      "model": "model-name",
      "messages": [["human: ...", "ai: ..."]],   // serialised chat turns
      "response": "the full generated text"
    }
"""

from __future__ import annotations

import datetime
import json
import logging
import pathlib
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

logger = logging.getLogger(__name__)


class PromptAuditCallback(BaseCallbackHandler):
    """Append every LLM call's prompt and response to a JSONL audit log.

    The log is written atomically one JSON object per line so the file
    stays valid even if the process is killed mid-run.

    Args:
        log_path: Path to the output ``.jsonl`` file.  Parent directories
            are created automatically.  Existing content is preserved
            (the file is opened in append mode).
    """

    def __init__(self, log_path: str) -> None:
        super().__init__()
        self._log_path = pathlib.Path(log_path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        # Temporary store: run_id → partial record (populated on start)
        self._pending: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # LangChain callback hooks
    # ------------------------------------------------------------------

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Capture the prompt messages before the model is called."""
        # Serialise chat messages to plain strings for storage
        serialised_turns = [
            [f"{m.type}: {m.content}" for m in turn] for turn in messages
        ]
        self._pending[str(run_id)] = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "run_id": str(run_id),
            "model": serialized.get("name") or serialized.get("id", ["unknown"])[-1],
            "messages": serialised_turns,
        }

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Attach the response to the pending record and flush to disk."""
        record = self._pending.pop(str(run_id), {})
        if not record:
            # on_llm_end can fire without a matching on_chat_model_start
            # (e.g. for non-chat models); still log what we can.
            record = {
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "run_id": str(run_id),
                "model": "unknown",
                "messages": [],
            }

        # Extract the generated text from the first generation
        try:
            gen = response.generations[0][0]
            record["response"] = getattr(gen, "text", None) or getattr(
                gen.message, "content", str(gen)
            )
        except (IndexError, AttributeError):
            record["response"] = ""

        self._write(record)

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Log LLM errors so the pending record is not silently dropped."""
        record = self._pending.pop(str(run_id), {})
        record.setdefault("timestamp", datetime.datetime.now(datetime.timezone.utc).isoformat())
        record["error"] = str(error)
        self._write(record)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write(self, record: dict[str, Any]) -> None:
        """Append *record* as a single JSON line to the audit log."""
        try:
            with self._log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.error("PromptAuditCallback: failed to write audit log: %s", exc)
