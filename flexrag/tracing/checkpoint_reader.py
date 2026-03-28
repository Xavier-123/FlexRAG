"""
Helper for reading persisted LangGraph checkpoints from the SQLite database
written by :class:`~langgraph.checkpoint.sqlite.SqliteSaver`.

Usage::

    from flexrag.tracing import CheckpointReader

    reader = CheckpointReader("./data/checkpoints.db")

    # Print a human-readable summary of one run
    reader.print_run("your-thread-id-here")

    # Or iterate over all snapshots programmatically
    for snap in reader.get_history("your-thread-id-here"):
        print(snap.metadata, snap.values.get("node_trace"))
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any, Iterator

import logging

logger = logging.getLogger(__name__)


class _Snapshot:
    """Lightweight wrapper around a raw checkpoint row."""

    def __init__(self, row: sqlite3.Row) -> None:
        self._row = row
        # The 'checkpoint' column contains JSON-serialised state values
        raw = row["checkpoint"]
        try:
            data: dict[str, Any] = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
        except (json.JSONDecodeError, TypeError):
            data = {}
        # LangGraph stores state values under the "channel_values" key
        self.values: dict[str, Any] = data.get("channel_values", data)
        # The 'metadata' column contains source / writes information
        raw_meta = row["metadata"]
        try:
            self.metadata: dict[str, Any] = (
                json.loads(raw_meta) if isinstance(raw_meta, (str, bytes)) else raw_meta
            )
        except (json.JSONDecodeError, TypeError):
            self.metadata = {}

    @property
    def checkpoint_id(self) -> str:
        return str(self._row["checkpoint_id"])

    @property
    def step(self) -> int:
        """Execution step index (monotonically increasing)."""
        return int(self.metadata.get("step", -1))


class CheckpointReader:
    """Read-only view of a FlexRAG LangGraph checkpoint database.

    Opens the SQLite file at *db_path* and provides helpers to list thread
    IDs, retrieve state histories, and pretty-print execution traces.

    Args:
        db_path: Path to the SQLite file produced by
            :class:`~langgraph.checkpoint.sqlite.SqliteSaver`.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row

    def close(self) -> None:
        """Close the underlying database connection."""
        self._conn.close()

    def __enter__(self) -> "CheckpointReader":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_threads(self) -> list[str]:
        """Return all thread IDs stored in the database.

        Returns:
            Sorted list of thread ID strings.
        """
        cur = self._conn.execute(
            "SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id"
        )
        return [row[0] for row in cur.fetchall()]

    def get_history(self, thread_id: str) -> list[_Snapshot]:
        """Return all checkpoints for *thread_id* ordered chronologically.

        Each checkpoint corresponds to the state of the graph immediately
        *after* one node has finished executing.

        Args:
            thread_id: The thread identifier used when calling
                ``graph.ainvoke(..., config={"configurable": {"thread_id": ...}})``.

        Returns:
            List of :class:`_Snapshot` objects, oldest first.
        """
        cur = self._conn.execute(
            """
            SELECT checkpoint_id, checkpoint, metadata
            FROM checkpoints
            WHERE thread_id = ?
            ORDER BY checkpoint_id ASC
            """,
            (thread_id,),
        )
        rows = cur.fetchall()
        return [_Snapshot(row) for row in rows]

    def get_final_state(self, thread_id: str) -> dict[str, Any]:
        """Return the state values from the last checkpoint of *thread_id*.

        Args:
            thread_id: The thread identifier.

        Returns:
            The ``values`` dict of the final graph state, or ``{}`` if no
            checkpoints are found for the given thread.
        """
        history = self.get_history(thread_id)
        return history[-1].values if history else {}

    def print_run(self, thread_id: str) -> None:
        """Pretty-print a complete execution trace for *thread_id*.

        Prints:
        - Per-node trace entries recorded in ``node_trace``
        - The final answer and evidence
        - A summary of how many checkpoints were saved

        Args:
            thread_id: The thread identifier to inspect.
        """
        history = self.get_history(thread_id)
        if not history:
            print(f"[CheckpointReader] No checkpoints found for thread_id={thread_id!r}")
            return

        print(f"\n{'=' * 60}")
        print(f"Execution trace for thread_id: {thread_id}")
        print(f"Total checkpoints: {len(history)}")
        print("=" * 60)

        # Collect all node_trace entries across checkpoints
        seen_traces: list[dict[str, Any]] = []
        for snap in history:
            for entry in snap.values.get("node_trace", []):
                if entry not in seen_traces:
                    seen_traces.append(entry)

        if seen_traces:
            print("\n--- Node execution trace ---")
            for i, entry in enumerate(seen_traces, 1):
                node = entry.get("node", "unknown")
                details = {k: v for k, v in entry.items() if k != "node"}
                details_str = "  ".join(f"{k}={v!r}" for k, v in details.items())
                print(f"  Step {i:>2}: [{node}]  {details_str}")

        # Final state summary
        final = history[-1].values
        print("\n--- Final state ---")
        print(f"  Query          : {final.get('original_query', '')!r}")
        print(f"  Iterations     : {final.get('iteration_count', 0)}")
        print(f"  Answer         : {str(final.get('answer', ''))[:200]}")
        evidence = final.get("evidence", [])
        if evidence:
            print(f"  Evidence ({len(evidence)})  :")
            for j, ev in enumerate(evidence, 1):
                print(f"    [{j}] {str(ev)[:120]}")
        if final.get("error"):
            print(f"  Error          : {final['error']}")
        print("=" * 60)
