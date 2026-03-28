#!/usr/bin/env python
"""
inspect_checkpoints.py – CLI tool for browsing FlexRAG execution traces.

Reads the SQLite checkpoint database written by the LangGraph SqliteSaver
and prints human-readable execution traces.

Usage
-----
List all stored thread IDs::

    python scripts/inspect_checkpoints.py --db ./data/checkpoints.db --list

Inspect a specific run by thread ID::

    python scripts/inspect_checkpoints.py --db ./data/checkpoints.db \\
        --thread <thread-id>

Show the final state (answer + evidence only) for a run::

    python scripts/inspect_checkpoints.py --db ./data/checkpoints.db \\
        --thread <thread-id> --final-only

Show the last N runs (most recently stored threads)::

    python scripts/inspect_checkpoints.py --db ./data/checkpoints.db \\
        --last 5
"""

from __future__ import annotations

import argparse
import json
import sys

from flexrag.tracing import CheckpointReader


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="inspect_checkpoints",
        description="Browse FlexRAG LangGraph execution traces stored in SQLite.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--db",
        required=True,
        metavar="PATH",
        help="Path to the SQLite checkpoint database (e.g. ./data/checkpoints.db)",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--list",
        action="store_true",
        help="List all thread IDs stored in the database",
    )
    group.add_argument(
        "--thread",
        metavar="THREAD_ID",
        help="Print the full execution trace for THREAD_ID",
    )
    group.add_argument(
        "--last",
        metavar="N",
        type=int,
        help="Print execution traces for the last N threads",
    )

    parser.add_argument(
        "--final-only",
        action="store_true",
        help="(Used with --thread) Print only the final answer and evidence",
    )
    parser.add_argument(
        "--json",
        dest="output_json",
        action="store_true",
        help="(Used with --thread --final-only) Output final state as JSON",
    )
    return parser


def cmd_list(reader: CheckpointReader) -> None:
    threads = reader.list_threads()
    if not threads:
        print("No threads found in the database.")
        return
    print(f"Found {len(threads)} thread(s):\n")
    for tid in threads:
        print(f"  {tid}")


def cmd_thread(
    reader: CheckpointReader,
    thread_id: str,
    final_only: bool,
    output_json: bool,
) -> None:
    if final_only:
        state = reader.get_final_state(thread_id)
        if not state:
            print(f"No checkpoints found for thread_id={thread_id!r}", file=sys.stderr)
            sys.exit(1)
        if output_json:
            print(json.dumps(state, ensure_ascii=False, default=str, indent=2))
        else:
            print(f"Query   : {state.get('original_query', '')!r}")
            print(f"Answer  : {state.get('answer', '')}")
            evidence = state.get("evidence", [])
            if evidence:
                print("Evidence:")
                for i, ev in enumerate(evidence, 1):
                    print(f"  [{i}] {str(ev)[:200]}")
    else:
        reader.print_run(thread_id)


def cmd_last(reader: CheckpointReader, n: int) -> None:
    threads = reader.list_threads()
    if not threads:
        print("No threads found in the database.")
        return
    # Threads are sorted alphabetically (UUIDs sort by creation time approx.)
    for tid in threads[-n:]:
        reader.print_run(tid)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    try:
        reader = CheckpointReader(args.db)
    except Exception as exc:
        print(f"[ERROR] Could not open database {args.db!r}: {exc}", file=sys.stderr)
        sys.exit(1)

    with reader:
        if args.list:
            cmd_list(reader)
        elif args.thread:
            cmd_thread(reader, args.thread, args.final_only, args.output_json)
        elif args.last is not None:
            cmd_last(reader, args.last)


if __name__ == "__main__":
    main()
