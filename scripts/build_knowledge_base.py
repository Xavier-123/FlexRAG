#!/usr/bin/env python
"""
Build a FAISS vector knowledge base from local documents.

This standalone script loads documents from a directory (or individual files),
chunks them, embeds them via a remote embedding endpoint, and persists the
resulting FAISS index to disk.

The persisted index can later be loaded by
:class:`~flexrag.retrievers.LlamaIndexRetriever` for retrieval.

Configuration
-------------
All settings are read from environment variables (or a ``.env`` file).
The most important ones for this script are:

    EMBEDDING_BASE_URL   – Base URL of the embedding model endpoint
    EMBEDDING_API_KEY    – API key for the embedding endpoint
    VLLM_EMBEDDING_MODEL – Name of the embedding model

    KNOWLEDGE_PERSIST_DIR   – Where to save the index  (default: ./knowledge_base)
    KNOWLEDGE_CHUNK_SIZE    – Tokens per chunk           (default: 512)
    KNOWLEDGE_CHUNK_OVERLAP – Token overlap between chunks (default: 50)

Usage
-----
::

    # Build from a directory of .txt / .md / .pdf files
    python scripts/build_knowledge_base.py --input-dir ./my_docs

    # Build from specific files
    python scripts/build_knowledge_base.py --files doc1.txt doc2.pdf

    # Override the output directory and chunk settings
    python scripts/build_knowledge_base.py --input-dir ./my_docs \\
        --output-dir ./my_index --chunk-size 256 --chunk-overlap 32

    # Force overwrite an existing index
    python scripts/build_knowledge_base.py --input-dir ./my_docs --force
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time

# Ensure the project root is on sys.path so ``flexrag`` can be imported when
# running the script directly (e.g. ``python scripts/build_knowledge_base.py``).
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from flexrag.config import Settings
from flexrag.knowledge import FaissKnowledgeBuilder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build a FAISS vector knowledge base from local documents.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--input-dir",
        type=str,
        help="Path to a directory of documents (.txt, .md, .pdf).",
    )
    source.add_argument(
        "--files",
        nargs="+",
        type=str,
        help="One or more individual file paths to index.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Directory where the FAISS index will be saved. "
            "Defaults to the KNOWLEDGE_PERSIST_DIR setting."
        ),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Max tokens per chunk (default: from KNOWLEDGE_CHUNK_SIZE setting).",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help="Token overlap between chunks (default: from KNOWLEDGE_CHUNK_OVERLAP setting).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing index at the output directory.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


async def build(args: argparse.Namespace) -> None:
    """Execute the build pipeline according to parsed CLI *args*."""
    settings = Settings()

    output_dir = args.output_dir or settings.knowledge_persist_dir
    chunk_size = args.chunk_size if args.chunk_size is not None else settings.knowledge_chunk_size
    chunk_overlap = args.chunk_overlap if args.chunk_overlap is not None else settings.knowledge_chunk_overlap

    # ---- safety check ----
    if FaissKnowledgeBuilder.index_exists(output_dir) and not args.force:
        print(
            f"[ERROR] An index already exists at '{output_dir}'.\n"
            "       Use --force to overwrite it.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ---- builder ----
    builder = FaissKnowledgeBuilder(
        embed_base_url=settings.embedding_base_url,
        embed_model_name=settings.vllm_embedding_model,
        embed_api_key=settings.embedding_api_key,
    )

    # ---- load ----
    source = args.input_dir if args.input_dir else args.files
    t0 = time.perf_counter()
    doc_count = await builder.load_files(source)
    elapsed_load = time.perf_counter() - t0
    print(f"[INFO] Loaded {doc_count} document(s) in {elapsed_load:.1f}s.")

    # ---- build ----
    print(
        f"[INFO] Building index (chunk_size={chunk_size}, "
        f"chunk_overlap={chunk_overlap}) ..."
    )
    t1 = time.perf_counter()
    await builder.build_index(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elapsed_build = time.perf_counter() - t1
    print(f"[INFO] Index built in {elapsed_build:.1f}s.")

    # ---- save ----
    await builder.save(output_dir)
    print(f"[INFO] Knowledge base saved to '{output_dir}'.")
    print(f"[INFO] Total time: {time.perf_counter() - t0:.1f}s.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """Parse arguments, configure logging, and run the async build pipeline."""
    args = _parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    asyncio.run(build(args))


if __name__ == "__main__":
    main()
