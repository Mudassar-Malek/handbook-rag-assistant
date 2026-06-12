"""CLI entry point for Phase 1 ingestion.

Usage
-----
Parse + chunk only (no embeddings), dump chunks.json + a stats report:

    python -m ingestion.ingest <pdf_path> --dry-run

Full run (also embed + upsert into ChromaDB):

    python -m ingestion.ingest <pdf_path>
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from . import config
from .chunk import (
    Chunk,
    build_front_matter_chunks,
    build_policy_chunks,
    chunk_stats,
    inject_leave_table,
    parse_toc_section_numbers,
)
from .clean import clean_page
from .extract import extract_pages, extract_table_markdown


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def build_chunks(pdf_path: str, sort: bool) -> tuple[list[Chunk], dict[int, str]]:
    """Run extract -> clean -> chunk. Returns (chunks, cleaned_pages_by_number)."""
    raw_pages = extract_pages(pdf_path, sort=sort)
    cleaned = {p.number: clean_page(p.text) for p in raw_pages}
    source = Path(pdf_path).name

    # Tables that need find_tables() fallback.
    table_markdown = {
        config.REVISION_HISTORY_PAGE: extract_table_markdown(
            pdf_path, config.REVISION_HISTORY_PAGE
        ),
    }
    leave_table = extract_table_markdown(pdf_path, config.LEAVE_TABLE_PAGE)

    chunks: list[Chunk] = []
    chunks += build_front_matter_chunks(cleaned, source, table_markdown)
    chunks += build_policy_chunks(cleaned, source)
    inject_leave_table(chunks, leave_table)
    return chunks, cleaned


# ---------------------------------------------------------------------------
# Validation report (part of --dry-run)
# ---------------------------------------------------------------------------
def print_report(chunks: list[Chunk], cleaned: dict[int, str]) -> None:
    stats = chunk_stats(chunks)
    print("\n" + "=" * 70)
    print("INGESTION REPORT")
    print("=" * 70)
    print(f"Total chunks:        {stats['count']}")
    print(
        "Chunk size (tokens): "
        f"min={stats['min_tokens']}  median={stats['median_tokens']}  max={stats['max_tokens']}"
    )
    if stats["split_sections"]:
        print(f"Sections split:      {', '.join(stats['split_sections'])}")
    else:
        print("Sections split:      (none)")

    # Cross-check the TOC against produced chunks.
    toc_text = "\n".join(cleaned.get(p, "") for p in config.TOC_PAGES)
    toc_numbers = parse_toc_section_numbers(toc_text)
    produced = {c.section_number for c in chunks}
    missing = sorted(
        (n for n in toc_numbers if n not in produced),
        key=lambda s: tuple(int(x) for x in s.split(".")),
    )
    print(f"\nTOC subsections found: {len(toc_numbers)}")
    if missing:
        print(f"WARNING: {len(missing)} TOC section(s) produced no chunk:")
        print("         " + ", ".join(missing))
    else:
        print("All TOC subsections produced at least one chunk.")

    # Sanity checks on the must-survive tables.
    rev = next((c for c in chunks if c.section_number == "0.6"), None)
    if rev and "|" in rev.body:
        print("\nRevision History (p.9): table rendered as markdown. OK")
    else:
        print("\nWARNING: Revision History table may not have rendered as markdown.")

    leave = next(
        (c for c in chunks if c.section_number.startswith("5.") and 24 in c.pages),
        None,
    )
    if leave and "15 Days" in leave.body:
        print(f"Leave table (p.24): present in chunk {leave.section_number}. OK")
    else:
        print("WARNING: Leave entitlement table (p.24) not found in any section-5 chunk.")
    print("=" * 70 + "\n")


def write_chunks_json(chunks: list[Chunk], path: str) -> None:
    payload = [dataclasses.asdict(c) for c in chunks]
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"Wrote {len(chunks)} chunks to {path}")


# ---------------------------------------------------------------------------
# Embedding + storage
# ---------------------------------------------------------------------------
def embed_and_store(chunks: list[Chunk], args) -> None:
    from .embeddings import describe_provider, get_embedding_function
    from .store import get_collection, upsert_chunks

    print(f"Embedding with: {describe_provider(args.provider)}")
    ef = get_embedding_function(args.provider)
    collection = get_collection(args.persist_dir, args.collection, ef)
    n = upsert_chunks(collection, chunks)
    print(f"Upserted {n} chunks into collection '{args.collection}' at {args.persist_dir}")
    print(f"Collection now holds {collection.count()} documents.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ingestion.ingest",
        description="Phase 1 ingestion for the Employee Handbook 2026.",
    )
    parser.add_argument("pdf_path", help="Path to the handbook PDF.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse + chunk only; dump chunks.json and the stats report (no embeddings).",
    )
    parser.add_argument(
        "--out", default="chunks.json", help="Where to write chunks.json."
    )
    parser.add_argument(
        "--persist-dir",
        default=config.DEFAULT_PERSIST_DIR,
        help="ChromaDB persistence directory.",
    )
    parser.add_argument(
        "--collection",
        default=config.DEFAULT_COLLECTION,
        help="ChromaDB collection name.",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="Embedding provider override (local|openai). Defaults to .env / local.",
    )
    parser.add_argument(
        "--sort",
        action="store_true",
        help="Use get_text(sort=True) (legacy column-sorted order; interleaves columns).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not Path(args.pdf_path).is_file():
        print(f"ERROR: file not found: {args.pdf_path}", file=sys.stderr)
        return 2

    sort = args.sort or config.DEFAULT_SORT
    print(f"Extracting {args.pdf_path} (sort={sort}) ...")
    chunks, cleaned = build_chunks(args.pdf_path, sort=sort)

    write_chunks_json(chunks, args.out)
    print_report(chunks, cleaned)

    if args.dry_run:
        print("Dry run complete. No embeddings written.")
        return 0

    embed_and_store(chunks, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
