"""CLI for manual retrieval testing.

    python -m retrieval.search "how many earned leaves do I get?" --k 5 --debug

--debug prints a score table (vector / bm25 / fused / rerank) and the origin of
each result so you can SEE why each chunk was retrieved.
"""

from __future__ import annotations

import argparse
import textwrap

from .config import get_settings
from .retriever import Retriever
from .types import RetrievalResult


def _fmt(score: float | None) -> str:
    return f"{score:.4f}" if score is not None else "  -  "


def print_results(query: str, results: list[RetrievalResult]) -> None:
    print(f'\nQuery: "{query}"\n')
    for i, r in enumerate(results, start=1):
        tag = "" if r.origin == "hybrid" else f"  ({r.origin})"
        part = f" [part {r.part}/{r.total_parts}]" if r.total_parts > 1 else ""
        print(f"{i}. {r.citation}{part}{tag}")
        snippet = textwrap.shorten(r.display_text.replace("\n", " "), width=220)
        print(f"   {snippet}")
        if r.sibling_sections:
            related = ", ".join(f"{n} {t}" for n, t in r.sibling_sections[:6])
            print(f"   related: {related}")
        print()


def print_debug_table(results: list[RetrievalResult]) -> None:
    print("=" * 104)
    print("DEBUG: why each chunk was retrieved")
    print("=" * 104)
    header = f"{'#':>2}  {'section':<8} {'title':<30} {'vector':>8} {'bm25':>8} {'fused':>8} {'rerank':>8}  origin"
    print(header)
    print("-" * 104)
    for i, r in enumerate(results, start=1):
        title = textwrap.shorten(r.section_title, width=30)
        print(
            f"{i:>2}  {r.section_number:<8} {title:<30} "
            f"{_fmt(r.scores.vector):>8} {_fmt(r.scores.bm25):>8} "
            f"{_fmt(r.scores.fused):>8} {_fmt(r.scores.rerank):>8}  {r.origin}"
        )
    print("=" * 104)
    print(
        "vector = cosine similarity (1 - distance) | bm25 = lexical score | "
        "fused = RRF score | rerank = cross-encoder logit"
    )
    print()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="retrieval.search",
        description="Hybrid retrieval over the Employee Handbook vector store.",
    )
    parser.add_argument("query", help="The search query.")
    parser.add_argument("--k", type=int, default=None, help="Number of results (top_k).")
    parser.add_argument("--debug", action="store_true", help="Print the score table.")
    parser.add_argument(
        "--no-rerank", action="store_true", help="Disable the cross-encoder reranker."
    )
    parser.add_argument(
        "--no-expansion", action="store_true", help="Disable cross-reference expansion."
    )
    parser.add_argument(
        "--no-stitching", action="store_true", help="Disable multi-part stitching."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    overrides = {"top_k": args.k}
    if args.no_rerank:
        overrides["use_reranker"] = False
    if args.no_expansion:
        overrides["use_expansion"] = False
    if args.no_stitching:
        overrides["use_stitching"] = False

    settings = get_settings(**overrides)
    retriever = Retriever(settings)
    results = retriever.retrieve(args.query)

    print_results(args.query, results)
    if args.debug:
        print_debug_table(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
