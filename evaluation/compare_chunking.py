"""Chunking experiment: section-aware (A) vs RecursiveCharacterTextSplitter (B).

Runs retrieval-only metrics on both strategies with the SAME retriever config,
and writes results/chunking_comparison.md with metrics side by side plus two
concrete examples showing the actual retrieved text (so you can SEE whether
naive chunking bleeds one leave type into another).
"""

from __future__ import annotations

import argparse
import os

from . import metrics as M
from .config import get_eval_settings
from .dataset import load_dataset
from .recursive_baseline import build_recursive_collection, collection_count

# Probe questions used ONLY for the qualitative side-by-side examples (these
# target the Sick-vs-Casual leave separation the experiment is meant to expose).
PROBE_QUESTIONS = [
    "How many sick leaves do I get?",
    "How many casual leaves do I get?",
]


def _retriever(collection_name=None):
    from retrieval.config import get_settings
    from retrieval.retriever import Retriever

    return Retriever(get_settings(collection_name=collection_name))


def _retrieve(retriever, question):
    results = retriever.retrieve(question)
    return results


def run_comparison(rebuild: bool = False) -> dict:
    settings = get_eval_settings()
    dataset = load_dataset(settings.dataset_path)

    if rebuild or collection_count(settings) == 0:
        build_recursive_collection(settings)

    ret_a = _retriever()  # section-aware (default collection)
    store_a = ret_a.store
    ret_b = _retriever(collection_name=settings.recursive_collection)

    a_section_rows, a_page_rows, b_page_rows = [], [], []
    per_q = []
    for item in dataset:
        gt_sections = item["ground_truth_sections"]
        gt_pages = M.sections_to_pages(store_a, gt_sections)

        res_a = _retrieve(ret_a, item["question"])
        a_sections = [r.section_number for r in res_a]
        a_pages = {p for r in res_a for p in r.pages}

        res_b = _retrieve(ret_b, item["question"])
        b_pages = {p for r in res_b for p in r.pages}

        a_section_rows.append((gt_sections, a_sections))
        a_page_rows.append((gt_pages, a_pages))
        b_page_rows.append((gt_pages, b_pages))
        per_q.append({
            "question": item["question"],
            "gt_sections": gt_sections,
            "gt_pages": sorted(gt_pages),
            "a_sections": a_sections,
            "a_pages": sorted(a_pages),
            "b_pages": sorted(b_pages),
            "a_section_hit": M.section_hit(gt_sections, a_sections),
            "a_page_hit": M.page_hit(gt_pages, a_pages, subset=True),
            "b_page_hit": M.page_hit(gt_pages, b_pages, subset=True),
        })

    aggregates = {
        "A_section_hit_rate": M.section_hit_rate(a_section_rows),
        "A_page_hit_rate_subset": M.page_hit_rate(a_page_rows, subset=True),
        "A_page_overlap_rate": M.page_hit_rate(a_page_rows, subset=False),
        "B_page_hit_rate_subset": M.page_hit_rate(b_page_rows, subset=True),
        "B_page_overlap_rate": M.page_hit_rate(b_page_rows, subset=False),
    }

    examples = _build_examples(ret_a, ret_b)
    return {"aggregates": aggregates, "per_question": per_q, "examples": examples}


def _build_examples(ret_a, ret_b) -> list[dict]:
    out = []
    for q in PROBE_QUESTIONS:
        a = _retrieve(ret_a, q)[:2]
        b = _retrieve(ret_b, q)[:2]
        out.append({
            "question": q,
            "a": [{"label": f"{r.section_number} {r.section_title}".strip(),
                   "pages": r.pages, "text": r.display_text[:500]} for r in a],
            "b": [{"label": f"pages {r.pages}", "pages": r.pages,
                   "text": r.display_text[:500]} for r in b],
        })
    return out


def write_markdown(result: dict, path: str) -> None:
    agg = result["aggregates"]
    lines = [
        "# Chunking comparison — section-aware (A) vs recursive baseline (B)",
        "",
        "Same retriever config (hybrid vector+BM25, RRF, cross-encoder rerank, top_k=5) "
        "over two collections. Strategy A = Phase 1 heading-aware chunks. Strategy B = "
        "RecursiveCharacterTextSplitter (1000 tokens, 150 overlap) with page metadata only.",
        "",
        "## Aggregate metrics",
        "",
        "| Metric | Strategy A (section-aware) | Strategy B (recursive) |",
        "|---|---|---|",
        f"| section_hit_rate (sections ⊆ retrieved) | {agg['A_section_hit_rate']:.3f} | n/a (no sections) |",
        f"| page_hit_rate (gt pages ⊆ retrieved) | {agg['A_page_hit_rate_subset']:.3f} | {agg['B_page_hit_rate_subset']:.3f} |",
        f"| page_overlap_rate (gt pages ∩ retrieved) | {agg['A_page_overlap_rate']:.3f} | {agg['B_page_overlap_rate']:.3f} |",
        "",
        "_`page_*` is the apples-to-apples comparison (B has no sections, so "
        "ground-truth sections are mapped to pages via Strategy A's metadata)._",
        "",
        "## Per-question (page-based)",
        "",
        "| Question | gt pages | A pages | A hit | B pages | B hit |",
        "|---|---|---|:--:|---|:--:|",
    ]
    for q in result["per_question"]:
        lines.append(
            f"| {q['question']} | {q['gt_pages']} | {q['a_pages']} | "
            f"{'✓' if q['a_page_hit'] else '✗'} | {q['b_pages']} | "
            f"{'✓' if q['b_page_hit'] else '✗'} |"
        )

    lines += ["", "## Example retrieved contexts (does B bleed sections together?)", ""]
    for ex in result["examples"]:
        lines.append(f"### Q: {ex['question']}")
        lines.append("")
        lines.append("**Strategy A (section-aware) top results:**")
        for r in ex["a"]:
            lines.append(f"- **[{r['label']}]** (p.{r['pages']})")
            lines.append(f"  > {r['text'].strip()}".replace("\n", " "))
        lines.append("")
        lines.append("**Strategy B (recursive) top results:**")
        for r in ex["b"]:
            lines.append(f"- **[{r['label']}]**")
            lines.append(f"  > {r['text'].strip()}".replace("\n", " "))
        lines.append("")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="evaluation.compare_chunking")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild Strategy B collection.")
    args = parser.parse_args(argv)

    settings = get_eval_settings()
    result = run_comparison(rebuild=args.rebuild)

    print("\nAGGREGATES")
    for k, v in result["aggregates"].items():
        print(f"  {k:26} {v:.3f}")

    path = os.path.join(settings.results_dir, "chunking_comparison.md")
    write_markdown(result, path)
    print(f"\nWrote {path}")

    a_sub = result["aggregates"]["A_page_hit_rate_subset"]
    b_sub = result["aggregates"]["B_page_hit_rate_subset"]
    verdict = ("Heading-aware (A) wins" if a_sub > b_sub
               else "Recursive (B) matches/wins" if b_sub >= a_sub else "tie")
    print(f"\nRead: page_hit_rate A={a_sub:.3f} vs B={b_sub:.3f} -> {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
