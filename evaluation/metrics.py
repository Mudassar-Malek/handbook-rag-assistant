"""Custom, LLM-free retrieval metrics.

`section_hit_rate` is the most interpretable number in the whole project: it
ignores LLM judgment entirely and just asks "did the retriever surface the
section(s) that actually contain the answer?".
"""

from __future__ import annotations


def section_hit(gt_sections: list[str], retrieved_sections: list[str]) -> bool:
    """True iff every ground-truth section was retrieved (gt ⊆ retrieved)."""
    return set(gt_sections) <= set(retrieved_sections)


def section_hit_rate(rows: list[tuple[list[str], list[str]]]) -> float:
    """Fraction of questions where ground_truth_sections ⊆ retrieved_sections."""
    if not rows:
        return 0.0
    return sum(section_hit(gt, got) for gt, got in rows) / len(rows)


def page_hit(gt_pages: set[int], retrieved_pages: set[int], subset: bool = True) -> bool:
    """Page-based hit for chunking strategies without section metadata.

    subset=True  -> all ground-truth pages were retrieved (gt ⊆ retrieved)
    subset=False -> at least one ground-truth page overlaps (gt ∩ retrieved ≠ ∅)
    """
    if not gt_pages:
        return False
    return gt_pages <= retrieved_pages if subset else bool(gt_pages & retrieved_pages)


def page_hit_rate(rows: list[tuple[set[int], set[int]]], subset: bool = True) -> float:
    if not rows:
        return 0.0
    return sum(page_hit(gt, got, subset=subset) for gt, got in rows) / len(rows)


def sections_to_pages(store, sections: list[str]) -> set[int]:
    """Map section_numbers to the set of pages they appear on (via Strategy A)."""
    pages: set[int] = set()
    for sec in sections:
        for chunk in store.get_section(sec):
            pages.update(chunk.pages)
    return pages


def worst_questions(per_question: list[dict], metric: str, n: int = 3) -> list[dict]:
    """Return the n lowest-scoring questions for a metric (skips missing values)."""
    scored = [q for q in per_question if isinstance(q.get(metric), (int, float))]
    scored.sort(key=lambda q: q[metric])
    return scored[:n]


def mean(values: list[float]) -> float:
    vals = [v for v in values if isinstance(v, (int, float))]
    return sum(vals) / len(vals) if vals else 0.0
