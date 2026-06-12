"""Reciprocal Rank Fusion (RRF) for merging ranked lists.

RRF combines lists using only the *rank* of each item, not its raw score, which
neatly sidesteps the fact that cosine similarities and BM25 scores live on
totally different scales. Each list contributes 1 / (k + rank) for every item;
contributions are summed across lists. The constant k (default 60) dampens the
influence of top ranks so a single list can't dominate.
"""

from __future__ import annotations

from .base import Hit


def reciprocal_rank_fusion(ranked_lists: list[list[Hit]], k: int = 60) -> dict[str, float]:
    """Return {chunk_id: fused_score}, summing 1/(k+rank) over all lists.

    `rank` is 1-based (the top hit in each list has rank 1).
    """
    fused: dict[str, float] = {}
    for hits in ranked_lists:
        for rank, (chunk_id, _score) in enumerate(hits, start=1):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return fused
