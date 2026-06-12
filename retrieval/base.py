"""Common interface for the individual searchers (vector, BM25)."""

from __future__ import annotations

from typing import Protocol


# A single ranked hit: (chunk_id, score). Higher score = more relevant.
Hit = tuple[str, float]


class Searcher(Protocol):
    """Anything that can rank chunk ids for a query.

    Both VectorSearch and BM25Search implement this, which is what lets the
    Retriever treat them interchangeably and fuse their outputs.
    """

    name: str

    def search(self, query: str, n: int) -> list[Hit]:
        """Return the top-`n` (chunk_id, score) hits, highest score first."""
        ...
