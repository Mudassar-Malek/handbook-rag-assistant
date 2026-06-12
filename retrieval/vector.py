"""Dense vector search over the ChromaDB collection (cosine similarity)."""

from __future__ import annotations

from .base import Hit


class VectorSearch:
    """Embeds the query and finds the nearest chunks by cosine similarity.

    Chroma returns a cosine *distance* (0 = identical, 2 = opposite); we convert
    it to a similarity (1 - distance) so higher always means more relevant,
    matching the BM25 convention.
    """

    name = "vector"

    def __init__(self, collection):
        self.collection = collection

    def search(self, query: str, n: int) -> list[Hit]:
        result = self.collection.query(
            query_texts=[query],
            n_results=n,
            include=["distances"],
        )
        ids = result["ids"][0]
        distances = result["distances"][0]
        return [(cid, 1.0 - dist) for cid, dist in zip(ids, distances)]
