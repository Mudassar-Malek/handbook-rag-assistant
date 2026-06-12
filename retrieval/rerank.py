"""Cross-encoder reranking.

The fusion step is cheap but approximate. A cross-encoder reads the query and a
candidate together (not as separate vectors) and scores their relevance
directly, which is far more accurate — at the cost of one model call per
candidate. So we only rerank the fused top-N, then keep the best top_k.
"""

from __future__ import annotations

from .base import Hit
from .types import Chunk


class CrossEncoderReranker:
    """Wraps a sentence-transformers CrossEncoder; loaded lazily on first use."""

    name = "rerank"

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None  # lazy: avoids importing torch unless reranking is on

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, candidates: list[Chunk]) -> list[Hit]:
        """Score (query, title+body) for each candidate; return sorted hits."""
        if not candidates:
            return []
        model = self._ensure_model()
        pairs = [(query, c.rerank_text) for c in candidates]
        scores = model.predict(pairs)
        ranked = sorted(
            zip((c.id for c in candidates), scores),
            key=lambda kv: kv[1],
            reverse=True,
        )
        return [(cid, float(score)) for cid, score in ranked]
