"""The Retriever: composes vector + BM25 + RRF + reranker + post-processing.

Pipeline:
    1. Vector search (top vector_n) and BM25 search (top bm25_n) in parallel.
    2. Reciprocal Rank Fusion merges the two ranked lists.
    3. Rerank the fused top fetch_k with a cross-encoder (optional); keep top_k.
    4. Post-process: cross-reference expansion, multi-part stitching, siblings.
"""

from __future__ import annotations

import contextlib

from .base import Hit
from .bm25 import BM25Search

# Optional: trace the BM25 index build if the observability module is present.
# Guarded so the retrieval layer still works standalone (Phases 1-2).
try:
    from observability import span as _obs_span
except Exception:  # pragma: no cover - module absent or import error
    @contextlib.contextmanager
    def _obs_span(*_a, **_k):
        yield None
from .config import Settings
from .fusion import reciprocal_rank_fusion
from .loader import ChunkStore
from .postprocess import (
    attach_sibling_hints,
    expand_cross_references,
    stitch_multiparts,
)
from .rerank import CrossEncoderReranker
from .types import RetrievalResult, Scores
from .vector import VectorSearch


class Retriever:
    """High-level hybrid retriever. Construct once, call `retrieve()` per query."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.collection = _open_collection(settings)
        self.store = ChunkStore(self.collection)
        self.vector = VectorSearch(self.collection)
        # BM25 index build is a one-time startup cost worth seeing in a trace.
        with _obs_span("bm25_index_load", metadata={"n_chunks": len(self.store.chunks)}):
            self.bm25 = BM25Search(self.store)
        self.reranker = (
            CrossEncoderReranker(settings.reranker_model)
            if settings.use_reranker
            else None
        )

    def retrieve(self, query: str) -> list[RetrievalResult]:
        s = self.settings

        # 1. Two independent ranked lists.
        vector_hits = self.vector.search(query, s.vector_n)
        bm25_hits = self.bm25.search(query, s.bm25_n)
        vector_scores = dict(vector_hits)
        bm25_scores = dict(bm25_hits)

        # 2. Fuse with RRF.
        fused = reciprocal_rank_fusion([vector_hits, bm25_hits], k=s.rrf_k)
        fused_ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
        candidate_ids = [cid for cid, _ in fused_ranked[: s.fetch_k]]

        # 3. Rerank (or fall back to fused order), keep top_k.
        rerank_scores: dict[str, float] = {}
        if self.reranker is not None:
            candidates = [self.store.get(cid) for cid in candidate_ids]
            candidates = [c for c in candidates if c is not None]
            reranked = self.reranker.rerank(query, candidates)
            rerank_scores = dict(reranked)
            ordered_ids = [cid for cid, _ in reranked][: s.top_k]
        else:
            ordered_ids = candidate_ids[: s.top_k]

        # 4. Build primary results with the full score breakdown.
        results: list[RetrievalResult] = []
        for cid in ordered_ids:
            chunk = self.store.get(cid)
            if chunk is None:
                continue
            results.append(
                RetrievalResult(
                    id=chunk.id,
                    section_number=chunk.section_number,
                    section_title=chunk.section_title,
                    parent_section=chunk.parent_section,
                    pages=chunk.pages,
                    display_text=chunk.body,
                    part=chunk.part,
                    total_parts=chunk.total_parts,
                    origin="hybrid",
                    scores=Scores(
                        vector=vector_scores.get(cid),
                        bm25=bm25_scores.get(cid),
                        fused=fused.get(cid),
                        rerank=rerank_scores.get(cid),
                    ),
                )
            )

        # 5. Document-specific post-processing (after reranking).
        if self.settings.use_stitching:
            results = stitch_multiparts(results, self.store)
        if self.settings.use_expansion:
            results = expand_cross_references(results, self.store)
        attach_sibling_hints(results, self.store)
        return results


def _open_collection(settings: Settings):
    import chromadb

    # Reuse Phase 1's embedding-function factory so queries are embedded with
    # the exact same model the documents were embedded with.
    from ingestion.embeddings import get_embedding_function

    client = chromadb.PersistentClient(path=settings.chroma_path)
    return client.get_collection(
        name=settings.collection_name,
        embedding_function=get_embedding_function(settings.embedding_provider),
    )
