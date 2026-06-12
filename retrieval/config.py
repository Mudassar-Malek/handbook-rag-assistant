"""Retrieval settings, loaded from environment / .env via pydantic-settings.

Every knob has a sensible default so the retriever runs with zero config.
Override any of them in `.env` (see README for the full list), e.g.:

    TOP_K=8
    USE_RERANKER=false
    USE_EXPANSION=true
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", case_sensitive=False, extra="ignore"
    )

    # --- Where the Phase 1 vector store lives ---
    chroma_path: str = "chroma_db"
    collection_name: str = "employee_handbook_2026"
    # Reuse the SAME embedding model Phase 1 used (shared EMBEDDING_PROVIDER var).
    embedding_provider: str = "local"

    # --- Candidate pool sizes ---
    vector_n: int = 20          # vector search top-N
    bm25_n: int = 20            # BM25 search top-N
    fetch_k: int = 20           # fused candidates fed to the reranker
    top_k: int = 5              # final results returned to the caller

    # --- Fusion ---
    rrf_k: int = 60             # Reciprocal Rank Fusion constant

    # --- Reranker (cross-encoder) ---
    use_reranker: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- Document-specific post-processing ---
    use_expansion: bool = True  # cross-reference expansion ("see Section 5.11")
    use_stitching: bool = True  # fetch sibling parts of split sections


def get_settings(**overrides) -> Settings:
    """Build Settings, applying any explicit overrides (e.g. from CLI flags)."""
    return Settings(**{k: v for k, v in overrides.items() if v is not None})
