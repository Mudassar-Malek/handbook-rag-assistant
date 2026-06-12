"""Embedding-function factory.

The embedding function is kept behind a small factory so we can swap providers
via .env without touching the pipeline. The default is a fully local model
(no API key required) so the project runs out of the box; set EMBEDDING_PROVIDER
to "openai" later to use OpenAI embeddings.

All providers return objects that satisfy ChromaDB's EmbeddingFunction
interface (callable: list[str] -> list[list[float]]), so they plug straight
into a collection.
"""

from __future__ import annotations

import os

from chromadb.utils import embedding_functions

try:  # Loading .env is optional; the pipeline works without it.
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is a convenience only
    pass


def get_embedding_function(provider: str | None = None):
    """Return a ChromaDB-compatible embedding function.

    provider:
      * "local"  (default) -> all-MiniLM-L6-v2 ONNX model, downloaded once, runs
                              offline thereafter. No API key needed.
      * "openai"           -> OpenAI embeddings (needs OPENAI_API_KEY).
    """
    provider = (provider or os.getenv("EMBEDDING_PROVIDER", "local")).lower()

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "EMBEDDING_PROVIDER=openai but OPENAI_API_KEY is not set."
            )
        model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        return embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key, model_name=model
        )

    if provider == "local":
        model = os.getenv("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        return embedding_functions.DefaultEmbeddingFunction()  # all-MiniLM-L6-v2

    raise ValueError(f"Unknown EMBEDDING_PROVIDER: {provider!r}")


def describe_provider(provider: str | None = None) -> str:
    provider = (provider or os.getenv("EMBEDDING_PROVIDER", "local")).lower()
    if provider == "openai":
        return f"openai:{os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')}"
    return "local:all-MiniLM-L6-v2"
