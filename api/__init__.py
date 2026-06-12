"""Phase 6 — FastAPI backend.

Exposes the Phase 3 agent over HTTP: a streaming `/ask` (SSE), `/feedback`
(Langfuse user_feedback scores), `/health`, `/sections`, and a `/threads`
introspection endpoint. Heavy resources (retriever, BM25, cross-encoder,
compiled graph) load ONCE at startup via the FastAPI lifespan.
"""

__version__ = "0.1.0"
