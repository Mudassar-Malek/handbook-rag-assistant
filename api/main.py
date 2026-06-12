"""FastAPI app: lifespan-loaded agent + streaming SSE chat.

Run locally:
    uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from sse_starlette.sse import EventSourceResponse

import observability as obs
from agents.config import get_agent_settings
from agents.graph import HandbookAgent

from .schemas import AskRequest, FeedbackRequest

log = logging.getLogger("api")
logging.basicConfig(level=logging.INFO)

limiter = Limiter(key_func=get_remote_address)


def _effective_settings():
    """Use the configured provider, but fall back to the offline 'fake' model
    if the chosen provider needs a key that isn't set — so the API always boots."""
    s = get_agent_settings()
    provider = s.llm_provider.lower()
    has_key = (
        (provider == "anthropic" and os.getenv("ANTHROPIC_API_KEY"))
        or (provider == "openai" and os.getenv("OPENAI_API_KEY"))
    )
    if provider in ("anthropic", "openai") and not has_key:
        log.warning("No %s API key set — API running with the offline 'fake' provider.", provider)
        return get_agent_settings(llm_provider="fake")
    return s


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all heavy resources exactly once."""
    obs.init_observability()
    settings = _effective_settings()

    from retrieval.config import get_settings as get_retrieval_settings
    from retrieval.retriever import Retriever

    log.info("Loading retriever (vector + BM25 + cross-encoder)…")
    retriever = Retriever(get_retrieval_settings(top_k=settings.retrieval_k))

    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    saver_cm = AsyncSqliteSaver.from_conn_string(settings.checkpoint_db)
    saver = await saver_cm.__aenter__()

    agent = HandbookAgent(settings, retriever=retriever, checkpointer=saver)

    app.state.settings = settings
    app.state.retriever = retriever
    app.state.agent = agent
    app.state._saver_cm = saver_cm
    log.info("Startup complete (provider=%s, langfuse=%s).",
             settings.llm_provider, "on" if obs.is_enabled() else "off")
    try:
        yield
    finally:
        await saver_cm.__aexit__(None, None, None)


app = FastAPI(title="Employee Handbook Assistant API", version="0.1.0", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, lambda r, e: JSONResponse(
    {"type": "error", "message": "rate limit exceeded (10/min on /ask)"}, status_code=429))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # open for the local Streamlit UI
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- ask
@app.post("/ask")
@limiter.limit("10/minute")
async def ask(request: Request, body: AskRequest):
    agent: HandbookAgent = request.app.state.agent
    thread_id = body.thread_id or uuid.uuid4().hex

    async def event_source():
        # Tell the client its thread id up front so follow-ups can continue.
        yield {"data": json.dumps({"type": "status", "node": "start", "detail": thread_id})}
        try:
            async for ev in agent.astream_turn(body.question, thread_id, body.user_id):
                yield {"data": json.dumps(ev)}
        except Exception as e:  # never drop the connection — emit + close cleanly
            log.exception("stream error")
            yield {"data": json.dumps({"type": "error", "message": str(e)})}

    return EventSourceResponse(event_source())


# ----------------------------------------------------------------------- feedback
@app.post("/feedback")
async def feedback(body: FeedbackRequest):
    obs.create_score(body.trace_id, "user_feedback", float(body.value),
                     comment="thumbs_up" if body.value == 1 else "thumbs_down")
    obs.flush()
    return {"ok": True, "langfuse_enabled": obs.is_enabled()}


# ------------------------------------------------------------------------- health
@app.get("/health")
async def health(request: Request):
    retriever = getattr(request.app.state, "retriever", None)
    settings = getattr(request.app.state, "settings", None)
    components: dict = {}
    healthy = True

    # ChromaDB reachable + collection non-empty (critical).
    try:
        count = retriever.collection.count()
        up = count > 0
        components["chromadb"] = {"status": "up" if up else "empty", "count": count, "critical": True}
        healthy = healthy and up
    except Exception as e:
        components["chromadb"] = {"status": "down", "error": str(e), "critical": True}
        healthy = False

    # BM25 index loaded (critical).
    try:
        n = len(retriever.store.chunks)
        components["bm25"] = {"status": "up" if n > 0 else "down", "n_chunks": n, "critical": True}
        healthy = healthy and n > 0
    except Exception as e:
        components["bm25"] = {"status": "down", "error": str(e), "critical": True}
        healthy = False

    # LLM availability (non-critical: 'fake' still serves answers).
    provider = settings.llm_provider if settings else "unknown"
    key_ok = provider in ("fake", "ollama") or (
        (provider == "anthropic" and os.getenv("ANTHROPIC_API_KEY"))
        or (provider == "openai" and os.getenv("OPENAI_API_KEY"))
    )
    components["llm"] = {"status": "up" if key_ok else "no_key", "provider": provider, "critical": False}
    components["langfuse"] = {"status": "up" if obs.is_enabled() else "disabled", "critical": False}

    return JSONResponse(
        {"status": "ok" if healthy else "unhealthy", "components": components},
        status_code=200 if healthy else 503,
    )


# ----------------------------------------------------------------------- sections
@app.get("/sections")
async def sections(request: Request):
    from retrieval.loader import _num_key

    store = request.app.state.retriever.store
    seen: dict[str, str] = {}
    for c in store.chunks:
        seen.setdefault(c.section_number, c.section_title)
    items = [{"section_number": sn, "section_title": st} for sn, st in seen.items()]
    items.sort(key=lambda x: _num_key(x["section_number"]))
    return {"sections": items}


# ------------------------------------------------------------- thread introspection
@app.get("/threads/{thread_id}")
async def thread_state(request: Request, thread_id: str):
    """Expose checkpointer state for a thread (proves memory + powers tests)."""
    agent: HandbookAgent = request.app.state.agent
    config = {"configurable": {"thread_id": thread_id}}
    snap = await agent.graph.aget_state(config)
    messages = (snap.values.get("messages", []) if snap and snap.values else [])
    history = [h async for h in agent.graph.aget_state_history(config)]
    return {
        "thread_id": thread_id,
        "message_count": len(messages),
        "history_length": len(history),
    }
