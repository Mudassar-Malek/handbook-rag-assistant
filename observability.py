"""Langfuse observability — a thin, fail-safe wrapper used across the project.

Design goals:
- **Graceful degradation:** if LANGFUSE_* env vars are missing (or the SDK
  errors), everything here becomes a no-op. Tracing must NEVER crash or block
  the app.
- **One import for both layers:** `agents/` and `retrieval/` both import this
  module so spans nest into the same per-turn trace (Langfuse v3 propagates the
  active span via OpenTelemetry context vars).
- **No per-LLM hand-instrumentation:** the LangChain CallbackHandler
  (`get_callback_handler()`) is passed through the graph's invoke config and
  turns every node / LLM / tool call into a nested observation automatically.

Langfuse SDK: v3 (OTEL-based).
"""

from __future__ import annotations

import contextlib
import logging
import os

log = logging.getLogger("observability")

# Name of the answer system prompt managed in Langfuse Prompt Management.
ANSWER_PROMPT_NAME = "handbook-answer-v1"
# How long the Langfuse client caches a fetched prompt before refreshing in the
# background. Lets you edit the prompt in the UI and see it without redeploying,
# while never doing a network fetch on the hot path.
PROMPT_CACHE_TTL_SECONDS = 60
DEFAULT_HOST = "https://cloud.langfuse.com"

_state: dict = {"enabled": False, "client": None, "init_done": False}


class _NoopSpan:
    """Stand-in span when tracing is disabled; every method is a no-op."""

    def update(self, *a, **k):
        return self

    def update_trace(self, *a, **k):
        return self

    def score_trace(self, *a, **k):
        return self

    def end(self, *a, **k):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def init_observability() -> object | None:
    """Initialize the Langfuse client once. Returns the client or None.

    Reads keys from the environment (loading .env first). Missing keys -> a
    single warning and tracing stays disabled.
    """
    if _state["init_done"]:
        return _state["client"]
    _state["init_done"] = True

    # Make LANGFUSE_* from .env visible as real env vars for the SDK.
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass

    pub = os.getenv("LANGFUSE_PUBLIC_KEY")
    sec = os.getenv("LANGFUSE_SECRET_KEY")
    if not (pub and sec):
        log.warning(
            "Langfuse tracing DISABLED — set LANGFUSE_PUBLIC_KEY and "
            "LANGFUSE_SECRET_KEY (and optionally LANGFUSE_HOST) to enable it."
        )
        return None

    try:
        from langfuse import Langfuse

        client = Langfuse(
            public_key=pub,
            secret_key=sec,
            host=os.getenv("LANGFUSE_HOST", DEFAULT_HOST),
        )
        _state["client"] = client
        _state["enabled"] = True
        log.info("Langfuse tracing enabled (host=%s).", os.getenv("LANGFUSE_HOST", DEFAULT_HOST))
        return client
    except Exception as e:  # never let tracing setup break the app
        log.warning("Langfuse init failed (%s); continuing without tracing.", e)
        return None


def is_enabled() -> bool:
    return bool(_state["enabled"])


def get_callback_handler():
    """The LangChain/LangGraph callback handler to pass via invoke config."""
    if not _state["enabled"]:
        return None
    try:
        from langfuse.langchain import CallbackHandler

        # update_trace=False: we own trace-level attributes (session/user/tags)
        # via update_current_trace, so the handler must not overwrite them.
        return CallbackHandler(update_trace=False)
    except Exception as e:
        log.warning("Langfuse callback handler unavailable: %s", e)
        return None


@contextlib.contextmanager
def turn_trace(name: str, session_id: str, user_id: str, input_text: str):
    """Enclosing span that becomes the trace root for ONE user turn.

    session_id = thread_id groups a whole conversation into one Langfuse session.
    """
    c = _state["client"]
    if not _state["enabled"] or c is None:
        yield _NoopSpan()
        return
    try:
        with c.start_as_current_span(name=name, input=input_text) as s:
            try:
                c.update_current_trace(
                    name=name, session_id=session_id, user_id=user_id, input=input_text
                )
            except Exception as e:
                log.debug("update_current_trace(init) failed: %s", e)
            yield s
    except Exception as e:
        log.debug("turn_trace error: %s", e)
        yield _NoopSpan()


@contextlib.contextmanager
def span(name: str, **kwargs):
    """A nested span under the current trace (no-op when disabled)."""
    c = _state["client"]
    if not _state["enabled"] or c is None:
        yield _NoopSpan()
        return
    try:
        with c.start_as_current_span(name=name, **kwargs) as s:
            yield s
    except Exception as e:
        log.debug("span(%s) error: %s", name, e)
        yield _NoopSpan()


def update_current_trace(**kwargs) -> None:
    c = _state["client"]
    if _state["enabled"] and c is not None:
        try:
            c.update_current_trace(**kwargs)
        except Exception as e:
            log.debug("update_current_trace failed: %s", e)


def score_current_trace(name: str, value, comment: str | None = None,
                        data_type: str = "NUMERIC") -> None:
    c = _state["client"]
    if _state["enabled"] and c is not None:
        try:
            c.score_current_trace(name=name, value=value, comment=comment, data_type=data_type)
        except Exception as e:
            log.debug("score_current_trace(%s) failed: %s", name, e)


def create_score(trace_id: str | None, name: str, value, comment: str | None = None,
                 data_type: str = "NUMERIC") -> None:
    """Attach a score to a specific (already-finished) trace, e.g. user feedback."""
    c = _state["client"]
    if _state["enabled"] and c is not None and trace_id:
        try:
            c.create_score(trace_id=trace_id, name=name, value=value, comment=comment,
                           data_type=data_type)
        except Exception as e:
            log.debug("create_score(%s) failed: %s", name, e)


def current_trace_id() -> str | None:
    c = _state["client"]
    if _state["enabled"] and c is not None:
        try:
            return c.get_current_trace_id()
        except Exception:
            return None
    return None


def get_answer_system_prompt(fallback: str) -> str:
    """Fetch the managed ANSWER prompt (cached); fall back to the hardcoded text.

    Uses Langfuse's built-in TTL cache, so this is safe to call per answer: it
    only hits the network when the cache is cold/stale, and edits made in the
    Langfuse UI appear within PROMPT_CACHE_TTL_SECONDS — no redeploy.
    """
    c = _state["client"]
    if not (_state["enabled"] and c is not None):
        return fallback
    try:
        p = c.get_prompt(
            ANSWER_PROMPT_NAME,
            type="text",
            fallback=fallback,
            cache_ttl_seconds=PROMPT_CACHE_TTL_SECONDS,
        )
        return getattr(p, "prompt", None) or p.compile()
    except Exception as e:
        log.warning("Langfuse prompt '%s' fetch failed (%s); using fallback.", ANSWER_PROMPT_NAME, e)
        return fallback


def flush() -> None:
    """Flush buffered traces — call on CLI exit so short runs don't lose data."""
    c = _state["client"]
    if c is not None:
        try:
            c.flush()
        except Exception as e:
            log.debug("flush failed: %s", e)
