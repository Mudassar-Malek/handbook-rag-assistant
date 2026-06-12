"""Thin HTTP client for the FastAPI backend (the UI talks ONLY to the API)."""

from __future__ import annotations

import json
from typing import Iterator

import httpx


class APIError(Exception):
    """Raised when the backend is unreachable or returns an error status."""


def get_health(base_url: str, timeout: float = 5.0) -> dict:
    try:
        r = httpx.get(f"{base_url}/health", timeout=timeout)
        return {"status_code": r.status_code, **r.json()}
    except Exception as e:
        raise APIError(f"API not reachable at {base_url}: {e}") from e


def get_sections(base_url: str, timeout: float = 10.0) -> list[dict]:
    try:
        r = httpx.get(f"{base_url}/sections", timeout=timeout)
        r.raise_for_status()
        return r.json().get("sections", [])
    except Exception as e:
        raise APIError(f"Could not load sections: {e}") from e


def send_feedback(base_url: str, trace_id: str, value: int, timeout: float = 10.0) -> dict:
    try:
        r = httpx.post(f"{base_url}/feedback", json={"trace_id": trace_id, "value": value},
                       timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        raise APIError(f"Feedback failed: {e}") from e


def ask_stream(base_url: str, question: str, thread_id: str | None, user_id: str = "anon"
               ) -> Iterator[dict]:
    """Yield typed events from the /ask SSE stream."""
    payload = {"question": question, "thread_id": thread_id, "user_id": user_id}
    try:
        with httpx.stream("POST", f"{base_url}/ask", json=payload, timeout=None) as r:
            if r.status_code != 200:
                body = r.read().decode("utf-8", "replace")
                raise APIError(f"/ask returned {r.status_code}: {body}")
            for line in r.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if not data:
                    continue
                try:
                    yield json.loads(data)
                except json.JSONDecodeError:
                    continue
    except APIError:
        raise
    except Exception as e:
        raise APIError(f"Streaming from API failed: {e}") from e
