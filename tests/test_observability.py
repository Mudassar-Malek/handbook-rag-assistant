"""Tests for Phase 4 observability.

1. The app runs cleanly with NO Langfuse env vars (graceful degradation).
2. With a MOCKED Langfuse client, one chat turn produces exactly one trace,
   one retrieval span per sub-query, and grader scores attached.

These run offline against the deterministic `fake` LLM provider (retrieval is
real, against ChromaDB), so no Langfuse account or API key is needed.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import observability as obs
from agents.config import get_agent_settings
from agents.graph import HandbookAgent


def _disable_obs():
    obs._state.update({"enabled": False, "client": None, "init_done": True})


class _CtxMock:
    """Context manager whose __enter__ returns the given span mock."""

    def __init__(self, span):
        self.span = span

    def __enter__(self):
        return self.span

    def __exit__(self, *a):
        return False


def _make_mock_client(records: dict) -> MagicMock:
    """A stand-in Langfuse client that records the calls we assert on."""
    client = MagicMock()
    span = MagicMock()

    def start_as_current_span(*, name, **kw):
        records.setdefault("spans", []).append(name)
        return _CtxMock(span)

    def score_current_trace(*, name, value, **kw):
        records.setdefault("scores", []).append((name, value))

    def update_current_trace(**kw):
        records.setdefault("trace_updates", []).append(kw)

    client.start_as_current_span.side_effect = start_as_current_span
    client.score_current_trace.side_effect = score_current_trace
    client.update_current_trace.side_effect = update_current_trace
    client.get_current_trace_id.return_value = "trace-abc123"
    # Force the prompt-management fallback path (no prompt server in tests).
    client.get_prompt.side_effect = Exception("no prompt server in tests")
    return client


class DegradationTests(unittest.TestCase):
    def tearDown(self):
        _disable_obs()

    def test_runs_without_langfuse(self):
        # Neutralize init so no .env is loaded and tracing stays disabled.
        with patch.object(obs, "init_observability", return_value=None):
            _disable_obs()
            agent = HandbookAgent(get_agent_settings(llm_provider="fake"))
            self.assertFalse(obs.is_enabled())
            state = agent.chat("how much earned leave do I get", "deg-thread")
            self.assertTrue(state.get("final_answer"))
            self.assertIsNone(agent.last_trace_id)


class MockedClientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Build once (loads the real retriever / reranker a single time).
        with patch.object(obs, "init_observability", return_value=None):
            cls.agent = HandbookAgent(get_agent_settings(llm_provider="fake"))

    def setUp(self):
        self.records: dict = {}
        obs._state.update(
            {"enabled": True, "client": _make_mock_client(self.records), "init_done": True}
        )
        # The real LangChain handler needs a live client; use a no-op in tests.
        self._patch_handler = patch.object(obs, "get_callback_handler", return_value=None)
        self._patch_handler.start()

    def tearDown(self):
        self._patch_handler.stop()
        _disable_obs()

    def test_single_turn_trace_spans_and_scores(self):
        state = self.agent.chat("how much earned leave do I get", "mock-thread", user_id="tester")

        spans = self.records.get("spans", [])
        self.assertEqual(spans.count("handbook-turn"), 1, "exactly one trace per turn")
        n_sub = len(state["rewritten_queries"])
        self.assertEqual(spans.count("retrieve_subquery"), n_sub, "one span per sub-query")

        score_names = [n for n, _ in self.records.get("scores", [])]
        self.assertIn("grader_grounded", score_names)
        self.assertIn("grader_complete", score_names)
        self.assertIn("retrieval_top_score", score_names)

        updates = self.records.get("trace_updates", [])
        all_tags = [t for u in updates for t in (u.get("tags") or [])]
        self.assertIn("handbook", all_tags)

    def test_user_feedback_score(self):
        self.agent.chat("how much earned leave do I get", "mock-thread-2")
        self.assertEqual(self.agent.last_trace_id, "trace-abc123")

        obs.create_score(self.agent.last_trace_id, "user_feedback", 1.0, comment="thumbs up")
        client = obs._state["client"]
        client.create_score.assert_called_once()
        _, kwargs = client.create_score.call_args
        self.assertEqual(kwargs.get("name"), "user_feedback")
        self.assertEqual(kwargs.get("value"), 1.0)
        self.assertEqual(kwargs.get("trace_id"), "trace-abc123")


if __name__ == "__main__":
    unittest.main()
