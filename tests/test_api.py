"""Phase 6 API tests.

Run against the REAL local ChromaDB + retriever, with the LLM mocked by the
deterministic offline `fake` provider (no network, no key). Langfuse is forced
off for determinism; the feedback test patches the Langfuse wrapper to assert
the client is called.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import patch

# Configure the app BEFORE importing it (settings are read at startup).
os.environ["LLM_PROVIDER"] = "fake"
os.environ["CHECKPOINT_DB"] = os.path.join(tempfile.gettempdir(), "test_api_ckpt.sqlite")
os.environ.pop("LANGFUSE_PUBLIC_KEY", None)
os.environ.pop("LANGFUSE_SECRET_KEY", None)

from fastapi.testclient import TestClient  # noqa: E402

from api.main import app  # noqa: E402


def _collect_sse(client, payload: dict) -> list[dict]:
    events: list[dict] = []
    with client.stream("POST", "/ask", json=payload) as r:
        assert r.status_code == 200, r.read()
        for line in r.iter_lines():
            line = line if isinstance(line, str) else line.decode("utf-8")
            if line.startswith("data:"):
                data = line[len("data:"):].strip()
                if data:
                    events.append(json.loads(data))
    return events


class APITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Entering the context manager runs the lifespan (loads retriever once).
        cls._cm = TestClient(app)
        cls.client = cls._cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._cm.__exit__(None, None, None)

    def test_health_ok(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["components"]["chromadb"]["status"], "up")
        self.assertEqual(body["components"]["bm25"]["status"], "up")
        self.assertGreater(body["components"]["chromadb"]["count"], 0)

    def test_ask_empty_question_422(self):
        self.assertEqual(self.client.post("/ask", json={"question": ""}).status_code, 422)
        self.assertEqual(self.client.post("/ask", json={"question": "   "}).status_code, 422)

    def test_ask_too_long_422(self):
        self.assertEqual(
            self.client.post("/ask", json={"question": "x" * 1001}).status_code, 422
        )

    def test_ask_happy_path_sse_order(self):
        events = _collect_sse(
            self.client, {"question": "How many earned leaves do I get per year?"}
        )
        types = [e["type"] for e in events]
        self.assertIn("status", types)
        self.assertIn("token", types)
        self.assertIn("citations", types)
        self.assertIn("done", types)

        # Ordering contract: status -> token -> citations -> done.
        self.assertLess(types.index("status"), types.index("token"))
        self.assertLess(types.index("token"), types.index("citations"))
        self.assertEqual(types[-1], "done")

        done = next(e for e in events if e["type"] == "done")
        self.assertTrue(done.get("thread_id"))

        citations = next(e for e in events if e["type"] == "citations")
        self.assertTrue(citations["items"])  # handbook question -> has sources

    def test_two_turns_share_checkpointer_state(self):
        tid = "test-thread-memory"
        _collect_sse(self.client, {"question": "How long is maternity leave?", "thread_id": tid})
        _collect_sse(self.client, {"question": "and what about adoption leave?", "thread_id": tid})

        body = self.client.get(f"/threads/{tid}").json()
        # 2 user + 2 assistant messages persisted under the same thread.
        self.assertGreaterEqual(body["message_count"], 4)
        self.assertGreaterEqual(body["history_length"], 2)

    def test_feedback_calls_langfuse_client(self):
        with patch("observability.create_score") as mock_score:
            r = self.client.post("/feedback", json={"trace_id": "fake-trace-123", "value": 1})
            self.assertEqual(r.status_code, 200)
            mock_score.assert_called_once()
            args, kwargs = mock_score.call_args
            self.assertIn("fake-trace-123", args)

    def test_feedback_bad_value_422(self):
        self.assertEqual(
            self.client.post("/feedback", json={"trace_id": "t", "value": 5}).status_code, 422
        )

    def test_sections_listed(self):
        body = self.client.get("/sections").json()
        sections = body["sections"]
        self.assertGreater(len(sections), 10)
        self.assertIn("5.1", [s["section_number"] for s in sections])


if __name__ == "__main__":
    unittest.main()
