"""Integration tests for the Phase 3 multi-agent graph.

By default these run against the deterministic offline `fake` provider (no API
key needed) so CI is reproducible. Retrieval is REAL (it queries the ChromaDB
index from Phase 1), so the numbers asserted below come from the actual handbook.

To run against a real model instead:

    AGENT_TEST_PROVIDER=anthropic ANTHROPIC_API_KEY=... pytest tests/test_agents.py

The graph wiring, routing, memory, and no-fabrication behavior are identical
across providers; only answer fluency differs.
"""

from __future__ import annotations

import os
import re
import unittest
import uuid
from unittest.mock import MagicMock

from agents.config import get_agent_settings
from agents.graph import HandbookAgent

PROVIDER = os.getenv("AGENT_TEST_PROVIDER", "fake")


def _sections(text: str) -> set[str]:
    """Distinct cited section numbers, e.g. {'5.1', '5.8'} from the answer."""
    return set(re.findall(r"\[(\d+(?:\.\d+)?)\b", text))


def _require_collection() -> None:
    """Skip (don't error) when the ChromaDB collection isn't populated yet.

    These are integration tests against the ingested sample handbook; a fresh
    clone has no data until `python -m ingestion.ingest <pdf>` is run.
    """
    import chromadb

    from retrieval.config import get_settings

    settings = get_settings()
    try:
        client = chromadb.PersistentClient(path=settings.chroma_path)
        col = client.get_collection(settings.collection_name)
        if col.count() == 0:
            raise unittest.SkipTest("Collection is empty; run Phase 1 ingestion first.")
    except unittest.SkipTest:
        raise
    except Exception as exc:  # collection missing / chroma not set up
        raise unittest.SkipTest(f"ChromaDB collection unavailable: {exc}")


class AgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _require_collection()
        # One shared agent (the real retriever / reranker loads only once).
        cls.agent = HandbookAgent(get_agent_settings(llm_provider=PROVIDER))

    def _thread(self) -> str:
        return f"test-{uuid.uuid4().hex[:8]}"

    def test_chitchat_routes_without_retrieval(self):
        # Use a dedicated agent with a spy retriever to prove it is never called.
        agent = HandbookAgent(get_agent_settings(llm_provider=PROVIDER))
        spy = MagicMock()
        agent._retriever = spy
        state = agent.chat("hi there", self._thread())
        self.assertEqual(state["route"], "chitchat")
        spy.retrieve.assert_not_called()

    def test_out_of_scope_routes(self):
        state = self.agent.chat("what's the capital of France", self._thread())
        self.assertEqual(state["route"], "out_of_scope")

    def test_earned_leave_amount_and_citation(self):
        state = self.agent.chat("how much earned leave do I get", self._thread())
        ans = state["final_answer"]
        self.assertIn("15", ans)
        self.assertIn("5.1", _sections(ans))

    def test_probation_notice_period_units(self):
        state = self.agent.chat("I'm on probation, what's my notice period", self._thread())
        ans = state["final_answer"].lower()
        self.assertIn("30 calendar days", ans)
        # Must not swap the units: 30 is calendar (not "30 business days").
        self.assertNotIn("30 business days", ans)

    def test_multipart_cites_two_sections(self):
        state = self.agent.chat(
            "what leaves do I get and can I encash them when I quit", self._thread()
        )
        self.assertGreaterEqual(len(_sections(state["final_answer"])), 2)

    def test_followup_memory_resolves_topic(self):
        thread = self._thread()
        self.agent.chat("what is maternity leave duration", thread)
        state = self.agent.chat("and for adoption?", thread)
        ans = state["final_answer"].lower()
        self.assertEqual(state["route"], "handbook")
        # Second answer is about ADOPTION (12 weeks, section 5.5), not maternity.
        self.assertTrue("adoption" in ans or "12 weeks" in ans)
        self.assertIn("5.5", _sections(state["final_answer"]))

    def test_no_fabrication_for_absent_policy(self):
        state = self.agent.chat("what is the policy on office pets", self._thread())
        self.assertIn("couldn't find", state["final_answer"].lower())


if __name__ == "__main__":
    unittest.main()
