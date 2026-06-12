"""Integration tests for the Phase 2 retrieval layer.

These require the populated ChromaDB collection from Phase 1 (run the ingestion
pipeline first). They build the full default retriever ONCE (vector + BM25 +
cross-encoder reranker + post-processing) and assert handbook-specific outcomes.

Run:  python -m pytest tests/test_retrieval.py -q
(The first run downloads the cross-encoder model.)
"""

from __future__ import annotations

import unittest

from retrieval.bm25 import expand_query, tokenize
from retrieval.config import get_settings
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.postprocess import find_references
from retrieval.types import format_pages, parse_pages


# --- Fast unit tests (no ChromaDB needed) ---------------------------------
class TestPureFunctions(unittest.TestCase):
    def test_tokenize_strips_punctuation(self):
        self.assertEqual(tokenize("Comp-Off / LOP!"), ["comp", "off", "lop"])

    def test_synonym_expansion_wfh(self):
        expanded = expand_query("WFH policy")
        self.assertIn("teleworking", expanded.lower())

    def test_synonym_expansion_comp_off(self):
        self.assertIn("compensatory", expand_query("comp off rules").lower())

    def test_no_expansion_when_no_match(self):
        self.assertEqual(expand_query("gratuity calculation"), "gratuity calculation")

    def test_rrf_combines_ranks(self):
        # "a" is rank1 in both lists -> should win; k=60.
        list1 = [("a", 0.9), ("b", 0.8)]
        list2 = [("a", 5.0), ("c", 1.0)]
        fused = reciprocal_rank_fusion([list1, list2], k=60)
        self.assertAlmostEqual(fused["a"], 1 / 61 + 1 / 61)
        self.assertAlmostEqual(fused["b"], 1 / 62)
        self.assertGreater(fused["a"], fused["b"])
        self.assertGreater(fused["a"], fused["c"])

    def test_find_references(self):
        text = "Please refer 5.11 and see Section 2.12 for details."
        self.assertEqual(find_references(text), ["5.11", "2.12"])

    def test_parse_and_format_pages(self):
        self.assertEqual(parse_pages("24,25"), [24, 25])
        self.assertEqual(format_pages([25]), "p.25")
        self.assertEqual(format_pages([24, 25]), "pp.24-25")


# --- Integration tests (require the populated collection) ------------------
class TestRetrievalIntegration(unittest.TestCase):
    retriever = None

    @classmethod
    def setUpClass(cls):
        import chromadb

        settings = get_settings()
        try:
            client = chromadb.PersistentClient(path=settings.chroma_path)
            col = client.get_collection(settings.collection_name)
            if col.count() == 0:
                raise unittest.SkipTest("Collection is empty; run Phase 1 ingestion.")
        except unittest.SkipTest:
            raise
        except Exception as exc:  # collection missing / chroma not set up
            raise unittest.SkipTest(f"ChromaDB collection unavailable: {exc}")

        from retrieval.retriever import Retriever

        cls.retriever = Retriever(settings)

    # Helpers ----------------------------------------------------------------
    def sections(self, query: str) -> list[str]:
        return [r.section_number for r in self.retriever.retrieve(query)]

    def top_n_sections(self, query: str, n: int) -> list[str]:
        return self.sections(query)[:n]

    # Assertions -------------------------------------------------------------
    def test_earned_leave_days_top3(self):
        self.assertIn("5.1", self.top_n_sections("how many earned leaves do I get", 3))

    def test_carry_forward_retrieves_5_1(self):
        self.assertIn("5.1", self.sections("can I carry forward unused leave"))

    def test_notice_period_probation_top3(self):
        top3 = self.top_n_sections("notice period during probation", 3)
        self.assertTrue({"7.3", "7.4"} & set(top3), f"got {top3}")

    def test_report_harassment_top3(self):
        self.assertIn("2.9", self.top_n_sections("who do I report harassment to", 3))

    def test_comp_off_top3_via_bm25(self):
        self.assertIn("5.10", self.top_n_sections("comp off rules", 3))

    def test_wfh_top3_via_synonym(self):
        self.assertIn("5.13", self.top_n_sections("WFH policy", 3))

    def test_what_changed_2026_top5(self):
        self.assertIn("0.6", self.top_n_sections("what changed in 2026", 5))

    def test_multipart_stitching_returns_all_parts(self):
        # 5.13 Teleworking is split into 3 parts; once any part is retrieved,
        # stitching must surface all of them.
        results = self.retriever.retrieve("teleworking work from home equipment policy")
        parts_5_13 = [r for r in results if r.section_number == "5.13"]
        self.assertTrue(parts_5_13, "5.13 not retrieved at all")
        total = parts_5_13[0].total_parts
        self.assertEqual(len(parts_5_13), total, f"expected {total} parts, got {len(parts_5_13)}")


if __name__ == "__main__":
    unittest.main()
