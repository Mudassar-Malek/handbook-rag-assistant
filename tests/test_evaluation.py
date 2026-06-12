"""Tests for Phase 5 evaluation.

- Dataset loader validates the schema (every item has the three fields).
- section_hit_rate computes correctly on a tiny synthetic fixture.
- `run --check` exits nonzero against a doctored baseline.

These run fully offline (no LLM judge): they exercise schema, the LLM-free
metric, and the regression guard with a fake retriever.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from evaluation import dataset as ds
from evaluation import metrics as M
from evaluation import runner
from evaluation.config import EvalSettings


class SchemaTests(unittest.TestCase):
    def test_core_dataset_is_valid(self):
        for item in ds.CORE_DATASET:
            ds.validate_item(item)  # must not raise
        self.assertEqual(len(ds.CORE_DATASET), 12)

    def test_loader_validates_and_roundtrips(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "dataset.jsonl")
            ds.save_dataset(path, ds.CORE_DATASET)
            loaded = ds.load_dataset(path)
            self.assertEqual(len(loaded), 12)
            for item in loaded:
                self.assertIn("question", item)
                self.assertIn("ground_truth", item)
                self.assertIn("ground_truth_sections", item)

    def test_missing_field_raises(self):
        with self.assertRaises(ValueError):
            ds.validate_item({"question": "q", "ground_truth": "a"})  # no sections
        with self.assertRaises(ValueError):
            ds.validate_item({"question": "q", "ground_truth": "a", "ground_truth_sections": []})


class SectionHitRateTests(unittest.TestCase):
    def test_section_hit_subset_semantics(self):
        self.assertTrue(M.section_hit(["5.1"], ["5.1", "5.0"]))      # subset
        self.assertTrue(M.section_hit(["7.3", "7.4"], ["7.4", "7.3", "5.1"]))
        self.assertFalse(M.section_hit(["7.3", "7.4"], ["7.3", "5.1"]))  # 7.4 missing

    def test_section_hit_rate_fixture(self):
        rows = [
            (["5.1"], ["5.1", "5.0"]),        # hit
            (["7.3", "7.4"], ["7.3"]),        # miss (7.4 absent)
            (["3.7"], ["3.7"]),               # hit
            (["9.2"], ["9.3", "9.1"]),        # miss
        ]
        self.assertAlmostEqual(M.section_hit_rate(rows), 0.5)

    def test_page_hit_modes(self):
        self.assertTrue(M.page_hit({24, 25}, {24, 25, 26}, subset=True))
        self.assertFalse(M.page_hit({24, 25}, {24}, subset=True))
        self.assertTrue(M.page_hit({24, 25}, {24}, subset=False))  # overlap


class RegressionGuardTests(unittest.TestCase):
    def test_check_regression_detects_drop(self):
        baseline = {"retrieval": {"section_hit_rate": 1.0, "context_precision": 0.9}}
        current = {"section_hit_rate": 0.80, "context_precision": 0.89}  # 20% drop on first
        violations = runner.check_regression(baseline, "retrieval", current, threshold=0.05)
        names = [v[0] for v in violations]
        self.assertIn("section_hit_rate", names)        # dropped >5%
        self.assertNotIn("context_precision", names)    # ~1% drop, within tolerance

    def test_run_check_exits_nonzero_on_doctored_baseline(self):
        class _FakeResult:
            def __init__(self, sn, pages):
                self.section_number = sn
                self.pages = pages
                self.display_text = "fake"
                self.scores = SimpleNamespace(rerank=0.5)

        class FakeRetriever:
            def __init__(self, *a, **k):
                pass

            def retrieve(self, q):
                # Never returns the right section -> low section_hit_rate.
                return [_FakeResult("0.0", [1])]

        with tempfile.TemporaryDirectory() as d:
            data_path = os.path.join(d, "dataset.jsonl")
            ds.save_dataset(data_path, ds.CORE_DATASET)
            base_path = os.path.join(d, "baseline.json")
            with open(base_path, "w") as f:
                json.dump({"retrieval": {"section_hit_rate": 1.0}}, f)  # doctored: perfect

            settings = EvalSettings(
                dataset_path=data_path, baseline_path=base_path, results_dir=os.path.join(d, "res")
            )
            from evaluation import run as run_mod

            with patch.object(run_mod, "get_eval_settings", return_value=settings), \
                 patch("retrieval.retriever.Retriever", FakeRetriever), \
                 patch("retrieval.config.get_settings", return_value=None):
                code = run_mod.main(["--target", "retrieval", "--check"])
            self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
