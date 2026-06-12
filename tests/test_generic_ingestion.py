"""Offline tests for the generic multi-source ingester (no ChromaDB / no models)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ingestion.generic import (
    build_generic_chunks,
    load_path,
    normalize_text,
)


class NormalizeTests(unittest.TestCase):
    def test_collapses_blank_lines_and_trailing_space(self):
        out = normalize_text("a  \n\n\n\nb   c\r\n")
        self.assertEqual(out, "a\n\nb c")


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        (self.dir / "policy.md").write_text(
            "# Vacation\n\nYou get 20 days.\n\n## Carryover\n\nUp to 5 days carry over.\n"
        )
        (self.dir / "benefits.html").write_text(
            "<html><head><style>.x{}</style></head><body>"
            "<h1>Benefits</h1><p>Health insurance and 4% match.</p>"
            "<script>ignore()</script><h2>Wellness</h2><p>Gym up to $50.</p></body></html>"
        )
        (self.dir / "notes.txt").write_text("Parking is free in lot B.")

    def tearDown(self):
        self.tmp.cleanup()

    def test_markdown_splits_on_headings(self):
        docs = load_path(str(self.dir / "policy.md"))
        self.assertEqual(len(docs), 1)
        titles = [s.title for s in docs[0].sections]
        self.assertEqual(titles, ["Vacation", "Carryover"])

    def test_html_sections_and_tag_stripping(self):
        docs = load_path(str(self.dir / "benefits.html"))
        sections = docs[0].sections
        titles = [s.title for s in sections]
        self.assertEqual(titles, ["Benefits", "Wellness"])
        joined = " ".join(s.text for s in sections)
        self.assertIn("4% match", joined)
        self.assertIn("$50", joined)
        self.assertNotIn("ignore()", joined)   # <script> stripped
        self.assertNotIn(".x{}", joined)        # <style> stripped

    def test_plain_text_single_section(self):
        docs = load_path(str(self.dir / "notes.txt"))
        self.assertEqual(len(docs[0].sections), 1)
        self.assertIsNone(docs[0].sections[0].title)

    def test_directory_recurses_supported_files(self):
        docs = load_path(str(self.dir))
        self.assertEqual({d.source for d in docs}, {"policy.md", "benefits.html", "notes.txt"})

    def test_missing_path_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_path(str(self.dir / "nope.md"))


class ChunkSchemaTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        (self.dir / "a.md").write_text("# One\n\nAlpha text here.\n\n# Two\n\nBeta text here.\n")
        (self.dir / "b.txt").write_text("Gamma plain text.")

    def tearDown(self):
        self.tmp.cleanup()

    def test_chunks_match_shared_schema(self):
        docs = load_path(str(self.dir))
        chunks = build_generic_chunks(docs, chunk_size=500, chunk_overlap=50)
        self.assertTrue(chunks)
        for c in chunks:
            self.assertTrue(c.section_number and "." in c.section_number)
            self.assertTrue(c.section_title)
            self.assertTrue(c.parent_section)
            self.assertTrue(c.body)
            self.assertTrue(c.embedding_text)   # auto-filled by Chunk.__post_init__
            self.assertTrue(c.id)               # deterministic id
            self.assertEqual(c.pages, [])       # non-PDF -> no page numbers

    def test_chunk_ids_are_unique(self):
        docs = load_path(str(self.dir))
        chunks = build_generic_chunks(docs)
        ids = [c.id for c in chunks]
        self.assertEqual(len(ids), len(set(ids)))

    def test_sections_grouped_per_document(self):
        docs = load_path(str(self.dir))
        chunks = build_generic_chunks(docs)
        # Each doc gets its own leading index (1.x, 2.x).
        doc_indexes = {c.section_number.split(".")[0] for c in chunks}
        self.assertEqual(doc_indexes, {"1", "2"})


if __name__ == "__main__":
    unittest.main()
