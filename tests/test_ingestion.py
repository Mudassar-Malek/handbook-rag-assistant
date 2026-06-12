"""Unit tests for the cleaning regexes and the heading-aware splitter.

These use small inline text fixtures (NOT the real PDF) so they run fast and
stay deterministic. Run with:  python -m pytest -q   (or: python -m unittest)
"""

from __future__ import annotations

import unittest

from ingestion.chunk import (
    HEADING_RE,
    TOPLEVEL_RE,
    build_policy_chunks,
    estimate_tokens,
    parent_for,
    parse_toc_section_numbers,
)
from ingestion.clean import (
    clean_page,
    collapse_whitespace,
    deglue,
    smart_title_case,
    strip_boilerplate,
)


class TestCleaningRegexes(unittest.TestCase):
    def test_deglue_lower_to_upper_boundary(self):
        self.assertEqual(deglue("theCompany"), "the Company")
        self.assertEqual(deglue("ofstakeholders"), "ofstakeholders")  # no boundary
        self.assertEqual(deglue("companyinreturn"), "companyinreturn")
        self.assertEqual(
            deglue("growth and opportunitytohelpsociety"),
            "growth and opportunitytohelpsociety",
        )

    def test_deglue_after_punctuation(self):
        self.assertEqual(deglue("satisfaction.We settle"), "satisfaction. We settle")
        self.assertEqual(deglue("Note:Important"), "Note: Important")
        self.assertEqual(deglue("2.1.EQUAL"), "2.1. EQUAL")

    def test_deglue_preserves_decimals_and_abbreviations(self):
        # Decimals and lowercase abbreviations must NOT be split.
        self.assertEqual(deglue("see 5.13 below"), "see 5.13 below")
        self.assertEqual(deglue("e.g. something"), "e.g. something")
        self.assertEqual(deglue("at 20:00 hrs"), "at 20:00 hrs")

    def test_deglue_protects_product_names(self):
        # The lowercase->Uppercase rule would split AcmeApp; it must be restored.
        self.assertEqual(deglue("useAcmeApp daily"), "use AcmeApp daily")
        # All-caps acronyms have no lower->upper boundary, so they're untouched.
        self.assertEqual(deglue("login on ESS and HRMS"), "login on ESS and HRMS")

    def test_strip_boilerplate(self):
        text = "EMPLOYEE\nHANDBOOK\nReal content here.\n24\n______\nMore content."
        cleaned = strip_boilerplate(text)
        self.assertNotIn("EMPLOYEE", cleaned)
        self.assertNotIn("HANDBOOK", cleaned)
        self.assertNotIn("______", cleaned)
        # Bare page number "24" removed, but content survives.
        self.assertNotIn("\n24\n", "\n" + cleaned + "\n")
        self.assertIn("Real content here.", cleaned)
        self.assertIn("More content.", cleaned)

    def test_strip_boilerplate_keeps_numbered_content_lines(self):
        # A line that is more than a bare number must survive.
        text = "5. LEAVE POLICIES\n10 Days\n12"
        cleaned = strip_boilerplate(text)
        self.assertIn("5. LEAVE POLICIES", cleaned)
        self.assertIn("10 Days", cleaned)
        self.assertNotIn("\n12", "\n" + cleaned)  # bare "12" removed

    def test_collapse_whitespace(self):
        text = "a   b\t\tc\n\n\n\nd   "
        self.assertEqual(collapse_whitespace(text), "a b c\n\nd")

    def test_clean_page_pipeline(self):
        raw = "EMPLOYEE\nHANDBOOK\ntheCompany providesgreat   service.\n24"
        cleaned = clean_page(raw)
        self.assertIn("the Company", cleaned)
        self.assertNotIn("EMPLOYEE", cleaned)
        self.assertNotIn("24", cleaned)


class TestSmartTitleCase(unittest.TestCase):
    def test_acronym_in_parentheses(self):
        self.assertEqual(smart_title_case("EARNED LEAVE (EL)"), "Earned Leave (EL)")
        self.assertEqual(smart_title_case("SICK LEAVE (SL)"), "Sick Leave (SL)")

    def test_plain_words(self):
        self.assertEqual(
            smart_title_case("BRING YOUR OWN DEVICE"), "Bring Your Own Device"
        )

    def test_hyphenated(self):
        self.assertEqual(
            smart_title_case("NON-HARASSMENT / NON-DISCRIMINATION"),
            "Non-Harassment / Non-Discrimination",
        )


class TestHeadingRegexes(unittest.TestCase):
    def test_subsection_heading_matches(self):
        m = HEADING_RE.match("5.1. EARNED LEAVE (EL)")
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), "5")
        self.assertEqual(m.group(2), "1")
        self.assertEqual(m.group(3), "EARNED LEAVE (EL)")

    def test_subsection_heading_two_digit(self):
        m = HEADING_RE.match("10.13. BRING YOUR OWN DEVICE")
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), "10")
        self.assertEqual(m.group(2), "13")

    def test_toplevel_heading_matches(self):
        m = TOPLEVEL_RE.match("7. DISCIPLINE POLICY")
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), "7")

    def test_toplevel_does_not_match_ordinary_sentence(self):
        # Sentences beginning with a number must not look like a heading.
        self.assertIsNone(TOPLEVEL_RE.match("1st January to 31st December"))


class TestHeadingSplitter(unittest.TestCase):
    SAMPLE = """5. LEAVE POLICIES
This is the leave overview. It applies to all full-time employees of the
Company and is credited on a half-yearly basis for confirmed employees only.
This preamble is intentionally written to be comfortably longer than the
minimum word threshold so that the splitter captures it as its own dedicated
section overview chunk during this validation test, which exercises the exact
code path used for the real leave policy introduction on page twenty four of
the handbook document under analysis here for the ingestion pipeline tests.
5.1. EARNED LEAVE (EL)
Eligibility: all permanent confirmed employees. EL is 15 days per year and is
calculated for the days worked during the previous calendar year.
5.2. SICK LEAVE (SL)
Sick leave is 12 days per year. It is credited on a half-yearly basis.
7. DISCIPLINE POLICY
7.1. GROUNDS FOR DISCIPLINARY ACTION
Misconduct includes theft and insubordination.
"""

    def setUp(self):
        self.chunks = build_policy_chunks({10: self.SAMPLE}, "Test.pdf")
        self.by_number = {c.section_number: c for c in self.chunks}

    def test_one_chunk_per_subsection(self):
        for num in ("5.1", "5.2", "7.1"):
            self.assertIn(num, self.by_number)

    def test_section_overview_captured_when_long(self):
        self.assertIn("5.0", self.by_number)
        self.assertIn("leave overview", self.by_number["5.0"].body)

    def test_embedded_top_level_heading_stripped_from_body(self):
        # "7. DISCIPLINE POLICY" should not leak into the 7.1 body.
        body = self.by_number["7.1"].body
        self.assertNotIn("DISCIPLINE POLICY", body)
        self.assertIn("Misconduct includes", body)

    def test_titles_are_smart_cased(self):
        self.assertEqual(self.by_number["5.1"].section_title, "Earned Leave (EL)")
        self.assertEqual(self.by_number["5.2"].section_title, "Sick Leave (SL)")

    def test_parent_section_derived(self):
        self.assertEqual(self.by_number["5.1"].parent_section, "5. Leave Policies")
        self.assertEqual(self.by_number["7.1"].parent_section, "7. Discipline Policy")

    def test_embedding_text_has_section_path_prefix(self):
        et = self.by_number["5.1"].embedding_text
        self.assertTrue(et.startswith("Leave Policies > 5.1 Earned Leave (EL):"))
        # Body is included after the prefix.
        self.assertIn("Eligibility", et)

    def test_pages_provenance_recorded(self):
        self.assertEqual(self.by_number["5.1"].pages, [10])

    def test_deterministic_ids(self):
        again = build_policy_chunks({10: self.SAMPLE}, "Test.pdf")
        ids_a = [c.id for c in self.chunks]
        ids_b = [c.id for c in again]
        self.assertEqual(ids_a, ids_b)


class TestOversizedSplitting(unittest.TestCase):
    def test_long_section_splits_on_paragraphs(self):
        para = ("word " * 400).strip()  # ~400 tokens
        body = f"5.13. TELEWORKING / WORK FROM HOME\n{para}\n\n{para}\n\n{para}\n"
        chunks = build_policy_chunks({28: body}, "Test.pdf")
        tele = [c for c in chunks if c.section_number == "5.13"]
        self.assertGreater(len(tele), 1)  # split into multiple parts
        parts = sorted(c.part for c in tele)
        self.assertEqual(parts, list(range(1, len(tele) + 1)))
        self.assertTrue(all(c.total_parts == len(tele) for c in tele))


class TestTocCrosscheck(unittest.TestCase):
    def test_parse_toc_section_numbers(self):
        toc = "5.1. EARNED LEAVE 24  5.13. TELEWORKING 28  10.13. BYOD 44"
        nums = parse_toc_section_numbers(toc)
        self.assertEqual(nums, {"5.1", "5.13", "10.13"})


class TestHelpers(unittest.TestCase):
    def test_estimate_tokens(self):
        self.assertGreater(estimate_tokens("one two three four five"), 0)

    def test_parent_for_front_matter(self):
        self.assertEqual(parent_for("0.6"), "Front Matter")

    def test_parent_for_policy(self):
        self.assertEqual(parent_for("10.13"), "10. General Computer Usage")


if __name__ == "__main__":
    unittest.main()
