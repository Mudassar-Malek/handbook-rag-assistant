"""Document-specific configuration for the Employee Handbook 2026 (sample).

Everything that is *specific to this one PDF* lives here so the rest of the
pipeline stays generic. These values were derived by analysing a specific
PDF (page layout, table-of-contents, and per-page heading positions). Adapt
them to your own document before ingesting.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Page roles (1-indexed, matching the printed page numbers in the PDF)
# ---------------------------------------------------------------------------
TOTAL_PAGES = 47

# Pages we ignore entirely: cover, the two-page TOC, and the back contact page.
SKIP_PAGES = {1, 2, 3, 47}

# Table-of-contents pages, parsed only for cross-checking which sections exist.
TOC_PAGES = (2, 3)

# Graphic / table front-matter pages (4-9). They have no numbered headings, so
# each is captured as ONE whole-page chunk with a hand-assigned title and a
# synthetic "0.x" section number under the "Front Matter" parent.
FRONT_MATTER = {
    4: ("0.1", "Welcome Aboard"),
    5: ("0.2", "About the Company"),
    6: ("0.3", "Message from our CEO"),
    7: ("0.4", "Leadership Team"),
    8: ("0.5", "Our Core Values"),
    9: ("0.6", "Revision History"),
}
FRONT_MATTER_PARENT = "Front Matter"

# Policy body pages with numbered subsections (10-46 inclusive).
POLICY_PAGE_START = 10
POLICY_PAGE_END = 46

# Pages that contain a real table which PyMuPDF's plain-text extraction mangles.
# We extract these with page.find_tables() and render them as markdown.
#   9  -> Revision History (the whole page IS the table; answers "what changed in 2026").
#   24 -> Leave entitlement table (EL 15 / SL+CL 12 / 10 holidays) inside section 5.
REVISION_HISTORY_PAGE = 9
LEAVE_TABLE_PAGE = 24

# ---------------------------------------------------------------------------
# The 11 top-level sections (hardcoded; derived from the TOC + body dividers).
# parent_section is derived from the leading digit of a subsection number.
# ---------------------------------------------------------------------------
TOP_LEVEL_SECTIONS = {
    1: "Welcome",
    2: "Workplace Commitments",
    3: "Company Policy & Procedures",
    4: "Attendance Policies",
    5: "Leave Policies",
    6: "Performance & Reviews",
    7: "Discipline Policy",
    8: "Employee Health and Safety",
    9: "Employee Benefits",
    10: "General Computer Usage",
    11: "Other Important Updates",
}

# ---------------------------------------------------------------------------
# Cleaning configuration
# ---------------------------------------------------------------------------
# Product / acronym names that the de-gluing pass must NOT split or lowercase.
# (e.g. the lowercase->Uppercase rule would otherwise turn "AcmeApp" into
# "Acme App".) Keys are the *spaced* form produced by de-gluing, values are
# the correct form to restore. Add your own document's product names here.
PROTECTED_TERMS = {
    "Acme App": "AcmeApp",
}

# Short tokens that should stay uppercase when title-casing a heading
# (so "EARNED LEAVE (EL)" -> "Earned Leave (EL)", not "(El)").
ACRONYMS = {
    "EL", "SL", "CL", "ML", "LOP", "IJP", "BYOD", "HRMS", "ESS", "CEO",
    "IT", "HR", "PF", "ESI", "POSH", "NDA", "CCTV", "WFH",
}

# ---------------------------------------------------------------------------
# Chunking configuration
# ---------------------------------------------------------------------------
# Soft cap on chunk size. Only a couple of sections (5.13 Teleworking,
# 10.13 BYOD) exceed this and get split on paragraph boundaries.
SOFT_CAP_TOKENS = 1000

# A "section overview" chunk (e.g. "5.0") is only emitted for the text that
# appears between a top-level heading and its first subsection if that text is
# at least this many words. Keeps us from creating empty overview chunks.
MIN_PREAMBLE_WORDS = 50

# ---------------------------------------------------------------------------
# Storage / embedding defaults (overridable via .env or CLI)
# ---------------------------------------------------------------------------
DEFAULT_COLLECTION = "employee_handbook_2026"
DEFAULT_PERSIST_DIR = "chroma_db"

# Extraction reading order. The PDF is 2-3 columns; the content stream is
# already in natural reading order, so sort=False produces clean, coherent
# bodies with headings on their own lines. sort=True sorts blocks by vertical
# position and interleaves the columns into incoherent text. Default to the
# clean mode; the CLI exposes --sort to force the legacy sorted behaviour.
DEFAULT_SORT = False
