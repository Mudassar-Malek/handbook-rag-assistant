"""PDF extraction with PyMuPDF (fitz).

Two responsibilities:
1. Extract per-page text so every chunk keeps page provenance.
2. Recover real tables (Revision History p.9, Leave entitlement p.24) that the
   plain-text extractor mangles, rendering them as markdown.
"""

from __future__ import annotations

from dataclasses import dataclass

import fitz  # PyMuPDF

from . import config


@dataclass
class RawPage:
    """One extracted page (1-indexed `number`) and its raw text."""

    number: int
    text: str


def extract_pages(pdf_path: str, sort: bool = config.DEFAULT_SORT) -> list[RawPage]:
    """Extract every page as raw text in reading order.

    `sort=False` (default) keeps the content-stream order, which for this
    multi-column PDF is the natural reading order. `sort=True` reproduces the
    legacy `get_text("text", sort=True)` behaviour (sorts blocks by vertical
    position, which interleaves columns).
    """
    pages: list[RawPage] = []
    with fitz.open(pdf_path) as doc:
        for index in range(doc.page_count):
            text = doc[index].get_text("text", sort=sort)
            pages.append(RawPage(number=index + 1, text=text))
    return pages


def extract_table_markdown(pdf_path: str, page_number: int) -> str | None:
    """Return the first table on `page_number` rendered as a markdown table.

    Used as a fallback for pages whose tables don't survive text extraction.
    Returns None if no table is detected.
    """
    with fitz.open(pdf_path) as doc:
        page = doc[page_number - 1]
        found = page.find_tables()
        if not found.tables:
            return None
        rows = found.tables[0].extract()
    return _rows_to_markdown(rows)


def _rows_to_markdown(rows: list[list[str | None]]) -> str:
    """Render a list of row-lists as a GitHub-style markdown table."""
    def cell(value: str | None) -> str:
        # Flatten in-cell newlines and escape pipes so the table stays valid.
        text = (value or "").replace("\n", " ").strip()
        return text.replace("|", "\\|")

    if not rows:
        return ""
    header = [cell(c) for c in rows[0]]
    width = len(header)
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    for row in rows[1:]:
        cells = [cell(c) for c in row]
        # Pad/truncate ragged rows so every row matches the header width.
        cells = (cells + [""] * width)[:width]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
