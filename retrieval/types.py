"""Typed data structures shared across the retrieval layer."""

from __future__ import annotations

from dataclasses import dataclass, field


def parse_pages(pages: str | int | list) -> list[int]:
    """Parse the stringified pages metadata ("24,25") back into ints."""
    if isinstance(pages, list):
        return [int(p) for p in pages]
    if isinstance(pages, int):
        return [pages]
    return [int(p) for p in str(pages).split(",") if p.strip().isdigit()]


def format_pages(pages: list[int]) -> str:
    """Human-friendly page string: "p.25", "pp.24-25", or "pp.24, 26"."""
    if not pages:
        return ""
    if len(pages) == 1:
        return f"p.{pages[0]}"
    if pages == list(range(pages[0], pages[-1] + 1)):
        return f"pp.{pages[0]}-{pages[-1]}"
    return "pp." + ", ".join(str(p) for p in pages)


@dataclass
class Chunk:
    """One chunk as loaded from ChromaDB (the single source of truth)."""

    id: str
    document: str          # the path-prefixed text that was embedded
    body: str              # the clean display text
    section_number: str    # "5.1"
    section_title: str     # "Earned Leave (EL)"
    parent_section: str    # "5. Leave Policies"
    pages: list[int]
    part: int
    total_parts: int
    source: str

    @property
    def rerank_text(self) -> str:
        """Text the cross-encoder scores against: title + display body.

        Deliberately NOT the path-prefixed `document` string — the reranker
        should judge relevance on the real content, not the synthetic prefix.
        """
        return f"{self.section_title}. {self.body}"


@dataclass
class Scores:
    """Per-stage scores for one result (None when a stage didn't touch it)."""

    vector: float | None = None
    bm25: float | None = None
    fused: float | None = None
    rerank: float | None = None


@dataclass
class RetrievalResult:
    """A single retrieved result with provenance and a ready-made citation."""

    id: str
    section_number: str
    section_title: str
    parent_section: str
    pages: list[int]
    display_text: str
    scores: Scores
    part: int = 1
    total_parts: int = 1
    # "hybrid" | "expanded_from:5.10" | "stitched:5.13"
    origin: str = "hybrid"
    # Other sections under the same parent, so the answer layer can cross-mention.
    sibling_sections: list[tuple[str, str]] = field(default_factory=list)

    @property
    def citation(self) -> str:
        label = section_label(self.section_number, self.section_title)
        pages = format_pages(self.pages)
        # Sources without page numbers (Markdown/HTML/Confluence) omit the page part.
        return f"[{label}, {pages}]" if pages else f"[{label}]"


def section_label(section_number: str, section_title: str) -> str:
    """"5.2" + "Sick Leave (SL)" -> "5.2 Sick Leave (SL)"."""
    return f"{section_number} {section_title}".strip()
