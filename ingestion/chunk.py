"""Heading-aware chunking for the Employee Handbook.

Strategy (NOT fixed-size recursive splitting):
  * Front-matter pages 4-9 -> one whole-page chunk each (hand-assigned titles).
  * Policy pages 10-46      -> one chunk per numbered subsection ("5.1", "10.13"),
                               split on subsection headings at line starts.
  * A "section overview" chunk ("5.0") captures substantial text that appears
    between a top-level heading and its first subsection (e.g. the leave intro).
  * Oversized sections (> SOFT_CAP_TOKENS) are split on paragraph boundaries.

Every chunk carries page provenance and rich metadata, and gets a deterministic
id (section number + content hash) so re-ingestion upserts instead of duplicating.
"""

from __future__ import annotations

import hashlib
import re
import statistics
from dataclasses import dataclass, field

from . import config
from .clean import smart_title_case

# A subsection heading at the start of a line: "5.1. EARNED LEAVE (EL)".
# Captures the number ("5.1") and the raw title.
HEADING_RE = re.compile(r"^(\d{1,2})\.(\d{1,2})\.?\s+(\S.*)$")

# Tolerant variants used by the splitter. Some headings carry an uppercase
# watermark artifact glued in front of the number (e.g. "ETHHI3.1. CODE OF ...")
# and some put the number on its own line with the title reflowed below it
# (e.g. "10.8." then "STATEMENT" / "INTERNET/INTRANET").
_JUNK_PREFIX = r"[A-Z]{0,8}"
HEADING_INLINE_RE = re.compile(
    rf"^{_JUNK_PREFIX}(\d{{1,2}})\.(\d{{1,2}})\.?\s+([A-Z]\S.*)$"
)
HEADING_NUMONLY_RE = re.compile(rf"^{_JUNK_PREFIX}(\d{{1,2}})\.(\d{{1,2}})\.?\s*$")
# A line that is purely an uppercase title fragment (used to rebuild reflowed titles).
_UPPER_TITLE_RE = re.compile(r"^[A-Z0-9][A-Z0-9 &/'\u2019\-\u2013\u2014,()]{1,70}$")

# An embedded top-level heading line: "5. LEAVE POLICIES", "10. GENERAL COMPUTER
# USAGE". Title must be (mostly) uppercase so we don't eat ordinary sentences.
TOPLEVEL_RE = re.compile(r"^(\d{1,2})\.?\s+([A-Z][A-Z0-9 &/'\u2019\-\u2013\u2014,()]{2,60})$")


@dataclass
class Chunk:
    section_number: str          # "5.1", "0.6", "5.0"
    section_title: str           # "Earned Leave (EL)"
    parent_section: str          # "5. Leave Policies" / "Front Matter"
    pages: list[int]             # provenance, e.g. [24] or [28, 29]
    source: str                  # source filename
    body: str                    # clean display text
    part: int = 1                # 1-indexed part number
    total_parts: int = 1
    embedding_text: str = ""     # section path prefix + body (what we embed)
    id: str = ""

    def __post_init__(self) -> None:
        if not self.embedding_text:
            self.embedding_text = build_embedding_text(self)
        if not self.id:
            self.id = build_chunk_id(self)


# ---------------------------------------------------------------------------
# Token estimation (heuristic; good enough for a soft cap)
# ---------------------------------------------------------------------------
def estimate_tokens(text: str) -> int:
    """Rough token count using the common ~4/3 tokens-per-word heuristic."""
    return max(1, round(len(text.split()) * 4 / 3))


# ---------------------------------------------------------------------------
# Section path helpers
# ---------------------------------------------------------------------------
def parent_for(section_number: str) -> str:
    """Derive the parent-section label from the leading digit."""
    if section_number.startswith("0."):
        return config.FRONT_MATTER_PARENT
    top = int(section_number.split(".")[0])
    name = config.TOP_LEVEL_SECTIONS.get(top, f"Section {top}")
    return f"{top}. {name}"


def _parent_title_only(parent_section: str) -> str:
    """"5. Leave Policies" -> "Leave Policies"; "Front Matter" -> "Front Matter"."""
    return re.sub(r"^\d+\.\s*", "", parent_section)


def build_embedding_text(chunk: Chunk) -> str:
    """Section-path prefix + body, e.g. "Leave Policies > 5.1 Earned Leave (EL):\\n<body>".

    This string is what we embed; `body` is stored separately as display text.
    """
    prefix = f"{_parent_title_only(chunk.parent_section)} > {chunk.section_number} {chunk.section_title}"
    if chunk.total_parts > 1:
        prefix += f" (Part {chunk.part}/{chunk.total_parts})"
    return f"{prefix}:\n{chunk.body}"


def build_chunk_id(chunk: Chunk) -> str:
    """Deterministic id: source + section number + part + content hash."""
    stem = re.sub(r"\.pdf$", "", chunk.source, flags=re.IGNORECASE)
    digest = hashlib.sha1(chunk.body.encode("utf-8")).hexdigest()[:10]
    return f"{stem}-{chunk.section_number}-p{chunk.part}-{digest}"


# ---------------------------------------------------------------------------
# Front matter
# ---------------------------------------------------------------------------
def build_front_matter_chunks(
    cleaned_pages: dict[int, str], source: str, table_markdown: dict[int, str | None]
) -> list[Chunk]:
    """One whole-page chunk per front-matter page (4-9)."""
    chunks: list[Chunk] = []
    for page_no, (number, title) in config.FRONT_MATTER.items():
        # Revision History (p.9) is a table -> prefer the markdown rendering.
        body = ""
        if page_no == config.REVISION_HISTORY_PAGE and table_markdown.get(page_no):
            body = table_markdown[page_no]
        if not body:
            body = cleaned_pages.get(page_no, "").strip()
        if not body:
            body = title  # graphic page with no extractable text
        chunks.append(
            Chunk(
                section_number=number,
                section_title=title,
                parent_section=config.FRONT_MATTER_PARENT,
                pages=[page_no],
                source=source,
                body=body,
            )
        )
    return chunks


# ---------------------------------------------------------------------------
# Policy body chunking
# ---------------------------------------------------------------------------
@dataclass
class _Accumulator:
    number: str | None = None
    title: str = ""
    pages: set[int] = field(default_factory=set)
    lines: list[str] = field(default_factory=list)

    def body(self) -> str:
        return _join_lines(self.lines)


def _join_lines(lines: list[str]) -> str:
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def build_policy_chunks(cleaned_pages: dict[int, str], source: str) -> list[Chunk]:
    """Split policy pages (10-46) into one chunk per numbered subsection."""
    chunks: list[Chunk] = []
    current = _Accumulator()           # open subsection
    preamble = _Accumulator()          # text after a top-level heading, before 1st sub

    def flush_subsection() -> None:
        nonlocal current
        if current.number and current.body():
            chunks.extend(_finalize(current, source))
        current = _Accumulator()

    def flush_preamble() -> None:
        nonlocal preamble
        if (
            preamble.number
            and len(preamble.body().split()) >= config.MIN_PREAMBLE_WORDS
        ):
            chunks.extend(_finalize(preamble, source))
        preamble = _Accumulator()

    # Flatten policy pages into a (page, line) stream so headings whose title is
    # reflowed onto the next line can be reassembled.
    stream: list[tuple[int, str]] = []
    for page_no in range(config.POLICY_PAGE_START, config.POLICY_PAGE_END + 1):
        for raw_line in cleaned_pages.get(page_no, "").splitlines():
            stream.append((page_no, raw_line.strip()))

    i = 0
    while i < len(stream):
        page_no, line = stream[i]
        if not line:
            target = current if current.number else preamble
            if target.lines and target.lines[-1] != "":
                target.lines.append("")
            i += 1
            continue

        heading = _detect_heading(stream, i)
        if heading is not None:
            number, title, consumed = heading
            flush_subsection()
            flush_preamble()
            current = _Accumulator(
                number=number, title=smart_title_case(title), pages={page_no}
            )
            i += consumed
            continue

        top = TOPLEVEL_RE.match(line)
        if top and int(top.group(1)) in config.TOP_LEVEL_SECTIONS:
            flush_subsection()
            flush_preamble()
            top_n = int(top.group(1))
            preamble = _Accumulator(
                number=f"{top_n}.0",
                title=f"{config.TOP_LEVEL_SECTIONS[top_n]} (Overview)",
                pages={page_no},
            )
            i += 1
            continue

        # Ordinary content line.
        if current.number:
            current.lines.append(line)
            current.pages.add(page_no)
        elif preamble.number:
            preamble.lines.append(line)
            preamble.pages.add(page_no)
        # else: stray text before any heading is dropped (cover bleed, etc.)
        i += 1

    flush_subsection()
    flush_preamble()
    return chunks


def _detect_heading(stream: list[tuple[int, str]], i: int) -> tuple[str, str, int] | None:
    """Detect a subsection heading at stream index `i`.

    Returns (section_number, raw_title, lines_consumed) or None. Handles:
      * normal "5.1. EARNED LEAVE (EL)"
      * uppercase-watermark prefixes: "ETHHI3.1. CODE OF PROFESSIONAL CONDUCT"
      * number-only lines whose title is reflowed below: "10.8." / "STATEMENT".
    """
    line = stream[i][1]

    inline = HEADING_INLINE_RE.match(line)
    if inline and int(inline.group(1)) in config.TOP_LEVEL_SECTIONS:
        number = f"{int(inline.group(1))}.{int(inline.group(2))}"
        return number, inline.group(3).strip(), 1

    numonly = HEADING_NUMONLY_RE.match(line)
    if numonly and int(numonly.group(1)) in config.TOP_LEVEL_SECTIONS:
        # Rebuild the title from up to 3 following uppercase-only fragments.
        title_parts: list[str] = []
        j = i + 1
        while j < len(stream) and len(title_parts) < 3:
            nxt = stream[j][1]
            if nxt and _UPPER_TITLE_RE.match(nxt):
                title_parts.append(nxt)
                j += 1
            else:
                break
        if title_parts:
            number = f"{int(numonly.group(1))}.{int(numonly.group(2))}"
            return number, " ".join(title_parts), (j - i)
    return None


# ---------------------------------------------------------------------------
# Finalization: title path, soft-cap splitting, metadata
# ---------------------------------------------------------------------------
def _finalize(acc: _Accumulator, source: str) -> list[Chunk]:
    body = _strip_embedded_headings(acc.body())
    pages = sorted(acc.pages)
    parent = parent_for(acc.number)

    if estimate_tokens(body) <= config.SOFT_CAP_TOKENS:
        return [
            Chunk(
                section_number=acc.number,
                section_title=acc.title,
                parent_section=parent,
                pages=pages,
                source=source,
                body=body,
            )
        ]

    # Oversized: split on paragraph (then sentence) boundaries, never mid-sentence.
    parts = _split_oversized(body, config.SOFT_CAP_TOKENS)
    total = len(parts)
    return [
        Chunk(
            section_number=acc.number,
            section_title=acc.title,
            parent_section=parent,
            pages=pages,
            source=source,
            body=part_body,
            part=i,
            total_parts=total,
        )
        for i, part_body in enumerate(parts, start=1)
    ]


def _strip_embedded_headings(body: str) -> str:
    """Remove any embedded top-level heading lines (e.g. "7. DISCIPLINE POLICY")."""
    kept = []
    for line in body.splitlines():
        stripped = line.strip()
        m = TOPLEVEL_RE.match(stripped)
        if m and int(m.group(1)) in config.TOP_LEVEL_SECTIONS:
            continue
        kept.append(line)
    return _join_lines(kept)


def _split_oversized(body: str, cap_tokens: int) -> list[str]:
    """Split a too-large body into parts under the cap, never mid-sentence.

    Prefer paragraph boundaries; if a single "paragraph" still exceeds the cap
    (common here, since PDF line-wraps leave no blank lines), fall back to
    sentence boundaries. Units are then greedily packed.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    units: list[str] = []
    for para in paragraphs:
        if estimate_tokens(para) <= cap_tokens:
            units.append(para)
        else:
            units.extend(_sentence_groups(para, cap_tokens))

    parts: list[str] = []
    buf: list[str] = []
    for unit in units:
        candidate = "\n\n".join(buf + [unit])
        if buf and estimate_tokens(candidate) > cap_tokens:
            parts.append("\n\n".join(buf))
            buf = [unit]
        else:
            buf.append(unit)
    if buf:
        parts.append("\n\n".join(buf))
    return parts or [body]


def _sentence_groups(text: str, cap_tokens: int) -> list[str]:
    """De-wrap a paragraph and group whole sentences into <=cap_tokens blocks."""
    flat = re.sub(r"\s*\n\s*", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", flat)
    groups: list[str] = []
    buf: list[str] = []
    for sentence in sentences:
        candidate = " ".join(buf + [sentence])
        if buf and estimate_tokens(candidate) > cap_tokens:
            groups.append(" ".join(buf))
            buf = [sentence]
        else:
            buf.append(sentence)
    if buf:
        groups.append(" ".join(buf))
    return groups or [flat]


# ---------------------------------------------------------------------------
# Leave-table injection (page 24)
# ---------------------------------------------------------------------------
def inject_leave_table(chunks: list[Chunk], table_markdown: str | None) -> None:
    """Append the page-24 leave entitlement table to the section-5 chunk on p.24.

    Modifies chunks in place. Targets the lowest-numbered section-5 chunk that
    covers page 24 (the "5.0" overview if present, else "5.1").
    """
    if not table_markdown:
        return
    candidates = [
        c
        for c in chunks
        if c.section_number.startswith("5.") and config.LEAVE_TABLE_PAGE in c.pages
    ]
    if not candidates:
        return
    # The raw values may already appear in the extracted text, but the row/column
    # relationships are lost; append the clean markdown so it is unambiguous.
    target = min(candidates, key=lambda c: _num_key(c.section_number))
    target.body = f"{target.body}\n\nLeave entitlement table:\n{table_markdown}"
    # Body changed -> recompute derived fields so the id stays content-addressed.
    target.embedding_text = build_embedding_text(target)
    target.id = build_chunk_id(target)


def _num_key(section_number: str) -> tuple[int, ...]:
    return tuple(int(p) for p in section_number.split("."))


# ---------------------------------------------------------------------------
# TOC parsing + validation
# ---------------------------------------------------------------------------
def parse_toc_section_numbers(toc_text: str) -> set[str]:
    """Pull every "N.M" subsection number out of the (messy) TOC text."""
    deglued = re.sub(r"([.:;])([A-Za-z])", r"\1 \2", toc_text)
    return set(re.findall(r"\b(\d{1,2}\.\d{1,2})\.", deglued))


def chunk_stats(chunks: list[Chunk]) -> dict:
    """Size distribution + which sections got split, for the dry-run report."""
    sizes = [estimate_tokens(c.body) for c in chunks]
    split = sorted(
        {c.section_number for c in chunks if c.total_parts > 1},
        key=_num_key,
    )
    return {
        "count": len(chunks),
        "min_tokens": min(sizes) if sizes else 0,
        "median_tokens": int(statistics.median(sizes)) if sizes else 0,
        "max_tokens": max(sizes) if sizes else 0,
        "split_sections": split,
    }
