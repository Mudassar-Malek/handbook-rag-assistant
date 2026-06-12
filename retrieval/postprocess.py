"""Document-specific post-processing, applied AFTER reranking.

These steps exploit the structure of THIS handbook to make the retrieved set
more complete for the downstream answer agent:

  a. Cross-reference expansion — the body text says things like "refer 5.11" or
     "see Section 2.12"; pull those referenced sections in too.
  b. Multi-part stitching — split sections (5.13, 10.11, 10.13) live as several
     part chunks; if one part is retrieved, fetch its siblings so the whole
     policy is visible.
  c. Sibling hints — annotate each result with the other sections under its
     parent, so the answer can mention related policies.
"""

from __future__ import annotations

import re

from .loader import ChunkStore
from .types import Chunk, RetrievalResult, Scores

# Matches an in-text reference to a numbered section, e.g.
# "refer 5.11", "refer to 5.11", "see Section 2.12", "according to section 5.10".
_REF_RE = re.compile(
    r"(?:refer(?:\s+to)?|see|sections?|as\s+per|according\s+to|per|clause|point)"
    r"\s+(?:section\s+)?(\d{1,2}\.\d{1,2})",
    re.IGNORECASE,
)


def find_references(text: str) -> list[str]:
    """Return distinct section numbers referenced inside `text`."""
    seen: list[str] = []
    for match in _REF_RE.finditer(text):
        num = match.group(1)
        if num not in seen:
            seen.append(num)
    return seen


def _result_from_chunk(chunk: Chunk, store: ChunkStore, origin: str) -> RetrievalResult:
    return RetrievalResult(
        id=chunk.id,
        section_number=chunk.section_number,
        section_title=chunk.section_title,
        parent_section=chunk.parent_section,
        pages=chunk.pages,
        display_text=chunk.body,
        scores=Scores(),  # supplemental chunks have no search scores
        part=chunk.part,
        total_parts=chunk.total_parts,
        origin=origin,
        sibling_sections=store.siblings(chunk.parent_section, chunk.section_number),
    )


def expand_cross_references(
    results: list[RetrievalResult], store: ChunkStore
) -> list[RetrievalResult]:
    """Append chunks referenced (by section number) inside the current results."""
    present = {r.id for r in results}
    present_sections = {r.section_number for r in results}
    additions: list[RetrievalResult] = []

    for result in list(results):
        for ref in find_references(result.display_text):
            if ref in present_sections:
                continue  # already retrieved, don't duplicate the section
            for chunk in store.get_section(ref):
                if chunk.id in present:
                    continue
                additions.append(
                    _result_from_chunk(
                        chunk, store, origin=f"expanded_from:{result.section_number}"
                    )
                )
                present.add(chunk.id)
            present_sections.add(ref)

    return results + additions


def stitch_multiparts(
    results: list[RetrievalResult], store: ChunkStore
) -> list[RetrievalResult]:
    """For any retrieved part of a split section, fetch its missing siblings."""
    present = {r.id for r in results}
    additions: list[RetrievalResult] = []

    for result in list(results):
        if result.total_parts <= 1:
            continue
        for chunk in store.get_section(result.section_number):
            if chunk.id in present:
                continue
            additions.append(
                _result_from_chunk(
                    chunk, store, origin=f"stitched:{result.section_number}"
                )
            )
            present.add(chunk.id)

    return results + additions


def attach_sibling_hints(results: list[RetrievalResult], store: ChunkStore) -> None:
    """Fill in sibling_sections for the primary results (in place)."""
    for result in results:
        if not result.sibling_sections:
            result.sibling_sections = store.siblings(
                result.parent_section, result.section_number
            )
