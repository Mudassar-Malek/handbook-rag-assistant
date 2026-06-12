"""MCP server exposing the Employee Handbook retriever as a Cursor tool.

This deliberately exposes ONLY retrieval (no LLM, no API key): it returns ranked
handbook excerpts with ready-made citations, and the calling agent (Cursor's
model) does the routing, answering, and grounding. Run by Cursor via stdio; see
the `mcpServers` entry in ~/.cursor/mcp.json.

    python -m mcp_server      # (with PYTHONPATH=<this dir>)
"""

from __future__ import annotations

import os
from pathlib import Path

# Resolve relative paths (chroma_db/, model caches) against the project dir,
# regardless of the cwd Cursor launches us from.
PROJECT_DIR = Path(__file__).resolve().parent
os.chdir(PROJECT_DIR)

from mcp.server.fastmcp import FastMCP  # noqa: E402

from retrieval.config import get_settings  # noqa: E402
from retrieval.retriever import Retriever  # noqa: E402

mcp = FastMCP("employee-handbook")

# The retriever loads an embedding model + cross-encoder, so build it once and
# reuse it across tool calls (the server process is long-lived).
_RETRIEVER: Retriever | None = None
# Build with a slightly larger pool than the typical request so we can serve
# different k values without reconstructing the heavy models.
_MAX_K = 8

_GROUNDING_RULES = (
    "Answer ONLY from these excerpts. Cite the `citation` field inline, e.g. "
    "'You accrue 15 days [5.1 Earned Leave (EL), p.24]'. Quote numbers, dates and "
    "durations verbatim and preserve UNITS (the handbook distinguishes calendar "
    "days vs business days). If a policy varies by employee type (probationer vs "
    "confirmed, full-time vs consultant/intern), state the conditions. If the "
    "excerpts don't contain the answer, say you couldn't find it in the handbook "
    "and suggest contacting HR at hr@example.com - never guess."
)


def _retriever() -> Retriever:
    global _RETRIEVER
    if _RETRIEVER is None:
        _RETRIEVER = Retriever(get_settings(top_k=_MAX_K))
    return _RETRIEVER


@mcp.tool()
def search_handbook(query: str, k: int = 5) -> dict:
    """Search the Acme Corp Employee Handbook 2026 and return the most
    relevant policy excerpts with citations.

    Use this for ANY question about company HR policy: leave (earned/sick/casual/
    maternity/adoption/paternity/bereavement), encashment, attendance, notice
    periods, probation, resignation/termination, full-and-final settlement,
    benefits, gratuity, referrals, conduct/discipline, POSH, IT/security
    (BYOD, teleworking, email/internet use), holidays, and the exit process.
    Resolve conversational follow-ups into a specific query before calling (e.g.
    "and for adoption?" -> "adoption leave duration"). For multi-part questions,
    call once per sub-topic.

    Args:
        query: A focused, self-contained search query in handbook vocabulary.
        k: Number of excerpts to return (1-8; default 5).

    Returns a dict with `results` (each: citation, section_number, section_title,
    parent_section, pages, text, rerank_score, origin, related_sections) and the
    `grounding_rules` you MUST follow when composing the answer.
    """
    k = max(1, min(int(k), _MAX_K))
    results = _retriever().retrieve(query)[:k]
    return {
        "query": query,
        "results": [
            {
                "citation": r.citation,
                "section_number": r.section_number,
                "section_title": r.section_title,
                "parent_section": r.parent_section,
                "pages": r.pages,
                "text": r.display_text,
                "rerank_score": r.scores.rerank,
                "origin": r.origin,  # hybrid | expanded_from:X | stitched:X
                "related_sections": [f"{n} {t}" for n, t in r.sibling_sections],
            }
            for r in results
        ],
        "grounding_rules": _GROUNDING_RULES,
    }


if __name__ == "__main__":
    mcp.run()  # stdio transport
