"""The graph state and helpers for turning retrieval results into excerpts.

LangGraph threads a single state dict through every node; each node returns a
partial dict that is merged in. `messages` uses the add_messages reducer so the
conversation history accumulates across turns (this is what the checkpointer
persists per thread). All other fields are reset at the start of each turn.
"""

from __future__ import annotations

from typing import Annotated, Any, Optional, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    # Conversation history (persisted per thread_id by the checkpointer).
    messages: Annotated[list[AnyMessage], add_messages]

    original_query: str           # the latest user question (verbatim)
    route: str                    # "handbook" | "chitchat" | "out_of_scope"
    rewritten_queries: list[str]  # current sub-queries
    previous_rewrites: list[str]  # all sub-queries tried this turn (for retries)
    retrieval: list[dict]         # serialized excerpts (see result_to_excerpt)
    draft_answer: str
    grade: Optional[dict]         # GradeOutput.model_dump()
    retry_count: int
    final_answer: str
    trace: list[str]              # node-firing trace for the CLI


def fresh_turn_state(query: str) -> dict[str, Any]:
    """Per-turn input that resets everything except the (appended) messages."""
    from langchain_core.messages import HumanMessage

    return {
        "messages": [HumanMessage(content=query)],
        "original_query": query,
        "route": "",
        "rewritten_queries": [],
        "previous_rewrites": [],
        "retrieval": [],
        "draft_answer": "",
        "grade": None,
        "retry_count": 0,
        "final_answer": "",
        "trace": [],
    }


def result_to_excerpt(result) -> dict:
    """Serialize a Phase 2 RetrievalResult into a plain dict for state/prompts."""
    return {
        "id": result.id,
        "section_number": result.section_number,
        "section_title": result.section_title,
        "parent_section": result.parent_section,
        "pages": result.pages,
        "citation": result.citation,
        "text": result.display_text,
        "rerank": result.scores.rerank,
    }


def format_excerpts(excerpts: list[dict]) -> str:
    """Render excerpts as a numbered, citation-tagged block for the LLM prompt."""
    blocks = []
    for i, ex in enumerate(excerpts, start=1):
        blocks.append(
            f"[Excerpt {i}] citation: {ex['citation']}\n"
            f"parent: {ex['parent_section']}\n"
            f"{ex['text']}"
        )
    return "\n\n---\n\n".join(blocks)
