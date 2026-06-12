"""Pydantic schemas for structured agent decisions.

Every agent decision goes through one of these models (via the LLM's
structured-output mode) so we never parse free-form strings — the LLM is forced
to return well-typed, validated fields.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RouteDecision(BaseModel):
    """Router output: which lane the latest user turn belongs to."""

    route: Literal["handbook", "chitchat", "out_of_scope"] = Field(
        description="handbook = company HR policy question (incl. follow-ups); "
        "chitchat = greeting/thanks/small talk; "
        "out_of_scope = legal advice, other companies, general knowledge, etc."
    )
    reasoning: str = Field(description="One short sentence explaining the choice.")


class RewriteOutput(BaseModel):
    """Rewriter output: 1-3 self-contained, handbook-vocabulary search queries."""

    rewritten_queries: list[str] = Field(
        description="1 to 3 standalone search queries (coreferences resolved, "
        "casual terms mapped to handbook terms, multi-part questions decomposed)."
    )
    reasoning: str = Field(description="Why these queries (and how follow-ups were resolved).")


class GradeOutput(BaseModel):
    """Grader output: is the draft answer trustworthy and complete?"""

    grounded: bool = Field(
        description="True only if EVERY factual claim (especially numbers/dates/"
        "durations) is supported by the provided excerpts."
    )
    complete: bool = Field(
        description="True only if all parts of the user's question were addressed."
    )
    reasoning: str = Field(
        description="What is missing or unsupported (used to guide a retry)."
    )

    @property
    def passed(self) -> bool:
        return self.grounded and self.complete
