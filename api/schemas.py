"""Request models with validation — meaningful 4xx on bad input."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    thread_id: str | None = None
    user_id: str = "anon"

    @field_validator("question")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question must not be blank")
        return v


class FeedbackRequest(BaseModel):
    trace_id: str = Field(..., min_length=1)
    value: int

    @field_validator("value")
    @classmethod
    def _thumb(cls, v: int) -> int:
        if v not in (1, -1):
            raise ValueError("value must be 1 (thumbs up) or -1 (thumbs down)")
        return v
