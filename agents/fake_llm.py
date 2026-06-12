"""Deterministic, offline LLM stand-in (no API key required).

This is NOT a language model — it's a rule-based double that implements the same
interface as the real client so the graph, routing, memory, and grounding
behavior can be exercised offline and in CI. Crucially, it reads from the REAL
retrieved excerpts (so any numbers it quotes come from the actual handbook, not
from hardcoded answers). For genuinely fluent answers, configure a real provider.
"""

from __future__ import annotations

import re

from langchain_core.messages import AIMessage, HumanMessage

from .config import AgentSettings
from .schemas import GradeOutput, RewriteOutput, RouteDecision

_GREETINGS = {"hi", "hello", "hey", "thanks", "thank", "thankyou", "bye", "yo", "sup", "hiya"}

# Words too generic to indicate the user found a matching policy section.
_GENERIC = {
    "policy", "policies", "office", "handbook", "acme", "employee", "employees",
    "company", "rule", "rules", "what", "whats", "is", "the", "do", "i", "my", "me",
    "a", "an", "of", "on", "for", "to", "and", "can", "how", "much", "many", "get",
    "are", "when", "where", "who", "does", "about", "during", "if", "will", "with",
    "you", "your", "this", "that", "it", "them", "they", "there", "here",
}

# Casual -> handbook vocabulary (applied during rewrite).
_VOCAB = [
    (r"\bvacation\b", "Earned Leave EL"),
    (r"\bpto\b", "Earned Leave EL"),
    (r"\bsick day(s)?\b", "Sick Leave SL"),
    (r"\bwfh\b", "Teleworking Work from Home"),
    (r"\bwork from home\b", "Teleworking Work from Home"),
    (r"\bquit\b", "resignation separation"),
    (r"\bresign\b", "resignation separation"),
    (r"\bfired\b", "termination"),
    (r"\breferral bonus\b", "employee referral reward"),
]

_LEAVE_TYPES = ["maternity", "paternity", "adoption", "sick", "casual", "earned", "bereavement"]
_FOLLOWUP_RE = re.compile(r"^\s*(and|what about|how about|also|for)\b", re.IGNORECASE)
# Split on sentence punctuation only — handbook bodies are reflowed, so single
# newlines fall mid-sentence (e.g. "30 calendar\ndays") and must NOT split.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _GENERIC and len(w) > 2}


def _match_count(qwords: set[str], text: str) -> int:
    """Lenient overlap: a query word matches if its 5-char stem is a substring of
    the text. This bridges casual vs handbook vocabulary ("leaves"~"leave",
    "encash"~"encashment") without a full stemmer."""
    tl = re.sub(r"\s+", " ", text.lower())
    return sum(1 for w in qwords if w[:5] in tl)


class FakeLLM:
    def __init__(self, settings: AgentSettings):
        self.settings = settings

    # --- ROUTER: greeting -> chitchat; handbook keyword/follow-up -> handbook;
    #     otherwise -> out_of_scope. ---
    def route(self, messages: list, query: str) -> RouteDecision:
        tokens = re.findall(r"[a-z']+", query.lower())
        has_handbook = bool(_content_words(query) & _HANDBOOK_HINTS) or "policy" in query.lower()
        if has_handbook:
            return RouteDecision(route="handbook", reasoning="mentions a handbook topic")
        if _FOLLOWUP_RE.match(query) and _prior_human(messages):
            return RouteDecision(route="handbook", reasoning="follow-up to a prior handbook turn")
        if any(t in _GREETINGS for t in tokens):
            return RouteDecision(route="chitchat", reasoning="greeting/small talk only")
        return RouteDecision(route="out_of_scope", reasoning="not answerable from an HR handbook")

    # --- REWRITER: coreference (slot-fill leave types), vocab mapping, decompose. ---
    def rewrite(self, messages, query, previous_rewrites, missing) -> RewriteOutput:
        prior = _prior_human(messages)
        resolved = query

        # Coreference: a follow-up like "and for adoption?" inherits the prior
        # question's frame, swapping in the new leave type.
        if _FOLLOWUP_RE.match(query) and prior:
            new_type = next((t for t in _LEAVE_TYPES if t in query.lower()), None)
            old_type = next((t for t in _LEAVE_TYPES if t in prior.lower()), None)
            if new_type and old_type:
                resolved = re.sub(old_type, new_type, prior, flags=re.IGNORECASE)
            elif new_type:
                resolved = f"{new_type} {prior}"
            else:
                resolved = f"{prior} {query}"

        # Decompose on conjunctions (max 3 sub-queries).
        parts = [p.strip(" ?.") for p in re.split(r"\band\b", resolved, flags=re.IGNORECASE)]
        parts = [p for p in parts if p][:3] or [resolved]

        # Vocabulary translation + light de-coref ("them"/"it" dropped).
        subs = []
        for part in parts:
            for pat, repl in _VOCAB:
                part = re.sub(pat, repl, part, flags=re.IGNORECASE)
            part = re.sub(r"\b(them|it|they)\b", "", part, flags=re.IGNORECASE).strip()
            subs.append(part)

        # On retry, ensure a different formulation by appending a broadening hint.
        if previous_rewrites:
            subs = [f"{s} eligibility entitlement rules" for s in subs]

        # Dedupe while preserving order.
        seen, unique = set(), []
        for s in subs:
            key = s.lower()
            if key not in seen and s:
                seen.add(key)
                unique.append(s)
        return RewriteOutput(rewritten_queries=unique[:3] or [resolved], reasoning="rule-based rewrite")

    # --- ANSWER: compose from the REAL excerpts; refuse if nothing matches. ---
    def answer(self, query: str, excerpts: list[dict], system: str | None = None) -> str:
        # `system` (the managed prompt) is ignored by the rule-based double.
        qwords = _content_words(query)
        scored = []
        for ex in excerpts:
            overlap = _match_count(qwords, ex["text"] + " " + ex["section_title"])
            if overlap:
                scored.append((overlap, ex))
        if not scored:
            return (
                "I couldn't find this in the handbook. Please contact HR at "
                f"{self.settings.hr_email} for help with this."
            )
        scored.sort(key=lambda t: t[0], reverse=True)

        lines = []
        for _, ex in scored[:3]:
            sentence = _best_sentence(ex["text"], qwords)
            if sentence:
                lines.append(f"{sentence} {ex['citation']}")
        if not lines:  # matched only on title; cite the top section anyway
            top = scored[0][1]
            lines.append(f"See {top['section_title']} {top['citation']}.")
        return " ".join(lines)

    # --- GRADER: an answer that cites sections (or honestly declines) passes. ---
    def grade(self, query: str, draft: str, excerpts: list[dict]) -> GradeOutput:
        if "couldn't find" in draft.lower():
            return GradeOutput(grounded=True, complete=True, reasoning="honest no-answer is acceptable")
        grounded = "[" in draft  # cites at least one section
        return GradeOutput(
            grounded=grounded,
            complete=grounded,
            reasoning="cited" if grounded else "no citation found",
        )

    # --- SMALLTALK: canned warm / scope responses. ---
    def smalltalk(self, route: str, messages: list, query: str) -> str:
        if route == "chitchat":
            return (
                "Hello! I'm the Acme Corp Employee Handbook assistant. "
                "Ask me anything about leave, attendance, benefits, conduct, or IT policies."
            )
        return (
            "I can only help with questions about Acme Corp HR policies "
            "(leave, attendance, conduct, benefits, IT/security, exit process). "
            f"For anything else, please contact HR at {self.settings.hr_email}."
        )


# Topic hints that mark a message as a handbook question (for the fake router).
_HANDBOOK_HINTS = {
    "leave", "leaves", "earned", "sick", "casual", "maternity", "paternity", "adoption",
    "bereavement", "salary", "payroll", "notice", "probation", "resignation", "termination",
    "separation", "exit", "harassment", "posh", "discipline", "gratuity", "referral",
    "teleworking", "wfh", "telework", "encash", "encashment", "comp", "holiday", "holidays",
    "benefit", "benefits", "attendance", "tardiness", "dress", "conduct", "byod", "device",
    "internet", "email", "loan", "loans", "increment", "appraisal", "shift", "overtime",
    "vacation", "pto", "quit", "fired", "reward", "bonus", "carryover", "carry",
}


def _prior_human(messages: list) -> str | None:
    """The most recent prior user message (for coreference resolution)."""
    humans = [m.content for m in messages if isinstance(m, HumanMessage)]
    return humans[-2] if len(humans) >= 2 else None


def _best_sentence(text: str, qwords: set[str]) -> str:
    """Pick the sentence most relevant to the query, preferring ones with numbers."""
    best, best_score = "", -1.0
    normalized = re.sub(r"\s+", " ", text)
    for sent in _SENTENCE_RE.split(normalized):
        s = sent.strip(" |-")
        if len(s) < 5:
            continue
        score = float(_match_count(qwords, s))
        if re.search(r"\d", s):  # numbers are the highest-stakes facts
            score += 1.5
        if score > best_score:
            best, best_score = s, score
    return best
