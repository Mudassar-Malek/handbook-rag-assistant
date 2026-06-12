"""LLM client behind a small interface, so nodes don't touch provider details.

Each node calls one high-level method (route / rewrite / answer / grade /
smalltalk). The real client (`LangChainLLM`) drives an Anthropic or OpenAI chat
model through LangChain's structured-output mode; the offline client
(`FakeLLM`, in fake_llm.py) implements the same interface deterministically.
"""

from __future__ import annotations

from typing import Protocol

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from .config import AgentSettings
from .prompts import ANSWER_SYSTEM, GRADER_SYSTEM, REWRITER_SYSTEM, ROUTER_SYSTEM
from .schemas import GradeOutput, RewriteOutput, RouteDecision
from .state import format_excerpts


class LLMClient(Protocol):
    def route(self, messages: list, query: str) -> RouteDecision: ...
    def rewrite(
        self, messages: list, query: str, previous_rewrites: list[str], missing: str
    ) -> RewriteOutput: ...
    def answer(self, query: str, excerpts: list[dict], system: str | None = None) -> str: ...
    def grade(self, query: str, draft: str, excerpts: list[dict]) -> GradeOutput: ...
    def smalltalk(self, route: str, messages: list, query: str) -> str: ...


def _history_text(messages: list, limit: int = 6) -> str:
    """Render recent turns as plain text for prompts (excludes the latest)."""
    lines = []
    for m in messages[-(limit + 1):-1]:  # skip the latest (passed separately)
        role = "User" if isinstance(m, HumanMessage) else "Assistant"
        if isinstance(m, (HumanMessage, AIMessage)):
            lines.append(f"{role}: {m.content}")
    return "\n".join(lines) if lines else "(no prior turns)"


class LangChainLLM:
    """Wraps a LangChain chat model (Anthropic or OpenAI)."""

    def __init__(self, model, settings: AgentSettings, structured_method: str | None = None):
        self.model = model
        self.settings = settings
        # Some providers (Ollama) need an explicit structured-output method;
        # hosted providers infer it, so we leave it None for them.
        self.structured_method = structured_method

    # --- helpers ---
    def _structured(self, schema, system: str, user: str):
        model = (
            self.model.with_structured_output(schema, method=self.structured_method)
            if self.structured_method
            else self.model.with_structured_output(schema)
        )
        return model.invoke([SystemMessage(content=system), HumanMessage(content=user)])

    def _generate(self, system: str, user: str) -> str:
        resp = self.model.invoke(
            [SystemMessage(content=system), HumanMessage(content=user)]
        )
        return resp.content if isinstance(resp.content, str) else str(resp.content)

    # --- interface ---
    def route(self, messages: list, query: str) -> RouteDecision:
        user = f"Conversation so far:\n{_history_text(messages)}\n\nLatest user message: {query}"
        return self._structured(RouteDecision, ROUTER_SYSTEM, user)

    def rewrite(self, messages, query, previous_rewrites, missing) -> RewriteOutput:
        retry_note = ""
        if previous_rewrites:
            retry_note = (
                f"\n\nThis is a RETRY. Previously tried queries: {previous_rewrites}.\n"
                f"What was missing/unsupported last time: {missing or 'unknown'}.\n"
                "Produce a DIFFERENT formulation."
            )
        user = (
            f"Conversation so far:\n{_history_text(messages)}\n\n"
            f"User question to rewrite: {query}{retry_note}"
        )
        return self._structured(RewriteOutput, REWRITER_SYSTEM, user)

    def answer(self, query: str, excerpts: list[dict], system: str | None = None) -> str:
        user = (
            f"Handbook excerpts:\n\n{format_excerpts(excerpts)}\n\n"
            f"Question: {query}\n\nAnswer (cite sections inline):"
        )
        return self._generate(system or ANSWER_SYSTEM, user)

    def grade(self, query: str, draft: str, excerpts: list[dict]) -> GradeOutput:
        user = (
            f"User question: {query}\n\n"
            f"Draft answer:\n{draft}\n\n"
            f"Handbook excerpts the answer must rely on:\n\n{format_excerpts(excerpts)}"
        )
        return self._structured(GradeOutput, GRADER_SYSTEM, user)

    def smalltalk(self, route: str, messages: list, query: str) -> str:
        if route == "chitchat":
            system = (
                "You are a warm, brief assistant for the Acme Corp "
                "Employee Handbook. Reply to small talk in 1-2 friendly sentences "
                "and invite the user to ask about HR policies. Do not invent policy."
            )
        else:  # out_of_scope
            system = (
                "You are the Acme Corp Employee Handbook assistant. The "
                "user asked something outside your scope. In 1-3 sentences, politely "
                "explain you can only answer questions about company HR policies "
                "(leave, attendance, conduct, benefits, IT/security, exit process) "
                f"and suggest contacting HR at {self.settings.hr_email}. Do not answer "
                "the out-of-scope question."
            )
        return self._generate(system, query)


def build_llm(settings: AgentSettings) -> LLMClient:
    """Construct the configured LLM client."""
    provider = settings.llm_provider.lower()

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        model = ChatAnthropic(model=settings.anthropic_model, temperature=settings.temperature)
        return LangChainLLM(model, settings)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        model = ChatOpenAI(model=settings.openai_model, temperature=settings.temperature)
        return LangChainLLM(model, settings)

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        model = ChatOllama(
            model=settings.ollama_model,
            temperature=settings.temperature,
            base_url=settings.ollama_base_url,
        )
        # Ollama structured outputs are most reliable via JSON schema (Ollama >= 0.5).
        return LangChainLLM(model, settings, structured_method="json_schema")

    if provider == "fake":
        from .fake_llm import FakeLLM

        return FakeLLM(settings)

    raise ValueError(f"Unknown LLM_PROVIDER: {settings.llm_provider!r}")
