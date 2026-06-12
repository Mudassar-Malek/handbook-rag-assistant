"""Agent settings (LLM provider, models, retry policy), loaded from .env."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict

# The HR contact the handbook gives for employee queries; used in fallbacks.
HR_EMAIL = "hr@example.com"


class AgentSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", case_sensitive=False, extra="ignore"
    )

    # Provider: "anthropic" | "openai" | "ollama" | "fake".
    #   anthropic/openai -> hosted LLM (needs the matching API key in .env)
    #   ollama           -> local LLM via Ollama (no key; needs `ollama serve`)
    #   fake             -> deterministic offline model (no key); for tests/demos
    llm_provider: str = "anthropic"
    anthropic_model: str = "claude-3-5-sonnet-latest"
    openai_model: str = "gpt-4o-mini"
    # Local Ollama model — must support structured outputs / tools (e.g.
    # llama3.1, qwen2.5, mistral-nemo). Pull it first: `ollama pull llama3.1`.
    ollama_model: str = "llama3.1"
    ollama_base_url: str = "http://localhost:11434"
    temperature: float = 0.0  # deterministic: HR facts must not drift

    # Retry loop: how many times the grader may bounce a bad draft back to the
    # rewriter before we finalize with an honest "couldn't fully answer".
    max_retries: int = 2

    # Retrieval.
    retrieval_k: int = 5          # results per sub-query (Phase 2 default)
    max_context_chunks: int = 10  # cap on excerpts handed to the answer node

    # Per-thread memory store.
    checkpoint_db: str = "agent_checkpoints.sqlite"

    hr_email: str = HR_EMAIL


def get_agent_settings(**overrides) -> AgentSettings:
    return AgentSettings(**{k: v for k, v in overrides.items() if v is not None})
