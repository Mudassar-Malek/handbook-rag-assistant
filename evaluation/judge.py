"""Build the RAGAS evaluator LLM + embeddings from the Phase 3 provider config.

The judge follows the same .env provider pattern as the agents (anthropic /
openai). If no key is configured, `build_judge()` returns (None, None) and the
LLM-based RAGAS metrics are skipped — the LLM-free section_hit_rate still runs.
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger("evaluation.judge")


def _build_chat_model(provider: str):
    """A raw LangChain chat model for the judge (temperature 0), or None."""
    if provider == "anthropic" and os.getenv("ANTHROPIC_API_KEY"):
        from langchain_anthropic import ChatAnthropic

        model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")
        return ChatAnthropic(model=model, temperature=0)
    if provider == "openai" and os.getenv("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI

        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        return ChatOpenAI(model=model, temperature=0)
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "llama3.1"),
            temperature=0,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )
    return None


def _build_embeddings():
    """Local HF embeddings for RAGAS (avoids paid embedding calls). None if unavailable."""
    try:
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    except Exception:
        pass
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings  # deprecated but works

        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    except Exception as e:
        log.warning("No local embeddings available for RAGAS (%s).", e)
        return None


def judge_model_id() -> str:
    """A stable identifier for the judge, used in the cache key."""
    from agents.config import get_agent_settings

    s = get_agent_settings()
    provider = s.llm_provider
    if provider == "anthropic":
        return f"anthropic:{os.getenv('ANTHROPIC_MODEL', s.anthropic_model)}"
    if provider == "openai":
        return f"openai:{os.getenv('OPENAI_MODEL', s.openai_model)}"
    if provider == "ollama":
        return f"ollama:{os.getenv('OLLAMA_MODEL', s.ollama_model)}"
    return provider


def build_judge():
    """Return (ragas_llm, ragas_embeddings) wrappers, or (None, None) if no key."""
    from agents.config import get_agent_settings

    provider = get_agent_settings().llm_provider
    model = _build_chat_model(provider)
    if model is None:
        log.warning(
            "No evaluator LLM configured (provider=%s, key missing) — LLM-based "
            "RAGAS metrics will be skipped; section_hit_rate still runs.", provider
        )
        return None, None

    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    emb = _build_embeddings()
    return (
        LangchainLLMWrapper(model),
        LangchainEmbeddingsWrapper(emb) if emb is not None else None,
    )


def judge_available() -> bool:
    from agents.config import get_agent_settings

    return _build_chat_model(get_agent_settings().llm_provider) is not None
