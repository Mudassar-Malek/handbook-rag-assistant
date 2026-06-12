"""Interactive CLI for the handbook assistant.

    python -m agents.chat --thread demo1 --user-id alice

Each turn prints a trace line showing which nodes fired and the route taken,
e.g.  trace: router(handbook)->rewriter->retriever(2 queries)->answer->grader(pass)->finalize

If Langfuse is configured, each turn is one trace (session = thread id); after
each answer you can type "+1" / "-1" to attach a user_feedback score.
"""

from __future__ import annotations

import argparse

import observability as obs

from .config import get_agent_settings
from .graph import HandbookAgent


def run_turn(agent: HandbookAgent, query: str, thread_id: str, user_id: str) -> None:
    state = agent.chat(query, thread_id, user_id=user_id)
    print(f"\ntrace: {'->'.join(state.get('trace', []))}")
    if state.get("rewritten_queries"):
        print(f"sub-queries: {state['rewritten_queries']}")
    print(f"\nAssistant: {state.get('final_answer', '')}\n")


def collect_feedback(agent: HandbookAgent) -> None:
    """Optionally attach a +1/-1 user_feedback score to the last turn's trace."""
    if not obs.is_enabled() or not agent.last_trace_id:
        return
    try:
        fb = input("Feedback (+1 / -1, Enter to skip): ").strip()
    except EOFError:
        return
    if fb in ("+1", "1"):
        obs.create_score(agent.last_trace_id, "user_feedback", 1.0, comment="thumbs up")
        print("  recorded +1")
    elif fb in ("-1",):
        obs.create_score(agent.last_trace_id, "user_feedback", -1.0, comment="thumbs down")
        print("  recorded -1")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="agents.chat", description="Employee Handbook assistant.")
    p.add_argument("--thread", default="demo1", help="Thread id (memory key + Langfuse session).")
    p.add_argument("--user-id", default="dev", help="User id attached to each Langfuse trace.")
    p.add_argument("--provider", default=None, help="Override LLM provider (anthropic|openai|ollama|fake).")
    p.add_argument("--model", default=None,
                   help="Override the model for the chosen provider "
                        "(e.g. --provider ollama --model llama3.2:3b).")
    p.add_argument("--once", default=None, help="Run a single query and exit (non-interactive).")
    return p.parse_args(argv)


def _model_override(provider: str | None, model: str | None) -> dict:
    """Map --model onto the provider-specific model setting, if given."""
    if not model:
        return {}
    return {
        "ollama": {"ollama_model": model},
        "openai": {"openai_model": model},
        "anthropic": {"anthropic_model": model},
    }.get(provider or "", {})


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = get_agent_settings(
        llm_provider=args.provider, **_model_override(args.provider, args.model)
    )
    agent = HandbookAgent(settings)

    try:
        if args.once is not None:
            run_turn(agent, args.once, args.thread, args.user_id)
            return 0

        print(f"Employee Handbook assistant (thread='{args.thread}', user='{args.user_id}', "
              f"provider='{settings.llm_provider}').")
        print("Ask about HR policies. Type 'exit' or Ctrl-D to quit.\n")
        while True:
            try:
                query = input("You: ").strip()
            except EOFError:
                break
            if not query:
                continue
            if query.lower() in {"exit", "quit"}:
                break
            run_turn(agent, query, args.thread, args.user_id)
            collect_feedback(agent)
        return 0
    finally:
        # Short-lived CLI runs must flush buffered traces before exiting.
        obs.flush()


if __name__ == "__main__":
    raise SystemExit(main())
