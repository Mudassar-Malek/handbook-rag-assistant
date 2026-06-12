"""Evaluation CLI.

    python -m evaluation.run --target retrieval
    python -m evaluation.run --target rag
    python -m evaluation.run --target retrieval --check          # CI regression guard
    python -m evaluation.run --target retrieval --update-baseline
"""

from __future__ import annotations

import argparse
import sys

from .config import get_eval_settings
from .dataset import load_dataset
from .runner import (
    RAG_METRICS,
    RETRIEVAL_METRICS,
    check_regression,
    format_table,
    load_baseline,
    make_run_id,
    push_langfuse,
    run_rag,
    run_retrieval,
    save_baseline,
    write_results,
)
from .metrics import worst_questions


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="evaluation.run")
    p.add_argument("--target", choices=["rag", "retrieval"], default="retrieval")
    p.add_argument("--check", action="store_true", help="Exit nonzero on >5% regression vs baseline.")
    p.add_argument("--update-baseline", action="store_true", help="Save current aggregates as baseline.")
    p.add_argument("--provider", default=None, help="Override LLM provider (rag target).")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    settings = get_eval_settings()
    dataset = load_dataset(settings.dataset_path)
    print(f"Loaded {len(dataset)} questions from {settings.dataset_path}\n")

    if args.target == "retrieval":
        from retrieval.config import get_settings as get_retrieval_settings
        from retrieval.retriever import Retriever

        retriever = Retriever(get_retrieval_settings())
        result = run_retrieval(dataset, retriever)
        metric_names = RETRIEVAL_METRICS
    else:
        from agents.config import get_agent_settings
        from agents.graph import HandbookAgent

        agent = HandbookAgent(get_agent_settings(llm_provider=args.provider))
        result = run_rag(dataset, agent)
        metric_names = RAG_METRICS + ["section_hit_rate"]

    run_id = make_run_id()
    print(format_table(result["per_question"], metric_names))
    print("\nAGGREGATES")
    for m, v in result["aggregates"].items():
        print(f"  {m:24} {v:.3f}")

    print("\n3 WORST QUESTIONS PER METRIC")
    for m in result["aggregates"]:
        worst = worst_questions(result["per_question"], m, n=3)
        print(f"  [{m}]")
        for w in worst:
            print(f"     {w[m]:.3f}  {w['question']}")

    path = write_results(settings, run_id, args.target, result)
    print(f"\nWrote {path}")
    push_langfuse(run_id, args.target, result["aggregates"])

    if args.update_baseline:
        save_baseline(settings.baseline_path, args.target, result["aggregates"])
        print(f"Updated baseline ({args.target}) in {settings.baseline_path}")

    if args.check:
        baseline = load_baseline(settings.baseline_path)
        if not baseline.get(args.target):
            save_baseline(settings.baseline_path, args.target, result["aggregates"])
            print(f"No baseline for '{args.target}' yet — saved current as baseline. PASS.")
            return 0
        violations = check_regression(baseline, args.target, result["aggregates"],
                                      settings.regression_threshold)
        if violations:
            print(f"\nREGRESSION (> {settings.regression_threshold:.0%} drop):")
            for metric, base_val, cur in violations:
                print(f"  {metric}: baseline {base_val:.3f} -> current {cur:.3f}")
            return 1
        print(f"\nNo regression beyond {settings.regression_threshold:.0%}. PASS.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
