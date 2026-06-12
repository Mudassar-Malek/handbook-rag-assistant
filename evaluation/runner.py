"""Orchestration for eval runs: retrieval-only and full-pipeline, plus IO,
Langfuse score push, and the regression guard."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone

import observability as obs

from . import metrics as M
from .config import EvalSettings
from .judge import build_judge, judge_model_id
from .judge_cache import JudgeCache

RETRIEVAL_METRICS = ["section_hit_rate", "context_precision", "context_recall"]
RAG_METRICS = ["faithfulness", "answer_relevancy", "answer_correctness"]


# --------------------------------------------------------------------------- ids
def git_hash() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(__file__)),
        )
        return out.stdout.strip() or "nogit"
    except Exception:
        return "nogit"


def make_run_id() -> str:
    return f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{git_hash()}"


# ----------------------------------------------------------------- retrieval run
def retrieve(retriever, question: str):
    """Return (sections, pages, contexts) for a raw question (no LLM rewrite)."""
    results = retriever.retrieve(question)
    sections = [r.section_number for r in results]
    pages = sorted({p for r in results for p in r.pages})
    contexts = [r.display_text for r in results]
    top_rerank = max((r.scores.rerank for r in results if r.scores.rerank is not None), default=None)
    return sections, pages, contexts, top_rerank


def run_retrieval(dataset: list[dict], retriever) -> dict:
    per_q: list[dict] = []
    samples: list[dict] = []
    for item in dataset:
        sections, pages, contexts, top = retrieve(retriever, item["question"])
        hit = M.section_hit(item["ground_truth_sections"], sections)
        per_q.append({
            "question": item["question"],
            "ground_truth_sections": item["ground_truth_sections"],
            "retrieved_sections": sections,
            "section_hit_rate": 1.0 if hit else 0.0,
            "retrieval_top_score": top,
        })
        samples.append({
            "question": item["question"],
            "contexts": contexts,
            "reference": item["ground_truth"],
        })

    _maybe_ragas(per_q, samples, ["context_precision", "context_recall"])
    aggregates = _aggregate(per_q, RETRIEVAL_METRICS)
    return {"per_question": per_q, "aggregates": aggregates}


# ----------------------------------------------------------------------- rag run
def run_rag(dataset: list[dict], agent) -> dict:
    per_q: list[dict] = []
    samples: list[dict] = []
    for i, item in enumerate(dataset):
        state = agent.chat(item["question"], thread_id=f"eval-rag-{i}", user_id="eval")
        answer = state.get("final_answer", "")
        contexts = [ex["text"] for ex in state.get("retrieval", [])]
        sections = [ex["section_number"] for ex in state.get("retrieval", [])]
        per_q.append({
            "question": item["question"],
            "answer": answer,
            "retrieved_sections": sections,
            "section_hit_rate": 1.0 if M.section_hit(item["ground_truth_sections"], sections) else 0.0,
        })
        samples.append({
            "question": item["question"],
            "answer": answer,
            "contexts": contexts,
            "reference": item["ground_truth"],
        })

    _maybe_ragas(per_q, samples, ["faithfulness", "answer_relevancy", "answer_correctness"])
    aggregates = _aggregate(per_q, RAG_METRICS + ["section_hit_rate"])
    return {"per_question": per_q, "aggregates": aggregates}


# --------------------------------------------------------------------- ragas glue
def _maybe_ragas(per_q: list[dict], samples: list[dict], metric_names: list[str]) -> None:
    """Attach RAGAS metric scores to per_q in place (no-op if no judge)."""
    llm, emb = build_judge()
    if llm is None:
        return
    from ragas.metrics import (
        answer_correctness, answer_relevancy, context_precision, context_recall, faithfulness,
    )
    from .config import EvalSettings
    from .ragas_eval import run_metric

    registry = {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "answer_correctness": answer_correctness,
        "context_precision": context_precision,
        "context_recall": context_recall,
    }
    cache = JudgeCache(EvalSettings().judge_cache_path)
    model_id = judge_model_id()
    for name in metric_names:
        metric = registry[name]
        scores = run_metric(metric, name, samples, llm, emb, cache, model_id)
        for q, sc in zip(per_q, scores):
            q[name] = sc


def _aggregate(per_q: list[dict], metric_names: list[str]) -> dict:
    agg = {}
    for name in metric_names:
        vals = [q[name] for q in per_q if isinstance(q.get(name), (int, float))]
        if vals:
            agg[name] = sum(vals) / len(vals)
    return agg


# --------------------------------------------------------------------------- IO
def write_results(settings: EvalSettings, run_id: str, target: str, result: dict) -> str:
    os.makedirs(settings.results_dir, exist_ok=True)
    path = os.path.join(settings.results_dir, f"{run_id}.json")
    payload = {
        "run_id": run_id,
        "target": target,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git": git_hash(),
        "aggregates": result["aggregates"],
        "worst": {
            m: [
                {"question": w["question"], m: w.get(m)}
                for w in M.worst_questions(result["per_question"], m, n=3)
            ]
            for m in result["aggregates"]
        },
        "per_question": result["per_question"],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


def format_table(per_q: list[dict], metric_names: list[str]) -> str:
    cols = ["question"] + [m for m in metric_names if any(m in q for q in per_q)]
    widths = {c: len(c) for c in cols}
    for q in per_q:
        widths["question"] = max(widths["question"], min(len(q["question"]), 60))
    lines = []
    header = " | ".join(c.ljust(widths.get(c, 12)) if c == "question" else c for c in cols)
    lines.append(header)
    lines.append("-" * len(header))
    for q in per_q:
        cells = [q["question"][:60].ljust(widths["question"])]
        for c in cols[1:]:
            v = q.get(c)
            cells.append("  -  " if v is None else f"{v:.3f}")
        lines.append(" | ".join(cells))
    return "\n".join(lines)


# ------------------------------------------------------------------- Langfuse
def push_langfuse(run_id: str, target: str, aggregates: dict) -> None:
    obs.init_observability()
    if not obs.is_enabled():
        return
    with obs.span("eval-run", input={"run_id": run_id, "target": target}):
        obs.update_current_trace(
            name=f"eval:{target}", tags=["eval", target],
            metadata={"run_id": run_id, "target": target, "git": git_hash()},
        )
        for name, value in aggregates.items():
            obs.score_current_trace(name, float(value), comment=f"eval {target} {run_id}")
    obs.flush()


# ------------------------------------------------------------- regression guard
def load_baseline(path: str) -> dict:
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_baseline(path: str, target: str, aggregates: dict) -> None:
    data = load_baseline(path)
    data[target] = aggregates
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def check_regression(baseline: dict, target: str, aggregates: dict, threshold: float):
    """Return list of (metric, baseline, current) that dropped > threshold."""
    violations = []
    base = baseline.get(target, {})
    for metric, base_val in base.items():
        cur = aggregates.get(metric)
        if cur is None:
            continue
        if base_val > 0 and cur < base_val * (1 - threshold):
            violations.append((metric, base_val, cur))
    return violations
