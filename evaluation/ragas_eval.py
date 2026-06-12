"""Run RAGAS metrics per-sample with judge-call caching.

RAGAS evaluates all samples for a metric in one call; to cache at the
(question, answer, metric) granularity we run one metric at a time over only
the *uncached* samples, then persist each result. Re-running after a
retrieval-only change therefore re-pays only for samples whose inputs changed.
"""

from __future__ import annotations

import logging

from .judge_cache import JudgeCache

log = logging.getLogger("evaluation.ragas")


def _metric_column(df, metric) -> str:
    name = getattr(metric, "name", None)
    if name and name in df.columns:
        return name
    input_cols = {
        "user_input", "response", "retrieved_contexts", "reference",
        "reference_contexts", "multi_responses", "rubrics",
    }
    candidates = [c for c in df.columns if c not in input_cols]
    return candidates[-1] if candidates else df.columns[-1]


def run_metric(metric, metric_name: str, samples: list[dict], llm, embeddings,
               cache: JudgeCache, model_id: str) -> list[float | None]:
    """Score every sample for one metric (cache-aware).

    samples: list of {question, answer, contexts, reference}.
    Returns per-sample scores aligned with `samples`.
    """
    from ragas import EvaluationDataset, SingleTurnSample, evaluate

    results: list[float | None] = [None] * len(samples)
    to_eval: list[int] = []
    for i, s in enumerate(samples):
        key = cache.key(metric_name, model_id, s["question"], s.get("answer", ""),
                        s.get("contexts"), s.get("reference", ""))
        cached = cache.get(key)
        if cached is not None:
            results[i] = cached
        else:
            to_eval.append(i)

    if to_eval:
        ds = EvaluationDataset(samples=[
            SingleTurnSample(
                user_input=samples[i]["question"],
                response=samples[i].get("answer", ""),
                retrieved_contexts=samples[i].get("contexts", []),
                reference=samples[i].get("reference", ""),
            )
            for i in to_eval
        ])
        try:
            res = evaluate(dataset=ds, metrics=[metric], llm=llm, embeddings=embeddings,
                           show_progress=False)
            df = res.to_pandas()
            col = _metric_column(df, metric)
            for row_pos, i in enumerate(to_eval):
                val = df.iloc[row_pos][col]
                score = float(val) if val is not None and not _is_nan(val) else None
                results[i] = score
                if score is not None:
                    key = cache.key(metric_name, model_id, samples[i]["question"],
                                    samples[i].get("answer", ""), samples[i].get("contexts"),
                                    samples[i].get("reference", ""))
                    cache.set(key, score)
            cache.save()
        except Exception as e:
            log.warning("RAGAS metric '%s' failed (%s); leaving scores empty.", metric_name, e)

    return results


def _is_nan(x) -> bool:
    try:
        return x != x
    except Exception:
        return False
