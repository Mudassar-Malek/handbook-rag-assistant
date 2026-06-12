"""A tiny on-disk cache for LLM-judge metric scores.

Keyed by a hash of (question, answer, contexts, reference, metric, judge model),
so re-running an eval after a retrieval-only change doesn't re-pay for judgments
that didn't change. Stored as plain JSON so it's easy to inspect/delete.
"""

from __future__ import annotations

import hashlib
import json
import os


class JudgeCache:
    def __init__(self, path: str):
        self.path = path
        self._data: dict[str, float] = {}
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    self._data = json.load(f)
            except Exception:
                self._data = {}

    @staticmethod
    def key(metric: str, model_id: str, question: str, answer: str,
            contexts: list[str] | None = None, reference: str = "") -> str:
        payload = json.dumps(
            {
                "metric": metric,
                "model": model_id,
                "question": question,
                "answer": answer,
                "contexts": contexts or [],
                "reference": reference,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, key: str):
        return self._data.get(key)

    def set(self, key: str, value: float) -> None:
        self._data[key] = value

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2)
