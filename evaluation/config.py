"""Evaluation settings (paths, thresholds, experiment knobs)."""

from __future__ import annotations

import os

from pydantic_settings import BaseSettings, SettingsConfigDict

EVAL_DIR = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(os.path.dirname(EVAL_DIR), "results")


class EvalSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")

    # Dataset files (the golden set is hand-curated; synthetic is never auto-merged).
    dataset_path: str = os.path.join(EVAL_DIR, "dataset.jsonl")
    synthetic_path: str = os.path.join(EVAL_DIR, "dataset_synthetic_unreviewed.jsonl")

    # Outputs.
    results_dir: str = RESULTS_DIR
    baseline_path: str = os.path.join(EVAL_DIR, "baseline.json")
    judge_cache_path: str = os.path.join(EVAL_DIR, ".judge_cache.json")

    # Regression guard: fail if any aggregate drops more than this fraction.
    regression_threshold: float = 0.05

    # Synthetic generation target (manual review required before promotion).
    n_synthetic: int = 13

    # Chunking experiment (Strategy B).
    recursive_collection: str = "recursive_baseline"
    recursive_chunk_size: int = 1000   # tokens
    recursive_chunk_overlap: int = 150  # tokens
    pdf_path: str = os.path.join(os.path.dirname(EVAL_DIR), "Employee_Handbook_2026.pdf")


def get_eval_settings(**overrides) -> EvalSettings:
    return EvalSettings(**{k: v for k, v in overrides.items() if v is not None})
