"""Phase 5 — evaluation.

A golden dataset + RAGAS metrics + a custom, LLM-free `section_hit_rate`, a
chunking A/B experiment, and a regression guard. CLIs:

    python -m evaluation.build_dataset
    python -m evaluation.run --target retrieval
    python -m evaluation.run --target rag
    python -m evaluation.run --check
    python -m evaluation.compare_chunking
"""

__version__ = "0.1.0"
