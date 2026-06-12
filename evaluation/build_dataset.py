"""Build the evaluation dataset.

- Writes the 12 hand-verified CORE items to dataset.jsonl (the golden set).
  Existing dataset.jsonl is preserved unless --force (so manually promoted
  synthetic questions are never clobbered).
- Generates ~13 synthetic questions with RAGAS TestsetGenerator over the
  ingested chunks and writes them to dataset_synthetic_unreviewed.jsonl for
  MANUAL review. They are NEVER auto-merged: a wrong synthetic ground truth
  silently poisons every future eval.
"""

from __future__ import annotations

import argparse
import json
import os

from .config import get_eval_settings
from .dataset import CORE_DATASET, save_dataset


def write_core(settings, force: bool) -> None:
    if os.path.exists(settings.dataset_path) and not force:
        print(f"{settings.dataset_path} exists — keeping it (use --force to overwrite with CORE only).")
        return
    save_dataset(settings.dataset_path, CORE_DATASET)
    print(f"Wrote {len(CORE_DATASET)} CORE items to {settings.dataset_path}")


def generate_synthetic(settings) -> None:
    from .judge import build_judge, judge_available

    if not judge_available():
        print(
            "\nSynthetic generation SKIPPED: no evaluator LLM configured.\n"
            "Set LLM_PROVIDER + the matching API key in .env, then re-run "
            "`python -m evaluation.build_dataset` to generate "
            f"{settings.n_synthetic} questions for manual review."
        )
        return

    print(f"\nGenerating ~{settings.n_synthetic} synthetic questions with RAGAS TestsetGenerator...")
    from langchain_core.documents import Document
    from ragas.testset import TestsetGenerator

    # Build LangChain documents from the ingested chunks (keep section metadata
    # so we can recover ground_truth_sections from the generated contexts).
    from retrieval.config import get_settings as get_retrieval_settings
    from retrieval.loader import ChunkStore
    from retrieval.retriever import _open_collection

    store = ChunkStore(_open_collection(get_retrieval_settings()))
    docs = [
        Document(
            page_content=c.body,
            metadata={"section_number": c.section_number, "section_title": c.section_title},
        )
        for c in store.chunks
    ]

    llm, emb = build_judge()
    generator = TestsetGenerator(llm=llm, embedding_model=emb)
    testset = generator.generate_with_langchain_docs(docs, testset_size=settings.n_synthetic)

    items = []
    for sample in testset.to_list():
        question = sample.get("user_input") or sample.get("question", "")
        reference = sample.get("reference") or sample.get("ground_truth", "")
        # Best-effort: recover sections from the reference contexts' metadata.
        sections = sorted({
            m for ctx in (sample.get("reference_contexts") or [])
            for m in _sections_for_context(ctx, store)
        })
        items.append({
            "question": question,
            "ground_truth": reference,
            "ground_truth_sections": sections or ["UNKNOWN"],
        })

    with open(settings.synthetic_path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(items)} UNREVIEWED synthetic questions to {settings.synthetic_path}")
    print("REVIEW THESE MANUALLY before promoting good ones into dataset.jsonl:\n")
    for it in items:
        print(f"  Q: {it['question']}")
        print(f"     gt: {it['ground_truth'][:100]}")
        print(f"     sections (verify!): {it['ground_truth_sections']}\n")


def _sections_for_context(ctx: str, store) -> list[str]:
    """Find which stored chunk a generated context came from (substring match)."""
    snippet = (ctx or "")[:120]
    if not snippet:
        return []
    return [c.section_number for c in store.chunks if snippet and snippet in c.body][:1]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="evaluation.build_dataset")
    parser.add_argument("--force", action="store_true", help="Overwrite dataset.jsonl with CORE only.")
    parser.add_argument("--no-synthetic", action="store_true", help="Skip synthetic generation.")
    args = parser.parse_args(argv)

    settings = get_eval_settings()
    write_core(settings, args.force)
    if not args.no_synthetic:
        generate_synthetic(settings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
