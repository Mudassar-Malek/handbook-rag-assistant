"""The golden evaluation dataset: schema, the hand-written core, and IO.

Each item has exactly three fields:
    question               : str
    ground_truth           : str   (the verified answer)
    ground_truth_sections  : list[str]  (section_numbers that contain the answer)

The CORE items below are ILLUSTRATIVE examples written against a fictional
"Acme Corp" employee handbook so this repo runs end-to-end without shipping any
real document. Replace them with answers hand-verified against YOUR own handbook
before trusting the eval numbers — an LLM must NEVER rewrite these ground truths
(a wrong ground truth silently poisons every future eval). Synthetic questions
are generated separately and kept in an *unreviewed* file for manual promotion.
"""

from __future__ import annotations

import json

REQUIRED_FIELDS = ("question", "ground_truth", "ground_truth_sections")

# --- HAND-WRITTEN CORE (illustrative sample; replace with your own) ---------
CORE_DATASET: list[dict] = [
    {
        "question": "How many earned leaves do I get per year?",
        "ground_truth": "15 days of Earned Leave (EL) per year.",
        "ground_truth_sections": ["5.1"],
    },
    {
        "question": "Can unused earned leave be carried forward?",
        "ground_truth": (
            "Yes. Earned Leave can be carried forward, but accumulation is capped "
            "at 45 days; any excess is compensated at the basic pay rate."
        ),
        "ground_truth_sections": ["5.1"],
    },
    {
        "question": "What is the notice period during probation versus after confirmation?",
        "ground_truth": (
            "30 calendar days during probation and 60 business days after "
            "confirmation (per your appointment letter)."
        ),
        "ground_truth_sections": ["7.3", "7.4"],
    },
    {
        "question": "Who can I report harassment to?",
        "ground_truth": (
            "You can report to the HR department, your supervisor, the local "
            "Internal Complaints Committee (ICC), or the third-party reporting "
            "hotline named in the policy."
        ),
        "ground_truth_sections": ["2.9", "2.10"],
    },
    {
        "question": "Can I take leave during my notice period?",
        "ground_truth": (
            "No. Leave is generally not approved during the notice period; in "
            "exceptional cases an approval may extend the notice period accordingly."
        ),
        "ground_truth_sections": ["5.0", "7.3"],
    },
    {
        "question": "How long is maternity leave?",
        "ground_truth": (
            "26 weeks of paid maternity leave for an employee with up to one "
            "surviving child; 12 weeks if she already has two or more children."
        ),
        "ground_truth_sections": ["5.4"],
    },
    {
        "question": "What is the probation period?",
        "ground_truth": "180 days, extendable by any time off exceeding 5 working days.",
        "ground_truth_sections": ["3.7"],
    },
    {
        "question": "What changed in the 2026 handbook revision?",
        "ground_truth": (
            "Updates to CEO information, Teleworking/Work From Home, Loans, and "
            "Earned Leave, effective January 1, 2026."
        ),
        "ground_truth_sections": ["0.6"],
    },
    {
        "question": "How much is the employee referral reward and when is it paid?",
        "ground_truth": (
            "$500: 50% after the referred hire completes 3 months and the "
            "remainder after 6 months, subject to conditions."
        ),
        "ground_truth_sections": ["9.3"],
    },
    {
        "question": "What is the minimum service required for a personal loan from the company?",
        "ground_truth": (
            "3 years of continuous service; loans are at management discretion "
            "and only for marriage or medical needs."
        ),
        "ground_truth_sections": ["9.2"],
    },
    {
        "question": "How many hours must I work weekly and what are the normal business hours?",
        "ground_truth": (
            "40 hours per week (8 hours/day excluding breaks); normal business "
            "hours are 10am-7pm, Monday to Friday."
        ),
        "ground_truth_sections": ["4.1"],
    },
    {
        "question": "Can consultants take paid leave?",
        "ground_truth": "No. Consultants are not entitled to any paid leave or holidays.",
        "ground_truth_sections": ["5.0"],
    },
]


def validate_item(item: dict) -> None:
    """Raise ValueError if an item doesn't match the dataset schema."""
    for field in REQUIRED_FIELDS:
        if field not in item:
            raise ValueError(f"dataset item missing '{field}': {item!r}")
    if not isinstance(item["question"], str) or not item["question"].strip():
        raise ValueError(f"'question' must be a non-empty string: {item!r}")
    if not isinstance(item["ground_truth"], str) or not item["ground_truth"].strip():
        raise ValueError(f"'ground_truth' must be a non-empty string: {item!r}")
    secs = item["ground_truth_sections"]
    if not isinstance(secs, list) or not secs or not all(isinstance(s, str) for s in secs):
        raise ValueError(f"'ground_truth_sections' must be a non-empty list[str]: {item!r}")


def load_dataset(path: str) -> list[dict]:
    """Load and validate a JSONL dataset file."""
    items: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON ({e})") from e
            validate_item(item)
            items.append(item)
    return items


def save_dataset(path: str, items: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
