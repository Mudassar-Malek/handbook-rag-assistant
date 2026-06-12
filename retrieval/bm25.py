"""BM25 keyword search over the same chunks, with query synonym expansion.

Why BM25 alongside vectors? Embeddings are great at meaning but can miss exact
short tokens users type verbatim — "LOP", "EL", "comp-off", "POSH", "ESS",
"AcmeApp", "PayApp". BM25 is a lexical match, so it nails those even when
the embedding similarity is mediocre.
"""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from .base import Hit
from .loader import ChunkStore

# Lowercase alphanumeric tokens. Splitting on non-alphanumerics means "comp-off"
# and "Comp Off" both tokenize to ["comp", "off"], and "HRMS/PayApp" becomes
# ["hrms", "payapp"], so verbatim terms match regardless of punctuation.
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


# Query-side synonym/expansion map. Keys are matched as whole words/phrases in
# the (lowercased) query; the value is appended before tokenizing for BM25.
# This bridges the gap between how people ask and how the handbook is worded.
SYNONYMS: dict[str, str] = {
    "wfh": "work from home teleworking",
    "work from home": "teleworking",
    "pto": "earned leave EL vacation",
    "vacation": "earned leave EL",
    "annual leave": "earned leave EL",
    "sick day": "sick leave SL",
    "sick days": "sick leave SL",
    "quit": "resignation separation notice period",
    "resign": "resignation separation notice period",
    "resignation": "separation notice period",
    "fired": "termination",
    "referral bonus": "employee referral",
    "harassment complaint": "POSH ICC reporting harassment",
    # Colloquial term whose canonical form was glued during extraction.
    "comp off": "compensatory off",
    "comp-off": "compensatory off",
    "compoff": "compensatory off",
}


def expand_query(query: str) -> str:
    """Append synonym expansions for any matched key (original query kept)."""
    low = query.lower()
    extra: list[str] = []
    for key, expansion in SYNONYMS.items():
        if re.search(rf"\b{re.escape(key)}\b", low):
            extra.append(expansion)
    return query + (" " + " ".join(extra) if extra else "")


class BM25Search:
    """A BM25Okapi index built over `section_title + body` of every chunk."""

    name = "bm25"

    def __init__(self, store: ChunkStore):
        self.store = store
        self.ids = [c.id for c in store.chunks]
        # Index title + display body so section titles ("Compensatory Off")
        # contribute strong keyword signal.
        corpus = [tokenize(f"{c.section_title} {c.body}") for c in store.chunks]
        self.bm25 = BM25Okapi(corpus)

    def search(self, query: str, n: int) -> list[Hit]:
        tokens = tokenize(expand_query(query))
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(zip(self.ids, scores), key=lambda kv: kv[1], reverse=True)
        # Drop zero-score hits (no lexical overlap at all).
        return [(cid, float(score)) for cid, score in ranked[:n] if score > 0.0]
