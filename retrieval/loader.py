"""Load all chunks from ChromaDB once and index them for fast lookup.

ChromaDB is the single source of truth: BM25, cross-reference expansion, and
multi-part stitching all read from the same in-memory chunk set loaded here.
We never re-read the PDF or chunks.json.
"""

from __future__ import annotations

from collections import defaultdict

from .types import Chunk, parse_pages


class ChunkStore:
    """In-memory view of every chunk in the collection, with lookups."""

    def __init__(self, collection):
        raw = collection.get(include=["documents", "metadatas"])
        self.chunks: list[Chunk] = []
        self._by_id: dict[str, Chunk] = {}
        self._by_section: dict[str, list[Chunk]] = defaultdict(list)
        self._by_parent: dict[str, list[Chunk]] = defaultdict(list)

        for cid, doc, meta in zip(raw["ids"], raw["documents"], raw["metadatas"]):
            chunk = Chunk(
                id=cid,
                document=doc,
                body=meta.get("body", ""),
                section_number=meta["section_number"],
                section_title=meta["section_title"],
                parent_section=meta["parent_section"],
                pages=parse_pages(meta.get("pages", "")),
                part=int(meta.get("part", 1)),
                total_parts=int(meta.get("total_parts", 1)),
                source=meta.get("source", ""),
            )
            self.chunks.append(chunk)
            self._by_id[cid] = chunk
            self._by_section[chunk.section_number].append(chunk)
            self._by_parent[chunk.parent_section].append(chunk)

        # Keep multi-part sections in part order so stitching is deterministic.
        for parts in self._by_section.values():
            parts.sort(key=lambda c: c.part)

    def __len__(self) -> int:
        return len(self.chunks)

    def get(self, chunk_id: str) -> Chunk | None:
        return self._by_id.get(chunk_id)

    def get_section(self, section_number: str) -> list[Chunk]:
        """All chunks (parts) for a section number, in part order."""
        return list(self._by_section.get(section_number, []))

    def siblings(self, parent_section: str, exclude_section: str) -> list[tuple[str, str]]:
        """Distinct (section_number, title) under the same parent, minus self."""
        seen: dict[str, str] = {}
        for chunk in self._by_parent.get(parent_section, []):
            if chunk.section_number == exclude_section:
                continue
            seen.setdefault(chunk.section_number, chunk.section_title)
        return sorted(seen.items(), key=lambda kv: _num_key(kv[0]))


def _num_key(section_number: str) -> tuple[int, ...]:
    try:
        return tuple(int(p) for p in section_number.split("."))
    except ValueError:
        return (999,)
