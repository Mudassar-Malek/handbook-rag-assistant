"""ChromaDB persistence: a cosine-space collection with deterministic upserts."""

from __future__ import annotations

import chromadb

from . import config
from .chunk import Chunk


def get_collection(
    persist_dir: str = config.DEFAULT_PERSIST_DIR,
    collection_name: str = config.DEFAULT_COLLECTION,
    embedding_function=None,
):
    """Open (or create) the persistent cosine-space collection."""
    client = chromadb.PersistentClient(path=persist_dir)
    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"},
    )


def chunk_metadata(chunk: Chunk) -> dict:
    """Flatten a chunk into Chroma-safe metadata (scalars only).

    Chroma metadata can't hold lists, so the pages list is stringified and we
    also expose page_start/page_end as integers for convenient range filters.
    """
    return {
        "section_number": chunk.section_number,
        "section_title": chunk.section_title,
        "parent_section": chunk.parent_section,
        "pages": ",".join(str(p) for p in chunk.pages),
        "page_start": chunk.pages[0] if chunk.pages else -1,
        "page_end": chunk.pages[-1] if chunk.pages else -1,
        "part": chunk.part,
        "total_parts": chunk.total_parts,
        "source": chunk.source,
    }


def upsert_chunks(collection, chunks: list[Chunk], batch_size: int = 100) -> int:
    """Upsert chunks; deterministic ids mean re-ingestion replaces, not duplicates.

    We embed `embedding_text` (section path + body) but the display `body` is
    preserved in metadata so retrieval can show clean text.
    """
    total = 0
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        collection.upsert(
            ids=[c.id for c in batch],
            documents=[c.embedding_text for c in batch],
            metadatas=[{**chunk_metadata(c), "body": c.body} for c in batch],
        )
        total += len(batch)
    return total
