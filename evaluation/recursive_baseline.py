"""Strategy B: a naive RecursiveCharacterTextSplitter baseline collection.

Reuses Phase 1's extraction + cleaning (only the SPLITTING differs), then splits
the continuous cleaned text into 1000-token chunks with 150-token overlap and
ingests them into a separate collection with PAGE metadata. Cross-page chunks
keep the set of pages they overlap, which is exactly what lets naive chunks
bleed one section into another.

Note: the Phase 2 loader requires section_number/title/parent fields, so we set
synthetic placeholders (section_number="R###", empty title/parent). The only
*real* provenance for Strategy B is the page set — as the spec requires.
"""

from __future__ import annotations

from ingestion import config as ing_config
from ingestion.clean import clean_page
from ingestion.embeddings import get_embedding_function
from ingestion.extract import extract_pages
from ingestion.store import get_collection


def _content_pages(pdf_path: str) -> list[tuple[int, str]]:
    """Cleaned text for every non-skipped page (front matter + policy body)."""
    pages = []
    for raw in extract_pages(pdf_path):
        if raw.number in ing_config.SKIP_PAGES:
            continue
        text = clean_page(raw.text)
        if text.strip():
            pages.append((raw.number, text))
    return pages


def _build_text_and_spans(pages: list[tuple[int, str]]) -> tuple[str, list[tuple[int, int, int]]]:
    """Concatenate page texts; return (full_text, [(start_char, end_char, page)])."""
    parts: list[str] = []
    spans: list[tuple[int, int, int]] = []
    cursor = 0
    sep = "\n\n"
    for page, text in pages:
        start = cursor
        parts.append(text)
        cursor += len(text)
        spans.append((start, cursor, page))
        parts.append(sep)
        cursor += len(sep)
    return "".join(parts), spans


def _pages_for_span(start: int, end: int, spans: list[tuple[int, int, int]]) -> list[int]:
    return sorted({pg for (ps, pe, pg) in spans if not (pe <= start or ps >= end)})


def build_recursive_collection(settings, verbose: bool = True) -> int:
    """(Re)build the Strategy B collection. Returns the number of chunks ingested."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    pages = _content_pages(settings.pdf_path)
    full_text, spans = _build_text_and_spans(pages)

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=settings.recursive_chunk_size,
        chunk_overlap=settings.recursive_chunk_overlap,
        strip_whitespace=False,  # keep exact substrings so we can map char offsets
    )
    chunks = splitter.split_text(full_text)

    collection = get_collection(
        persist_dir="chroma_db",
        collection_name=settings.recursive_collection,
        embedding_function=get_embedding_function(),
    )
    # Start clean so re-runs don't accumulate stale chunks.
    try:
        collection.delete(where={})
    except Exception:
        pass

    ids, documents, metadatas = [], [], []
    cursor = 0
    for idx, chunk in enumerate(chunks):
        loc = full_text.find(chunk, max(0, cursor - settings.recursive_chunk_overlap * 6))
        if loc < 0:
            loc = full_text.find(chunk)
        start = loc if loc >= 0 else cursor
        end = start + len(chunk)
        cursor = end
        pgs = _pages_for_span(start, end, spans)
        text = chunk.strip()
        if not text:
            continue
        ids.append(f"recursive-{idx:04d}")
        documents.append(text)
        metadatas.append({
            "section_number": f"R{idx:03d}",  # placeholder (loader requires it)
            "section_title": "",
            "parent_section": "",
            "pages": ",".join(str(p) for p in pgs),
            "page_start": pgs[0] if pgs else -1,
            "page_end": pgs[-1] if pgs else -1,
            "part": 1,
            "total_parts": 1,
            "source": "recursive_baseline",
            "body": text,
        })

    for i in range(0, len(ids), 100):
        collection.upsert(
            ids=ids[i:i + 100], documents=documents[i:i + 100], metadatas=metadatas[i:i + 100]
        )
    if verbose:
        print(f"Strategy B: ingested {len(ids)} recursive chunks into "
              f"'{settings.recursive_collection}' (size={settings.recursive_chunk_size}, "
              f"overlap={settings.recursive_chunk_overlap} tokens).")
    return len(ids)


def collection_count(settings) -> int:
    try:
        c = get_collection(persist_dir="chroma_db", collection_name=settings.recursive_collection)
        return c.count()
    except Exception:
        return 0
