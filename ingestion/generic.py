"""Generic, multi-source ingestion — use this RAG system with YOUR OWN documents.

`ingestion/ingest.py` is tuned to one specific handbook PDF's layout (page roles,
numbered "N.M" headings, hand-mapped tables). THIS module is the general-purpose
path: point it at arbitrary documents and it produces the SAME `Chunk` schema, so
the retrieval / agents / API / UI layers all work unchanged.

Supported sources (auto-detected by extension, or forced with --source-type):
  * PDF                      (.pdf)            — per-page text, keeps page numbers
  * Plain text               (.txt)
  * Markdown                 (.md, .markdown)  — split into sections on '#' headings
  * HTML / Confluence export (.html, .htm, .xml) — split on <h1>..<h4> headings
  * A directory              — recurses and ingests every supported file

Confluence: use Space/Page tools > "Export to HTML" (or PDF) and point this at the
export. For live pages, fetch them (e.g. via the Atlassian MCP) and save as .html.

Examples
--------
    python -m ingestion.generic ./my_docs --collection my_docs --reset
    python -m ingestion.generic guide.pdf
    python -m ingestion.generic confluence_export.html --source-type confluence
    python -m ingestion.generic notes.md --dry-run
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path

from . import config
from .chunk import Chunk

# Extension -> normalized source type.
_EXT_TYPE = {
    ".pdf": "pdf",
    ".txt": "text",
    ".md": "markdown",
    ".markdown": "markdown",
    ".html": "html",
    ".htm": "html",
    ".xml": "html",  # Confluence "storage format" export is XML/HTML-ish
}
SUPPORTED_EXTS = tuple(_EXT_TYPE)


@dataclass
class Section:
    """A titled span of text from one document (page is set for PDFs only)."""

    title: str | None
    text: str
    page: int | None = None


@dataclass
class LoadedDoc:
    title: str             # human title (file stem, title-cased)
    source: str            # file basename, used in chunk ids/citations
    sections: list[Section] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Text normalization (light — NOT the handbook-specific cleaner)
# ---------------------------------------------------------------------------
def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.rstrip() for ln in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)        # collapse blank-line runs
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _title_from_path(path: Path) -> str:
    name = path.stem.replace("_", " ").replace("-", " ").strip()
    return re.sub(r"\s+", " ", name) or path.name


def _truncate(text: str, limit: int = 80) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


# ---------------------------------------------------------------------------
# Loaders: each returns a list[Section]
# ---------------------------------------------------------------------------
def _load_pdf(path: Path) -> list[Section]:
    from .extract import extract_pages  # local import: PyMuPDF is heavy

    sections: list[Section] = []
    for page in extract_pages(str(path)):
        body = normalize_text(page.text)
        if body:
            sections.append(Section(title=None, text=body, page=page.number))
    return sections


def _split_on_markdown_headings(text: str) -> list[Section]:
    sections: list[Section] = []
    cur_title: str | None = None
    buf: list[str] = []

    def flush() -> None:
        body = normalize_text("\n".join(buf))
        if body:
            sections.append(Section(title=cur_title, text=body))

    for line in text.split("\n"):
        m = re.match(r"^#{1,6}\s+(.*\S)\s*$", line)
        if m:
            flush()
            cur_title = _truncate(m.group(1))
            buf = []
        else:
            buf.append(line)
    flush()
    return sections or [Section(title=None, text=normalize_text(text))]


def _load_text(path: Path) -> list[Section]:
    return [Section(title=None, text=normalize_text(path.read_text(encoding="utf-8", errors="replace")))]


def _load_markdown(path: Path) -> list[Section]:
    return _split_on_markdown_headings(path.read_text(encoding="utf-8", errors="replace"))


class _HTMLSectioner(HTMLParser):
    """Extract visible text, starting a new Section at each <h1>..<h4>."""

    _HEADINGS = {"h1", "h2", "h3", "h4"}
    _SKIP = {"script", "style", "head", "nav", "footer"}
    _BLOCK = {"p", "div", "br", "li", "tr", "table", "section", "article",
              "h1", "h2", "h3", "h4", "h5", "h6"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.sections: list[Section] = []
        self._title: str | None = None
        self._buf: list[str] = []
        self._heading: list[str] | None = None
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skip += 1
        elif tag in self._HEADINGS:
            self._flush()
            self._heading = []
        elif tag in self._BLOCK:
            self._buf.append("\n")

    def handle_endtag(self, tag):
        if tag in self._SKIP and self._skip:
            self._skip -= 1
        elif tag in self._HEADINGS and self._heading is not None:
            self._title = _truncate("".join(self._heading)) or None
            self._heading = None

    def handle_data(self, data):
        if self._skip:
            return
        (self._heading if self._heading is not None else self._buf).append(data)

    def _flush(self) -> None:
        body = normalize_text("".join(self._buf))
        if body:
            self.sections.append(Section(title=self._title, text=body))
        self._buf = []

    def result(self) -> list[Section]:
        self._flush()
        return self.sections


def _load_html(path: Path) -> list[Section]:
    parser = _HTMLSectioner()
    parser.feed(path.read_text(encoding="utf-8", errors="replace"))
    sections = parser.result()
    return sections or [Section(title=None, text="")]


_LOADERS = {
    "pdf": _load_pdf,
    "text": _load_text,
    "markdown": _load_markdown,
    "html": _load_html,
}


def detect_type(path: Path) -> str | None:
    return _EXT_TYPE.get(path.suffix.lower())


def load_path(path: str, source_type: str = "auto") -> list[LoadedDoc]:
    """Load a file or directory into LoadedDocs (recurses directories)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"path not found: {path}")

    files: list[Path]
    if p.is_dir():
        files = sorted(f for f in p.rglob("*") if f.is_file() and detect_type(f))
        if not files:
            raise ValueError(f"no supported files ({', '.join(SUPPORTED_EXTS)}) under {path}")
    else:
        files = [p]

    docs: list[LoadedDoc] = []
    for f in files:
        stype = source_type if source_type != "auto" else (detect_type(f) or "text")
        if stype == "confluence":
            stype = "html"
        loader = _LOADERS.get(stype)
        if loader is None:
            raise ValueError(f"unsupported source type: {stype!r}")
        sections = loader(f)
        if sections:
            docs.append(LoadedDoc(title=_title_from_path(f), source=f.name, sections=sections))
    return docs


# ---------------------------------------------------------------------------
# Chunking (recursive splitter -> the shared Chunk schema)
# ---------------------------------------------------------------------------
def build_generic_chunks(
    docs: list[LoadedDoc], chunk_size: int = 1200, chunk_overlap: int = 150
) -> list[Chunk]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[Chunk] = []
    for doc_idx, doc in enumerate(docs, start=1):
        counter = 0
        for sec in doc.sections:
            for piece in splitter.split_text(sec.text):
                piece = piece.strip()
                if not piece:
                    continue
                counter += 1
                title = sec.title or f"{doc.title} (part {counter})"
                chunks.append(
                    Chunk(
                        section_number=f"{doc_idx}.{counter}",
                        section_title=_truncate(title, 80),
                        parent_section=f"{doc_idx}. {doc.title}",
                        pages=[sec.page] if sec.page else [],
                        source=doc.source,
                        body=piece,
                    )
                )
    return chunks


# ---------------------------------------------------------------------------
# Report / storage
# ---------------------------------------------------------------------------
def print_report(docs: list[LoadedDoc], chunks: list[Chunk]) -> None:
    from .chunk import estimate_tokens

    print("\n" + "=" * 70)
    print("GENERIC INGESTION REPORT")
    print("=" * 70)
    print(f"Documents:  {len(docs)}  ({', '.join(d.source for d in docs[:8])}"
          f"{', …' if len(docs) > 8 else ''})")
    print(f"Chunks:     {len(chunks)}")
    if chunks:
        toks = sorted(estimate_tokens(c.body) for c in chunks)
        print(f"Tokens/chunk: min={toks[0]}  median={toks[len(toks)//2]}  max={toks[-1]}")
        sample = chunks[0]
        print(f"\nSample chunk: {sample.section_number} — {sample.section_title}")
        print(f"  citation key: [{sample.section_number} {sample.section_title}]")
        print(f"  body: {sample.body[:160].strip()}…")
    print("=" * 70 + "\n")


def reset_collection(persist_dir: str, collection: str) -> None:
    import chromadb

    client = chromadb.PersistentClient(path=persist_dir)
    try:
        client.delete_collection(collection)
        print(f"Reset: dropped existing collection '{collection}'.")
    except Exception:
        pass  # didn't exist — fine


def embed_and_store(chunks: list[Chunk], provider, persist_dir: str, collection: str) -> None:
    from .embeddings import describe_provider, get_embedding_function
    from .store import get_collection, upsert_chunks

    print(f"Embedding with: {describe_provider(provider)}")
    ef = get_embedding_function(provider)
    coll = get_collection(persist_dir, collection, ef)
    n = upsert_chunks(coll, chunks)
    print(f"Upserted {n} chunks into collection '{collection}' at {persist_dir}")
    print(f"Collection now holds {coll.count()} documents.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="ingestion.generic",
        description="Generic multi-source ingestion (PDF / text / Markdown / HTML / Confluence).",
    )
    p.add_argument("path", help="File or directory to ingest.")
    p.add_argument(
        "--source-type",
        default="auto",
        choices=["auto", "pdf", "text", "markdown", "html", "confluence"],
        help="Force a source type (default: auto-detect by extension).",
    )
    p.add_argument("--collection", default=config.DEFAULT_COLLECTION,
                   help="ChromaDB collection name (point retrieval at the same name).")
    p.add_argument("--persist-dir", default=config.DEFAULT_PERSIST_DIR,
                   help="ChromaDB persistence directory.")
    p.add_argument("--provider", default=None,
                   help="Embedding provider override (local|openai). Defaults to .env / local.")
    p.add_argument("--chunk-size", type=int, default=1200, help="Max chunk size (characters).")
    p.add_argument("--chunk-overlap", type=int, default=150, help="Chunk overlap (characters).")
    p.add_argument("--reset", action="store_true",
                   help="Drop the collection before ingesting (use when switching documents).")
    p.add_argument("--dry-run", action="store_true",
                   help="Parse + chunk only; print the report and write chunks.json (no embeddings).")
    p.add_argument("--out", default="chunks.json", help="Where to write chunks.json (dry-run).")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        docs = load_path(args.path, args.source_type)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    chunks = build_generic_chunks(docs, args.chunk_size, args.chunk_overlap)
    if not chunks:
        print("ERROR: no text could be extracted from the input.", file=sys.stderr)
        return 1

    print_report(docs, chunks)

    if args.dry_run:
        Path(args.out).write_text(
            json.dumps([dataclasses.asdict(c) for c in chunks], indent=2, ensure_ascii=False)
        )
        print(f"Dry run: wrote {len(chunks)} chunks to {args.out}. No embeddings written.")
        return 0

    if args.reset:
        reset_collection(args.persist_dir, args.collection)
    embed_and_store(chunks, args.provider, args.persist_dir, args.collection)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
