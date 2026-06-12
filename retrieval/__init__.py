"""Phase 2 retrieval layer for the Employee Handbook RAG project.

A hybrid retriever that combines dense vector search and BM25 keyword search,
fuses them with Reciprocal Rank Fusion, optionally reranks with a cross-encoder,
and applies document-specific post-processing (cross-reference expansion,
multi-part stitching, sibling hints).

Entry point: `python -m retrieval.search "<query>" --k 5 --debug`.
"""

__version__ = "0.1.0"
