"""Phase 1 ingestion pipeline for the Employee Handbook 2026.

Parse -> clean -> heading-aware chunk -> embed -> upsert into ChromaDB.

See README.md for the full design and the `ingest` module for the CLI.
"""

__version__ = "0.1.0"
