"""
app/ingestion/__init__.py
=========================
Core ingestion layer for loading, chunking, and hybrid indexing.

Exports:
  - DocumentLoader: Wikipedia and HotpotQA document loader and golden set compiler.
  - QdrantHybridIndexer: Native dense + sparse hybrid vector indexer in Qdrant.
  - SparseEncoder: Lexical (BM25) and neural expansion (SPLADE) sparse encoders.
"""

from app.ingestion.indexing.dense import QdrantHybridIndexer
from app.ingestion.indexing.sparse import SparseEncoder
from app.ingestion.loader import DocumentLoader

__all__ = [
    "DocumentLoader",
    "QdrantHybridIndexer",
    "SparseEncoder",
]
