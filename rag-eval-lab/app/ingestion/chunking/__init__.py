"""
app/ingestion/chunking/__init__.py
Clean public API for the chunking package.
"""
from app.ingestion.chunking.base import BaseChunker, ChunkingConfig, ChunkerResult
from app.ingestion.chunking.factory import CHUNKER_REGISTRY, get_chunker, list_strategies
from app.ingestion.chunking.fixed import FixedSizeChunker
from app.ingestion.chunking.recursive import RecursiveChunker
from app.ingestion.chunking.semantic import SemanticChunker
from app.ingestion.chunking.sentence import SentenceChunker

__all__ = [
    "BaseChunker",
    "ChunkingConfig",
    "ChunkerResult",
    "FixedSizeChunker",
    "RecursiveChunker",
    "SentenceChunker",
    "SemanticChunker",
    "get_chunker",
    "list_strategies",
    "CHUNKER_REGISTRY",
]
