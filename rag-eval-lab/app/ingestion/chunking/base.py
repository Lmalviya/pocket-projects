"""
app/ingestion/chunking/base.py
===============================
Abstract base class for all chunking strategies.

📚 LESSON — Why Abstract Base Classes (ABCs)?
----------------------------------------------
We have 4 chunking strategies (fixed, recursive, semantic, sentence).
Each one is a different class with different internals.

BUT — the rest of the pipeline (indexer, pipeline orchestrator) doesn't care
WHICH chunker is being used. It just needs to:
  1. Give it some documents
  2. Get back TextNodes

An ABC enforces this contract: any class that extends BaseChunker MUST
implement the `chunk()` method, or Python raises a TypeError at import time.

This is the "Liskov Substitution Principle" in practice:
  Any BaseChunker subclass can replace any other without breaking the pipeline.

📚 LESSON — LlamaIndex TextNode vs LangChain Document
------------------------------------------------------
LlamaIndex works with TextNode objects:
  TextNode(text="...", metadata={"source": "wiki", "title": "RAG"})

LangChain works with Document objects:
  Document(page_content="...", metadata={"source": "wiki"})

Our chunkers accept LlamaIndex Documents as INPUT (from the loader) and
return LlamaIndex TextNodes as OUTPUT (for the indexer). When we use
LangChain splitters internally (recursive, semantic), we convert back to
TextNodes before returning — the rest of the pipeline only speaks LlamaIndex.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from llama_index.core.schema import Document, TextNode

from app.tracing.langfuse import LangfuseTracer
from app.utils.logger import get_logger

# NOTE: chunkers use tracer.span() context manager directly (Langfuse v4 API)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class ChunkingConfig:
    """
    Parameters that control how documents are split into chunks.

    📚 LESSON — chunk_size vs chunk_overlap:
      Imagine a book page. If chunk_size=512 tokens:
        Chunk 1: tokens 0-511
        Chunk 2: tokens 462-973   ← starts 50 tokens before chunk 1 ended
        Chunk 3: tokens 924-1435  ← etc.

      The overlap (50 tokens) ensures that context isn't lost at boundaries.
      A sentence like "This relates to the previous point..." won't lose
      "the previous point" just because it falls at a chunk boundary.

      Trade-off:
        Larger overlap → less information loss, but more redundant chunks
        Smaller overlap → more efficient storage, but potential boundary issues
    """
    chunk_size: int = 512          # max tokens per chunk
    chunk_overlap: int = 50        # token overlap between consecutive chunks
    strategy: str = "fixed"        # which strategy produced these chunks


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ChunkerResult:
    """
    Output of a chunking operation.

    Wrapping the result in a dataclass (instead of returning a bare list)
    makes it easy to add metadata without changing function signatures.
    """
    nodes: list[TextNode]           # the actual chunks as LlamaIndex TextNodes
    strategy: str = "unknown"       # which chunker produced this
    metadata: dict[str, Any] = field(default_factory=dict)
    # e.g., {"doc_count": 5, "node_count": 47, "avg_node_length": 312}


# ---------------------------------------------------------------------------
# Abstract Base Class
# ---------------------------------------------------------------------------

class BaseChunker(ABC):
    """
    Abstract interface that every chunking strategy must implement.

    Subclasses must implement:
      - chunk(documents) → ChunkerResult
      - name (property) → str

    Subclasses inherit:
      - tracer / _trace: Langfuse tracing support
      - _make_text_node(): helper to create LlamaIndex TextNodes consistently
    """

    def __init__(
        self,
        config: ChunkingConfig,
        tracer: LangfuseTracer | None = None,
        trace=None,
    ) -> None:
        """
        Args:
            config: Chunking parameters (size, overlap, strategy name).
            tracer: Optional LangfuseTracer for span tracking.
            trace: Optional active Langfuse trace to attach spans to.
        """
        self.config = config
        self.tracer = tracer
        self._trace = trace

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Return the canonical name of this chunking strategy.
        Used as the Langfuse span name: e.g., "chunking.fixed".
        """
        ...

    @abstractmethod
    def chunk(self, documents: list[Document]) -> ChunkerResult:
        """
        Split a list of LlamaIndex Documents into TextNodes.

        This is the core contract. Every chunker MUST:
          1. Accept a list of LlamaIndex Document objects
          2. Return a ChunkerResult with a list of TextNodes

        📚 LESSON — What is a TextNode?
          A TextNode is LlamaIndex's unit of indexed content. It contains:
            - text: The actual chunk text
            - metadata: Dict with source info (title, url, chunk_index, etc.)
            - id_: A unique identifier (auto-generated UUID)
            - embedding: The vector (set later by the indexer, not here)

        Args:
            documents: List of LlamaIndex Document objects to chunk.

        Returns:
            ChunkerResult with the list of TextNodes and metadata.
        """
        ...

    def set_trace(self, trace) -> None:
        """
        Attach an active Langfuse trace to this chunker.

        Called by the pipeline before invoking chunk(), so spans created
        inside chunk() are correctly parented to the current query's trace.

        Args:
            trace: Active Langfuse trace object.
        """
        self._trace = trace

    @staticmethod
    def _make_text_node(
        text: str,
        metadata: dict[str, Any],
        chunk_index: int = 0,
    ) -> TextNode:
        """
        Helper to create a LlamaIndex TextNode with consistent metadata.

        📚 LESSON — Why a static helper?
          All chunkers need to create TextNodes. Instead of each one
          doing it differently (and potentially forgetting important metadata
          fields), this helper guarantees a consistent format.

        Args:
            text: The chunk text.
            metadata: Source document metadata (title, source, url, etc.).
            chunk_index: Position of this chunk within its source document.

        Returns:
            A LlamaIndex TextNode ready for indexing.
        """
        node_metadata = {
            **metadata,             # inherit all source document metadata
            "chunk_index": chunk_index,
        }
        return TextNode(text=text.strip(), metadata=node_metadata)

    def _log_chunk_stats(self, documents: list[Document], nodes: list[TextNode]) -> dict:
        """
        Log chunking statistics and return them as a metadata dict.

        Args:
            documents: Input documents.
            nodes: Output nodes.

        Returns:
            Dict with stats suitable for Langfuse span metadata.
        """
        if not nodes:
            return {}

        avg_len = sum(len(n.text) for n in nodes) / len(nodes)
        stats = {
            "doc_count": len(documents),
            "node_count": len(nodes),
            "avg_node_length_chars": round(avg_len, 1),
            "strategy": self.name,
        }

        logger.info(
            "[{strategy}] {doc_count} docs → {node_count} nodes (avg {avg:.0f} chars)",
            strategy=self.name,
            doc_count=stats["doc_count"],
            node_count=stats["node_count"],
            avg=avg_len,
        )

        return stats
