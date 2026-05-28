"""
app/ingestion/chunking/fixed.py
================================
Fixed-size chunking using LlamaIndex's SentenceSplitter.

📚 LESSON — Fixed-size Chunking
---------------------------------
This is the simplest and most common chunking strategy. The idea:
  "Give me chunks of exactly N tokens, with M tokens of overlap."

Strengths:
  ✅ Predictable, uniform chunk sizes → consistent embedding quality
  ✅ Fast — no ML model needed, pure text splitting
  ✅ Easy to reason about storage (N docs × avg_tokens / chunk_size ≈ node count)

Weaknesses:
  ❌ May split mid-sentence, mid-paragraph, or mid-concept
  ❌ Context boundary losses at chunk edges
  ❌ No semantic awareness (unrelated topics may land in the same chunk)

When to use:
  - Baseline experiments (always start here!)
  - Documents with uniform structure (legal, academic papers)
  - When you need speed over accuracy

📚 LESSON — Why SentenceSplitter instead of a naive character split?
---------------------------------------------------------------------
LlamaIndex's SentenceSplitter is "sentence-aware fixed" chunking:
  - It targets chunk_size tokens (using the model's tokenizer)
  - BUT it tries not to split in the middle of a sentence
  - It adds chunk_overlap tokens at the start of each chunk

This is slightly smarter than a naive "split every N characters" approach
because it respects sentence boundaries while still being size-limited.
"""

from __future__ import annotations

import time
from contextlib import nullcontext

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document

from app.ingestion.chunking.base import BaseChunker, ChunkingConfig, ChunkerResult
from app.tracing.langfuse import LangfuseTracer
from app.utils.logger import get_logger

logger = get_logger(__name__)


class FixedSizeChunker(BaseChunker):
    """
    Fixed-size chunker backed by LlamaIndex SentenceSplitter.

    The SentenceSplitter respects sentence boundaries while targeting
    a fixed token count per chunk. This is the recommended baseline.

    Example:
        config = ChunkingConfig(chunk_size=512, chunk_overlap=50)
        chunker = FixedSizeChunker(config)
        result = chunker.chunk(documents)
        print(f"Got {len(result.nodes)} nodes")
    """

    def __init__(
        self,
        config: ChunkingConfig,
        tracer: LangfuseTracer | None = None,
        trace=None,
    ) -> None:
        super().__init__(config, tracer, trace)

        # SentenceSplitter is LlamaIndex's built-in, sentence-aware text splitter.
        # chunk_size is in TOKENS (not characters).
        # chunk_overlap adds the last M tokens of the previous chunk to the start
        # of the next chunk — this is the "sliding window" effect.
        self._splitter = SentenceSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            paragraph_separator="\n\n",
        )

        logger.debug(
            "FixedSizeChunker ready: chunk_size={size}, overlap={overlap}",
            size=config.chunk_size,
            overlap=config.chunk_overlap,
        )

    @property
    def name(self) -> str:
        return "chunking.fixed"

    def chunk(self, documents: list[Document]) -> ChunkerResult:
        """
        Split documents into fixed-size nodes.

        📚 LESSON — What happens inside SentenceSplitter.get_nodes_from_documents():
          1. Each document is tokenized
          2. Tokens are grouped into windows of `chunk_size` with `chunk_overlap`
          3. Sentence boundaries are respected (won't split mid-sentence if possible)
          4. Each window becomes a TextNode with inherited metadata

        Args:
            documents: LlamaIndex Document objects to chunk.

        Returns:
            ChunkerResult with list of TextNode chunks.
        """
        start = time.perf_counter()

        # ── Langfuse v4 span (context manager) ──────────────────────────────
        # nullcontext() is a no-op context manager used when tracer is None,
        # so the `with` block works identically whether tracing is on or off.
        span_ctx = (
            self.tracer.span(
                self.name,
                input={"doc_count": len(documents), "chunk_size": self.config.chunk_size},
            )
            if self.tracer
            else nullcontext()
        )

        with span_ctx as span:
            # get_nodes_from_documents() is LlamaIndex's standard way to convert
            # Documents → TextNodes. It handles metadata inheritance automatically.
            nodes = self._splitter.get_nodes_from_documents(documents, show_progress=False)
            elapsed_ms = (time.perf_counter() - start) * 1000

            stats = self._log_chunk_stats(documents, nodes)
            stats["latency_ms"] = round(elapsed_ms, 2)

            # Update span with output data (v4 API: span.update())
            if span is not None:
                span.update(output={"node_count": len(nodes)}, metadata=stats)

        return ChunkerResult(nodes=nodes, strategy=self.name, metadata=stats)
