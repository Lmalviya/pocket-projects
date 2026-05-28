"""
app/ingestion/chunking/sentence.py
====================================
Sentence-aware chunking using LlamaIndex's SentenceSplitter in sentence mode.

📚 LESSON — Sentence Chunking vs Fixed-Size Chunking
------------------------------------------------------
Both use LlamaIndex's SentenceSplitter under the hood! The difference is in
what we're OPTIMIZING for:

  FixedSizeChunker:
    - Optimizes for uniform token count per chunk
    - May split a paragraph mid-sentence if needed to hit the size target
    - Good for: consistent embedding quality, predictable index size

  SentenceChunker:
    - Optimizes for complete sentences within each chunk
    - Detects sentence boundaries using NLTK's punkt tokenizer
    - Chunks are "complete thought" units: always end at a sentence boundary
    - Chunk sizes VARY — some sentences are 10 tokens, some are 80 tokens
    - Good for: question answering (answers often fit within 1-2 sentences)

📚 LESSON — Why complete sentences matter for RAG quality
----------------------------------------------------------
When an LLM reads retrieved context, truncated sentences cause confusion:
  ❌ "The capital of France, which has been the center of..."  ← cut off mid-thought
  ✅ "The capital of France is Paris."                         ← complete thought

Sentence chunking ensures the LLM always reads complete, coherent units.
The trade-off: your chunks vary in size, so some vectors represent tiny
chunks (low information density) and others represent very large chunks
(too broad for precise retrieval).

NLTK punkt tokenizer:
  - Pre-trained sentence boundary detector for English
  - Handles abbreviations (Dr., U.S.A.) correctly
  - Download once:  python -c "import nltk; nltk.download('punkt_tab')"
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


class SentenceChunker(BaseChunker):
    """
    Sentence-boundary-aware chunker.

    Produces chunks that always end at sentence boundaries. Chunk sizes vary
    but are bounded by chunk_size tokens. Uses NLTK punkt for sentence detection.

    Example:
        config = ChunkingConfig(chunk_size=256, chunk_overlap=30)
        chunker = SentenceChunker(config)
        result = chunker.chunk(documents)
    """

    def __init__(
        self,
        config: ChunkingConfig,
        tracer: LangfuseTracer | None = None,
        trace=None,
    ) -> None:
        super().__init__(config, tracer, trace)

        self._splitter = SentenceSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            paragraph_separator="\n\n",
            secondary_chunking_regex=r"[^,.;。？！]+[,.;。？！]?",
        )

        logger.debug(
            "SentenceChunker ready: chunk_size={size}, overlap={overlap}",
            size=config.chunk_size,
            overlap=config.chunk_overlap,
        )

    @property
    def name(self) -> str:
        return "chunking.sentence"

    def chunk(self, documents: list[Document]) -> ChunkerResult:
        """
        Split documents into sentence-boundary-respecting chunks.

        Args:
            documents: LlamaIndex Document objects.

        Returns:
            ChunkerResult with TextNodes — each ending at a sentence boundary.
        """
        start = time.perf_counter()

        span_ctx = (
            self.tracer.span(self.name, input={"doc_count": len(documents)})
            if self.tracer
            else nullcontext()
        )

        with span_ctx as span:
            nodes = self._splitter.get_nodes_from_documents(documents, show_progress=False)
            elapsed_ms = (time.perf_counter() - start) * 1000

            stats = self._log_chunk_stats(documents, nodes)
            stats["latency_ms"] = round(elapsed_ms, 2)

            # Calculate size variance — useful diagnostic for sentence chunking
            if nodes:
                lengths = [len(n.text) for n in nodes]
                stats["min_node_length"] = min(lengths)
                stats["max_node_length"] = max(lengths)
                stats["size_variance"] = round(max(lengths) - min(lengths), 1)

            if span is not None:
                span.update(output={"node_count": len(nodes)}, metadata=stats)

        return ChunkerResult(nodes=nodes, strategy=self.name, metadata=stats)
