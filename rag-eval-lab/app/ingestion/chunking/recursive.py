"""
app/ingestion/chunking/recursive.py
=====================================
Recursive character text splitting using LangChain.

📚 LESSON — Recursive Character Text Splitting
------------------------------------------------
This is LangChain's most popular splitter. The word "recursive" describes:

  1. Try to split on "\n\n" (paragraphs) first
  2. If any chunk is still too large, split on "\n" (lines)
  3. If still too large, split on ". " (sentences)
  4. If still too large, split on " " (words)
  5. As a last resort, split on "" (characters)

This hierarchy means the splitter PREFERS natural text boundaries and only
falls back to harder cuts when necessary. Result: chunks that are semantically
more coherent than simple fixed-size splitting.

Strengths:
  ✅ Respects text structure (paragraphs > lines > sentences > words)
  ✅ Works on any language/format (no NLP model needed)
  ✅ Still fast — pure string operations
  ✅ Great default for unstructured text (web pages, articles)

Weaknesses:
  ❌ Still no semantic understanding — just structural heuristics
  ❌ Very short paragraphs may be joined into odd combinations

When to use:
  - General-purpose baseline
  - Mixed-structure documents (web scrapes, PDFs)
  - Documents with clear paragraph structure

📚 LESSON — LangChain → LlamaIndex Bridge
-------------------------------------------
LangChain's RecursiveCharacterTextSplitter returns LangChain `Document`
objects. LlamaIndex's pipeline expects `TextNode` objects. We convert between
them here so the rest of the pipeline is unaware of LangChain internals.

The conversion is straightforward:
  LangChain Document.page_content → TextNode.text
  LangChain Document.metadata    → TextNode.metadata
"""

from __future__ import annotations

import time
from contextlib import nullcontext

from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_index.core.schema import Document, TextNode

from app.ingestion.chunking.base import BaseChunker, ChunkingConfig, ChunkerResult
from app.tracing.langfuse import LangfuseTracer
from app.utils.logger import get_logger

logger = get_logger(__name__)


class RecursiveChunker(BaseChunker):
    """
    Recursive character text splitter (LangChain) adapted for LlamaIndex.

    Uses LangChain's RecursiveCharacterTextSplitter internally, then
    converts the output to LlamaIndex TextNodes for pipeline compatibility.

    Example:
        config = ChunkingConfig(chunk_size=512, chunk_overlap=50)
        chunker = RecursiveChunker(config)
        result = chunker.chunk(documents)
    """

    # 📚 LESSON — These separators are tried IN ORDER from first to last.
    # The splitter keeps trying the next separator until chunks are small enough.
    DEFAULT_SEPARATORS = [
        "\n\n",   # paragraph break (most preferred)
        "\n",     # line break
        ". ",     # sentence boundary (period + space)
        "! ",     # exclamation sentence end
        "? ",     # question sentence end
        "; ",     # semicolon clause break
        ", ",     # comma clause break
        " ",      # word break
        "",       # character break (last resort)
    ]

    def __init__(
        self,
        config: ChunkingConfig,
        separators: list[str] | None = None,
        tracer: LangfuseTracer | None = None,
        trace=None,
    ) -> None:
        super().__init__(config, tracer, trace)

        # chunk_size here is in CHARACTERS not tokens, unlike LlamaIndex splitters.
        # A rough approximation: 1 token ≈ 4 characters for English text.
        char_chunk_size = config.chunk_size * 4
        char_chunk_overlap = config.chunk_overlap * 4

        self._splitter = RecursiveCharacterTextSplitter(
            separators=separators or self.DEFAULT_SEPARATORS,
            chunk_size=char_chunk_size,
            chunk_overlap=char_chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

        logger.debug(
            "RecursiveChunker ready: char_size={size}, overlap={overlap}",
            size=char_chunk_size,
            overlap=char_chunk_overlap,
        )

    @property
    def name(self) -> str:
        return "chunking.recursive"

    def chunk(self, documents: list[Document]) -> ChunkerResult:
        """
        Split documents using recursive character splitting.

        Args:
            documents: LlamaIndex Document objects to chunk.

        Returns:
            ChunkerResult with LlamaIndex TextNodes.
        """
        start = time.perf_counter()

        span_ctx = (
            self.tracer.span(
                self.name,
                input={"doc_count": len(documents)},
                metadata={"separators": self._splitter._separators[:4]},  # first 4 only
            )
            if self.tracer
            else nullcontext()
        )

        with span_ctx as span:
            nodes: list[TextNode] = []

            for doc_idx, doc in enumerate(documents):
                # split_text() returns a list of plain strings
                chunks: list[str] = self._splitter.split_text(doc.get_content())

                for chunk_idx, chunk_text in enumerate(chunks):
                    # Bridge: convert plain string → LlamaIndex TextNode
                    node = self._make_text_node(
                        text=chunk_text,
                        metadata={**doc.metadata, "doc_index": doc_idx},
                        chunk_index=chunk_idx,
                    )
                    nodes.append(node)

            elapsed_ms = (time.perf_counter() - start) * 1000
            stats = self._log_chunk_stats(documents, nodes)
            stats["latency_ms"] = round(elapsed_ms, 2)

            if span is not None:
                span.update(output={"node_count": len(nodes)}, metadata=stats)

        return ChunkerResult(nodes=nodes, strategy=self.name, metadata=stats)
