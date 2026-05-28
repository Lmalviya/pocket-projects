"""
app/ingestion/chunking/semantic.py
====================================
Semantic chunking using embedding-based similarity breakpoints.

📚 LESSON — Semantic Chunking: The Most Intelligent Strategy
-------------------------------------------------------------
The previous strategies (fixed, recursive, sentence) split text based on
STRUCTURE: character count, sentence boundaries, paragraphs.

Semantic chunking splits text based on MEANING:
  "Split here because the topic just changed significantly."

Algorithm:
  1. Split text into base units (usually sentences)
  2. Embed each sentence using an embedding model
  3. Compute cosine similarity between consecutive sentence embeddings
  4. Find "breakpoints" — places where similarity drops sharply (= topic shift)
  5. Group sentences between breakpoints into chunks

Visual example:
  Sentence 1: "Paris is the capital of France." ──┐
  Sentence 2: "It has a population of 2.1M."    ──┤ SIMILAR TOPIC → same chunk
  Sentence 3: "The Eiffel Tower was built 1889." ──┘
  Similarity drops ↓↓↓ (BREAKPOINT DETECTED)
  Sentence 4: "Python is a programming language." ──┐ NEW CHUNK
  Sentence 5: "It was created by Guido van Rossum" ──┘

Strengths:
  ✅ Chunks represent coherent topics
  ✅ Retrieval quality is much higher — queries match complete ideas

Weaknesses:
  ❌ Requires embedding every sentence → 5-50x slower than structural approaches
  ❌ Chunk sizes vary dramatically

When to use:
  - High-quality offline indexing where latency doesn't matter
  - Long documents with multiple distinct topics

📚 LESSON — Breakpoint Types
-----------------------------
LangChain SemanticChunker supports three threshold methods:

  "percentile"         → split at positions where similarity is in the bottom X%
  "standard_deviation" → split where similarity drops more than Z std deviations
  "interquartile"      → robust to outliers (below Q1 - 1.5*IQR)

Default: "percentile" at 95th percentile.
"""

from __future__ import annotations

import time
from contextlib import nullcontext

from langchain_community.embeddings import OllamaEmbeddings
from langchain_experimental.text_splitter import SemanticChunker as LCSemanticChunker
from llama_index.core.schema import Document, TextNode

from app.config.settings import get_settings
from app.ingestion.chunking.base import BaseChunker, ChunkingConfig, ChunkerResult
from app.tracing.langfuse import LangfuseTracer
from app.utils.logger import get_logger

logger = get_logger(__name__)


class SemanticChunker(BaseChunker):
    """
    Embedding-based semantic chunker using LangChain SemanticChunker.

    Detects topic shifts by comparing consecutive sentence embeddings.
    Uses Ollama to embed sentences locally (BAAI/bge-small-en-v1.5).

    Note:
        chunk_size and chunk_overlap are kept for API consistency but
        semantic chunking ignores fixed sizes — it groups by topic coherence.
    """

    VALID_BREAKPOINT_TYPES = ("percentile", "standard_deviation", "interquartile")

    def __init__(
        self,
        config: ChunkingConfig,
        breakpoint_threshold_type: str = "percentile",
        breakpoint_threshold_amount: float = 95.0,
        tracer: LangfuseTracer | None = None,
        trace=None,
    ) -> None:
        super().__init__(config, tracer, trace)

        if breakpoint_threshold_type not in self.VALID_BREAKPOINT_TYPES:
            raise ValueError(
                f"breakpoint_threshold_type must be one of {self.VALID_BREAKPOINT_TYPES}"
            )

        self.breakpoint_threshold_type = breakpoint_threshold_type
        self.breakpoint_threshold_amount = breakpoint_threshold_amount

        settings = get_settings()

        # 📚 LESSON — OllamaEmbeddings from LangChain Community is used here
        # because SemanticChunker is a LangChain class requiring a LangChain
        # embeddings interface. We run the same bge-small model via Ollama.
        self._embed_model = OllamaEmbeddings(
            model=settings.ollama_embed_model,
            base_url=settings.ollama_base_url,
        )

        self._splitter = LCSemanticChunker(
            embeddings=self._embed_model,
            breakpoint_threshold_type=breakpoint_threshold_type,
            breakpoint_threshold_amount=breakpoint_threshold_amount,
        )

        logger.debug(
            "SemanticChunker ready: model={model}, threshold={type}@{amount}",
            model=settings.ollama_embed_model,
            type=breakpoint_threshold_type,
            amount=breakpoint_threshold_amount,
        )

    @property
    def name(self) -> str:
        return "chunking.semantic"

    def chunk(self, documents: list[Document]) -> ChunkerResult:
        """
        Split documents based on semantic topic shifts.

        ⚠️  This is the SLOWEST chunker — it embeds every sentence.
        Expect 10-60 seconds for 10 Wikipedia articles. Quality improvement
        often justifies the wait for offline indexing.

        Args:
            documents: LlamaIndex Document objects.

        Returns:
            ChunkerResult with topic-coherent TextNodes.
        """
        logger.info(
            "SemanticChunker starting (slow — embeds every sentence)... {n} docs",
            n=len(documents),
        )

        start = time.perf_counter()

        span_ctx = (
            self.tracer.span(
                self.name,
                input={
                    "doc_count": len(documents),
                    "breakpoint_type": self.breakpoint_threshold_type,
                    "threshold_amount": self.breakpoint_threshold_amount,
                },
            )
            if self.tracer
            else nullcontext()
        )

        with span_ctx as span:
            nodes: list[TextNode] = []

            for doc_idx, doc in enumerate(documents):
                doc_text = doc.get_content()
                if not doc_text.strip():
                    continue

                try:
                    # SemanticChunker.create_documents() embeds every sentence
                    # then groups by topic similarity. Returns LangChain Documents.
                    lc_docs = self._splitter.create_documents([doc_text])
                except Exception as e:
                    # Graceful degradation: fall back to newline split
                    logger.warning(
                        "Semantic chunking failed for doc {idx} (falling back): {err}",
                        idx=doc_idx,
                        err=str(e),
                    )
                    lc_docs = [
                        type("FB", (), {"page_content": t})()
                        for t in doc_text.split("\n\n")
                        if t.strip()
                    ]

                # Convert LangChain Document → LlamaIndex TextNode
                for chunk_idx, lc_doc in enumerate(lc_docs):
                    node = self._make_text_node(
                        text=lc_doc.page_content,
                        metadata={
                            **doc.metadata,
                            "doc_index": doc_idx,
                            "breakpoint_type": self.breakpoint_threshold_type,
                        },
                        chunk_index=chunk_idx,
                    )
                    nodes.append(node)

            elapsed_ms = (time.perf_counter() - start) * 1000
            stats = self._log_chunk_stats(documents, nodes)
            stats["latency_ms"] = round(elapsed_ms, 2)
            stats["breakpoint_type"] = self.breakpoint_threshold_type
            stats["threshold_amount"] = self.breakpoint_threshold_amount

            if span is not None:
                span.update(output={"node_count": len(nodes)}, metadata=stats)

        return ChunkerResult(nodes=nodes, strategy=self.name, metadata=stats)
