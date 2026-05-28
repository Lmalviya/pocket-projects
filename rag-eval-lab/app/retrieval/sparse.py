"""
app/retrieval/sparse.py
========================
Sparse vector retriever using LlamaIndex and lexical/learned keyword search in Qdrant.

📚 LESSON — Sparse Index Retrieval:
Sparse retrieval uses an inverted index to query a sparse vocabulary table.
  - When query_mode is set to `"sparse"`, LlamaIndex intercepts the lookup.
  - It triggers our custom `sparse_query_fn` callback, which converts the string
    query into a sparse vector `{token_ids: weights}` (via lexical BM25 or neural SPLADE).
  - Qdrant queries its sparse index in real-time, matching and scoring exact keywords
    (or expanded concepts in the case of SPLADE) with sub-millisecond latencies.
"""

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore

from app.retrieval.base import BaseRetriever
from app.tracing.langfuse import LangfuseTracer
from app.utils.logger import get_logger

logger = get_logger(__name__)


class SparseRetriever(BaseRetriever):
    """
    Retriever that queries Qdrant using sparse vectors (BM25 or SPLADE).
    """

    def __init__(
        self,
        index: VectorStoreIndex,
        top_k: int = 5,
        tracer: LangfuseTracer | None = None,
    ) -> None:
        """
        Initialize the sparse retriever.

        Args:
            index: Pluggable LlamaIndex VectorStoreIndex (connected to Qdrant).
            top_k: Number of candidate chunks to retrieve.
            tracer: Optional LangfuseTracer instance.
        """
        self._index = index
        self._top_k = top_k
        self._tracer = tracer

        # 📚 LESSON — LlamaIndex Sparse Query Mode:
        # We configure `vector_store_query_mode="sparse"`. This tells LlamaIndex to bypass
        # the dense HNSW index entirely, trigger the `sparse_query_fn` callback we supplied during
        # indexing, and execute a sparse-only search against Qdrant's inverted indexes.
        self._retriever = self._index.as_retriever(
            similarity_top_k=top_k,
            vector_store_query_mode="sparse",
        )

        logger.info("SparseRetriever initialized successfully with top_k={k}", k=top_k)

    def retrieve(self, query: str) -> list[NodeWithScore]:
        """
        Retrieve candidate chunks lexically matching the query.

        Args:
            query: User search query.

        Returns:
            List of LlamaIndex NodeWithScore objects.
        """
        logger.info("Executing sparse retrieval for query: '{query}'", query=query)

        span_ctx = (
            self._tracer.span(
                name="retrieval.sparse",
                input={"query": query, "top_k": self._top_k},
            )
            if self._tracer
            else None
        )

        try:
            results = self._retriever.retrieve(query)
            
            logger.info("Sparse retrieval finished. Retrieved {count} nodes.", count=len(results))

            if span_ctx:
                span_ctx.update(output={"retrieved_count": len(results)})

            return results

        except Exception as e:
            logger.error("Sparse retrieval lookup failed: {err}", err=str(e))
            if span_ctx:
                span_ctx.update(level="ERROR", status_message=str(e))
            raise

    @property
    def name(self) -> str:
        return "sparse"
