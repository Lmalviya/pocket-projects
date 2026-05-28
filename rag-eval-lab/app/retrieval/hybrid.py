"""
app/retrieval/hybrid.py
========================
Hybrid retriever fusing dense semantic and sparse lexical lookups via Reciprocal Rank Fusion.

📚 LESSON — Hybrid Retrieval & Client-Side Fusion:
To overcome the limitations of dense-only (which misses exact codes/names) and 
sparse-only (which misses synonyms/concepts) retrieval, we perform Hybrid Search.

We execute:
  1. Dense Semantic lookup to get `top_k * 2` candidate nodes.
  2. Sparse Lexical lookup to get `top_k * 2` candidate nodes.
  3. Python-side Reciprocal Rank Fusion (RRF) to merge the candidate lists,
     normalize their scoring hierarchies, and extract the top-K consolidated nodes.
     
Fusing on the client side ensures 100% control over the RRF smoothing constant `k`
and guarantees absolute compatibility across any Qdrant server version.
"""

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore

from app.retrieval.base import BaseRetriever
from app.retrieval.dense import DenseRetriever
from app.retrieval.sparse import SparseRetriever
from app.retrieval.utils import reciprocal_rank_fusion
from app.tracing.langfuse import LangfuseTracer
from app.utils.logger import get_logger

logger = get_logger(__name__)


class HybridRetriever(BaseRetriever):
    """
    Retriever that merges semantic (dense) and lexical (sparse) lookups using RRF.
    """

    def __init__(
        self,
        index: VectorStoreIndex,
        top_k: int = 20,
        rrf_k: int = 60,
        tracer: LangfuseTracer | None = None,
    ) -> None:
        """
        Initialize the hybrid retriever.

        Args:
            index: Pluggable LlamaIndex VectorStoreIndex (connected to Qdrant).
            top_k: Final consolidated candidates count.
            rrf_k: RRF rank smoothing constant (default: 60).
            tracer: Optional LangfuseTracer instance.
        """
        self._top_k = top_k
        self._rrf_k = rrf_k #hyper-parameter
        self._tracer = tracer

        # 📚 LESSON — Candidate Oversampling for Fusion:
        # When merging candidate lists, we fetch `top_k * 2` items from each individual
        # retriever (dense and sparse). This ensures that RRF has a rich overlapping pool of candidates
        # to fuse, preventing relevant nodes that rank slightly lower from being cut off before fusion.
        oversampled_k = top_k * 2
        self._dense_retriever = DenseRetriever(index, top_k=oversampled_k, tracer=None)
        self._sparse_retriever = SparseRetriever(index, top_k=oversampled_k, tracer=None)

        logger.info(
            "HybridRetriever initialized successfully: top_k={k}, rrf_k={rrf_k}, candidate_oversampling={oversample}",
            k=top_k,
            rrf_k=rrf_k,
            oversample=oversampled_k,
        )

    def retrieve(self, query: str) -> list[NodeWithScore]:
        """
        Retrieve fused dense and sparse candidates.

        Args:
            query: User search query.

        Returns:
            List of fused NodeWithScore objects sorted by RRF scores.
        """
        logger.info("Executing hybrid (dense + sparse) retrieval for query: '{query}'", query=query)

        span_ctx = (
            self._tracer.span(
                name="retrieval.hybrid",
                input={"query": query, "top_k": self._top_k, "rrf_k": self._rrf_k},
            )
            if self._tracer
            else None
        )

        try:
            # 1. Fetch dense and sparse candidates in parallel
            dense_results = self._dense_retriever.retrieve(query)
            sparse_results = self._sparse_retriever.retrieve(query)

            # 2. Fuse the results mathematically using RRF
            results = reciprocal_rank_fusion(
                results_list=[dense_results, sparse_results],
                k=self._rrf_k,
                top_k=self._top_k,
            )

            logger.info("Hybrid retrieval finished. Fused {count} nodes.", count=len(results))

            if span_ctx:
                span_ctx.update(output={"retrieved_count": len(results)})

            return results

        except Exception as e:
            logger.error("Hybrid retrieval lookup failed: {err}", err=str(e))
            if span_ctx:
                span_ctx.update(level="ERROR", status_message=str(e))
            raise

    @property
    def name(self) -> str:
        return "hybrid"
