"""
app/retrieval/reranker/cross_encoder.py
=========================================
Local Cross-Encoder reranker using sentence-transformers.

📚 LESSON — Cross-Encoders vs Bi-Encoders:
  - Bi-Encoders (like our dense retriever) embed queries and documents INDEPENDENTLY.
    This is extremely fast for HNSW vector database lookups (takes microseconds), but
    since the query and document cannot interact during embedding, it misses fine-grained alignments.
  - Cross-Encoders take the query and document TOGETHER as a single combined input
    and pass them through all self-attention layers of a transformer (e.g. BERT).
    This allows full attention interaction between query words and document words,
    making it incredibly accurate, though much slower to compute (so we only run it
    on a small subset of candidate nodes, e.g. top 20).
"""

from sentence_transformers import CrossEncoder
from llama_index.core.schema import NodeWithScore

from app.retrieval.reranker.base import BaseReranker
from app.tracing.langfuse import LangfuseTracer
from app.utils.logger import get_logger

logger = get_logger(__name__)


class CrossEncoderReranker(BaseReranker):
    """
    Reranker using a local MS-MARCO Cross-Encoder model to re-score candidate nodes.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k: int = 5,
        tracer: LangfuseTracer | None = None,
    ) -> None:
        """
        Initialize the Cross-Encoder reranker.

        Args:
            model_name: Pretrained CrossEncoder model identifier.
            top_k: Number of final sorted nodes to return.
            tracer: Optional LangfuseTracer instance.
        """
        self._model_name = model_name
        self._top_k = top_k
        self._tracer = tracer

        logger.info("Loading Cross-Encoder model '{name}'...", name=model_name)
        try:
            self._model = CrossEncoder(model_name)
            logger.info("Cross-Encoder model loaded successfully.")
        except Exception as e:
            err_msg = (
                f"❌ Failed to load Cross-Encoder model '{model_name}'.\n"
                f"Reason: {str(e)}\n"
                f"Please check internet connection for initial weights download."
            )
            logger.error(err_msg)
            raise ConnectionError(err_msg) from e

    def rerank(self, query: str, nodes: list[NodeWithScore]) -> list[NodeWithScore]:
        """
        Re-scores and sorts candidate nodes using the Cross-Encoder.

        Args:
            query: The user search query.
            nodes: Initial retrieved candidates list.

        Returns:
            Re-sorted and sliced list of NodeWithScore objects.
        """
        if not nodes:
            logger.debug("Reranking bypassed - empty nodes list passed.")
            return []

        logger.info(
            "Reranking {count} candidates for query: '{query}' using local Cross-Encoder...",
            count=len(nodes),
            query=query,
        )

        span_ctx = (
            self._tracer.span(
                name="reranker.cross_encoder",
                input={"query": query, "node_count": len(nodes), "model": self._model_name},
            )
            if self._tracer
            else None
        )

        try:
            # 1. Format candidate inputs as [query, document_text] pairs for the transformer
            pairs = [[query, node.node.get_content()] for node in nodes]

            # 2. Predict relevance scores (unbounded log-likelihood outputs)
            scores = self._model.predict(pairs)

            # 3. Reconstruct NodeWithScore list with new scores
            reranked_nodes = []
            for node, score in zip(nodes, scores):
                reranked_nodes.append(
                    NodeWithScore(
                        node=node.node,
                        score=float(score),  # Map numpy float -> Python float
                    )
                )

            # 4. Sort candidates descending by Cross-Encoder scores
            reranked_nodes = sorted(reranked_nodes, key=lambda x: x.score, reverse=True)

            # 5. Slice to top-K
            results = reranked_nodes[:self._top_k]

            logger.info("Reranking complete. Selected top {top_k} candidates.", top_k=len(results))

            if span_ctx:
                span_ctx.update(output={"reranked_count": len(results)})

            return results

        except Exception as e:
            logger.error("Cross-Encoder reranking execution failed: {err}", err=str(e))
            if span_ctx:
                span_ctx.update(level="ERROR", status_message=str(e))
            raise

    @property
    def name(self) -> str:
        return "cross_encoder"
