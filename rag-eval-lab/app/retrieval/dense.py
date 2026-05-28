"""
app/retrieval/dense.py
=======================
Dense vector retriever using LlamaIndex and cosine semantic similarity in Qdrant.

📚 LESSON — Dense Vector Retrieval:
Dense retrieval transforms the user's string query into a semantic embedding vector 
(using Ollama's BAAI/bge-small-en-v1.5) and searches our Qdrant HNSW vector space.
It calculates similarity using cosine distance to identify chunks that share
conceptual meaning, regardless of whether they share any exact word spellings.
"""

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore

from app.retrieval.base import BaseRetriever
from app.tracing.langfuse import LangfuseTracer, get_safe_tracer
from app.utils.logger import get_logger

logger = get_logger(__name__)


class DenseRetriever(BaseRetriever):
    """
    Retriever that queries Qdrant for semantically similar chunks (dense only).
    """

    def __init__(
        self,
        index: VectorStoreIndex,
        top_k: int = 5,
        tracer: LangfuseTracer | None = None,
    ) -> None:
        """
        Initialize the dense retriever.

        Args:
            index: Pluggable LlamaIndex VectorStoreIndex (connected to Qdrant).
            top_k: Number of candidate chunks to retrieve.
            tracer: Optional LangfuseTracer instance.
        """
        self._index = index
        self._top_k = top_k
        self._tracer = get_safe_tracer(tracer)

        # 📚 LESSON — LlamaIndex Query Modes:
        # In LlamaIndex, we instantiate a retriever from our index using `as_retriever()`.
        # Passing `vector_store_query_mode="default"` configures the underlying store (Qdrant)
        # to perform purely dense cosine similarity search.
        self._retriever = self._index.as_retriever(
            similarity_top_k=top_k,
            vector_store_query_mode="default",
        )

        logger.info("DenseRetriever initialized successfully with top_k={k}", k=top_k)

    def retrieve(self, query: str) -> list[NodeWithScore]:
        """
        Retrieve candidate chunks semantically similar to the query.

        Args:
            query: User search query.

        Returns:
            List of LlamaIndex NodeWithScore objects.
        """
        logger.info("Executing dense retrieval for query: '{query}'", query=query)

        span_ctx = (
            self._tracer.span(
                name="retrieval.dense",
                input={"query": query, "top_k": self._top_k},
            )
            if self._tracer
            else None
        )

        try:
            results = self._retriever.retrieve(query)
            
            logger.info("Dense retrieval finished. Retrieved {count} nodes.", count=len(results))

            if span_ctx:
                span_ctx.update(output={"retrieved_count": len(results)})

            return results

        except Exception as e:
            logger.error("Dense retrieval lookup failed: {err}", err=str(e))
            if span_ctx:
                span_ctx.update(level="ERROR", status_message=str(e))
            raise

    @property
    def name(self) -> str:
        return "dense"
