"""
app/retrieval/reranker/cohere.py
=================================
Hosted Cohere API reranker.

📚 LESSON — API-Based Hosted Reranking:
  - Local cross-encoders (like MS-MARCO) are excellent for small applications and data sizes,
    but they consume local RAM/GPU and can slow down your application server.
  - Hosted API rerankers (like Cohere Rerank v3) offload this heavy deep-learning computation
    to optimized cloud hardware. 
  - Cohere Rerank v3 is the industry standard for high-performance enterprise RAG. It handles
    extremely long chunks and supports multi-lingual queries out-of-the-box.
"""

import cohere
from llama_index.core.schema import NodeWithScore

from app.retrieval.reranker.base import BaseReranker
from app.tracing.langfuse import LangfuseTracer, get_safe_tracer
from app.utils.logger import get_logger

logger = get_logger(__name__)


class CohereReranker(BaseReranker):
    """
    Reranker that calls Cohere's hosted Rerank API to re-score candidate nodes.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "rerank-english-v3.0",
        top_k: int = 5,
        tracer: LangfuseTracer | None = None,
    ) -> None:
        """
        Initialize the Cohere Reranker.

        Args:
            api_key: Optional Cohere API key. If None, loaded from central settings.
            model_name: Hosted Cohere model to use (default: "rerank-english-v3.0").
            top_k: Number of final sorted nodes to return.
            tracer: Optional LangfuseTracer instance.
        """
        self._model_name = model_name
        self._top_k = top_k
        self._tracer = get_safe_tracer(tracer)

        # 1. Resolve API key
        if not api_key:
            from app.config.settings import get_settings
            settings = get_settings()
            api_key = settings.cohere_api_key

        if not api_key:
            raise ValueError(
                "❌ Cohere API Key is missing.\n"
                "To use the Cohere Reranker, you must either:\n"
                "  1. Define COHERE_API_KEY in your local `.env` file.\n"
                "  2. Pass `api_key` directly to the CohereReranker constructor."
            )

        # 2. Instantiate Cohere Client
        logger.info("Initializing Cohere client using model '{model}'...", model=model_name)
        try:
            self._client = cohere.Client(api_key=api_key)
            logger.info("Cohere client initialized successfully.")
        except Exception as e:
            logger.error("Failed to initialize Cohere client: {err}", err=str(e))
            raise

    def rerank(self, query: str, nodes: list[NodeWithScore]) -> list[NodeWithScore]:
        """
        Re-scores candidate nodes using Cohere's hosted Rerank API.

        Args:
            query: The user search query.
            nodes: Initial retrieved candidates list.

        Returns:
            Re-sorted list of NodeWithScore objects.
        """
        if not nodes:
            logger.debug("Reranking bypassed - empty nodes list passed.")
            return []

        logger.info(
            "Reranking {count} candidates for query: '{query}' using Cohere Rerank API...",
            count=len(nodes),
            query=query,
        )

        span_ctx = (
            self._tracer.span(
                name="reranker.cohere",
                input={"query": query, "node_count": len(nodes), "model": self._model_name},
            )
            if self._tracer
            else None
        )

        try:
            # 1. Format candidate inputs as plain text document strings
            documents = [node.node.get_content() for node in nodes]

            # 2. Query Cohere Rerank API
            # In Cohere SDK v5+, `client.rerank` returns a RerankResponse object.
            # response.results contains a list of objects with fields: index, relevance_score
            response = self._client.rerank(
                model=self._model_name,
                query=query,
                documents=documents,
                top_n=self._top_k,
            )

            # 3. Map result indices and relevance scores back to LlamaIndex nodes
            reranked_nodes = []
            for result in response.results:
                orig_node = nodes[result.index]
                reranked_nodes.append(
                    NodeWithScore(
                        node=orig_node.node,
                        score=float(result.relevance_score),
                    )
                )

            logger.info("Cohere rerank complete. Retrieved top {count} candidates.", count=len(reranked_nodes))

            if span_ctx:
                span_ctx.update(output={"reranked_count": len(reranked_nodes)})

            return reranked_nodes

        except Exception as e:
            logger.error("Cohere Rerank API call failed: {err}", err=str(e))
            if span_ctx:
                span_ctx.update(level="ERROR", status_message=str(e))
            raise

    @property
    def name(self) -> str:
        return "cohere"
