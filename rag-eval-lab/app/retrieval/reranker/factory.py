"""
app/retrieval/reranker/factory.py
==================================
Registry and factory for pluggable Reranking strategies.

📚 LESSON — Factory Pattern for Rerankers:
A modular RAG architecture never hardcodes components.
By using a central Factory, the main execution pipeline only asks for "a reranker"
by its strategy name. If we want to switch from a local Cross-Encoder to a hosted
Cohere API, we simply change the name in `experiment.yaml`.
"""

from typing import Type

from app.retrieval.reranker.base import BaseReranker
from app.retrieval.reranker.cohere import CohereReranker
from app.retrieval.reranker.cross_encoder import CrossEncoderReranker
from app.tracing.langfuse import LangfuseTracer
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Registry mapping strategy names to Reranker classes
RERANKER_REGISTRY: dict[str, Type[BaseReranker]] = {
    "cross_encoder": CrossEncoderReranker,
    "cohere": CohereReranker,
}


def get_reranker(
    strategy: str,
    top_k: int = 5,
    tracer: LangfuseTracer | None = None,
) -> BaseReranker | None:
    """
    Dynamically instantiates and returns a reranker based on strategy name.

    Args:
        strategy: Reranker strategy name: "none" | "cross_encoder" | "cohere".
        top_k: Number of final sorted nodes to return.
        tracer: Optional LangfuseTracer instance.

    Returns:
        BaseReranker instance, or None if strategy is "none" or empty.
    """
    strategy_clean = strategy.strip().lower()

    if not strategy_clean or strategy_clean == "none":
        logger.info("Reranker strategy is set to 'none' — skipping Stage 2 Reranking.")
        return None

    if strategy_clean not in RERANKER_REGISTRY:
        allowed = list(RERANKER_REGISTRY.keys()) + ["none"]
        err_msg = f"Unknown reranker strategy '{strategy}'. Allowed strategies are: {allowed}"
        logger.error(err_msg)
        raise ValueError(err_msg)

    logger.info("Instantiating reranker strategy: '{strategy}'...", strategy=strategy_clean)
    
    # Dynamically look up and instantiate class from registry
    reranker_class = RERANKER_REGISTRY[strategy_clean]
    return reranker_class(top_k=top_k, tracer=tracer)


def list_reranker_strategies() -> list[str]:
    """Returns a list of all registered reranker strategies."""
    return list(RERANKER_REGISTRY.keys())
