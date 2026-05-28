"""
app/ingestion/chunking/factory.py
===================================
Factory for creating chunker instances from a strategy name string.

📚 LESSON — The Factory Pattern
---------------------------------
The Factory Pattern is a creational design pattern that centralizes object
construction. Instead of this scattered across the codebase:

  ❌ In pipeline.py:
    if strategy == "fixed":
        chunker = FixedSizeChunker(config)
    elif strategy == "recursive":
        chunker = RecursiveChunker(config)
    elif strategy == "sentence":
        chunker = SentenceChunker(config)
    elif strategy == "semantic":
        chunker = SemanticChunker(config)

  ✅ With factory:
    chunker = get_chunker("fixed", config)

Benefits:
  1. Pipeline code is decoupled from chunker implementations
  2. Adding a new chunker = adding one entry to CHUNKER_REGISTRY
  3. Config-driven selection: read strategy from experiment.yaml → done
  4. Easy to test: mock the factory, inject a test chunker

📚 LESSON — Registry Pattern
------------------------------
We store chunker classes in a dict (the "registry"):
  CHUNKER_REGISTRY = {"fixed": FixedSizeChunker, "recursive": RecursiveChunker, ...}

get_chunker() looks up the class, instantiates it with the given config,
and returns the instance. This is a simple but powerful form of dependency
injection — the caller never knows which class was instantiated.
"""

from __future__ import annotations

from app.ingestion.chunking.base import BaseChunker, ChunkingConfig
from app.ingestion.chunking.fixed import FixedSizeChunker
from app.ingestion.chunking.recursive import RecursiveChunker
from app.ingestion.chunking.semantic import SemanticChunker
from app.ingestion.chunking.sentence import SentenceChunker
from app.tracing.langfuse import LangfuseTracer
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Registry: maps strategy name (from experiment.yaml) → chunker class
# ---------------------------------------------------------------------------
# To add a new chunker:
#   1. Create app/ingestion/chunking/my_new.py with MyNewChunker(BaseChunker)
#   2. Add "my_new": MyNewChunker to this dict
#   3. That's it — no other code changes needed.

CHUNKER_REGISTRY: dict[str, type[BaseChunker]] = {
    "fixed": FixedSizeChunker,
    "recursive": RecursiveChunker,
    "sentence": SentenceChunker,
    "semantic": SemanticChunker,
}


def get_chunker(
    strategy: str,
    config: ChunkingConfig,
    tracer: LangfuseTracer | None = None,
    trace=None,
    **kwargs,
) -> BaseChunker:
    """
    Instantiate and return a chunker for the given strategy.

    Args:
        strategy: Strategy name matching a key in CHUNKER_REGISTRY.
            Valid values: "fixed", "recursive", "sentence", "semantic"
        config: ChunkingConfig with chunk_size, chunk_overlap, etc.
        tracer: Optional LangfuseTracer for span tracking.
        trace: Optional active Langfuse trace.
        **kwargs: Extra kwargs passed to the chunker constructor
            (e.g., breakpoint_threshold_type for SemanticChunker).

    Returns:
        A BaseChunker instance ready to call .chunk(documents).

    Raises:
        ValueError: If strategy is not in the registry.

    Example:
        config = ChunkingConfig(chunk_size=512, chunk_overlap=50, strategy="fixed")
        chunker = get_chunker("fixed", config, tracer=my_tracer)
        result = chunker.chunk(documents)
    """
    strategy = strategy.lower().strip()

    if strategy not in CHUNKER_REGISTRY:
        available = list(CHUNKER_REGISTRY.keys())
        raise ValueError(
            f"Unknown chunking strategy: {strategy!r}. "
            f"Available strategies: {available}"
        )

    chunker_class = CHUNKER_REGISTRY[strategy]
    logger.info("Creating chunker: strategy={strategy}, class={cls}", strategy=strategy, cls=chunker_class.__name__)

    return chunker_class(config=config, tracer=tracer, trace=trace, **kwargs)


def list_strategies() -> list[str]:
    """Return a list of all registered chunking strategy names."""
    return list(CHUNKER_REGISTRY.keys())
