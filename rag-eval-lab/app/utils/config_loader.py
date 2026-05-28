"""
app/utils/config_loader.py
===========================
Loads and validates the experiment.yaml configuration file.

📚 LESSON — experiment.yaml vs settings.py (what goes where?)
--------------------------------------------------------------
We have TWO config files. Understanding why is important:

  settings.py (Settings class)
    → Infrastructure config: API keys, URLs, credentials.
    → Changes rarely. Set once per environment.
    → Sensitive — loaded from .env (gitignored).
    → Example: nvidia_api_key, qdrant_url, langfuse_host

  experiment.yaml (ExperimentConfig)
    → Experiment parameters: which chunker? which retriever? top-k?
    → Changes OFTEN — every time you run a different experiment.
    → NOT sensitive — committed to git so experiments are reproducible.
    → Example: chunking=semantic, retrieval=hybrid, top_k=10

This separation lets you: version-control experiment configs, reproduce any
past experiment exactly, and switch strategies without touching secrets.
"""

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from app.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses for each section of experiment.yaml
# ---------------------------------------------------------------------------
# 📚 LESSON — We use plain Python dataclasses (not Pydantic) here because:
#   - experiment.yaml is developer-controlled (not user-facing API input)
#   - We want simple, readable classes without validation overhead
#   - Pydantic is reserved for external inputs (API payloads, .env vars)
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Which LLM and embedding model to use for this experiment run."""
    llm: str = "nvidia/llama-3.1-nemotron-nano-8b-instruct"
    embedding: str = "bge-small-en-v1.5"


@dataclass
class IngestionConfig:
    """How to chunk documents. The key knob for experimenting."""
    chunking: str = "fixed"      # fixed | semantic | recursive | sentence
    chunk_size: int = 512        # tokens per chunk
    chunk_overlap: int = 50      # overlap between consecutive chunks
    stage: int = 1               # Evaluation dataset scale stage (1, 2, or 3)


@dataclass
class RetrievalConfig:
    """How to retrieve relevant chunks given a user query."""
    method: str = "dense"        # dense | sparse | hybrid
    top_k: int = 5               # how many chunks to retrieve initially
    reranker: str = "none"       # none | cross_encoder | cohere
    reranker_top_k: int = 5      # how many chunks to keep after reranking
    sparse_strategy: str = "splade" # sparse representation strategy: splade | bm25


@dataclass
class GenerationConfig:
    """LLM generation settings."""
    mode: str = "single_turn"    # single_turn | multi_turn
    prompt_version: str = "v1"   # maps to a prompt template file


@dataclass
class TracingConfig:
    """Langfuse tracing settings for this experiment."""
    enabled: bool = True
    session_tag: str = "default"


@dataclass
class ExperimentConfig:
    """
    Full experiment configuration, mirroring the structure of experiment.yaml.

    Every RAG pipeline run is driven by this config — it determines exactly
    which strategy combination is being tested.
    """
    experiment_name: str = "baseline"
    model: ModelConfig = field(default_factory=ModelConfig)
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    tracing: TracingConfig = field(default_factory=TracingConfig)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """
    Parse experiment.yaml and return a typed ExperimentConfig.

    Args:
        path: Path to the experiment YAML file.

    Returns:
        ExperimentConfig populated from the YAML values.

    Raises:
        FileNotFoundError: If the YAML file doesn't exist.
        yaml.YAMLError: If the YAML file is malformed.

    Example:
        config = load_experiment_config("app/config/experiment.yaml")
        print(config.ingestion.chunking)  # "fixed"
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Experiment config not found: {path.resolve()}")

    logger.info("Loading experiment config from {path}", path=str(path))

    with path.open("r", encoding="utf-8") as f:
        raw: dict = yaml.safe_load(f) or {}

    # Build each sub-config from the relevant YAML section.
    # .get("section", {}) means "use empty dict if section is missing",
    # which lets the dataclass defaults kick in gracefully.
    config = ExperimentConfig(
        experiment_name=raw.get("experiment_name", "baseline"),
        model=ModelConfig(**raw.get("model", {})),
        ingestion=IngestionConfig(**raw.get("ingestion", {})),
        retrieval=RetrievalConfig(**raw.get("retrieval", {})),
        generation=GenerationConfig(**raw.get("generation", {})),
        tracing=TracingConfig(**raw.get("tracing", {})),
    )

    logger.info(
        "Experiment '{name}' loaded: chunking={chunking}, retrieval={retrieval}",
        name=config.experiment_name,
        chunking=config.ingestion.chunking,
        retrieval=config.retrieval.method,
    )

    return config
