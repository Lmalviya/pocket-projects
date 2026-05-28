"""
app/config/settings.py
=======================
Central configuration for the RAG Eval Lab.

📚 LESSON — Why Pydantic Settings?
------------------------------------
Instead of scattering `os.getenv("KEY")` calls across the codebase, we define
ALL environment variables in ONE place as a typed Pydantic model.

Benefits:
  ✅ Type safety  — wrong type raises an error at startup, not buried in a call
  ✅ Validation   — you can add validators (e.g., URL must start with http)
  ✅ Autocompletion — your IDE knows the type of every setting
  ✅ Single source of truth — no hunting for which env vars the app needs

Usage anywhere in the codebase:
    from app.config.settings import get_settings
    settings = get_settings()
    print(settings.nvidia_api_key)
"""

from functools import lru_cache

from pydantic import Field, HttpUrl, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables and/or a .env file.

    Pydantic-Settings automatically reads from:
      1. Real environment variables (e.g., set in shell or Docker)
      2. A .env file in the project root (via `env_file = ".env"` below)

    Fields with defaults are optional in .env.
    Fields WITHOUT defaults are REQUIRED — app will fail fast at startup if missing.
    """

    model_config = SettingsConfigDict(
        env_file=".env",          # reads .env in the working directory
        env_file_encoding="utf-8",
        case_sensitive=False,     # NVIDIA_API_KEY and nvidia_api_key both work
        extra="ignore",           # silently ignore any extra vars in .env
    )

    # ── NVIDIA API ─────────────────────────────────────────────────────
    # 📚 LESSON — NVIDIA's API is OpenAI-compatible, meaning it uses the same
    # request/response format. We just point the OpenAI client to a different
    # base_url. No new SDK needed!
    nvidia_api_key: str = Field(..., description="NVIDIA API key from build.nvidia.com")
    nvidia_base_url: str = Field(
        default="https://integrate.api.nvidia.com/v1",
        description="NVIDIA API base URL (OpenAI-compatible endpoint)",
    )
    nvidia_model: str = Field(
        default="nvidia/llama-3.1-nemotron-nano-8b-instruct",
        description="NVIDIA model name to use for generation",
    )

    # ── Ollama Embeddings ──────────────────────────────────────────────────
    # 📚 LESSON — Ollama lets you run models locally. The embedding model
    # BAAI/bge-small-en-v1.5 produces 384-dimensional vectors and is fast + free.
    # "Embedding" = converting text into a list of numbers (a vector) that
    # captures semantic meaning. Similar texts → similar vectors.
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama server URL",
    )
    ollama_embed_model: str = Field(
        default="bge-small-en-v1.5",
        description="Ollama embedding model name",
    )
    embed_dim: int = Field(
        default=384,
        description="Output dimension of the embedding model (bge-small = 384)",
    )

    # ── Qdrant Vector Database ───────────────────────────────────────────────
    # 📚 LESSON — A vector database is optimized for similarity search.
    # Instead of "find rows where id = 5", it does "find the 10 vectors most
    # similar to this query vector" — that's the core of dense retrieval.
    qdrant_url: str = Field(
        default="http://localhost:6333",
        description="Qdrant REST API URL",
    )
    qdrant_collection_name: str = Field(
        default="rag_eval_lab",
        description="Qdrant collection name (like a table in a relational DB)",
    )

    # ── Langfuse Tracing ──────────────────────────────────────────────────────
    # 📚 LESSON — Langfuse is an observability platform for LLM apps.
    # It records every RAG call as a "trace" with nested "spans".
    langfuse_public_key: str = Field(..., description="Langfuse project public key")
    langfuse_secret_key: str = Field(..., description="Langfuse project secret key")
    langfuse_host: str = Field(
        default="http://localhost:3000",
        description="Langfuse server URL (self-hosted)",
    )

    # ── Cohere API (optional) ─────────────────────────────────────────────────
    cohere_api_key: str | None = Field(
        default=None,
        description="Cohere API key for Cohere Rerank (optional)",
    )

    # ── Application ───────────────────────────────────────────────────────────
    experiment_config_path: str = Field(
        default="app/config/experiment.yaml",
        description="Path to the active experiment configuration file",
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level: DEBUG | INFO | WARNING | ERROR",
    )

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in allowed:
            raise ValueError(f"log_level must be one of {allowed}, got: {v!r}")
        return v.upper()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the singleton Settings instance.

    📚 LESSON — @lru_cache(maxsize=1) means this function's result is cached
    after the first call. Every subsequent call returns the SAME object
    without re-reading .env. This is the standard Python singleton pattern
    for settings — simple, thread-safe, and testable (you can clear the
    cache in tests with get_settings.cache_clear()).
    """
    return Settings()
