"""
app/tracing/langfuse.py
========================
Central Langfuse tracing setup for the RAG Eval Lab.

📚 LESSON — What is distributed tracing and why does it matter for RAG?
------------------------------------------------------------------------
A RAG pipeline has MANY steps: load → chunk → index → retrieve → rerank → generate.
Each step takes time and can fail. Without tracing, debugging is a nightmare:
  ❌ "The answer was wrong — was it bad retrieval? A bad prompt? Wrong chunks?"

With Langfuse tracing, every query creates a "trace" — a tree of timed spans:
  ✅ You can see EXACTLY which chunks were retrieved for each query
  ✅ You can compare chunking strategies side-by-side by experiment name
  ✅ You see token usage and latency for each LLM call
  ✅ You can search traces by experiment, session, score

📚 LESSON — Langfuse v4 API (context-manager based)
-----------------------------------------------------
Langfuse v4 uses the OpenTelemetry (OTel) standard under the hood. Spans are
managed via context managers and the API changed significantly from v2:

  v2 (old):  trace = client.trace(); span = trace.span(); span.end()
  v4 (new):  with langfuse.start_as_current_observation(name=...) as span: ...

Key v4 concepts:
  Langfuse.start_observation()              → creates span (must call .end() manually)
  Langfuse.start_as_current_observation()   → context manager, auto-ends span
  Langfuse.get_current_trace_id()           → get the active trace ID
  Langfuse.set_current_trace_io()           → set input/output on the current trace
  Langfuse.flush()                          → flush buffered events to server

Architecture of this module:
  - get_langfuse_client()  → singleton Langfuse SDK client
  - LangfuseTracer         → helper class wrapping v4 API cleanly
  - trace_span()           → context manager for named spans
  - get_langfuse_callback()→ LangChain CallbackHandler (auto-instruments chains)

ALL other modules import ONLY from this file.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Generator

from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

from app.config.settings import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Singleton Langfuse client
# ---------------------------------------------------------------------------

_langfuse_client: Langfuse | None = None


def get_langfuse_client() -> Langfuse:
    """
    Return the singleton Langfuse client, initializing it on first call.

    📚 LESSON — Singleton pattern:
      We create the client ONCE and reuse it. Creating a new client per
      request would be wasteful (TCP connection setup, auth handshake, etc.)
      and could also cause concurrency issues with trace IDs.

    Returns:
        Initialized Langfuse client instance.
    """
    global _langfuse_client

    if _langfuse_client is None:
        settings = get_settings()
        _langfuse_client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
        logger.info("Langfuse client initialized → {host}", host=settings.langfuse_host)

    return _langfuse_client


def get_langfuse_callback() -> CallbackHandler:
    """
    Return a Langfuse LangChain CallbackHandler for v4.

    📚 LESSON — LangChain Callbacks in Langfuse v4:
      In v4, the CallbackHandler automatically instruments LangChain chains
      as observations (spans) within the current active trace context.
      It hooks into LangChain's callback system (on_chain_start, on_llm_end, etc.)
      so you get full LLM call visibility with zero manual code.

      Usage:
        callbacks = [get_langfuse_callback()]
        chain.invoke({"question": "..."}, config={"callbacks": callbacks})

    Returns:
        A Langfuse CallbackHandler for LangChain chains.
    """
    settings = get_settings()
    return CallbackHandler(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
    )


# ---------------------------------------------------------------------------
# LangfuseTracer — clean wrapper around the v4 API
# ---------------------------------------------------------------------------

class LangfuseTracer:
    """
    Thin wrapper around the Langfuse v4 SDK for clean span lifecycle management.

    📚 LESSON — Why wrap the SDK?
      The raw Langfuse v4 SDK is powerful but can be verbose. This wrapper:
        1. Provides a consistent API used across all pipeline modules
        2. Handles the "tracing disabled" case gracefully (no-ops)
        3. Catches SDK errors so tracing failures NEVER crash the app
        4. Adds default metadata (experiment name, model info)

    📚 LESSON — Langfuse v4 Trace Lifecycle:
      In v4, a "trace" is implicitly created when the first top-level span
      is started. The trace ID is accessible via get_current_trace_id().
      You don't create a trace object explicitly anymore.

      Typical usage:
        tracer = LangfuseTracer("my-experiment")
        with tracer.span("chunking.fixed", input={"docs": 5}):
            nodes = chunker.chunk(documents)
        # span auto-ends when with-block exits

    Usage:
        tracer = LangfuseTracer(experiment_name="fixed-dense-v1")

        with tracer.span("rag-query", input={"question": q}) as root:
            with tracer.span("chunking.fixed", input={"doc_count": 5}):
                nodes = chunker.chunk(docs)
            with tracer.span("retrieval.dense", input={"query": q}):
                results = retriever.retrieve(q)

        trace_url = tracer.get_trace_url()
    """

    def __init__(self, experiment_name: str = "default", enabled: bool = True) -> None:
        self.experiment_name = experiment_name
        self.enabled = enabled
        self._client = get_langfuse_client() if enabled else None

    @contextmanager
    def span(
        self,
        name: str,
        input: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        as_type: str = "span",
    ) -> Generator:
        """
        Context manager that creates a named Langfuse span.

        📚 LESSON — Context managers for spans:
          Using `with tracer.span("name"):` instead of manual start/end
          guarantees the span is ALWAYS ended, even if an exception occurs.
          This prevents "orphaned" spans in Langfuse that never close.

        Args:
            name: Span name (e.g., "chunking.fixed", "retrieval.dense").
            input: Input data for this operation (logged in Langfuse UI).
            metadata: Extra context (config, sizes, etc.).
            as_type: Observation type — "span" | "generation" | "retriever".

        Yields:
            The active span object (call .update() on it for live updates).
        """
        if not self.enabled or self._client is None:
            yield None
            return

        full_metadata = {
            "experiment_name": self.experiment_name,
            **(metadata or {}),
        }

        try:
            with self._client.start_as_current_observation(
                name=name,
                as_type=as_type,
                input=input or {},
                metadata=full_metadata,
            ) as observation:
                start = time.perf_counter()
                try:
                    yield observation
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    # Update span with timing info after the block completes
                    if observation:
                        observation.update(
                            metadata={**full_metadata, "latency_ms": round(elapsed_ms, 2)}
                        )
                except Exception as e:
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    if observation:
                        observation.update(
                            level="ERROR",
                            status_message=str(e),
                            metadata={**full_metadata, "latency_ms": round(elapsed_ms, 2)},
                        )
                    raise

        except Exception as sdk_err:
            # If the Langfuse SDK itself fails, log and continue — NEVER crash the app
            logger.warning("Langfuse span '{name}' failed (non-fatal): {err}", name=name, err=str(sdk_err))
            yield None

    def set_trace_io(
        self,
        input: dict[str, Any] | None = None,
        output: dict[str, Any] | None = None,
    ) -> None:
        """
        Set the top-level input/output on the current active trace.

        Call this with the user's question (input) and the final answer (output)
        to make the trace's top card in Langfuse show the full Q&A.

        Args:
            input: Dict representing the overall trace input (e.g., {"question": "..."}).
            output: Dict representing the final result (e.g., {"answer": "..."}).
        """
        if not self.enabled or self._client is None:
            return
        try:
            self._client.set_current_trace_io(input=input, output=output)
        except Exception as e:
            logger.warning("Langfuse set_trace_io failed (non-fatal): {err}", err=str(e))

    def get_trace_url(self) -> str | None:
        """
        Get the URL to view the current trace in the Langfuse dashboard.

        Returns:
            Full Langfuse trace URL, or None if tracing is disabled.

        Example:
            url = tracer.get_trace_url()
            print(f"View trace: {url}")
            # http://localhost:3000/trace/abc-123-def-456
        """
        if not self.enabled or self._client is None:
            return None
        try:
            return self._client.get_trace_url()
        except Exception as e:
            logger.warning("Langfuse get_trace_url failed: {err}", err=str(e))
            return None

    def get_trace_id(self) -> str | None:
        """Return the current active trace ID."""
        if not self.enabled or self._client is None:
            return None
        try:
            return self._client.get_current_trace_id()
        except Exception as e:
            logger.warning("Langfuse get_trace_id failed: {err}", err=str(e))
            return None

    def flush(self) -> None:
        """
        Flush buffered Langfuse events to the server.

        📚 LESSON — Langfuse v4 buffers events for performance.
        Call flush() at the end of a script to ensure all spans are sent
        before the process exits. In a long-running server, this is less
        critical as the background flush runs automatically.
        """
        if self._client:
            try:
                self._client.flush()
                logger.debug("Langfuse flush complete")
            except Exception as e:
                logger.warning("Langfuse flush failed: {err}", err=str(e))
