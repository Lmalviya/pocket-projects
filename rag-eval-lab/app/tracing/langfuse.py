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
    # Ensure client singleton is initialized first so credentials are bound
    get_langfuse_client()
    return CallbackHandler()


class SpanContext:
    """
    Unified context manager and direct update container for Langfuse spans.
    Supports standard 'with' block usage and fallback direct update calls.
    """

    def __init__(
        self,
        client: Any,
        name: str,
        as_type: str,
        input_data: dict[str, Any] | None,
        full_metadata: dict[str, Any],
        enabled: bool,
    ) -> None:
        self.client = client
        self.name = name
        self.as_type = as_type
        self.input_data = input_data
        self.full_metadata = full_metadata
        self.enabled = enabled
        self.observation = None
        self.observation_ctx = None
        self.start_time = None

    def update(self, **kwargs: Any) -> SpanContext:
        """
        Dynamically updates the active span/observation with new payload parameters.
        No-ops safely if tracing is disabled or the span has not been entered.
        """
        if not self.enabled or self.observation is None:
            return self
        try:
            self.observation.update(**kwargs)
        except Exception as e:
            logger.warning("Failed to update span '{name}' (non-fatal): {err}", name=self.name, err=str(e))
        return self

    def __enter__(self) -> SpanContext:
        if not self.enabled or self.client is None:
            return self
        try:
            self.start_time = time.perf_counter()
            self.observation_ctx = self.client.start_as_current_observation(
                name=self.name,
                as_type=self.as_type,
                input=self.input_data or {},
                metadata=self.full_metadata,
            )
            self.observation = self.observation_ctx.__enter__()
        except Exception as e:
            logger.warning("Failed to start span '{name}' (non-fatal): {err}", name=self.name, err=str(e))
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        if not self.enabled or self.observation_ctx is None:
            return False
        elapsed_ms = (time.perf_counter() - self.start_time) * 1000
        try:
            if exc_type is not None:
                self.observation.update(
                    level="ERROR",
                    status_message=str(exc_val),
                    metadata={**self.full_metadata, "latency_ms": round(elapsed_ms, 2)}
                )
            else:
                self.observation.update(
                    metadata={**self.full_metadata, "latency_ms": round(elapsed_ms, 2)}
                )
        except Exception as e:
            logger.warning("Failed to update span metadata '{name}' (non-fatal): {err}", name=self.name, err=str(e))

        # Exit the underlying OTel/Langfuse context manager
        try:
            self.observation_ctx.__exit__(exc_type, exc_val, exc_tb)
        except Exception as e:
            logger.warning("Failed to close underlying span '{name}' (non-fatal): {err}", name=self.name, err=str(e))
        return False


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

    def span(
        self,
        name: str,
        input: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        as_type: str = "span",
    ) -> SpanContext:
        """
        Creates a named Langfuse SpanContext that can be used either as a 
        standard context manager ('with') or as a fallback direct-call container.

        Args:
            name: Span name (e.g., "chunking.fixed", "retrieval.dense").
            input: Input data for this operation (logged in Langfuse UI).
            metadata: Extra context (config, sizes, etc.).
            as_type: Observation type — "span" | "generation" | "retriever".

        Returns:
            The active SpanContext object.
        """
        full_metadata = {
            "experiment_name": self.experiment_name,
            **(metadata or {}),
        }
        return SpanContext(
            client=self._client,
            name=name,
            as_type=as_type,
            input_data=input,
            full_metadata=full_metadata,
            enabled=self.enabled,
        )

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


class NoOpTracer:
    """
    A tracer that implements the LangfuseTracer interface but does absolutely nothing.
    This serves as a safe fallback (Null Object pattern) to prevent AttributeError
    and remove boilerplate 'if tracer' guards across the codebase.
    """

    def __init__(self) -> None:
        self.experiment_name = "noop"
        self.enabled = False

    def span(
        self,
        name: str,
        input: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        as_type: str = "span",
    ) -> SpanContext:
        return SpanContext(
            client=None,
            name=name,
            as_type=as_type,
            input_data=input,
            full_metadata={},
            enabled=False,
        )

    def set_trace_io(
        self,
        input: dict[str, Any] | None = None,
        output: dict[str, Any] | None = None,
    ) -> None:
        pass

    def get_trace_url(self) -> str | None:
        return None

    def get_trace_id(self) -> str | None:
        return None

    def flush(self) -> None:
        pass


_noop_tracer = None


def get_safe_tracer(tracer: Any = None) -> LangfuseTracer | NoOpTracer:
    """
    Safely resolves a tracer instance.
    If the provided tracer is a valid LangfuseTracer, returns it.
    Otherwise, returns a NoOpTracer instance.
    This guarantees that components can always call tracer.span() without checking for None
    or encountering type/attribute errors.
    """
    global _noop_tracer
    if isinstance(tracer, LangfuseTracer):
        return tracer
    if tracer is not None and hasattr(tracer, "span") and callable(tracer.span):
        return tracer
    if _noop_tracer is None:
        _noop_tracer = NoOpTracer()
    return _noop_tracer

