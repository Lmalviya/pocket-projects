"""
app/pipeline/base.py
=====================
Abstract Base Class and unified result schemas for all execution pipelines.

📚 LESSON — Top-Level Pipelines & Unified Tracing Contexts:
In information retrieval, a Pipeline binds the individual modules (Ingestion, Retrieval, 
Reranking, Generation) into a cohesive end-to-end execution flow.
  - The Pipeline starts the top-level Langfuse **Trace** (the root of the telemetry tree).
  - All nested steps (e.g. retrieval, LLM generation) automatically register as children Spans
    within this active trace context.
  - The result is a unified `PipelineResult` container, enabling the interactive CLI loop
    to present the synthesized answer, listing detailed candidate chunk sources, 
    latencies, and the clickable Langfuse web dashboard trace URL!
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from llama_index.core.schema import NodeWithScore


@dataclass
class PipelineResult:
    """
    Unified result container for end-to-end RAG pipeline executions.
    """
    query: str
    answer: str
    retrieved_nodes: list[NodeWithScore]
    trace_id: str | None = None
    trace_url: str | None = None
    latency_ms: float = 0.0
    metadata: dict[str, Any] | None = None


class BasePipeline(ABC):
    """
    Abstract Base Class defining the execution contract for all RAG pipelines.
    """

    @abstractmethod
    def run(self, query: str) -> PipelineResult:
        """
        Executes the end-to-end RAG pipeline for a given user query.

        Args:
            query: The user search query string.

        Returns:
            PipelineResult container.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Return the unique name of this pipeline strategy.
        """
        pass
