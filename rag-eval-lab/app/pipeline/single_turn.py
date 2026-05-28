"""
app/pipeline/single_turn.py
===========================
End-to-end single-turn factual RAG execution pipeline.

📚 LESSON — Orchestration Flow & Telemetry:
In single-turn search, the pipeline retrieves candidates, applies an optional Stage-2 reranking,
passes the curated context and user query to the SingleTurnGenerator to synthesize an answer,
and tracks the entire execution as nested spans within a unified Langfuse Trace.
"""

import time

from app.config.settings import Settings
from app.generation.single_turn import SingleTurnGenerator
from app.pipeline.base import BasePipeline, PipelineResult
from app.retrieval.base import BaseRetriever
from app.retrieval.reranker.base import BaseReranker
from app.tracing.langfuse import LangfuseTracer
from app.utils.logger import get_logger

logger = get_logger(__name__)


class SingleTurnPipeline(BasePipeline):
    """
    Coordinates end-to-end single-turn factual RAG Q&A executions.
    """

    def __init__(
        self,
        settings: Settings,
        retriever: BaseRetriever,
        generator: SingleTurnGenerator,
        reranker: BaseReranker | None = None,
        tracer: LangfuseTracer | None = None,
    ) -> None:
        """
        Initialize the single-turn pipeline.

        Args:
            settings: Central Settings instance.
            retriever: Pluggable retriever (dense/sparse/hybrid).
            generator: Single-turn generator (OpenAI-compatible LLM).
            reranker: Optional Stage-2 reranker (cross_encoder/cohere).
            tracer: Optional LangfuseTracer instance.
        """
        self.settings = settings
        self.retriever = retriever
        self.generator = generator
        self.reranker = reranker
        self.tracer = tracer

        logger.info(
            "SingleTurnPipeline initialized: retriever={ret}, reranker={rer}, tracer_enabled={trace}",
            ret=retriever.name,
            rer=reranker.name if reranker else "none",
            trace=bool(tracer and tracer.enabled),
        )

    def run(self, query: str) -> PipelineResult:
        """
        Executes a single-turn RAG pipeline:
        retrieval -> optional reranking -> generation.

        Args:
            query: User search query.

        Returns:
            PipelineResult container.
        """
        logger.info("Executing single-turn RAG pipeline for query: '{query}'", query=query)
        start_time = time.perf_counter()

        # Start top-level trace span
        span_ctx = (
            self.tracer.span(
                name="rag-query",
                input={"query": query},
            )
            if self.tracer
            else None
        )

        try:
            # 1. Retrieve candidate document chunks
            retrieved_nodes = self.retriever.retrieve(query)

            # 2. Apply Stage-2 Reranking if configured
            processed_nodes = retrieved_nodes
            if self.reranker:
                processed_nodes = self.reranker.rerank(query, retrieved_nodes)

            # 3. Generate response using NVIDIA LLM
            gen_res = self.generator.generate(
                query=query,
                nodes=processed_nodes,
            )

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            # Record top-level trace input and output on the Langfuse server
            if self.tracer:
                self.tracer.set_trace_io(
                    input={"query": query},
                    output={"answer": gen_res.answer},
                )
                if span_ctx:
                    span_ctx.update(output={"answer": gen_res.answer})

            logger.info("Single-turn RAG pipeline query complete. Latency: {lat}ms", lat=round(elapsed_ms, 2))

            return PipelineResult(
                query=query,
                answer=gen_res.answer,
                retrieved_nodes=processed_nodes,
                trace_id=self.tracer.get_trace_id() if self.tracer else None,
                trace_url=self.tracer.get_trace_url() if self.tracer else None,
                latency_ms=round(elapsed_ms, 2),
                metadata={
                    "prompt_version": gen_res.prompt_version,
                },
            )

        except Exception as e:
            logger.error("Single-turn RAG pipeline query failed: {err}", err=str(e))
            if span_ctx:
                span_ctx.update(level="ERROR", status_message=str(e))
            raise

    @property
    def name(self) -> str:
        return "single_turn_pipeline"
