"""
app/pipeline/multi_turn.py
===========================
End-to-end multi-turn stateful conversational RAG execution pipeline.

📚 LESSON — Conversational Memory & Stateful Session Orchestration:
In multi-turn chat sessions, we must preserve conversational state:
  - The `MultiTurnPipeline` maintains a session history (`self.history` list of LangChain messages).
  - For each new turn, it runs the retriever to pull relevant context documents.
  - It invokes `MultiTurnGenerator.generate_turn(query, nodes, history)`, which runs
    our compiled LangGraph state machine to incorporate conversation memory.
  - It appends the new question (HumanMessage) and answer (AIMessage) to `self.history`
    to persist memory for the next turn.
  - It provides a `clear_history()` utility to reset the session.
"""

import time

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from app.config.settings import Settings
from app.generation.multi_turn import MultiTurnGenerator
from app.pipeline.base import BasePipeline, PipelineResult
from app.retrieval.base import BaseRetriever
from app.retrieval.reranker.base import BaseReranker
from app.tracing.langfuse import LangfuseTracer
from app.utils.logger import get_logger

logger = get_logger(__name__)


class MultiTurnPipeline(BasePipeline):
    """
    Coordinates end-to-end stateful conversational RAG Q&A executions.
    """

    def __init__(
        self,
        settings: Settings,
        retriever: BaseRetriever,
        generator: MultiTurnGenerator,
        reranker: BaseReranker | None = None,
        tracer: LangfuseTracer | None = None,
    ) -> None:
        """
        Initialize the conversational pipeline.

        Args:
            settings: Central Settings instance.
            retriever: Pluggable retriever (dense/sparse/hybrid).
            generator: Multi-turn graph generator (LangGraph).
            reranker: Optional Stage-2 reranker (cross_encoder/cohere).
            tracer: Optional LangfuseTracer instance.
        """
        self.settings = settings
        self.retriever = retriever
        self.generator = generator
        self.reranker = reranker
        self.tracer = tracer

        # 📚 LESSON — Session Memory Storage:
        # We store the conversation history locally as a list of LangChain message objects.
        self.history: list[BaseMessage] = []

        logger.info(
            "MultiTurnPipeline initialized: retriever={ret}, reranker={rer}, tracer_enabled={trace}",
            ret=retriever.name,
            rer=reranker.name if reranker else "none",
            trace=bool(tracer and tracer.enabled),
        )

    def clear_history(self) -> None:
        """
        Resets the conversation history, starting a fresh chat session.
        """
        logger.info("Clearing conversation memory history...")
        self.history.clear()

    def run(self, query: str) -> PipelineResult:
        """
        Executes a single conversational RAG turn:
        retrieval -> optional reranking -> LangGraph conversational generation.

        Args:
            query: Latest user message query.

        Returns:
            PipelineResult container.
        """
        logger.info("Executing conversational RAG MultiTurnPipeline for query: '{query}'", query=query)
        start_time = time.perf_counter()

        # 📚 LESSON — Root Trace per Conversational Turn:
        # Each query turn starts a new Langfuse trace. We can link these turns on the server
        # by passing a session ID tag in later phases to group full chat sessions.
        span_ctx = (
            self.tracer.span(
                name="rag-conversational-turn",
                input={"query": query, "history_size": len(self.history)},
            )
            if self.tracer
            else None
        )

        try:
            # 1. Retrieve Candidate document chunks based on user's latest query
            retrieved_nodes = self.retriever.retrieve(query)

            # 2. Apply Stage-2 Reranking if configured
            processed_nodes = retrieved_nodes
            if self.reranker:
                processed_nodes = self.reranker.rerank(query, retrieved_nodes)

            # 3. Execute conversational LangGraph turn (passes history and context)
            gen_res = self.generator.generate_turn(
                query=query,
                nodes=processed_nodes,
                history=self.history,
            )

            # 4. Persist Turn to Session History
            # We append both the HumanMessage and AIMessage to our local memory history
            self.history.append(HumanMessage(content=query))
            self.history.append(AIMessage(content=gen_res.answer))

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            # 5. Record top-level trace input and output on the Langfuse server
            if self.tracer:
                self.tracer.set_trace_io(
                    input={"query": query, "history_size": len(self.history) - 2},
                    output={"answer": gen_res.answer},
                )
                if span_ctx:
                    span_ctx.update(output={"answer": gen_res.answer})

            logger.info("Conversational pipeline turn complete. Latency: {lat}ms", lat=round(elapsed_ms, 2))

            return PipelineResult(
                query=query,
                answer=gen_res.answer,
                retrieved_nodes=processed_nodes,
                trace_id=self.tracer.get_trace_id() if self.tracer else None,
                trace_url=self.tracer.get_trace_url() if self.tracer else None,
                latency_ms=round(elapsed_ms, 2),
                metadata={
                    "prompt_version": gen_res.prompt_version,
                    "history_size": len(self.history),
                },
            )

        except Exception as e:
            logger.error("Conversational RAG pipeline turn failed: {err}", err=str(e))
            if span_ctx:
                span_ctx.update(level="ERROR", status_message=str(e))
            raise

    @property
    def name(self) -> str:
        return "multi_turn_pipeline"
