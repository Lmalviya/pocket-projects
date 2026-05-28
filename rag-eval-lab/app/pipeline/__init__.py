"""
app/pipeline/__init__.py
========================
Core execution pipelines of the RAG Eval Lab.

Re-exports:
  - BasePipeline: Abstract Base Class for RAG pipelines.
  - PipelineResult: Data schema carrying consolidated RAG query execution stats.
  - SingleTurnPipeline: Coordinates single-turn Q&A sessions.
  - MultiTurnPipeline: Coordinates stateful multi-turn conversational chat sessions.
"""

from app.pipeline.base import BasePipeline, PipelineResult
from app.pipeline.multi_turn import MultiTurnPipeline
from app.pipeline.single_turn import SingleTurnPipeline

__all__ = [
    "BasePipeline",
    "PipelineResult",
    "SingleTurnPipeline",
    "MultiTurnPipeline",
]
