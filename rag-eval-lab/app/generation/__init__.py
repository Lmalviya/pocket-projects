"""
app/generation/__init__.py
==========================
Core generation layer of the RAG Eval Lab.

Re-exports:
  - GenerationResult: Dataclass carrying LLM generated answers.
  - SingleTurnGenerator: Factual answer generator using NVIDIA API and Langfuse prompts.
  - MultiTurnGenerator: Stateful conversational RAG generator using LangGraph workflows.
"""

from app.generation.multi_turn import MultiTurnGenerator
from app.generation.single_turn import GenerationResult, SingleTurnGenerator

__all__ = [
    "GenerationResult",
    "SingleTurnGenerator",
    "MultiTurnGenerator",
]
