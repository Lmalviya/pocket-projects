"""
app/retrieval/reranker/base.py
===============================
Abstract Base Class for all reranking strategies in the RAG Eval Lab.

📚 LESSON — Two-Stage Retrieval:
RAG retrieval is typically engineered as a two-stage process for latency/precision trade-offs:
  - Stage 1 (Retrieval): Use fast vector/lexical lookups to pull a broad candidate pool
    (e.g., top 20 or 50 chunks). This is fast but might include somewhat irrelevant chunks.
  - Stage 2 (Reranking): Run a heavier, context-aware cross-encoder model over just those 
    top 20-50 candidates to evaluate exact query-document alignment, re-sorting them 
    and slicing down to a final consolidated set (e.g., top 5).
    
Every reranker:
  1. Accepts a `query` and a list of `NodeWithScore` candidates.
  2. Re-scores the candidates, sorts them descending by new scores, and slices to top-K.
  3. Returns the re-ordered list of `NodeWithScore` objects.
"""

from abc import ABC, abstractmethod
from llama_index.core.schema import NodeWithScore


class BaseReranker(ABC):
    """
    Abstract Base Class defining the contract for all reranking components.
    """

    @abstractmethod
    def rerank(self, query: str, nodes: list[NodeWithScore]) -> list[NodeWithScore]:
        """
        Re-scores and re-ranks a list of candidate nodes against a search query.

        Args:
            query: The user search query.
            nodes: List of initial retrieved NodeWithScore objects.

        Returns:
            Sorted and pruned list of NodeWithScore objects.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Return the unique name of this reranking strategy.
        """
        pass
