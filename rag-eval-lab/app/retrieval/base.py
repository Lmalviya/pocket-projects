"""
app/retrieval/base.py
======================
Abstract Base Class for all retrieval strategies in the RAG Eval Lab.

📚 LESSON — Unified Retrieval Interface:
To evaluate different search strategies (Dense, Sparse, Hybrid) side-by-side,
they must implement a single unified contract. This allows our downstream pipelines
and evaluation scripts to remain 100% independent of the underlying retrieval math.

Every retriever:
  1. Accepts a string `query`.
  2. Returns a list of LlamaIndex `NodeWithScore` objects.
     - `NodeWithScore` combines the `TextNode` (document fragment + metadata)
       with a floating-point `score` representing relevance.
"""

from abc import ABC, abstractmethod
from llama_index.core.schema import NodeWithScore


class BaseRetriever(ABC):
    """
    Abstract Base Class defining the contract for all retrieval components.
    """

    @abstractmethod
    def retrieve(self, query: str) -> list[NodeWithScore]:
        """
        Retrieves the top-K most relevant document nodes for a given query.

        Args:
            query: The search query string.

        Returns:
            List of LlamaIndex NodeWithScore objects.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Return the unique name of this retrieval strategy (used for logs and Langfuse tags).
        """
        pass
