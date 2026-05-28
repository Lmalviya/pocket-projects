"""
app/retrieval/__init__.py
========================
Core retrieval and reranking layer of the RAG Eval Lab.

Re-exports:
  - BaseRetriever: Abstract Base Class for all retrievers.
  - DenseRetriever: Semantic cosine vector retriever.
  - SparseRetriever: Lexical keyphrase retriever.
  - HybridRetriever: Consolidated vector & keyphrase RRF fusion retriever.
  - reciprocal_rank_fusion: Python-side RRF algorithm helper.
  - BaseReranker: Abstract Base Class for all Stage-2 rerankers.
  - CrossEncoderReranker: Local MS-MARCO Cross-Encoder reranker.
  - CohereReranker: Hosted Cohere API Rerank v3 reranker.
  - get_reranker: Reranker factory.
"""

from app.retrieval.base import BaseRetriever
from app.retrieval.dense import DenseRetriever
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.sparse import SparseRetriever
from app.retrieval.utils import reciprocal_rank_fusion
from app.retrieval.reranker.base import BaseReranker
from app.retrieval.reranker.cohere import CohereReranker
from app.retrieval.reranker.cross_encoder import CrossEncoderReranker
from app.retrieval.reranker.factory import get_reranker

__all__ = [
    "BaseRetriever",
    "DenseRetriever",
    "SparseRetriever",
    "HybridRetriever",
    "reciprocal_rank_fusion",
    "BaseReranker",
    "CrossEncoderReranker",
    "CohereReranker",
    "get_reranker",
]
