"""
app/retrieval/utils.py
=======================
Helper utilities for information retrieval and score fusion.

📚 LESSON — Reciprocal Rank Fusion (RRF) Math:
Dense search and sparse search output scores on entirely different scales:
  - Dense (Cosine): ranges from -1.0 to 1.0 (usually 0.3 to 0.9 in practice).
  - Sparse (BM25 or SPLADE): unbound positive numbers (e.g. 0.0 to 30.0+).

We cannot simply add them. Instead, we use Reciprocal Rank Fusion (RRF).
RRF ignores the raw scores and fuses candidate lists using the **rank position** of each document:
  Score(d) = sum( 1 / (k + rank_i(d)) )
  
Where:
  - `rank_i(d)` is the 1-based index position of document `d` in retriever list `i`.
  - `k` is a smoothing parameter (constant, default: 60) that penalizes candidates
    ranking at the very top (e.g. Rank 1) from completely overriding other candidates.
"""

from llama_index.core.schema import NodeWithScore

from app.utils.logger import get_logger

logger = get_logger(__name__)


def reciprocal_rank_fusion(
    results_list: list[list[NodeWithScore]],
    k: int = 60,
    top_k: int = 20,
) -> list[NodeWithScore]:
    """
    Fuses multiple lists of ranked NodeWithScore objects into a single unified ranked list.

    Args:
        results_list: List of candidate NodeWithScore lists (one list per retriever strategy).
        k: RRF rank-smoothing constant (default: 60).
        top_n: Maximum number of fused nodes to return.

    Returns:
        List of fused NodeWithScore objects sorted by RRF score descending.
    """
    logger.debug("Running Reciprocal Rank Fusion on {count} retrieval lists...", count=len(results_list))

    rrf_scores: dict[str, float] = {}  # node_id -> RRF fused score
    nodes_map: dict[str, NodeWithScore] = {}  # node_id -> original NodeWithScore reference

    # 1. Accumulate RRF scores across all retrieval candidate lists
    for retriever_results in results_list:
        for rank, node_with_score in enumerate(retriever_results, start=1):
            node_id = node_with_score.node.node_id
            nodes_map[node_id] = node_with_score

            # Add RRF score contribution for this rank position: 1 / (k + rank)
            score_contribution = 1.0 / (k + rank)
            rrf_scores[node_id] = rrf_scores.get(node_id, 0.0) + score_contribution

    # 2. Sort candidate node IDs by their accumulated RRF score descending
    sorted_node_ids = sorted(rrf_scores.keys(), key=lambda node_id: rrf_scores[node_id], reverse=True)

    # 3. Compile the top-N final candidates
    fused_results: list[NodeWithScore] = []
    for node_id in sorted_node_ids[:top_k]:
        orig_node = nodes_map[node_id]
        
        # We return a new NodeWithScore object carrying the merged node and the RRF score
        fused_results.append(
            NodeWithScore(
                node=orig_node.node,
                score=round(rrf_scores[node_id], 6),
            )
        )

    logger.debug("RRF complete. Consolidated {candidates} candidates down to {top_n}.", candidates=len(rrf_scores), top_n=len(fused_results))
    return fused_results
