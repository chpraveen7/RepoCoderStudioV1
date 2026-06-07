"""Retrieval metrics: precision@k, recall@k, and MRR@k (Stages 4 & 5).

Pure python. Each query provides a ranked list of retrieved item ids and the set of relevant ids.
"""

from __future__ import annotations

from typing import Sequence


def retrieval_metrics(
    retrieved: Sequence[Sequence],
    relevant: Sequence[set],
    k: int = 10,
) -> dict:
    """Compute mean precision@k, recall@k, and MRR@k over a set of queries.

    Args:
        retrieved: per-query ranked lists of item ids (best first).
        relevant: per-query sets of relevant item ids (ground truth).
        k: cutoff.
    """
    precisions, recalls, rr = [], [], []
    for ranked, gold in zip(retrieved, relevant):
        gold = set(gold)
        topk = list(ranked)[:k]
        hits = sum(1 for item in topk if item in gold)

        precisions.append(hits / k if k else 0.0)
        recalls.append(hits / len(gold) if gold else 0.0)

        # Reciprocal rank: 1/position of the first relevant item within top-k.
        reciprocal = 0.0
        for pos, item in enumerate(topk, start=1):
            if item in gold:
                reciprocal = 1.0 / pos
                break
        rr.append(reciprocal)

    n = len(precisions) or 1
    return {
        f"precision@{k}": sum(precisions) / n,
        f"recall@{k}": sum(recalls) / n,
        f"mrr@{k}": sum(rr) / n,
        "num_queries": len(precisions),
    }
