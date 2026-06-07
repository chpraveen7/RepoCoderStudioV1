"""ROUGE-L for documentation overlap (Stage 1).

Uses the ``rouge_score`` package when available; otherwise falls back to a dependency-free
longest-common-subsequence F-measure so the metric (and tests) still run offline. The fallback is
token-level LCS, which matches ROUGE-L's definition closely enough for a baseline.
"""

from __future__ import annotations

from typing import Sequence


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    """Length of the longest common subsequence of two token lists (DP)."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        curr = [0]
        for j, y in enumerate(b, start=1):
            curr.append(prev[j - 1] + 1 if x == y else max(prev[j], curr[j - 1]))
        prev = curr
    return prev[-1]


def _rouge_l_fallback(prediction: str, reference: str) -> float:
    pred_tokens = prediction.split()
    ref_tokens = reference.split()
    lcs = _lcs_length(pred_tokens, ref_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def rouge_l(predictions: Sequence[str], references: Sequence[str]) -> dict:
    """Mean ROUGE-L F-measure over prediction/reference pairs."""
    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = [
            scorer.score(ref, pred)["rougeL"].fmeasure
            for pred, ref in zip(predictions, references)
        ]
        backend = "rouge_score"
    except ImportError:
        scores = [_rouge_l_fallback(p, r) for p, r in zip(predictions, references)]
        backend = "fallback_lcs"

    return {
        "rouge_l": sum(scores) / len(scores) if scores else 0.0,
        "backend": backend,
        "n": len(scores),
    }
