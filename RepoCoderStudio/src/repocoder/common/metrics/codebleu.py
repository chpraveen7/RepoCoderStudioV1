"""CodeBLEU (Ren et al., 2020) — BLEU augmented with code syntax/data-flow terms.

Prefers the ``codebleu`` package (which uses tree-sitter for the AST/dataflow components). If it
isn't installed, falls back to a token-level BLEU-style n-gram overlap so Stage 1/2/3 pipelines and
tests still produce a number — clearly labelled via the ``backend`` field so it's never mistaken for
the real metric.
"""

from __future__ import annotations

from collections import Counter
from typing import Sequence


def _ngram_precision(pred: list[str], ref: list[str], n: int) -> float:
    if len(pred) < n:
        return 0.0
    pred_ngrams = Counter(tuple(pred[i : i + n]) for i in range(len(pred) - n + 1))
    ref_ngrams = Counter(tuple(ref[i : i + n]) for i in range(len(ref) - n + 1))
    overlap = sum((pred_ngrams & ref_ngrams).values())
    total = max(sum(pred_ngrams.values()), 1)
    return overlap / total


def _bleu_fallback(prediction: str, reference: str, max_n: int = 4) -> float:
    """Geometric mean of 1..4-gram precisions with a brevity penalty (a BLEU stand-in)."""
    import math

    pred, ref = prediction.split(), reference.split()
    if not pred or not ref:
        return 0.0
    precisions = [_ngram_precision(pred, ref, n) for n in range(1, max_n + 1)]
    precisions = [p for p in precisions if p > 0]
    if not precisions:
        return 0.0
    geo = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
    bp = min(1.0, math.exp(1 - len(ref) / len(pred)))
    return bp * geo


def code_bleu(
    predictions: Sequence[str],
    references: Sequence[str],
    lang: str = "python",
) -> dict:
    """Mean CodeBLEU over prediction/reference pairs.

    Returns ``{"codebleu": float, "backend": "codebleu" | "fallback_bleu", "n": int}``.
    """
    try:
        from codebleu import calc_codebleu

        result = calc_codebleu(list(references), list(predictions), lang=lang)
        return {"codebleu": result["codebleu"], "backend": "codebleu", "n": len(predictions)}
    except ImportError:
        scores = [_bleu_fallback(p, r) for p, r in zip(predictions, references)]
        return {
            "codebleu": sum(scores) / len(scores) if scores else 0.0,
            "backend": "fallback_bleu",
            "n": len(scores),
        }
