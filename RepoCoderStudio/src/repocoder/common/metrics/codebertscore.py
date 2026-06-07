"""CodeBERTScore (Zhang et al., 2023) — embedding-based semantic similarity for code/docs.

Uses ``bert-score`` with a code-aware model (``microsoft/codebert-base``) when available. This
metric genuinely needs an embedding model; without ``bert-score`` installed it raises
:class:`MetricUnavailable` rather than silently degrading (a semantic metric has no meaningful
token-overlap fallback).
"""

from __future__ import annotations

from typing import Sequence

from . import MetricUnavailable


def code_bert_score(
    predictions: Sequence[str],
    references: Sequence[str],
    lang: str = "python",
    model_type: str = "microsoft/codebert-base",
) -> dict:
    """Mean CodeBERTScore F1 over prediction/reference pairs."""
    try:
        from bert_score import score as bert_score
    except ImportError as exc:
        raise MetricUnavailable(
            "CodeBERTScore requires `bert-score` (pip install bert-score). "
            "Install requirements/stage1.txt."
        ) from exc

    P, R, F1 = bert_score(
        list(predictions),
        list(references),
        model_type=model_type,
        num_layers=12,
        lang=lang,
        rescale_with_baseline=False,
        verbose=False,
    )
    return {
        "codebertscore_f1": float(F1.mean()),
        "codebertscore_precision": float(P.mean()),
        "codebertscore_recall": float(R.mean()),
        "n": len(predictions),
    }
