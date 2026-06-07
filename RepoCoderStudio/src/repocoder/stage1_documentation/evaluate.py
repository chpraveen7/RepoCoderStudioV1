"""Stage 1 evaluation — score generated documentation against references.

Computes the three Stage 1 metrics (CodeBLEU, CodeBERTScore, ROUGE-L). CodeBERTScore is optional:
if ``bert-score`` isn't installed it's skipped with a note rather than failing the run, so the
pipeline still produces CodeBLEU + ROUGE-L offline.
"""

from __future__ import annotations

from typing import Sequence

from ..common.metrics import MetricUnavailable, code_bert_score, code_bleu, rouge_l


def evaluate(
    predictions: Sequence[str],
    references: Sequence[str],
    *,
    metrics: Sequence[str] = ("codebleu", "rouge_l", "codebertscore"),
    lang: str = "python",
) -> dict:
    """Return a flat dict of metric → value for documentation generation."""
    results: dict = {}

    if "rouge_l" in metrics:
        results.update(rouge_l(predictions, references))

    if "codebleu" in metrics:
        cb = code_bleu(predictions, references, lang=lang)
        results["codebleu"] = cb["codebleu"]
        results["codebleu_backend"] = cb["backend"]

    if "codebertscore" in metrics:
        try:
            results.update(code_bert_score(predictions, references, lang=lang))
        except MetricUnavailable as exc:
            results["codebertscore_skipped"] = str(exc)

    results["n"] = len(predictions)
    return results
