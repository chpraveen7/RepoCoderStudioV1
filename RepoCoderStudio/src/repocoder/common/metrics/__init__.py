"""Evaluation metrics, one module per family.

Heavy backends (codebleu, bert-score, tree-sitter, a JDK) are imported lazily *inside* each
function, so importing this package is cheap and the pure-python metrics (pass@k, execution,
retrieval) work with no extra dependencies. Where a backend is missing, functions raise a clear
``MetricUnavailable`` rather than failing obscurely.

A small name→callable registry (:data:`REGISTRY`) lets ``evaluate.py`` and the CLI select metrics
from a config list like ``metrics: [codebleu, rouge_l]``.
"""

from __future__ import annotations

from typing import Callable


class MetricUnavailable(RuntimeError):
    """Raised when a metric's backend dependency isn't installed."""


from .ast_similarity import ast_similarity            # noqa: E402
from .codebertscore import code_bert_score            # noqa: E402
from .codebleu import code_bleu                        # noqa: E402
from .compile_java import compile_and_run_java         # noqa: E402
from .execution import execution_accuracy, run_python  # noqa: E402
from .pass_k import pass_at_k                           # noqa: E402
from .retrieval import retrieval_metrics               # noqa: E402
from .rouge_l import rouge_l                            # noqa: E402

# Metric name -> callable. Used to resolve config `metrics: [...]` entries.
REGISTRY: dict[str, Callable] = {
    "codebleu": code_bleu,
    "codebertscore": code_bert_score,
    "rouge_l": rouge_l,
    "pass_k": pass_at_k,
    "execution_accuracy": execution_accuracy,
    "compilation_success": compile_and_run_java,
    "execution_correctness": compile_and_run_java,
    "ast_similarity": ast_similarity,
    "retrieval_precision": retrieval_metrics,
    "retrieval_recall": retrieval_metrics,
    "mrr_at_k": retrieval_metrics,
}


def get_metric(name: str) -> Callable:
    """Look up a metric callable by config name."""
    if name not in REGISTRY:
        raise KeyError(f"Unknown metric {name!r}. Known: {sorted(REGISTRY)}")
    return REGISTRY[name]


__all__ = [
    "MetricUnavailable",
    "REGISTRY",
    "get_metric",
    "code_bleu",
    "code_bert_score",
    "rouge_l",
    "pass_at_k",
    "execution_accuracy",
    "run_python",
    "compile_and_run_java",
    "ast_similarity",
    "retrieval_metrics",
]
