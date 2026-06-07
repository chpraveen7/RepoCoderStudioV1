"""pass@k — the standard code-synthesis metric (Chen et al., 2021).

Pure python, no dependencies. Given, per problem, how many candidate samples were generated (n) and
how many passed (c), the unbiased estimator of pass@k is::

    pass@k = 1 - C(n-c, k) / C(n, k)

averaged over problems.
"""

from __future__ import annotations

from typing import Sequence


def _pass_at_k_single(n: int, c: int, k: int) -> float:
    """Unbiased pass@k for one problem with ``n`` samples, ``c`` correct, target ``k``."""
    if k > n:
        raise ValueError(f"k={k} cannot exceed number of samples n={n}")
    if n - c < k:
        # Even the worst k-subset must contain a correct sample.
        return 1.0
    # Product form of 1 - C(n-c, k)/C(n, k), numerically stable for modest n.
    prob_all_fail = 1.0
    for i in range(k):
        prob_all_fail *= (n - c - i) / (n - i)
    return 1.0 - prob_all_fail


def pass_at_k(results: Sequence[Sequence[bool]], k: int | Sequence[int] = 1) -> dict:
    """Compute pass@k averaged over problems.

    Args:
        results: per-problem lists of pass/fail booleans (one entry per generated sample).
        k: a single k or a list of k values to report.

    Returns:
        ``{"pass@1": ..., "pass@10": ..., "num_problems": N}``.
    """
    ks = [k] if isinstance(k, int) else list(k)
    out: dict = {"num_problems": len(results)}
    for kk in ks:
        scores = []
        for samples in results:
            n = len(samples)
            c = sum(1 for s in samples if s)
            scores.append(_pass_at_k_single(n, c, kk))
        out[f"pass@{kk}"] = sum(scores) / len(scores) if scores else 0.0
    return out
