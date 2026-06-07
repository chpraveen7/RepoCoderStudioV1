"""Stage 1 data loading — code↔documentation pairs.

Normalizes several sources into a single record shape so ``run.py``/``evaluate.py`` don't care which
dataset is configured::

    {"id": str, "code": str, "reference": str, "language": str}

Sources (``cfg.dataset.name``):
- ``code_search_net`` — HF ``code_search_net`` (code + docstring).
- ``codocbench``       — HF CoDocBench code/doc pairs.
- ``synthetic``        — a tiny built-in set so the pipeline runs fully offline.

Any HF load failure (no network, gated, renamed) falls back to ``synthetic`` with a warning, so a
Colab cell never hard-fails just because a dataset is unreachable.
"""

from __future__ import annotations

import warnings
from typing import Optional

Record = dict

# A handful of hand-written pairs. Enough to exercise the full load→generate→score path with no
# downloads (used by tests and as the offline fallback).
SYNTHETIC: list[Record] = [
    {
        "id": "syn-1",
        "language": "python",
        "code": "def add(a, b):\n    return a + b",
        "reference": "Return the sum of two numbers a and b.",
    },
    {
        "id": "syn-2",
        "language": "python",
        "code": "def is_even(n):\n    return n % 2 == 0",
        "reference": "Return True if n is even, otherwise False.",
    },
    {
        "id": "syn-3",
        "language": "python",
        "code": "def reverse(s):\n    return s[::-1]",
        "reference": "Return the reversed copy of string s.",
    },
    {
        "id": "syn-4",
        "language": "python",
        "code": "def factorial(n):\n    r = 1\n    for i in range(2, n + 1):\n        r *= i\n    return r",
        "reference": "Compute the factorial of a non-negative integer n.",
    },
]


def load_dataset(cfg) -> list[Record]:
    """Load and normalize Stage 1 records according to ``cfg.dataset``.

    Honors ``cfg.limit`` (None = all). Always returns at least the synthetic set on failure.
    """
    name = cfg.get_path("dataset.name", "synthetic")
    limit = cfg.get("limit")

    if name == "synthetic":
        records = list(SYNTHETIC)
    elif name == "code_search_net":
        records = _load_code_search_net(cfg) or list(SYNTHETIC)
    elif name == "codocbench":
        records = _load_codocbench(cfg) or list(SYNTHETIC)
    else:
        warnings.warn(f"Unknown dataset {name!r}; using synthetic.")
        records = list(SYNTHETIC)

    return records[:limit] if limit else records


def _load_code_search_net(cfg) -> Optional[list[Record]]:
    lang = cfg.get_path("dataset.config", "python")
    split = cfg.get_path("dataset.split", "test")
    try:
        from datasets import load_dataset as hf_load

        ds = hf_load("code_search_net", lang, split=split, streaming=True)
        out: list[Record] = []
        cap = cfg.get("limit") or 200
        for i, row in enumerate(ds):
            doc = (row.get("func_documentation_string") or "").strip()
            code = (row.get("func_code_string") or "").strip()
            if not doc or not code:
                continue
            out.append({"id": f"csn-{i}", "language": lang, "code": code, "reference": doc})
            if len(out) >= cap:
                break
        return out or None
    except Exception as exc:  # network/schema/gated — degrade gracefully
        warnings.warn(f"code_search_net load failed ({exc}); falling back to synthetic.")
        return None


def _load_codocbench(cfg) -> Optional[list[Record]]:
    split = cfg.get_path("dataset.split", "test")
    try:
        from datasets import load_dataset as hf_load

        # CoDocBench schema varies by mirror; pull common field names defensively.
        ds = hf_load("Kunpai/codocbench", split=split, streaming=True)
        out: list[Record] = []
        cap = cfg.get("limit") or 200
        for i, row in enumerate(ds):
            code = (row.get("code") or row.get("function") or "").strip()
            doc = (row.get("docstring") or row.get("documentation") or "").strip()
            if not code or not doc:
                continue
            out.append({"id": f"cdb-{i}", "language": row.get("language", "python"),
                        "code": code, "reference": doc})
            if len(out) >= cap:
                break
        return out or None
    except Exception as exc:
        warnings.warn(f"codocbench load failed ({exc}); falling back to synthetic.")
        return None
