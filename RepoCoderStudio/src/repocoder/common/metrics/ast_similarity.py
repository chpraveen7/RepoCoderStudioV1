"""AST similarity — structural equivalence between two code snippets (Stages 3 & 4).

Compares the multiset of AST node types of prediction vs reference and reports a Jaccard-style
overlap in [0, 1].

- **Python** uses the stdlib ``ast`` module (always available).
- **Other languages** (e.g. Java) use ``tree_sitter_languages`` when installed.
- If parsing isn't possible, falls back to a token-bag overlap so a number is still produced
  (labelled via ``backend``).
"""

from __future__ import annotations

from collections import Counter
from typing import Sequence


def _python_node_types(code: str) -> Counter | None:
    import ast

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    return Counter(type(node).__name__ for node in ast.walk(tree))


def _treesitter_node_types(code: str, lang: str) -> Counter | None:
    try:
        from tree_sitter_languages import get_parser
    except ImportError:
        return None
    try:
        parser = get_parser(lang)
        tree = parser.parse(code.encode("utf-8"))
    except Exception:
        return None

    types: Counter = Counter()
    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        types[node.type] += 1
        stack.extend(node.children)
    return types


def _token_bag(code: str) -> Counter:
    import re

    return Counter(re.findall(r"\w+|[^\s\w]", code))


def _overlap(a: Counter, b: Counter) -> float:
    """Weighted Jaccard over two multisets of node types."""
    if not a and not b:
        return 1.0
    inter = sum((a & b).values())
    union = sum((a | b).values())
    return inter / union if union else 0.0


def ast_similarity(
    predictions: Sequence[str],
    references: Sequence[str],
    lang: str = "python",
) -> dict:
    """Mean AST node-type overlap over prediction/reference pairs."""
    scores: list[float] = []
    backend = "ast" if lang == "python" else "tree_sitter"
    used_fallback = False

    for pred, ref in zip(predictions, references):
        if lang == "python":
            pa, ra = _python_node_types(pred), _python_node_types(ref)
        else:
            pa, ra = _treesitter_node_types(pred, lang), _treesitter_node_types(ref, lang)

        if pa is None or ra is None:
            pa, ra = _token_bag(pred), _token_bag(ref)
            used_fallback = True
        scores.append(_overlap(pa, ra))

    return {
        "ast_similarity": sum(scores) / len(scores) if scores else 0.0,
        "backend": "fallback_tokens" if used_fallback else backend,
        "n": len(scores),
    }
