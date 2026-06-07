"""Sandboxed Python execution for functional-correctness metrics.

Generated code is **untrusted**. We never ``exec()`` it in the host process — every candidate runs
in a separate ``python`` subprocess with a hard wall-clock timeout, so a hang or crash can't take
down the evaluator. This backs Stage 2's execution accuracy and is reused by Stage 5.

Only the standard library is used.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass
class ExecResult:
    passed: bool
    timed_out: bool
    returncode: int
    stdout: str
    stderr: str


def run_python(program: str, *, timeout_s: float = 10.0) -> ExecResult:
    """Run a self-contained Python ``program`` in a subprocess.

    ``program`` should be a complete script that raises (e.g. via ``assert``) on failure and exits 0
    on success — exactly the shape produced by appending HumanEval/MBPP test code to a candidate
    solution. ``passed`` is True iff the process exits 0 within the timeout.
    """
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "candidate.py"
        script.write_text(program, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, str(script)],
                capture_output=True,
                text=True,
                timeout=timeout_s,
                cwd=tmp,
            )
        except subprocess.TimeoutExpired as exc:
            return ExecResult(False, True, -1, exc.stdout or "", exc.stderr or "")
        return ExecResult(proc.returncode == 0, False, proc.returncode, proc.stdout, proc.stderr)


def build_program(solution: str, test: str, entry_point: str | None = None) -> str:
    """Assemble a runnable script from a candidate ``solution`` and its ``test`` code.

    Mirrors the HumanEval convention: the solution defines a function, the test calls
    ``check(<entry_point>)`` or contains bare asserts.
    """
    parts = [solution, "", test]
    if entry_point and "check(" in test:
        parts.append(f"\ncheck({entry_point})")
    return "\n".join(parts)


def execution_accuracy(
    solutions: Sequence[str],
    tests: Sequence[str],
    *,
    entry_points: Sequence[str] | None = None,
    timeout_s: float = 10.0,
) -> dict:
    """Fraction of (solution, test) pairs that execute successfully.

    Returns ``{"execution_accuracy": float, "passed": int, "total": int, "per_item": [bool,...]}``.
    """
    entry_points = entry_points or [None] * len(solutions)
    per_item: list[bool] = []
    for sol, test, ep in zip(solutions, tests, entry_points):
        program = build_program(sol, test, ep)
        per_item.append(run_python(program, timeout_s=timeout_s).passed)
    total = len(per_item)
    passed = sum(per_item)
    return {
        "execution_accuracy": passed / total if total else 0.0,
        "passed": passed,
        "total": total,
        "per_item": per_item,
    }
