"""Java compile + run for translation correctness (Stage 3).

Writes translated Java to a temp dir, compiles with ``javac``, and (optionally) runs the resulting
class, all with timeouts. Requires a JDK on PATH; if ``javac`` is absent it raises
:class:`MetricUnavailable` with install guidance.

Returns both **compilation success** and **execution correctness** so the two metrics share one
toolchain invocation.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from . import MetricUnavailable


@dataclass
class JavaResult:
    compiled: bool
    ran: bool
    output_matched: bool | None  # None when no expected output was supplied
    stderr: str


def _public_class_name(code: str) -> str:
    """Best-effort extraction of the public class name (defaults to ``Main``)."""
    m = re.search(r"public\s+class\s+([A-Za-z_]\w*)", code)
    if m:
        return m.group(1)
    m = re.search(r"\bclass\s+([A-Za-z_]\w*)", code)
    return m.group(1) if m else "Main"


def _require_jdk(javac: str) -> None:
    if shutil.which(javac) is None:
        raise MetricUnavailable(
            f"`{javac}` not found on PATH. Install a JDK "
            "(Colab: `apt-get install -y default-jdk`, Mac: `brew install openjdk`)."
        )


def compile_and_run_java_one(
    code: str,
    *,
    expected_output: str | None = None,
    run: bool = True,
    javac: str = "javac",
    java: str = "java",
    compile_timeout_s: float = 30.0,
    run_timeout_s: float = 10.0,
) -> JavaResult:
    """Compile (and optionally run) a single Java source string."""
    _require_jdk(javac)
    cls = _public_class_name(code)

    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / f"{cls}.java"
        src.write_text(code, encoding="utf-8")

        try:
            cproc = subprocess.run(
                [javac, str(src)], capture_output=True, text=True,
                timeout=compile_timeout_s, cwd=tmp,
            )
        except subprocess.TimeoutExpired:
            return JavaResult(False, False, None, "compile timeout")
        if cproc.returncode != 0:
            return JavaResult(False, False, None, cproc.stderr)

        if not run:
            return JavaResult(True, False, None, "")

        try:
            rproc = subprocess.run(
                [java, "-cp", tmp, cls], capture_output=True, text=True,
                timeout=run_timeout_s, cwd=tmp,
            )
        except subprocess.TimeoutExpired:
            return JavaResult(True, False, None, "run timeout")

        ran_ok = rproc.returncode == 0
        matched = None
        if expected_output is not None:
            matched = rproc.stdout.strip() == expected_output.strip()
        return JavaResult(True, ran_ok, matched, rproc.stderr)


def compile_and_run_java(
    sources: Sequence[str],
    *,
    expected_outputs: Sequence[str] | None = None,
    run: bool = True,
    javac: str = "javac",
    java: str = "java",
    compile_timeout_s: float = 30.0,
    run_timeout_s: float = 10.0,
) -> dict:
    """Aggregate compilation success and execution correctness over many Java sources."""
    expected_outputs = expected_outputs or [None] * len(sources)
    results = [
        compile_and_run_java_one(
            code,
            expected_output=exp,
            run=run,
            javac=javac,
            java=java,
            compile_timeout_s=compile_timeout_s,
            run_timeout_s=run_timeout_s,
        )
        for code, exp in zip(sources, expected_outputs)
    ]
    total = len(results) or 1
    compiled = sum(r.compiled for r in results)
    ran = sum(r.ran for r in results)
    judged = [r for r in results if r.output_matched is not None]
    correct = sum(r.output_matched for r in judged)
    return {
        "compilation_success": compiled / total,
        "execution_correctness": (correct / len(judged)) if judged else (ran / total),
        "compiled": compiled,
        "ran": ran,
        "total": len(results),
        "per_item": [r.__dict__ for r in results],
    }
