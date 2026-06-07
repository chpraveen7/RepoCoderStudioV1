"""I/O helpers: JSONL read/write, run directories, and result logging.

Stages produce records (one JSON object per example) and a small metrics summary. Centralizing the
format here keeps every stage's outputs consistent and easy to compare across branches.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator


def read_jsonl(path: str | Path) -> Iterator[dict]:
    """Yield one parsed object per line from a JSONL file (blank lines skipped)."""
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, records: Iterable[dict]) -> int:
    """Write records as JSONL; returns the number of records written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            count += 1
    return count


def write_json(path: str | Path, obj: Any) -> None:
    """Write a single JSON document (e.g., a metrics summary)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2, default=str)


def new_run_dir(base: str | Path, tag: str = "") -> Path:
    """Create a fresh, numbered run directory under ``base`` and return it.

    Avoids ``datetime.now()`` (and keeps runs ordered) by scanning existing ``run-NNN`` dirs and
    incrementing — deterministic and dependency-free.
    """
    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)
    existing = [p.name for p in base.iterdir() if p.is_dir() and p.name.startswith("run-")]
    next_idx = 1 + max(
        (int(n.split("-")[1]) for n in existing if n.split("-")[1].isdigit()), default=0
    )
    name = f"run-{next_idx:03d}" + (f"-{tag}" if tag else "")
    run_dir = base / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
