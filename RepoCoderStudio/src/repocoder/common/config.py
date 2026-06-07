"""Tiny YAML config loader with dotted access and CLI overrides.

Configs live in ``configs/stageN.yaml``. This keeps loading them in one place so every stage and
the ``run_stage`` CLI read them identically.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class Config(dict):
    """A dict that also supports dotted access (``cfg.model.id``) for convenience."""

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:  # pragma: no cover - attribute error semantics
            raise AttributeError(item) from exc
        return Config(value) if isinstance(value, dict) else value

    def get_path(self, dotted: str, default: Any = None) -> Any:
        """Read a nested key by dotted path, e.g. ``cfg.get_path("model.id")``."""
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


def load_config(path: str | Path) -> Config:
    """Load a YAML config file into a :class:`Config`."""
    import yaml

    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} must be a mapping at top level, got {type(data)}")
    return Config(data)


def apply_overrides(cfg: Config, overrides: dict[str, Any]) -> Config:
    """Return a shallow copy of ``cfg`` with top-level keys replaced by ``overrides``.

    Used by the CLI to apply ``--limit`` / ``--model`` style flags without mutating the file.
    ``None`` values are ignored so unset CLI flags don't clobber config values.
    """
    merged = Config(cfg)
    for key, value in overrides.items():
        if value is not None:
            merged[key] = value
    return merged
