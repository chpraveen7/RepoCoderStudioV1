"""Platform detection and path resolution.

The whole point of this module is portability: the same stage code should run on Google Colab,
Kaggle, or a local machine with no edits. Anything platform-specific (where caches live, whether a
GPU/MPS device is present, where to persist outputs) is decided here and nowhere else.

    >>> from repocoder.common.env import detect_platform, get_paths, get_device
    >>> detect_platform()
    'local'
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

# Platform identifiers returned by detect_platform().
COLAB = "colab"
KAGGLE = "kaggle"
LOCAL = "local"


@lru_cache(maxsize=1)
def detect_platform() -> str:
    """Return one of ``"colab"``, ``"kaggle"``, or ``"local"``.

    Detection is based on import-ability / environment markers rather than anything that requires
    network access, so it is safe to call offline and in tests.
    """
    if "google.colab" in sys.modules or _can_import("google.colab"):
        return COLAB
    # Kaggle sets these env vars inside its notebook kernels.
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.path.isdir("/kaggle"):
        return KAGGLE
    return LOCAL


def _can_import(name: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


@dataclass(frozen=True)
class Paths:
    """Resolved, writable locations for this run.

    All paths are created on access via :func:`get_paths` so callers can rely on them existing.
    """

    root: Path        # project/data root for this platform
    data: Path        # datasets
    cache: Path       # HF / model cache (HF_HOME points here)
    outputs: Path     # run outputs, results, logs


@lru_cache(maxsize=1)
def get_paths() -> Paths:
    """Resolve and create the data/cache/output directories for the current platform.

    - **Colab:** prefer a mounted Google Drive (``/content/drive/MyDrive/repocoder``) so artifacts
      survive disconnects; fall back to ephemeral ``/content/repocoder`` if Drive isn't mounted.
    - **Kaggle:** use the persistent working dir ``/kaggle/working/repocoder``.
    - **Local:** use ``$REPOCODER_HOME`` if set, else ``./.repocoder`` under the current directory.
    """
    platform = detect_platform()

    if platform == COLAB:
        drive = Path("/content/drive/MyDrive")
        root = (drive / "repocoder") if drive.is_dir() else Path("/content/repocoder")
    elif platform == KAGGLE:
        root = Path("/kaggle/working/repocoder")
    else:
        root = Path(os.environ.get("REPOCODER_HOME", Path.cwd() / ".repocoder"))

    paths = Paths(
        root=root,
        data=root / "data",
        cache=root / "cache",
        outputs=root / "outputs",
    )
    for p in (paths.root, paths.data, paths.cache, paths.outputs):
        p.mkdir(parents=True, exist_ok=True)

    # Route Hugging Face caches at the resolved location so weights aren't re-downloaded.
    os.environ.setdefault("HF_HOME", str(paths.cache / "huggingface"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(paths.cache / "datasets"))
    return paths


def get_device() -> str:
    """Return the best available torch device string: ``"cuda"``, ``"mps"``, or ``"cpu"``.

    Imported lazily so this module is usable without torch installed (e.g., during light tests).
    """
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def supports_4bit() -> bool:
    """True only on CUDA hosts with bitsandbytes available (4-bit is unsupported on MPS/CPU)."""
    return get_device() == "cuda" and _can_import("bitsandbytes")


def summary() -> dict:
    """A small dict describing the environment — handy to log at the top of a run."""
    paths = get_paths()
    return {
        "platform": detect_platform(),
        "device": get_device(),
        "supports_4bit": supports_4bit(),
        "root": str(paths.root),
        "python": sys.version.split()[0],
    }


if __name__ == "__main__":  # `python -m repocoder.common.env`
    import json

    print(json.dumps(summary(), indent=2))
