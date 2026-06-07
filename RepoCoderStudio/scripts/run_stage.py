"""Single CLI entrypoint for any stage.

    python -m scripts.run_stage --stage 1 --config configs/stage1.yaml --limit 5

It loads the YAML config, applies CLI overrides, prints the environment summary, then dispatches to
the selected stage's ``run.run_stage(cfg)``. Every stage exposes the same entrypoint, so this file
never needs per-stage special cases.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys

from repocoder.common import env
from repocoder.common.config import apply_overrides, load_config

# Stage number -> package containing a `run` module with `run_stage(cfg) -> dict`.
STAGE_PACKAGES = {
    1: "repocoder.stage1_documentation",
    2: "repocoder.stage2_nl2python",
    3: "repocoder.stage3_py2java",
    4: "repocoder.stage4_repo_awareness",
    5: "repocoder.stage5_rag",
    6: "repocoder.stage6_agentic",
}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a RepoCoder Studio stage end-to-end.")
    p.add_argument("--stage", type=int, required=True, choices=sorted(STAGE_PACKAGES))
    p.add_argument("--config", required=True, help="Path to configs/stageN.yaml")
    p.add_argument("--limit", type=int, default=None, help="Cap number of examples")
    p.add_argument("--model", default=None, help="Override model.id from the config")
    p.add_argument(
        "--dataset",
        default=None,
        help="Override dataset.name from the config (e.g. 'synthetic' for an offline run)",
    )
    p.add_argument("--dry-run", action="store_true", help="Print resolved config + env, then exit")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, {"limit": args.limit})
    if args.model:
        cfg.setdefault("model", {})["id"] = args.model
    if args.dataset:
        cfg.setdefault("dataset", {})["name"] = args.dataset

    print("== Environment ==")
    print(json.dumps(env.summary(), indent=2))
    print(f"== Stage {args.stage}: {cfg.get('name', '?')} ==")

    if args.dry_run:
        print(json.dumps(dict(cfg), indent=2, default=str))
        return 0

    pkg = STAGE_PACKAGES[args.stage]
    try:
        run_mod = importlib.import_module(f"{pkg}.run")
    except ModuleNotFoundError as exc:
        print(
            f"Stage {args.stage} is not implemented on this branch ({exc}). "
            f"Check out branch stage-{args.stage}-* to run it.",
            file=sys.stderr,
        )
        return 2

    summary = run_mod.run_stage(cfg)
    print("== Results ==")
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
