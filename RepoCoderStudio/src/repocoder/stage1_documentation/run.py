"""Stage 1 — generate documentation and evaluate it.

``run_stage(cfg)`` loads code↔doc records, prompts the configured model (and any
``comparison_models``) for a one-line summary/docstring per function, scores the outputs, and writes
predictions + a metrics summary to a fresh run directory.

Designed so the model step is the only heavy part: data loading and evaluation work offline, which
is what the CPU smoke test exercises.
"""

from __future__ import annotations

import random
from typing import Sequence

from ..common import env, io_utils
from . import data, evaluate


def build_prompt(record: dict, tokenizer=None) -> str:
    """Build a documentation prompt for one code record.

    Uses a chat template for instruct models (if the tokenizer exposes one), else a base-model
    completion prompt suitable for CodeGen.
    """
    code = record["code"]
    instruction = (
        "Write a single-sentence summary describing what the following "
        f"{record.get('language', 'python')} function does.\n\n{code}\n\nSummary:"
    )
    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return instruction


def _clean(text: str) -> str:
    """Keep the first non-empty line of a generation as the summary."""
    for line in text.strip().splitlines():
        line = line.strip().lstrip("#").strip()
        if line:
            return line
    return text.strip()


def _generate_for_model(model_id: str, records: list[dict], cfg) -> list[str]:
    """Load a model, generate one summary per record, return cleaned predictions."""
    from ..common.generation import generate
    from ..common.model_loader import load_model

    lm = load_model(
        model_id,
        load_in_4bit=bool(cfg.get_path("model.load_in_4bit", False)),
        dtype=cfg.get_path("model.dtype", "auto"),
    )
    gen = cfg.get("generation", {})
    prompts = [build_prompt(r, lm.tokenizer) for r in records]
    outputs = generate(
        lm,
        prompts,
        max_new_tokens=int(gen.get("max_new_tokens", 128)),
        temperature=float(gen.get("temperature", 0.2)),
        top_p=float(gen.get("top_p", 0.95)),
        do_sample=bool(gen.get("do_sample", False)),
    )
    return [_clean(o[0]) for o in outputs]


def run_stage(cfg) -> dict:
    """Entry point used by ``scripts.run_stage``."""
    random.seed(int(cfg.get("seed", 42)))

    records = data.load_dataset(cfg)
    references = [r["reference"] for r in records]
    metrics = cfg.get("metrics", ["codebleu", "rouge_l", "codebertscore"])

    paths = env.get_paths()
    run_dir = io_utils.new_run_dir(paths.outputs / "stage1", tag=cfg.get("name", ""))

    model_ids: Sequence[str] = [cfg.get_path("model.id")] + list(cfg.get("comparison_models", []))
    per_model: dict = {}

    for model_id in model_ids:
        predictions = _generate_for_model(model_id, records, cfg)
        scores = evaluate.evaluate(predictions, references, metrics=metrics)

        # Persist per-example predictions for inspection / later RAG corpora.
        io_utils.write_jsonl(
            run_dir / f"predictions__{model_id.replace('/', '_')}.jsonl",
            (
                {"id": r["id"], "code": r["code"], "reference": r["reference"], "prediction": p}
                for r, p in zip(records, predictions)
            ),
        )
        per_model[model_id] = scores

    summary = {
        "stage": 1,
        "num_examples": len(records),
        "dataset": cfg.get_path("dataset.name", "synthetic"),
        "models": per_model,
        "run_dir": str(run_dir),
        "env": env.summary(),
    }
    io_utils.write_json(run_dir / "summary.json", summary)
    return summary
