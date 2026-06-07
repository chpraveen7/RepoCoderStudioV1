# Stage 1 — Code Understanding & Documentation

**Branch:** `stage-1-documentation` · **Phase:** 1 (Foundations)

## Goal
Automatically generate docstrings, summaries, and commit messages from source code to improve
maintainability. Establishes the project's evaluation harness and a **baseline floor**.

## Model
- `Salesforce/codegen-350M-multi` (default) — lightweight, reproducible on CPU/free T4.

## Datasets
- **CoDocBench** — multi-language code↔documentation pairs.
- **CodeSearchNet** — code↔natural-language alignment (semantic evaluation).
- *Synthetic fallback* — a few hand-written code/doc pairs so the pipeline runs offline.

## Metrics
- **CodeBLEU** — syntax + data-flow aware similarity.
- **CodeBERTScore** — embedding-based semantic similarity.
- **ROUGE-L** — overlap with reference summaries.

## Pipeline (`src/repocoder/stage1_documentation/`)
`data.py` → load code/doc records · `run.py` → prompt model for a docstring/summary ·
`evaluate.py` → score against references with the three metrics.

```bash
python -m scripts.run_stage --stage 1 --config configs/stage1.yaml --limit 20
```

## Notes / pitfalls
- Surface metrics (ROUGE-L) reward style overlap; pair with semantic metrics — see
  [`../04_challenges.md`](../04_challenges.md#stage-1--documentation).
- This baseline is meant to be **beaten** later by Qwen2.5-Coder + RAG (Stage 5).
