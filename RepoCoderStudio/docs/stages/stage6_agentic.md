# Stage 6 — Agentic Coding Assistant

**Branch:** `stage-6-agentic` · **Phase:** 3 (Generalizability)

## Goal
Orchestrate the earlier building blocks into multi-step workflows (bug fixing, feature integration)
via cooperating agents.

## Framework
**LangGraph** with four nodes:
- **Planner** — decomposes the task into steps.
- **Retriever** — uses Stage 4 index + Stage 5 RAG to fetch context.
- **Generator** — Qwen2.5-Coder produces code/edits.
- **Validator** — runs tests / static checks, gates each transition, can trigger bounded retry.

## Datasets
- **SWE-Bench Lite** + artifacts integrated from earlier stages.

## Metrics
- **Task success rate** — proportion of tasks completed correctly.
- **Workflow success rate** — proportion of multi-step workflows completed.
- **End-to-end accuracy** — correctness of final outputs.
- **User satisfaction** — qualitative usability (rubric).

## Pipeline (`src/repocoder/stage6_agentic/`)
`data.py` → load tasks · `run.py` → build & execute the LangGraph graph (Planner→Retriever→
Generator→Validator) · `evaluate.py` → success-rate metrics with per-node logs for replay.

```bash
python -m scripts.run_stage --stage 6 --config configs/stage6.yaml --limit 3
```

## Notes / pitfalls
- Cap steps and **checkpoint state between nodes** so a disconnect doesn't restart everything.
- The Validator must be able to *reject and retry*, not merely score — prevents error cascades.
- Long runs → use Lightning AI / Modal, not Colab (see [`../01_compute_platforms.md`](../01_compute_platforms.md)).
