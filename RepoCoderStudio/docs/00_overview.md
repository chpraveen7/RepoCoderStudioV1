# 00 — Overview & Architecture

## Problem

Modern software engineering struggles with maintaining large codebases, migrating legacy systems,
and letting non-technical users interact with code. Developers spend disproportionate time writing
documentation, translating code between languages, and navigating unfamiliar repositories.

**RepoCoder Studio** addresses this with a repository-aware assistant that:

1. Automates documentation and commit messages.
2. Generates executable code from natural language.
3. Translates code between languages for interoperability.
4. Understands repository structure for navigation.
5. Improves accuracy with retrieval-augmented generation (RAG).
6. Orchestrates multi-step workflows with agents.

## How the stages connect

The project is deliberately **incremental** — each stage produces an artifact that later stages
consume:

```
Stage 1  Documentation ─────────────┐
         (code → docs)              │  docs become a retrieval corpus
Stage 2  NL → Python ───────────────┤  generated functions become indexed examples
         (prompt → code)            │
Stage 3  Python → Java ─────────────┤  translations become bilingual retrieval pairs
         (code → code)              │
                                    ▼
Stage 4  Repository awareness  →  structural map (AST index) of a codebase
                                    │
Stage 5  RAG  →  grounds Stage 2/3 generation in Stage 1/3/4 corpora
                                    │
Stage 6  Agentic  →  Planner → Retriever → Generator → Validator over the above
```

This is why the git branches are stacked (`stage-N` branched off `stage-N-1`): the dependency
graph is linear, and each branch is a working superset of the previous.

## Three phases

| Phase | Stages | Theme | Goal |
|---|---|---|---|
| 1 — Foundations | 1, 2, 3 | Baseline capabilities | Establish documentation, synthesis, translation with measurable baselines |
| 2 — Integration | 4, 5 | Contextual accuracy | Add repository structure + retrieval grounding |
| 3 — Generalizability | 6 | Orchestration | Compose the above into real multi-step workflows |

## Software architecture

Every stage is implemented with the **same three-module shape**, driven by a YAML config:

- `data.py` — load and normalize the stage's dataset(s) into a common record format.
- `run.py` — load a model (via `common/model_loader.py`) and generate outputs.
- `evaluate.py` — score outputs with the stage's metrics (from `common/metrics/`).

Shared, reusable building blocks live in [`src/repocoder/common/`](../src/repocoder/common):

| Module | Responsibility |
|---|---|
| `env.py` | Detect Colab / Kaggle / local; resolve cache, data, and output paths |
| `model_loader.py` | Portable Hugging Face model/tokenizer loading (dtype, 4-bit, device map) |
| `generation.py` | Greedy/sampled generation, chat templating, fill-in-the-middle helper |
| `io_utils.py` | JSONL read/write, run directories, result logging |
| `metrics/` | CodeBLEU, ROUGE-L, CodeBERTScore, pass@k, execution, compile, AST similarity, retrieval |

A single CLI — [`scripts/run_stage.py`](../scripts/run_stage.py) — runs any stage end-to-end
(`load → generate → evaluate`). Notebooks are thin wrappers around this so the logic is tested
and version-controlled in `src/`, not buried in `.ipynb` cells.

## Design principles

- **Portable by default.** No code path assumes Colab; platform specifics are isolated in `env.py`.
- **Light repo.** Model weights and datasets are downloaded at runtime and git-ignored.
- **Offline-friendly tests.** Each stage ships a CPU-only smoke test using a tiny model or synthetic
  fixtures, so the pipeline can be validated without GPUs or large downloads.
- **Inference first.** We establish strong zero/few-shot baselines and a solid eval harness before
  investing in fine-tuning.
