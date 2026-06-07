# Stage 3 — Python → Java Translation

**Branch:** `stage-3-py2java` · **Phase:** 1 (Foundations)

## Goal
Translate Python to Java to enable enterprise migration — translations must be correct **and**
idiomatic, not merely compilable.

## Models
- `Qwen/Qwen2.5-Coder-7B-Instruct` (default), `deepseek-ai/deepseek-coder-6.7b-instruct`,
  `Salesforce/codegen-350M-multi` (baseline comparison).

## Datasets
- **XLCoST** — aligned Python↔Java pairs.
- **CodeNet** — large-scale multilingual samples.
- **TransCoder** — idiomatic translation pairs.

## Metrics
- **Compilation success** — translated Java compiles with `javac`.
- **Execution correctness** — runs and produces expected output.
- **AST similarity** — structural equivalence (Tree-Sitter Java grammar).
- **CodeBLEU** — translation quality.

## Pipeline (`src/repocoder/stage3_py2java/`)
`data.py` → load Py/Java pairs · `run.py` → translate · `evaluate.py` → compile (`javac`), run, and
score AST similarity + CodeBLEU.

```bash
python -m scripts.run_stage --stage 3 --config configs/stage3.yaml --limit 10
```

## Requirements
- A **JDK** must be on PATH (`javac`, `java`). See [`../03_environments.md`](../03_environments.md).
- Prefer a platform with a persistent JDK (Kaggle/Modal) — see
  [`../01_compute_platforms.md`](../01_compute_platforms.md).

## Notes / pitfalls
- "Compiles" ≠ "correct" ≠ "idiomatic" — gate on all three metrics together.
