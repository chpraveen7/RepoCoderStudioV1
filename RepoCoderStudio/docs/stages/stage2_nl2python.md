# Stage 2 — Natural Language → Python

**Branch:** `stage-2-nl2python` · **Phase:** 1 (Foundations)

## Goal
Convert natural-language problem statements into executable Python — supporting prototyping and
education. First head-to-head model comparison.

## Models
- `Qwen/Qwen2.5-Coder-1.5B-Instruct` (default; 7B on Colab Pro/Kaggle).
- `deepseek-ai/deepseek-coder-1.3b-instruct` (comparison).

## Datasets
- **HumanEval** — algorithmic problems with reference solutions + unit tests.
- **MBPP** — diverse Python tasks.
- **EvalPlus** — stricter test suites over HumanEval/MBPP.

## Metrics
- **pass@k** — probability a correct solution appears within k samples.
- **Execution accuracy** — candidate runs and passes the task's tests (sandboxed).
- **CodeBLEU** — structural similarity to reference.

## Pipeline (`src/repocoder/stage2_nl2python/`)
`data.py` → load problems (prompt + tests) · `run.py` → sample k completions per problem ·
`evaluate.py` → run tests in a **subprocess sandbox** (`common/metrics/execution.py`), compute
pass@k + CodeBLEU.

```bash
python -m scripts.run_stage --stage 2 --config configs/stage2.yaml --limit 10
```

## Notes / pitfalls
- **Never execute generated code in-process.** Use the timeout-bounded subprocess sandbox.
- Report temperature and k; `pass@1` (greedy) and `pass@10` (sampled) tell different stories.
