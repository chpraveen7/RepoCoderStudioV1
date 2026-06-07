# Stage 4 — Repository Awareness

**Branch:** `stage-4-repo-awareness` · **Phase:** 2 (Integration)

## Goal
Build structural understanding of a repository to enable dependency tracing and navigation — the
structural map that RAG (Stage 5) and the agent (Stage 6) rely on.

## Models / tools
- `deepseek-ai/deepseek-coder-6.7b-instruct` (repo-level, long context).
- **Tree-Sitter** for multi-language AST parsing and chunking.

## Datasets
- **RepoBench** — repository-level retrieval/navigation tasks.
- **SWE-Bench Lite** — bug-fixing/dependency-tracing tasks (subset by default).

## Metrics
- **Retrieval precision / recall** — relevance of retrieved files/symbols.
- **MRR@k** — ranking quality of retrieval.
- **File/Function identification accuracy** — correct location of repo components.

## Pipeline (`src/repocoder/stage4_repo_awareness/`)
`data.py` → clone/load repos · `run.py` → AST-chunk + build index (+ `networkx` dependency graph) ·
`evaluate.py` → retrieval metrics, with a **BM25 lexical baseline** alongside.

```bash
python -m scripts.run_stage --stage 4 --config configs/stage4.yaml --limit 5
```

## Notes / pitfalls
- Chunk by AST node (function/class), not fixed line windows — see
  [`../04_challenges.md`](../04_challenges.md#stage-4--repository-awareness).
- Index must persist across sessions → prefer Kaggle/Lightning over Colab.
