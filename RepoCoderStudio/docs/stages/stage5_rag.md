# Stage 5 — Retrieval-Augmented Generation (RAG)

**Branch:** `stage-5-rag` · **Phase:** 2 (Integration)

## Goal
Ground generation/translation in retrieved examples to reduce hallucination and improve correctness
over the Stage 2/3 baselines.

## Models / tools
- `Qwen/Qwen2.5-Coder-7B-Instruct` (generator).
- **FAISS** (dense index) + **bge-reranker** (`BAAI/bge-reranker-base`).
- Embeddings: `BAAI/bge-small-en-v1.5` (or a code-tuned embedder).

## Retrieval corpora (from earlier stages)
- HumanEval/MBPP solutions → NL→Python retrieval.
- XLCoST/CodeNet bilingual pairs → Python→Java retrieval.
- Repository files + AST chunks (Stage 4) → contextual retrieval.

## Metrics
- **Retrieval precision** — relevance of retrieved examples.
- **pass@k improvement** — gain over the no-RAG baseline.
- **Execution accuracy** — correctness of augmented outputs.
- **CodeBLEU / AST similarity gains** — translation-quality improvement.

## Pipeline (`src/repocoder/stage5_rag/`)
`data.py` → build/load corpus + FAISS index · `run.py` → retrieve→rerank→augment prompt→generate ·
`evaluate.py` → recompute Stage 2/3 metrics **and the no-RAG baseline** for attributable deltas.

```bash
python -m scripts.run_stage --stage 5 --config configs/stage5.yaml --limit 10
```

## Notes / pitfalls
- Always compute the same items **without** RAG so the gain is attributable.
- Ablate retrieval depth k; irrelevant context can hurt.
