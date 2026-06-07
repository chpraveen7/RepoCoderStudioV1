# 03 — Environment Requirements (per stage)

Dependencies are **layered**: `requirements/base.txt` is shared by every stage, and each
`requirements/stageN.txt` adds only what that stage introduces (and `-r`'s the previous stage so a
stage branch installs everything beneath it). This keeps Colab installs minimal and mirrors the
incremental branch structure.

```bash
pip install -r requirements/base.txt
pip install -r requirements/stage3.txt    # pulls in stages 1→3 transitively
pip install -e .                            # make `repocoder` importable
```

## Base (all stages)

| Package | Purpose |
|---|---|
| `torch` | Tensor backend (CUDA on Colab/Kaggle, MPS on Mac) |
| `transformers` | Model + tokenizer loading, chat templates, generation |
| `accelerate` | `device_map="auto"`, multi-GPU placement |
| `datasets` | Hugging Face dataset loading / streaming |
| `huggingface_hub` | Model/dataset downloads, auth for gated assets |
| `pyyaml` | Read `configs/*.yaml` |
| `tqdm`, `numpy`, `pandas` | Progress, arrays, tabular results |
| `pytest` | Smoke tests |

> **bitsandbytes** (4-bit quantization) is listed in base but is **CUDA-only**. On a Mac/MPS it is
> skipped automatically (`model_loader.py` falls back to fp16/fp32). Install it only on GPU hosts.

## Per-stage additions

| Stage | Adds | Why |
|---|---|---|
| **1 — Documentation** | `evaluate`, `rouge_score`, `sacrebleu`, `codebleu`, `bert-score`, `nltk`, `tree_sitter`, `tree_sitter_languages` | ROUGE-L, CodeBLEU, CodeBERTScore; tree-sitter powers CodeBLEU's syntax/dataflow terms |
| **2 — NL→Python** | `human-eval` *(or EvalPlus)*, `evalplus`, `datasets[humaneval/mbpp]` | Synthesis benchmarks + pass@k harness; execution runs in a sandboxed subprocess (stdlib, no extra dep) |
| **3 — Python→Java** | a **JDK** (`javac`/`java` on PATH — system, not pip), `javalang` | Compile + run translated Java; AST similarity reuses tree-sitter (Java grammar) |
| **4 — Repo awareness** | `tree_sitter_languages`, `gitpython`, `networkx`, `rank-bm25` | Multi-language AST parsing, repo cloning, dependency graphs, lexical retrieval baseline |
| **5 — RAG** | `faiss-cpu` (or `faiss-gpu`), `sentence-transformers`, `FlagEmbedding` | Dense index, embeddings, bge-reranker |
| **6 — Agentic** | `langgraph`, `langchain-core` | Multi-agent orchestration graph |

## Platform / system requirements

| Requirement | Colab | Kaggle | Local Mac (M-series) |
|---|---|---|---|
| Python | 3.10/3.11 (preinstalled) | 3.10/3.11 | 3.10+ (`brew`/`conda`) |
| CUDA | provided (T4/L4/A100) | provided (2×T4/P100) | ❌ MPS only |
| `bitsandbytes` 4-bit | ✅ | ✅ | ❌ (auto-skipped) |
| JDK for Stage 3 | `apt-get install -y default-jdk` | preinstalled / installable | `brew install openjdk` |
| Persistent disk | ❌ (mount Drive) | ✅ (`/kaggle/working`, datasets) | ✅ |

## Hugging Face authentication

Some datasets/models are gated (e.g., parts of SWE-Bench, certain Qwen sizes). Authenticate once:

```python
from huggingface_hub import login; login()           # paste a token, or:
# export HF_TOKEN=hf_xxx   (read by huggingface_hub automatically)
```

`common/env.py` sets `HF_HOME` to a cache under the resolved data root so repeated runs don't
re-download weights (and on Colab, points it at mounted Drive when available).

## Reproducibility notes

- Pin versions inside each `requirements/stageN.txt` once a stage is validated (left loose initially
  so Colab resolves compatible CUDA wheels).
- Set seeds via the config (`seed:` key) — `run.py` seeds `torch`, `numpy`, and Python `random`.
- Record the resolved environment per run: `run.py` writes `pip freeze` and the active config into
  the run's output directory.
