# 04 — Challenges & Recommendations (per stage)

The proposal lists seven cross-cutting challenges. This document maps each to the stage(s) where it
bites hardest and gives a concrete mitigation **and** a recommendation you can act on.

## Cross-cutting challenges → stages

| # | Challenge | Hits stages | Mitigation | Recommendation |
|---|---|---|---|---|
| 1 | Execution-environment complexity (running Python/Java safely) | 2, 3, 6 | Sandboxed subprocess with timeouts + resource caps; Docker/Modal for Java | Run untrusted generated code in an isolated process, never in the notebook kernel |
| 2 | Hallucinated APIs / functions | 2, 3, 5, 6 | AST/static validation + RAG grounding | Validate before executing; prefer retrieved real examples over free generation |
| 3 | Dataset coverage limits | all | Augment with CodeNet/RepoBench + synthetic data | Report results per-benchmark; don't over-claim from HumanEval alone |
| 4 | Translation quality (compiles ≠ idiomatic) | 3 | TransCoder idioms + AST similarity + human spot-check | Track compile **and** execution **and** AST similarity together |
| 5 | Repository complexity (deep dependency chains) | 4, 5 | Tree-Sitter AST indexing + FAISS + reranking | Chunk by AST node, not by line window; rerank top-k |
| 6 | Multi-agent error propagation | 6 | Validator agent gates each step | Make the Validator able to *reject and retry*, not just score |
| 7 | Resource constraints (memory) | all | 4-bit quant, gradient accumulation, LoRA, cloud burst | Default to 4-bit inference; reserve full precision for final numbers |

## Per-stage detail

### Stage 1 — Documentation
- **Challenge:** CodeGen-350M produces fluent but shallow docstrings; reference summaries vary in
  style, so surface metrics (ROUGE-L) can mislead.
- **Mitigation:** Pair ROUGE-L with **CodeBERTScore** (semantic) and **CodeBLEU** (syntax/dataflow);
  report all three. Few-shot prompt with one in-domain example.
- **Recommendation:** Treat Stage 1 numbers as a **baseline floor**; the same eval re-run with
  Qwen2.5-Coder + RAG (Stage 5) should beat it — that delta is the real story.

### Stage 2 — NL→Python
- **Challenge:** pass@k requires running model-generated code; a bad sample can hang or harm the host.
  HumanEval/MBPP also under-represent enterprise tasks.
- **Mitigation:** Execute each candidate in a **subprocess with a hard timeout** and captured I/O
  (`common/metrics/execution.py`); never `exec()` in-process. Use EvalPlus for stricter tests.
- **Recommendation:** Report `pass@1` and `pass@10` with sampling temperature noted; compare
  Qwen2.5-Coder vs DeepSeek-Coder head-to-head on identical prompts.

### Stage 3 — Python→Java
- **Challenge:** Output may compile but be non-idiomatic, or be idiomatic but wrong. Java toolchain
  setup is environment-sensitive.
- **Mitigation:** Three-metric gate — **compile success** (`javac`), **execution correctness**
  (run with sample I/O), **AST similarity** (tree-sitter Java grammar) — plus CodeBLEU.
- **Recommendation:** Use a platform with a persistent JDK (Kaggle/Modal); cache compiled artifacts.
  Don't accept "compiles" as success on its own.

### Stage 4 — Repository awareness
- **Challenge:** Large repos with deep dependency chains overwhelm naive retrieval; line-window
  chunking splits functions.
- **Mitigation:** **AST-aware chunking** (function/class nodes via Tree-Sitter) + dependency graph
  (`networkx`); evaluate retrieval with precision/recall and MRR@k.
- **Recommendation:** Keep a **lexical baseline (BM25)** alongside dense retrieval — on code,
  identifiers often make BM25 a strong, cheap competitor worth beating.

### Stage 5 — RAG
- **Challenge:** Irrelevant retrieved context can *hurt* generation; measuring "did RAG help?"
  requires a clean baseline.
- **Mitigation:** Two-stage retrieve→rerank (FAISS + bge-reranker); always compute the **no-RAG
  baseline** on the same items so the gain is attributable.
- **Recommendation:** Report **pass@k improvement** and **CodeBLEU/AST gains vs baseline**, not just
  absolute numbers. Ablate retrieval depth (k).

### Stage 6 — Agentic
- **Challenge:** Errors compound across Planner → Retriever → Generator → Validator; long runs are
  expensive and fragile on Colab.
- **Mitigation:** Validator gates each transition and can trigger a bounded retry; cap steps and log
  every node's I/O for replay.
- **Recommendation:** Run on a persistent platform (Lightning/Modal); checkpoint state between nodes
  so a disconnect doesn't restart the whole workflow.

## Fine-tuning (deferred)

Inference + evaluation come first. Add **LoRA/QLoRA** only where the eval harness shows a clear,
persistent gap that prompting + RAG don't close — most likely **Stage 3 (idiomatic Java)** and
**Stage 1 (house documentation style)**. Use QLoRA (4-bit base + LoRA adapters) with gradient
accumulation to stay within a single 16 GB GPU; burst to a rented A100 for larger bases.
