# 01 — Compute Platforms: Google Colab vs Alternatives

You asked specifically about **Google Colab** and whether anything else is better. Short answer:
**Colab is a good place to start (Stages 1–3), but it is the wrong tool for the long, stateful runs
in Stages 4–6.** Use Colab for prototyping and small-model inference, and move to a
persistent-storage platform (Kaggle or Lightning AI) — or a serverless GPU (Modal) — as the work
grows. The code in this repo is written to run on all of them unchanged (`common/env.py` detects
the platform).

## The options at a glance

| Platform | Free GPU | Session limit | Persistent storage | Best for | Watch out for |
|---|---|---|---|---|---|
| **Colab Free** | T4 16 GB (when available) | ~12 h, idle disconnects ~90 min | ❌ (Drive mount only) | Stages 1–3 prototyping, demos | Disconnects lose state; disk wiped each session; GPU not guaranteed |
| **Colab Pro / Pro+** | L4 / A100 40 GB | Longer, background exec (Pro+) | ❌ (Drive) | 7B-class models, longer eval runs | Paid; still no real persistence |
| **Kaggle Notebooks** | 2× T4 16 GB **or** P100 | 9 h/session, **30 h/week** | ✅ Datasets + `/kaggle/working` | Free 7B inference (2×T4), reproducible datasets | Weekly quota; 20 GB working dir |
| **Lightning AI Studios** | Free monthly credits (T4/A10G) | Persistent studio (pause/resume) | ✅ Full filesystem persists | Long Stage 4–6 runs, iterative dev, VS Code | Credits run out; setup learning curve |
| **Modal** | Pay-per-second (no free GPU tier to speak of) | None (serverless) | ✅ Volumes | Batch eval, **Java sandboxing**, parallel pass@k | Costs money; code must be containerized |
| **RunPod / Vast.ai** | Cheap rentals (A100/H100/4090) | None | ✅ Network volumes | Big models (33B), fine-tuning | Hourly billing; bring-your-own setup |
| **Local (M-series Mac)** | MPS (unified memory) | None | ✅ | Small models, dev, the CPU smoke tests | No CUDA; bitsandbytes 4-bit unsupported on MPS; slow for >3B |

## Per-stage recommendation

| Stage | Workload | Recommended | Why |
|---|---|---|---|
| 1 — Documentation | CodeGen-350M, short generations | **Colab Free** | Tiny model, fits T4 easily; fast feedback |
| 2 — NL→Python | Qwen2.5-Coder 1.5B/7B, pass@k (many samples) | **Colab Pro** or **Kaggle (2×T4)** | 7B needs ~16 GB in 4-bit; pass@k is sample-heavy |
| 3 — Python→Java | 7B inference **+ JDK to compile/run Java** | **Kaggle** or **Modal** | Need `javac`/`java` installed and sandboxed execution |
| 4 — Repo awareness | Indexing whole repos, AST parsing, retrieval | **Lightning AI** or **Kaggle** | Stateful index must persist across sessions |
| 5 — RAG | FAISS index + reranker + generation | **Lightning AI** | Persisted index + GPU together; iterative |
| 6 — Agentic | Long multi-step LangGraph runs, many model calls | **Lightning AI** or **Modal** | 12 h Colab limit and disconnects break long agent loops |

## Why Colab struggles past Stage 3

1. **No persistence.** Stages 4–6 build indexes and intermediate artifacts that must survive across
   sessions. On Colab everything outside a mounted Drive is wiped on disconnect.
2. **Session limits & disconnects.** A multi-agent SWE-Bench run can exceed 12 hours; an idle
   disconnect mid-run loses progress.
3. **Disk pressure.** Repo checkouts + model weights + FAISS indexes + JDK quickly exceed Colab's
   ephemeral disk.
4. **Java toolchain.** Stage 3 needs a JDK; installing it every session on Colab is wasteful versus
   a persistent environment or a container image.

## Practical guidance for this repo

- **Default workflow:** prototype each stage's notebook on **Colab**, then run the full evaluation
  on **Kaggle** (free, 2×T4, persistent datasets) or **Lightning AI** (persistent studio).
- **Mount Drive on Colab** to persist outputs:
  ```python
  from google.colab import drive; drive.mount('/content/drive')
  ```
  `common/env.py` will route caches/outputs to Drive when it detects Colab + a mounted path.
- **For Java (Stage 3)** prefer a platform where you control the image (Kaggle has a JDK; Modal lets
  you bake one in) so compilation is reproducible and sandboxed.
- **For big models / fine-tuning (later)**, rent an A100 on RunPod/Vast or use Colab Pro+ A100.

## Bottom line

> Start on **Colab** for Stages 1–3 (Qwen2.5-Coder fits in 4-bit on a T4). Graduate to **Kaggle**
> or **Lightning AI** for Stages 4–6 where persistence and long runtimes matter, and use **Modal**
> when you need reproducible Java sandboxing or parallel batch evaluation. The code is
> platform-agnostic, so switching is just a matter of where you launch the notebook.
