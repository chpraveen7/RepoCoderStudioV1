# 02 — Model Comparison

This project uses several open code models. **Qwen2.5-Coder is the default/preferred model** for
generation and translation; the others serve as a lightweight baseline (CodeGen, CodeParrot) or as
a repository/long-context specialist (DeepSeek-Coder). This document explains the differences and
*when to pick which*.

> Benchmark numbers below are **approximate, vendor-/paper-reported HumanEval pass@1** for the
> *instruct* variants and move over time. Treat them as ballpark, not ground truth — this repo
> measures them itself in Stage 2 (`evaluate.py`).

## At a glance

| Model | Params | Year / Org | Training focus | Context | Fill-in-Middle | HumanEval (≈) | Instruct? | License |
|---|---|---|---|---|---|---|---|---|
| **CodeGen-350M-Multi** | 350M | 2022, Salesforce | Multi-language GitHub code | 2,048 | No | ~12–15% | No (base) | BSD-3 / Apache |
| **CodeParrot** | 110M / 1.5B | 2021, HuggingFace | **Python-only** GitHub code | 1,024 | No | low (baseline) | No (base) | Apache-2.0 |
| **DeepSeek-Coder** | 1.3B / 6.7B / 33B | 2024, DeepSeek | **Repo-level** code (87% code, 2T tok) | 16,384 | Yes | ~65–79% (6.7B–33B) | Base + Instruct | DeepSeek (commercial OK) |
| **Qwen2.5-Coder** | 0.5–32B | 2024, Alibaba | Multi-language, 5.5T tok | 32,768 (→128K YaRN) | Yes | ~61–92% (1.5B→32B) | Base + Instruct | Apache-2.0 (3B: research) |

## Model-by-model

### CodeGen-350M-Multi (baseline)
A small GPT-style autoregressive model trained on multiple languages. Its value here is
**reproducibility on modest hardware** — it runs on CPU or a free T4, even Drive-free. It is weak at
correctness by modern standards, which is exactly why it's the **Stage 1 documentation baseline**:
it sets a floor that later, stronger models and RAG must beat.

### CodeParrot (baseline / sanity check)
GPT-2 architecture trained **only on Python**. Useful as a synthesis baseline and a sanity reference
on Python tasks, but its tiny context (1,024) and Python-only scope make it unsuitable for
translation or repo work. Used as a comparison point, not a primary model.

### DeepSeek-Coder (repository & long-context specialist)
Pretrained with **repository-level** objectives (files concatenated at the project level) and
**fill-in-the-middle**, with a 16K context. This makes it strong where *surrounding code matters*:
**Stage 4 (repo awareness)** and as a comparison model in **Stages 2–3**. The 1.3B variant fits a
free T4; 6.7B fits in 4-bit; 33B needs a rented A100.

### Qwen2.5-Coder (default for generation & translation)
The strongest open option across our sizes, with:
- A wide size ladder (0.5B → 32B) so we can fit the platform: **1.5B** on free Colab/Kaggle, **7B**
  in 4-bit on a T4/L4, **14B/32B** on rented A100s.
- **Top HumanEval/MBPP** scores among open models in its class, and solid multilingual coverage
  (good for **Python→Java** in Stage 3).
- **32K native context** (extendable), useful when we feed retrieved chunks in **Stage 5 (RAG)**.
- **FIM** support and an instruct chat template that the agentic stage (6) relies on.
- **Apache-2.0** for most sizes — clean for organizational use.

This combination of correctness, context length, multilingual ability, and permissive licensing is
why Qwen2.5-Coder is the project default.

## When to pick which

| If you need… | Pick | Stage |
|---|---|---|
| A cheap, reproducible baseline / floor | **CodeGen-350M-Multi** | 1 |
| A Python-only synthesis sanity check | **CodeParrot** | (comparison) |
| Best NL→code / code→code accuracy on limited compute | **Qwen2.5-Coder 1.5B/7B** | 2, 3, 5, 6 |
| Repo-level / long-context reasoning, FIM completion | **DeepSeek-Coder 6.7B** | 4 (and 2–3 comparison) |
| Maximum accuracy, compute available | **Qwen2.5-Coder 14B/32B** | any |

## Concrete model IDs used in configs

```
Salesforce/codegen-350M-multi
codeparrot/codeparrot-small            # 110M
Qwen/Qwen2.5-Coder-1.5B-Instruct       # free Colab/Kaggle default
Qwen/Qwen2.5-Coder-7B-Instruct         # Colab Pro / 4-bit on T4
deepseek-ai/deepseek-coder-1.3b-instruct
deepseek-ai/deepseek-coder-6.7b-instruct
```

Each `configs/stageN.yaml` picks a default plus a `comparison_models` list so the same harness can
run a head-to-head (e.g., Qwen2.5-Coder vs DeepSeek-Coder in Stage 2).
