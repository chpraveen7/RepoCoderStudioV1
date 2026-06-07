"""Generation helpers shared across stages.

Wraps the common patterns — plain completion, chat-templated instruct generation, and
fill-in-the-middle — so each stage's ``run.py`` stays a few lines. Operates on a
:class:`~repocoder.common.model_loader.LoadedModel`.
"""

from __future__ import annotations

from typing import Sequence

from .model_loader import LoadedModel


def generate(
    lm: LoadedModel,
    prompts: Sequence[str],
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.2,
    top_p: float = 0.95,
    do_sample: bool = False,
    num_return_sequences: int = 1,
) -> list[list[str]]:
    """Generate completions for a batch of raw prompts.

    Returns a list (per prompt) of lists (per returned sequence) of decoded *completion* strings —
    the prompt is stripped so callers get only the new text.
    """
    import torch

    tok = lm.tokenizer
    enc = tok(list(prompts), return_tensors="pt", padding=True, truncation=True)
    enc = {k: v.to(lm.model.device) for k, v in enc.items()}

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        num_return_sequences=num_return_sequences,
        pad_token_id=tok.pad_token_id,
    )
    if do_sample:
        gen_kwargs.update(temperature=temperature, top_p=top_p)

    with torch.no_grad():
        out = lm.model.generate(**enc, **gen_kwargs)

    # Strip the prompt tokens from each sequence, then regroup by prompt.
    input_len = enc["input_ids"].shape[1]
    completions = tok.batch_decode(out[:, input_len:], skip_special_tokens=True)

    grouped: list[list[str]] = []
    for i in range(len(prompts)):
        start = i * num_return_sequences
        grouped.append(completions[start : start + num_return_sequences])
    return grouped


def chat(
    lm: LoadedModel,
    messages_batch: Sequence[Sequence[dict]],
    **gen_kwargs,
) -> list[list[str]]:
    """Generate for instruct models using their chat template.

    ``messages_batch`` is a batch of message lists, each like
    ``[{"role": "user", "content": "..."}]``. Falls back to raw concatenation if the tokenizer has
    no chat template (e.g., base CodeGen).
    """
    tok = lm.tokenizer
    prompts: list[str] = []
    for messages in messages_batch:
        if getattr(tok, "chat_template", None):
            prompts.append(
                tok.apply_chat_template(list(messages), tokenize=False, add_generation_prompt=True)
            )
        else:
            prompts.append("\n".join(m["content"] for m in messages))
    return generate(lm, prompts, **gen_kwargs)


def fill_in_middle(lm: LoadedModel, prefix: str, suffix: str, **gen_kwargs) -> str:
    """Fill-in-the-middle completion for models that support FIM (Qwen2.5-Coder, DeepSeek-Coder).

    Uses each family's sentinel tokens when present, else falls back to prefix-only completion.
    """
    tok = lm.tokenizer
    vocab = tok.get_vocab()
    # Qwen2.5-Coder / DeepSeek-Coder FIM sentinels.
    if "<|fim_prefix|>" in vocab:
        prompt = f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>"
    elif "<｜fim▁begin｜>" in vocab:  # DeepSeek style
        prompt = f"<｜fim▁begin｜>{prefix}<｜fim▁hole｜>{suffix}<｜fim▁end｜>"
    else:
        prompt = prefix
    return generate(lm, [prompt], **gen_kwargs)[0][0]
