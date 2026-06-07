"""Portable Hugging Face model + tokenizer loading.

One function, :func:`load_model`, used by every stage. It hides the platform differences:

- chooses dtype/device sensibly (fp16 on GPU, fp32 on CPU),
- enables 4-bit quantization **only** when the host supports it (CUDA + bitsandbytes),
- sets ``device_map="auto"`` on GPU so large models shard across available memory.

torch/transformers are imported lazily so the rest of the package (env, io, pure-python metrics)
stays usable in lightweight environments and tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .env import get_device, supports_4bit


@dataclass
class LoadedModel:
    """A loaded model + tokenizer plus the resolved settings used to load it."""

    model: Any
    tokenizer: Any
    model_id: str
    device: str
    dtype: str
    quantized: bool


def load_model(
    model_id: str,
    *,
    load_in_4bit: bool = False,
    dtype: str = "auto",
    trust_remote_code: bool = True,
) -> LoadedModel:
    """Load ``model_id`` for causal LM generation, portably across Colab/Kaggle/local.

    Args:
        model_id: Hugging Face repo id (e.g. ``"Qwen/Qwen2.5-Coder-1.5B-Instruct"``).
        load_in_4bit: Request 4-bit quantization. Silently ignored if the host can't support it
            (non-CUDA or bitsandbytes missing) — we fall back to the best available precision.
        dtype: ``"auto"`` | ``"float16"`` | ``"bfloat16"`` | ``"float32"``.
        trust_remote_code: Needed by some code models that ship custom modeling files.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = get_device()
    want_4bit = load_in_4bit and supports_4bit()

    torch_dtype = _resolve_dtype(dtype, device, torch)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        # Many code models have no pad token; reuse EOS so batched generation works.
        tokenizer.pad_token = tokenizer.eos_token
    # Decoder-only causal LMs must be left-padded: with right-padding the model continues
    # generation after the pad tokens, corrupting batched outputs. All stages generate in batches.
    tokenizer.padding_side = "left"

    kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
    if want_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = "auto"
    else:
        kwargs["torch_dtype"] = torch_dtype
        if device == "cuda":
            kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if "device_map" not in kwargs:  # move explicitly when we didn't let accelerate place it
        model = model.to(device)
    model.eval()

    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        model_id=model_id,
        device=device,
        dtype=str(torch_dtype).replace("torch.", ""),
        quantized=want_4bit,
    )


def _resolve_dtype(dtype: str, device: str, torch) -> Any:
    """Map a dtype string to a torch dtype, defaulting to fp16 on GPU and fp32 on CPU."""
    if dtype == "auto":
        return torch.float16 if device in ("cuda", "mps") else torch.float32
    return {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }.get(dtype, torch.float32)
